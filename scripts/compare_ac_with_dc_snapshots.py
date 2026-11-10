#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Relax the approximation of DC approx by adding sampled voltages that vary by percent from nominal value,
effectively changing transmission capacities, and regarding sinius in coupling.
This compared the inital step after the triggering of a failure to see if that leads more often
to a cascade and how often.

The workflow is split into three separable, individually-saved stages so that a
generated AC system can be reproduced and its power flows re-run later:

1. ``setup_system_choosePVnPQ_distribtueQ`` - assign one Slack bus, a fraction of
   PV buses (both ranked by conventional generation *capacity*) and treat the rest
   as PQ buses. A power factor is sampled per PQ bus and used to derive reactive
   power for the loads. The resulting configuration is saved for reproducibility.
2. ``run_dc_power_flow`` - pin the optimized dispatch and run the DC (linear) power
   flow, saving the line flows.
3. ``run_ac_power_flow`` - apply the saved configuration, run the AC (Newton-Raphson)
   power flow per snapshot and save the results.

``compare_line_loadings`` then compares the DC and AC line loadings.
"""

import gzip
import logging
import multiprocessing as mp
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import pypsa

from tqdm import tqdm
from loguru import logger

from utils import send_mattermost_messages
from utils.data_handling import load_pypsa_network, get_co2_levels
from utils.config import path_to_ac_dc_comparison, mattermost_url

# Silence PyPSA's per-sub-network convergence chatter: we solve every sub-network
# but only report convergence for the one under analysis (see run_ac_power_flow).
logging.getLogger("pypsa.network.power_flow").setLevel(logging.ERROR)


# Conventional (dispatchable, inertia-carrying) carriers, mirroring
# scripts/calc_pre_outage_data.py. Used to rank buses for Slack/PV selection.
CONVENTIONAL_CARRIERS = [
    "nuclear",
    "oil",
    "OCGT",
    "CCGT",
    "coal",
    "lignite",
    "geothermal",
    "biomass",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _select_snapshots(n: pypsa.Network, snapshots=None) -> pd.Index:
    """Resolve a snapshot selection given by positional index or by date.

    Args:
        n (pypsa.Network): Network whose snapshots are selected from.
        snapshots: One of
            - ``None``: use all snapshots of the network.
            - a list of ``int``: positional indices into ``n.snapshots``.
            - a list of date-like values (str / ``pd.Timestamp``): snapshot labels.
            A scalar of any of these types is also accepted.

    Returns:
        pd.Index: The selected snapshot labels (subset of ``n.snapshots``).
    """
    if snapshots is None:
        sns = n.snapshots
    else:
        if isinstance(snapshots, (int, np.integer, str, pd.Timestamp)):
            snapshots = [snapshots]
        snapshots = list(snapshots)

        if all(isinstance(s, (int, np.integer)) for s in snapshots):
            sns = n.snapshots[[int(s) for s in snapshots]]
        else:
            sns = pd.DatetimeIndex([pd.Timestamp(s) for s in snapshots])
            missing = sns.difference(n.snapshots)
            if len(missing) > 0:
                raise ValueError(
                    f"Snapshots not found in the network: {list(missing)}"
                )

    logger.info(
        "Selected {} snapshot(s) ({} .. {}).",
        len(sns),
        sns[0],
        sns[-1],
    )
    return sns


def _pin_dispatch(n: pypsa.Network) -> None:
    """Pin the optimized dispatch into the ``*_set`` fields.

    ``lpf``/``pf`` read the dispatch from ``p_set`` (not from the solved ``p``),
    so the optimized operating point must be copied over before running a power
    flow. The solved dispatch is copied as a whole, **label-aligned** to the
    component index: assigning ``.loc[sns] = ....loc[sns]`` scrambles the values
    across components because ``*_t.p`` and a reindexed ``*_t.p_set`` need not
    share the same column order.

    The artificial ``load`` / ``art_load`` generators (load shedding and
    numerical slack from the SCLOPF preparation) are part of the balanced
    operating point and are kept; with this pinning ``lpf`` reproduces the stored
    SCLOPF line flows exactly.

    Args:
        n (pypsa.Network): Network to modify in place.
    """
    n.generators_t.p_set = n.generators_t.p.reindex(
        columns=n.generators.index, fill_value=0.0
    )
    n.storage_units_t.p_set = n.storage_units_t.p.reindex(
        columns=n.storage_units.index, fill_value=0.0
    )
    n.links_t.p_set = n.links_t.p0.reindex(columns=n.links.index, fill_value=0.0)


# A ``tag`` suffix (e.g. "_date0_pf3") distinguishes output files of different
# campaign runs (date realizations x power-factor seeds); "" keeps the plain names.
def _config_path(co2lvl: float, n_nodes: int, tag: str = "") -> str:
    return path_to_ac_dc_comparison + f"/ac_system_config_Co2L{co2lvl}_n{n_nodes}{tag}.pklz"


def _dc_results_path(co2lvl: float, n_nodes: int, tag: str = "") -> str:
    return path_to_ac_dc_comparison + f"/dc_powerflow_Co2L{co2lvl}_n{n_nodes}{tag}.h5"


def _ac_results_path(co2lvl: float, n_nodes: int, tag: str = "") -> str:
    return path_to_ac_dc_comparison + f"/ac_powerflow_Co2L{co2lvl}_n{n_nodes}{tag}.h5"


def _summary_results_path(co2lvl: float, n_nodes: int, tag: str = "") -> str:
    return path_to_ac_dc_comparison + f"/comparison_summary_Co2L{co2lvl}_n{n_nodes}{tag}.h5"


# Default HDF5 compression for the (float-heavy) result frames, matching the
# convention in scripts/plots/plot_rocof_statistics.py.
_COMPLIB = "blosc:zstd"
_COMPLEVEL = 9


def _write_dc_results(dc_p0: pd.DataFrame, co2lvl: float, n_nodes: int, tag: str = "") -> str:
    """Write DC line flows to a compressed HDF5 (key ``p0``). Returns the path."""
    path = _dc_results_path(co2lvl, n_nodes, tag)
    with pd.HDFStore(
        path, mode="w", complib=_COMPLIB, complevel=_COMPLEVEL
    ) as store:
        store.put("p0", dc_p0)
    logger.info("Saved DC results to {}", path)
    return path


def _write_ac_results(
    p0: pd.DataFrame,
    q0: pd.DataFrame,
    v_mag: pd.DataFrame,
    v_ang: pd.DataFrame,
    converged: pd.Series,
    co2lvl: float,
    n_nodes: int,
    tag: str = "",
) -> str:
    """Write AC flows, voltages and convergence flags to a compressed HDF5. Returns path."""
    path = _ac_results_path(co2lvl, n_nodes, tag)
    with pd.HDFStore(
        path, mode="w", complib=_COMPLIB, complevel=_COMPLEVEL
    ) as store:
        store.put("p0", p0)
        store.put("q0", q0)
        store.put("v_mag_pu", v_mag)
        store.put("v_ang", v_ang)
        store.put("converged", converged.to_frame())
    logger.info("Saved AC results to {}", path)
    return path


def save_ac_system_config(config: dict, co2lvl: float, n_nodes: int, tag: str = "") -> str:
    """Save the AC system configuration (bus-type assignment + power factors).

    Args:
        config (dict): Configuration returned by ``setup_system_choosePVnPQ_distribtueQ``.
        co2lvl (float): CO2 constraint (used in the filename).
        n_nodes (int): Aggregation level of nodes (used in the filename).
        tag (str): Optional filename suffix distinguishing campaign runs.

    Returns:
        str: Path to the written file.
    """
    path = _config_path(co2lvl, n_nodes, tag)
    with gzip.open(path, "wb") as fh:
        pickle.dump(config, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved AC system configuration to {}", path)
    return path


def load_ac_system_config(co2lvl: float, n_nodes: int, tag: str = "") -> dict:
    """Load a previously saved AC system configuration.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        tag (str): Optional filename suffix distinguishing campaign runs.

    Returns:
        dict: The stored configuration.
    """
    path = _config_path(co2lvl, n_nodes, tag)
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _apply_system_config(n: pypsa.Network, config: dict, sns: pd.Index) -> None:
    """Apply a system configuration to a network: generator controls + load Q.

    The bus-type assignment is static (identical for every snapshot); the reactive
    power of each load is derived per snapshot from its (time-varying) active power
    and the static, per-bus sampled power factor:  ``Q = P * tan(arccos(pf))``
    (inductive / lagging convention, so loads consume reactive power).

    Args:
        n (pypsa.Network): Network to modify in place.
        config (dict): Configuration from ``setup_system_choosePVnPQ_distribtueQ``.
        sns (pd.Index): Snapshots for which ``loads_t.q_set`` is populated.
    """
    # Static generator control assignment (Slack / PV / PQ).
    n.generators["control"] = (
        config["generator_control"].reindex(n.generators.index).fillna("PQ")
    )

    # Per-load reactive power from the static per-bus power factors.
    pf_bus = config["power_factors"]
    load_pf = n.loads.bus.map(pf_bus)
    load_tan = np.tan(np.arccos(load_pf))
    # Loads that do not sit on a PQ bus (i.e. on the Slack/PV buses) get no Q.
    load_tan = load_tan.fillna(0.0)

    p_set_t = n.get_switchable_as_dense("Load", "p_set", sns)
    n.loads_t["q_set"] = p_set_t.mul(load_tan, axis=1)


def _line_loading(flow: pd.DataFrame, s_ref: pd.Series) -> pd.DataFrame:
    """Line loading (|flow| / thermal reference) as a snapshots x lines frame."""
    return flow.abs().div(s_ref, axis=1)


def _sub_network_buses_lines(n: pypsa.Network, sub_network: str):
    """Return the (buses, lines) indices belonging to one sub_network.

    Args:
        n (pypsa.Network): Network (topology already determined by the loader).
        sub_network (str): Sub-network id (``"0"`` is the main Continental
            European AC grid).

    Returns:
        tuple[pd.Index, pd.Index]: bus indices and line indices of that sub_network.
    """
    sub = str(sub_network)
    if sub not in n.sub_networks.index.astype(str).tolist():
        raise ValueError(
            f"Sub-network {sub!r} not found; available: "
            f"{list(n.sub_networks.index)}"
        )
    buses = n.buses.index[n.buses.sub_network.astype(str) == sub]
    lines = n.lines.index[n.lines.sub_network.astype(str) == sub]
    return buses, lines


# --------------------------------------------------------------------------- #
# Stage 1: generate the AC-solvable system
# --------------------------------------------------------------------------- #
def setup_system_choosePVnPQ_distribtueQ(
    n: pypsa.Network,
    co2lvl: float,
    n_nodes: int,
    fraction_PV_nodes: float = 0.3,
    power_factor_mid: float = 0.95,
    power_factor_range: float = 0.05,
    seed: int = 0,
    sub_network: str = "0",
    conventional_carriers=CONVENTIONAL_CARRIERS,
    save: bool = True,
    tag: str = "",
) -> dict:
    """Set up a system for running AC simulations by assigning PV and PQ nodes and setting Q.

    The analysis is restricted to a single connected ``sub_network`` (default
    ``"0"``, the main Continental European AC grid). Its buses are ranked by
    installed *conventional generation capacity* (``p_nom``). The single
    highest-capacity bus becomes the Slack, the next
    ``round(fraction_PV_nodes * n_sub_buses)`` buses become PV, and the rest are
    PQ. A power factor is sampled once per PQ bus, uniformly in
    ``[power_factor_mid - power_factor_range/2, power_factor_mid + power_factor_range/2]``
    and clipped to ``[0, 1]``; it is later used to derive the reactive power of the
    loads on that bus. The assignment is static (reused for every snapshot).

    Args:
        n (pypsa.Network): Network with the optimized dispatch.
        co2lvl (float): CO2 constraint (used for saving).
        n_nodes (int): Aggregation level of nodes (used for saving).
        fraction_PV_nodes (float): Fraction of the sub-network's buses to turn
            into PV nodes.
        power_factor_mid (float): Centre of the sampled power-factor interval.
        power_factor_range (float): Width of the sampled power-factor interval.
        seed (int): Seed for the (reproducible) power-factor sampling.
        sub_network (str): Sub-network to analyse (``"0"`` = main AC grid).
        conventional_carriers (list): Carriers counted as conventional generation.
        save (bool): If ``True`` the configuration is written to disk.

    Returns:
        dict: Configuration with keys ``generator_control`` (pd.Series indexed by
        generator, values ``"PQ"``/``"PV"``/``"Slack"``), ``power_factors``
        (pd.Series indexed by PQ bus) and ``params`` (metadata dict).
    """
    sub_buses, _ = _sub_network_buses_lines(n, sub_network)

    conv_gens = n.generators[
        n.generators.carrier.isin(conventional_carriers)
        & n.generators.bus.isin(sub_buses)
    ]
    if conv_gens.empty:
        raise ValueError(
            f"No conventional generators in sub_network {sub_network!r} for "
            f"carriers {conventional_carriers}."
        )

    # Rank buses by conventional capacity (descending), keep only buses that host
    # conventional generation (only those can carry a Slack/PV control).
    cap = conv_gens.groupby("bus").p_nom.sum().sort_values(ascending=False)
    cap = cap[cap > 0]

    n_pv_requested = int(round(fraction_PV_nodes * len(sub_buses)))
    n_pv = min(n_pv_requested, len(cap) - 1)
    if n_pv < n_pv_requested:
        logger.warning(
            "Requested {} PV buses but only {} conventional buses available in "
            "sub_network {}; capping to {}.",
            n_pv_requested,
            len(cap),
            sub_network,
            n_pv,
        )

    slack_bus = cap.index[0]
    pv_buses = cap.index[1 : 1 + n_pv]

    # Generator control: everything PQ, one Slack generator at the slack bus
    # (the largest one there so it can carry the balance), PV on the PV buses.
    control = pd.Series("PQ", index=n.generators.index)
    slack_gen = conv_gens[conv_gens.bus == slack_bus].p_nom.idxmax()
    control[slack_gen] = "Slack"
    pv_gens = conv_gens.index[conv_gens.bus.isin(pv_buses)]
    control[pv_gens] = "PV"

    # PQ buses = sub-network buses that are neither Slack nor PV.
    non_pq = pd.Index([slack_bus]).append(pd.Index(pv_buses))
    pq_buses = sub_buses.difference(non_pq)

    # Sample one power factor per PQ bus, clipped to a realistic [0, 1] magnitude.
    rng = np.random.default_rng(seed)
    lo = power_factor_mid - power_factor_range / 2.0
    hi = power_factor_mid + power_factor_range / 2.0
    power_factors = pd.Series(
        np.clip(rng.uniform(lo, hi, size=len(pq_buses)), 0.0, 1.0),
        index=pq_buses,
        name="power_factor",
    )

    logger.info(
        "System setup (sub_network {}): 1 Slack bus ({}), {} PV buses, {} PQ buses.",
        sub_network,
        slack_bus,
        len(pv_buses),
        len(pq_buses),
    )

    config = {
        "generator_control": control,
        "power_factors": power_factors,
        "params": {
            "co2lvl": co2lvl,
            "n_nodes": n_nodes,
            "sub_network": str(sub_network),
            "fraction_PV_nodes": fraction_PV_nodes,
            "power_factor_mid": power_factor_mid,
            "power_factor_range": power_factor_range,
            "seed": seed,
            "slack_bus": slack_bus,
            "n_pv_buses": len(pv_buses),
            "n_pq_buses": len(pq_buses),
        },
    }

    if save:
        save_ac_system_config(config, co2lvl, n_nodes, tag)

    return config


# --------------------------------------------------------------------------- #
# Stage 2: DC power flow
# --------------------------------------------------------------------------- #
def run_dc_power_flow(
    co2lvl: float = 0.2,
    n_nodes: int = 600,
    snapshots=None,
    sub_network: str = "0",
    use_sclopf: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Run the DC (linear) power flow for the pinned dispatch and save the flows.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        snapshots: Snapshot selection (index or date), see ``_select_snapshots``.
        sub_network (str): Sub-network to solve (``"0"`` = main AC grid).
        use_sclopf (bool): Passed to ``load_pypsa_network``.
        save (bool): If ``True`` the DC line flows are written to disk.

    Returns:
        pd.DataFrame: DC line active-power flows (snapshots x lines) for the
        chosen sub-network.
    """
    logger.info(
        "Running DC power flow (Co2L={}, n={}, sub_network={}).",
        co2lvl,
        n_nodes,
        sub_network,
    )
    n = load_pypsa_network(co2lvl, n_nodes, use_sclopf=use_sclopf)
    sns = _select_snapshots(n, snapshots)
    _, sub_lines = _sub_network_buses_lines(n, sub_network)

    _pin_dispatch(n)
    # Network-level lpf solves every sub-network independently; we keep only the
    # requested one. (PyPSA 0.35's SubNetwork.lpf is broken when called directly.)
    n.lpf(snapshots=sns)

    dc_p0 = n.lines_t.p0.loc[sns, sub_lines].copy()

    loading = _line_loading(dc_p0, n.lines.s_nom.reindex(sub_lines))
    logger.info(
        "DC: max line loading = {:.3f}, overloaded (line,snapshot) pairs = {}.",
        loading.max().max(),
        int((loading > 1.0).sum().sum()),
    )

    if save:
        _write_dc_results(dc_p0, co2lvl, n_nodes)

    return dc_p0


# --------------------------------------------------------------------------- #
# Stage 3: AC power flow
# --------------------------------------------------------------------------- #
def run_ac_power_flow(
    co2lvl: float = 0.2,
    n_nodes: int = 600,
    config: dict = None,
    snapshots=None,
    sub_network: str = "0",
    use_sclopf: bool = True,
    x_tol: float = 1e-6,
    save: bool = True,
) -> dict:
    """Apply a system configuration and run the AC power flow per snapshot.

    Only the chosen ``sub_network`` is solved (via ``SubNetwork.pf``). The
    per-snapshot loop gives a progress bar and isolates snapshots that fail to
    converge. Each snapshot is seeded with its DC angle solution (``lpf`` before
    ``pf(use_seed=True)``), which greatly helps convergence.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        config (dict): System configuration; loaded from disk if ``None``.
        snapshots: Snapshot selection (index or date).
        sub_network (str): Sub-network to solve (``"0"`` = main AC grid).
        use_sclopf (bool): Passed to ``load_pypsa_network``.
        x_tol (float): Newton-Raphson tolerance for the AC power flow.
        save (bool): If ``True`` the AC results are written to disk.

    Returns:
        dict: Results with keys ``p0``, ``q0`` (snapshots x lines),
        ``v_mag_pu``, ``v_ang`` (snapshots x buses) and ``converged``
        (pd.Series indexed by snapshot). Non-converged snapshots are masked to
        ``NaN`` (diverged values are garbage).
    """
    logger.info(
        "Running AC power flow (Co2L={}, n={}, sub_network={}).",
        co2lvl,
        n_nodes,
        sub_network,
    )
    n = load_pypsa_network(co2lvl, n_nodes, use_sclopf=use_sclopf)
    sns = _select_snapshots(n, snapshots)
    sub_buses, sub_lines = _sub_network_buses_lines(n, sub_network)

    if config is None:
        config = load_ac_system_config(co2lvl, n_nodes)

    _pin_dispatch(n)
    _apply_system_config(n, config, sns)
    sub = str(sub_network)

    converged = {}
    for now in tqdm(sns, desc="AC power flow", unit="snapshot"):
        # Network-level lpf/pf solve each sub-network independently; we read only
        # the requested one. (PyPSA 0.35's SubNetwork.pf is broken standalone.)
        n.lpf(snapshots=[now])  # seed voltage angles with the DC solution
        res = n.pf(snapshots=[now], use_seed=True, x_tol=x_tol)
        conv_row = res["converged"].loc[now]
        conv_row.index = conv_row.index.astype(str)
        converged[now] = bool(conv_row.get(sub, False))
        if not converged[now]:
            logger.warning("AC power flow did not converge for snapshot {}.", now)

    # After the loop every selected snapshot row holds its AC solution.
    ac_p0 = n.lines_t.p0.loc[sns, sub_lines].copy()
    ac_q0 = n.lines_t.q0.loc[sns, sub_lines].copy()
    v_mag = n.buses_t.v_mag_pu.loc[sns, sub_buses].copy()
    v_ang = n.buses_t.v_ang.loc[sns, sub_buses].copy()
    converged_s = pd.Series(converged, name="converged").reindex(sns)

    # Mask non-converged snapshots (diverged values are garbage, e.g. ~1e150).
    bad_sns = converged_s.index[~converged_s.fillna(False)]
    for frame in (ac_p0, ac_q0, v_mag, v_ang):
        frame.loc[bad_sns] = np.nan

    logger.info(
        "AC: {}/{} snapshots converged.",
        int(converged_s.sum()),
        len(converged_s),
    )

    results = {
        "p0": ac_p0,
        "q0": ac_q0,
        "v_mag_pu": v_mag,
        "v_ang": v_ang,
        "converged": converged_s,
    }

    if save:
        _write_ac_results(ac_p0, ac_q0, v_mag, v_ang, converged_s, co2lvl, n_nodes)

    return results


# --------------------------------------------------------------------------- #
# Stage 4: comparison
# --------------------------------------------------------------------------- #
def compare_line_loadings(
    co2lvl: float = 0.2,
    n_nodes: int = 600,
    use_sclopf: bool = True,
    tag: str = "",
) -> pd.DataFrame:
    """Compare DC vs AC line loadings from the saved power-flow results.

    DC loading uses ``|P| / s_nom``; AC loading uses the apparent power
    ``sqrt(P^2 + Q^2) / s_nom``. Only snapshots that converged in the AC run are
    considered.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        use_sclopf (bool): Passed to ``load_pypsa_network``.
        tag (str): Optional filename suffix distinguishing campaign runs.

    Returns:
        pd.DataFrame: Per-line summary with mean/max DC and AC loading and the
        number of snapshots each line is overloaded (>1) under DC and AC.
    """
    dc_p0 = pd.read_hdf(_dc_results_path(co2lvl, n_nodes, tag), "p0")
    ac_p0 = pd.read_hdf(_ac_results_path(co2lvl, n_nodes, tag), "p0")
    ac_q0 = pd.read_hdf(_ac_results_path(co2lvl, n_nodes, tag), "q0")
    converged = pd.read_hdf(_ac_results_path(co2lvl, n_nodes, tag), "converged")[
        "converged"
    ].astype(bool)

    n = load_pypsa_network(co2lvl, n_nodes, use_sclopf=use_sclopf)
    s_ref = n.lines.s_nom

    # Restrict to snapshots that converged in AC and to lines present in both runs
    # (the DC/AC result files already hold only the analysed sub-network's lines).
    ok_sns = converged.index[converged.fillna(False)]
    lines = dc_p0.columns.intersection(ac_p0.columns)
    dc_p0 = dc_p0.loc[ok_sns, lines]
    ac_p0 = ac_p0.loc[ok_sns, lines]
    ac_q0 = ac_q0.loc[ok_sns, lines]
    s_ref = s_ref.reindex(lines)

    dc_loading = _line_loading(dc_p0, s_ref)
    ac_loading = np.sqrt(ac_p0**2 + ac_q0**2).div(s_ref, axis=1)

    logger.info(
        "Comparison over {} converged snapshots: "
        "overloaded (line,snapshot) pairs DC={} vs AC={}.",
        len(ok_sns),
        int((dc_loading > 1.0).sum().sum()),
        int((ac_loading > 1.0).sum().sum()),
    )

    summary = pd.DataFrame(
        {
            "dc_loading_mean": dc_loading.mean(),
            "dc_loading_max": dc_loading.max(),
            "ac_loading_mean": ac_loading.mean(),
            "ac_loading_max": ac_loading.max(),
            "dc_overloaded_snapshots": (dc_loading > 1.0).sum(),
            "ac_overloaded_snapshots": (ac_loading > 1.0).sum(),
        }
    )
    summary["max_loading_diff"] = (
        summary["ac_loading_max"] - summary["dc_loading_max"]
    )

    return summary


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def solve_power_flow_dc_vs_ac(
    co2_lvl: float = 0.2,
    n_nodes: int = 600,
    snapshots=None,
    fraction_PV_nodes: float = 0.3,
    power_factor_mid: float = 0.95,
    power_factor_range: float = 0.05,
    seed: int = 0,
    sub_network: str = "0",
    use_sclopf: bool = True,
) -> pd.DataFrame:
    """Outer wrapper: generate the AC system, solve DC and AC, then compare.

    The three stages remain independently callable and each saves its own output;
    this convenience wrapper just runs them in sequence for one ``sub_network``.

    Args:
        co2_lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        snapshots: Snapshot selection (index or date).
        fraction_PV_nodes (float): Fraction of the sub-network's buses set to PV.
        power_factor_mid (float): Centre of the sampled power-factor interval.
        power_factor_range (float): Width of the sampled power-factor interval.
        seed (int): Seed for the power-factor sampling.
        sub_network (str): Sub-network to analyse (``"0"`` = main AC grid).
        use_sclopf (bool): Passed to ``load_pypsa_network``.

    Returns:
        pd.DataFrame: The DC vs AC line-loading comparison summary.
    """
    n = load_pypsa_network(co2_lvl, n_nodes, use_sclopf=use_sclopf)
    setup_system_choosePVnPQ_distribtueQ(
        n,
        co2lvl=co2_lvl,
        n_nodes=n_nodes,
        fraction_PV_nodes=fraction_PV_nodes,
        power_factor_mid=power_factor_mid,
        power_factor_range=power_factor_range,
        seed=seed,
        sub_network=sub_network,
    )

    run_dc_power_flow(
        co2_lvl, n_nodes, snapshots=snapshots, sub_network=sub_network,
        use_sclopf=use_sclopf,
    )
    run_ac_power_flow(
        co2_lvl, n_nodes, snapshots=snapshots, sub_network=sub_network,
        use_sclopf=use_sclopf,
    )

    return compare_line_loadings(co2_lvl, n_nodes, use_sclopf=use_sclopf)


# --------------------------------------------------------------------------- #
# Parallel grid over CO2 levels x snapshots
# --------------------------------------------------------------------------- #
# Read-only data shared with the worker processes. On fork platforms
# (Linux/macOS) it is populated in the PARENT before the pool is forked, so
# workers inherit the (small) per-CO2 configs via copy-on-write; on spawn
# platforms it is handed to each worker once via ``_init_grid_worker``.
_WORKER_STATE = {}


def _init_grid_worker(state):
    """Pool initializer for the 'spawn' start method: receive shared read-only data."""
    _WORKER_STATE.update(state)


def _make_pool_context():
    """Return ``(ctx, pool_kwargs)`` for a worker pool that shares ``_WORKER_STATE``.

    Mirrors ``scripts/plots/plot_rocof_statistics.py``: fork platforms inherit
    ``_WORKER_STATE`` via copy-on-write (nothing passed through the initializer);
    spawn platforms snapshot the current state into each worker once. Call this
    AFTER ``_WORKER_STATE`` is populated.
    """
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork"), {}
    return (
        mp.get_context("spawn"),
        dict(initializer=_init_grid_worker, initargs=(dict(_WORKER_STATE),)),
    )


@lru_cache(maxsize=1)
def _load_pinned_network(co2lvl: float, n_nodes: int, use_sclopf: bool):
    """Load a network with dispatch pinned, cached one-per-worker.

    ``maxsize=1`` bounds each worker process to a single network in memory (so the
    whole run holds at most ``n_processes`` copies). Tasks are dispatched grouped by
    CO2 level so a worker mostly sees contiguous same-CO2 tasks and rarely reloads.
    """
    n = load_pypsa_network(co2lvl, n_nodes, use_sclopf=use_sclopf)
    _pin_dispatch(n)
    return n


def _solve_co2_snapshot(task: tuple) -> dict:
    """Solve DC and AC for one (CO2 level, snapshot) pair. Body of the flat grid.

    Returns the single-snapshot slices (cheap to pickle back to the parent). The
    network is fetched from the per-worker cache; only its copy-on-write pages are
    touched when ``lpf``/``pf`` write the time series.
    """
    co2lvl, snapshot, n_nodes, sub_network, use_sclopf, x_tol = task
    n = _load_pinned_network(co2lvl, n_nodes, use_sclopf)
    sub_buses, sub_lines = _sub_network_buses_lines(n, sub_network)
    config = _WORKER_STATE["configs"][co2lvl]

    _apply_system_config(n, config, pd.Index([snapshot]))
    n.lpf(snapshots=[snapshot])  # DC (also seeds voltage angles for AC)
    dc_p0 = n.lines_t.p0.loc[snapshot, sub_lines].copy()
    res = n.pf(snapshots=[snapshot], use_seed=True, x_tol=x_tol)
    conv_row = res["converged"].loc[snapshot]
    conv_row.index = conv_row.index.astype(str)

    return dict(
        co2lvl=co2lvl,
        snapshot=snapshot,
        dc_p0=dc_p0,
        ac_p0=n.lines_t.p0.loc[snapshot, sub_lines].copy(),
        ac_q0=n.lines_t.q0.loc[snapshot, sub_lines].copy(),
        v_mag=n.buses_t.v_mag_pu.loc[snapshot, sub_buses].copy(),
        v_ang=n.buses_t.v_ang.loc[snapshot, sub_buses].copy(),
        converged=bool(conv_row.get(str(sub_network), False)),
    )


def run_comparison_grid(
    co2_lvl_tup=None,
    n_nodes: int = 600,
    snapshots=None,
    n_processes: int = 1,
    fraction_PV_nodes: float = 0.3,
    power_factor_mid: float = 0.95,
    power_factor_range: float = 0.05,
    seed: int = 0,
    sub_network: str = "0",
    use_sclopf: bool = True,
    x_tol: float = 1e-6,
    overwrite: bool = True,
    notify: bool = True,
    tag: str = "",
) -> dict:
    """Run the DC/AC comparison over a grid of CO2 levels and snapshots in parallel.

    Every ``(co2_level, snapshot)`` pair is an independent task. With
    ``n_processes > 1`` the tasks are spread over a multiprocessing pool; each worker
    loads the network for its CO2 level, cached one-per-worker, so at most
    ``n_processes`` networks live in memory at once. With ``n_processes == 1`` a plain
    serial loop is used (identical results, easy to debug).

    For each CO2 level the DC flows, AC results and the DC-vs-AC comparison summary are
    saved (one file each, per level). If ``mattermost_url`` is set in ``utils.config`` a
    completion message is posted.

    Args:
        co2_lvl_tup: CO2 levels to run; ``None`` uses all available (``get_co2_levels``).
        n_nodes (int): Aggregation level of nodes.
        snapshots: Snapshot selection (index or date), applied to every CO2 level.
        n_processes (int): Number of worker processes (``1`` = sequential).
        fraction_PV_nodes (float): Fraction of the sub-network's buses set to PV.
        power_factor_mid (float): Centre of the sampled power-factor interval.
        power_factor_range (float): Width of the sampled power-factor interval.
        seed (int): Seed for the power-factor sampling.
        sub_network (str): Sub-network to analyse (``"0"`` = main AC grid).
        use_sclopf (bool): Passed to ``load_pypsa_network``.
        x_tol (float): Newton-Raphson tolerance for the AC power flow.
        overwrite (bool): If ``False``, skip CO2 levels whose AC result file exists.
        notify (bool): If ``True``, post a Mattermost message when done (if configured).
        tag (str): Optional filename suffix distinguishing this run's outputs.

    Returns:
        dict: ``{co2_level: comparison_summary_DataFrame}`` for the levels that ran.
    """
    if co2_lvl_tup is None:
        co2_levels = [float(c) for c in get_co2_levels(n_nodes)]
    else:
        co2_levels = list(co2_lvl_tup)

    if not overwrite:
        kept = [c for c in co2_levels if not os.path.exists(_ac_results_path(c, n_nodes, tag))]
        for c in set(co2_levels) - set(kept):
            logger.info("Skipping Co2L={} — AC result file exists.", c)
        co2_levels = kept
    if not co2_levels:
        logger.info("Nothing to do — all requested CO2 levels already computed.")
        return {}

    # --- Config phase (parent, one network in memory at a time, released after). ---
    configs, snapshots_by_co2, sub_meta = {}, {}, {}
    for co2 in co2_levels:
        n = load_pypsa_network(co2, n_nodes, use_sclopf=use_sclopf)
        configs[co2] = setup_system_choosePVnPQ_distribtueQ(
            n, co2lvl=co2, n_nodes=n_nodes,
            fraction_PV_nodes=fraction_PV_nodes, power_factor_mid=power_factor_mid,
            power_factor_range=power_factor_range, seed=seed, sub_network=sub_network,
            save=True, tag=tag,
        )
        sns = _select_snapshots(n, snapshots)
        snapshots_by_co2[co2] = sns
        sub_buses, sub_lines = _sub_network_buses_lines(n, sub_network)
        sub_meta[co2] = (sub_buses, sub_lines)
        del n

    # Flat task list, grouped by CO2 so each worker sees contiguous same-CO2 tasks.
    tasks = [
        (co2, snap, n_nodes, sub_network, use_sclopf, x_tol)
        for co2 in co2_levels
        for snap in snapshots_by_co2[co2]
    ]
    logger.info(
        "Solving {} (CO2, snapshot) tasks over {} CO2 level(s) with n_processes={}.",
        len(tasks), len(co2_levels), n_processes,
    )

    # --- Solve phase (flat, parallel or serial). ---
    _WORKER_STATE["configs"] = configs
    if n_processes == 1:
        outputs = [_solve_co2_snapshot(t) for t in tqdm(tasks, desc="DC/AC grid")]
    else:
        _load_pinned_network.cache_clear()
        chunksize = max(1, len(tasks) // (n_processes * 4))
        ctx, pool_kwargs = _make_pool_context()
        with ctx.Pool(processes=n_processes, **pool_kwargs) as pool:
            outputs = list(
                tqdm(
                    pool.imap_unordered(_solve_co2_snapshot, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="DC/AC grid",
                )
            )

    # --- Assemble, save and compare, per CO2 level. ---
    summaries = {}
    for co2 in co2_levels:
        outs = [o for o in outputs if o["co2lvl"] == co2]
        sns = snapshots_by_co2[co2]
        sub_buses, sub_lines = sub_meta[co2]

        dc_p0 = pd.DataFrame(
            {o["snapshot"]: o["dc_p0"] for o in outs}
        ).T.reindex(index=sns, columns=sub_lines)
        ac_p0 = pd.DataFrame({o["snapshot"]: o["ac_p0"] for o in outs}).T.reindex(
            index=sns, columns=sub_lines
        )
        ac_q0 = pd.DataFrame({o["snapshot"]: o["ac_q0"] for o in outs}).T.reindex(
            index=sns, columns=sub_lines
        )
        v_mag = pd.DataFrame({o["snapshot"]: o["v_mag"] for o in outs}).T.reindex(
            index=sns, columns=sub_buses
        )
        v_ang = pd.DataFrame({o["snapshot"]: o["v_ang"] for o in outs}).T.reindex(
            index=sns, columns=sub_buses
        )
        converged = pd.Series(
            {o["snapshot"]: o["converged"] for o in outs}, name="converged"
        ).reindex(sns)

        # Mask non-converged snapshots (diverged values are garbage).
        bad = converged.index[~converged.fillna(False)]
        for frame in (ac_p0, ac_q0, v_mag, v_ang):
            frame.loc[bad] = np.nan

        _write_dc_results(dc_p0, co2, n_nodes, tag)
        _write_ac_results(ac_p0, ac_q0, v_mag, v_ang, converged, co2, n_nodes, tag)
        logger.info(
            "Co2L={}: {}/{} snapshots converged.",
            co2, int(converged.sum()), len(converged),
        )

        summary = compare_line_loadings(co2, n_nodes, use_sclopf=use_sclopf, tag=tag)
        summary.to_hdf(
            _summary_results_path(co2, n_nodes, tag), key="summary", mode="w",
            complib=_COMPLIB, complevel=_COMPLEVEL,
        )
        logger.info("Saved comparison summary to {}", _summary_results_path(co2, n_nodes, tag))
        summaries[co2] = summary

    if notify and isinstance(mattermost_url, str):
        total_conv = sum(
            int(pd.Series({o["snapshot"]: o["converged"] for o in outputs
                           if o["co2lvl"] == c}).sum())
            for c in co2_levels
        )
        message_text = (
            f"Finished AC/DC comparison grid: {len(co2_levels)} CO2 level(s) "
            f"{tuple(co2_levels)}, {len(tasks)} (CO2, snapshot) tasks, n={n_nodes}, "
            f"sub_network={sub_network}, n_processes={n_processes}. "
            f"AC converged for {total_conv}/{len(tasks)} tasks."
        )
        send_mattermost_messages.post_message(message_text, mattermost_url)

    return summaries


# --------------------------------------------------------------------------- #
# Representative-scenario campaign
# --------------------------------------------------------------------------- #
# Meteorological seasons by month, and the four times of day sampled per day.
_SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
_TIME_LABELS = ("morning", "noon", "evening", "night")


def _snapshot_selection_path(n_nodes: int) -> str:
    return path_to_ac_dc_comparison + f"/snapshot_selection_n{n_nodes}.pklz"


def _snap_hours_to_grid(all_snapshots, target_hours) -> list:
    """Snap requested clock hours to the nearest hour present in the data.

    The networks are 3-hourly (00, 03, ... 21), so e.g. 07:00 -> 06:00 and
    17:00 -> 18:00. Distance is measured circularly (01:00 is closest to 00:00).
    """
    avail = sorted({int(h) for h in pd.DatetimeIndex(all_snapshots).hour})

    def nearest(t):
        return min(avail, key=lambda h: min(abs(h - t), 24 - abs(h - t)))

    return [nearest(int(t)) for t in target_hours]


def _build_snapshot_realizations(
    all_snapshots,
    n_date_realizations: int,
    n_weekdays_per_season: int,
    n_sundays_per_season: int,
    hours: list,
    base_seed: int,
) -> dict:
    """Draw ``n_date_realizations`` reproducible snapshot sets.

    Each realization samples, per season, ``n_weekdays_per_season`` distinct
    weekdays (Mon-Fri) and ``n_sundays_per_season`` distinct Sundays; every chosen
    day contributes the snapshots at ``hours``. Realization ``r`` uses RNG seed
    ``base_seed + r`` so the choice is reproducible and identical across CO2 levels
    and power-factor seeds ("same for all simulations" within a realization).

    Returns ``{r: DataFrame[season, day_type, time_label, snapshot]}``.
    """
    s = pd.DatetimeIndex(all_snapshots)
    days = pd.DatetimeIndex(sorted(set(s.normalize())))
    realizations = {}
    for r in range(n_date_realizations):
        rng = np.random.default_rng(base_seed + r)
        rows = []
        for season, months in _SEASONS.items():
            season_days = days[days.month.isin(months)]
            weekdays = season_days[season_days.dayofweek < 5]
            sundays = season_days[season_days.dayofweek == 6]
            chosen = [
                (pd.Timestamp(d), "weekday")
                for d in rng.choice(
                    weekdays, size=min(n_weekdays_per_season, len(weekdays)),
                    replace=False,
                )
            ] + [
                (pd.Timestamp(d), "sunday")
                for d in rng.choice(
                    sundays, size=min(n_sundays_per_season, len(sundays)),
                    replace=False,
                )
            ]
            for day, day_type in chosen:
                for hour, label in zip(hours, _TIME_LABELS):
                    ts = day + pd.Timedelta(hours=int(hour))
                    if ts in s:
                        rows.append((season, day_type, label, ts))
        realizations[r] = pd.DataFrame(
            rows, columns=["season", "day_type", "time_label", "snapshot"]
        )
    return realizations


def run_representative_scenario_campaign(
    fraction_PV_nodes: float = 0.3,
    power_factor_mid: float = 0.95,
    power_factor_range: float = 0.05,
    n_power_factor_seeds: int = 10,
    n_date_realizations: int = 5,
    n_weekdays_per_season: int = 2,
    n_sundays_per_season: int = 1,
    target_hours=(7, 12, 18, 0),
    n_nodes: int = 600,
    n_processes: int = 1,
    sub_network: str = "0",
    use_sclopf: bool = True,
    x_tol: float = 1e-6,
    date_seed: int = 254,
    overwrite: bool = True,
    notify: bool = True,
) -> dict:
    """Run the DC/AC comparison over a representative set of operating conditions.

    Covers **all CO2 levels** that have SCLOPF solutions (``get_co2_levels``). The
    snapshots span the four seasons; per season ``n_weekdays_per_season`` weekdays and
    ``n_sundays_per_season`` Sundays are sampled, each at the four times of day
    (``target_hours``, snapped to the 3-hourly grid). This snapshot set is drawn
    ``n_date_realizations`` times (different random dates each, but identical across CO2
    levels and power-factor seeds), and for each realization the per-bus power factors
    are drawn ``n_power_factor_seeds`` times.

    Every (date realization, power-factor seed) combination is a full
    :func:`run_comparison_grid` run, tagged ``_date{r}_pf{s}`` so its config / DC / AC /
    summary files do not collide. The random snapshot choices are saved once to
    ``snapshot_selection_n{n}.pklz``; the random power-factor choices are saved per run
    in the tagged ``ac_system_config`` files.

    Total AC solves = (#CO2 levels) x (#snapshots per realization) x
    ``n_date_realizations`` x ``n_power_factor_seeds`` — sizeable; use ``n_processes`` and
    ``overwrite=False`` (to resume) accordingly.

    Args:
        fraction_PV_nodes (float): Fraction of the sub-network's buses set to PV.
        power_factor_mid (float): Centre of the sampled power-factor interval.
        power_factor_range (float): Width of the sampled power-factor interval.
        n_power_factor_seeds (int): Number of power-factor realizations (seeds 0..N-1).
        n_date_realizations (int): Number of random snapshot-date realizations.
        n_weekdays_per_season (int): Weekdays sampled per season.
        n_sundays_per_season (int): Sundays sampled per season.
        target_hours (tuple): Clock hours for morning/noon/evening/night; snapped to
            the data's 3-hourly grid.
        n_nodes (int): Aggregation level of nodes.
        n_processes (int): Worker processes for each grid run (``1`` = sequential).
        sub_network (str): Sub-network to analyse (``"0"`` = main AC grid).
        use_sclopf (bool): Passed to ``load_pypsa_network``.
        x_tol (float): Newton-Raphson tolerance for the AC power flow.
        date_seed (int): Base RNG seed for the snapshot-date sampling.
        overwrite (bool): If ``False``, skip runs whose AC output already exists (resume).
        notify (bool): If ``True``, post one Mattermost message at the end (if configured).

    Returns:
        dict: ``{(date_realization, pf_seed): {co2_level: summary_DataFrame}}``.
    """
    co2_levels = [float(c) for c in get_co2_levels(n_nodes)]

    # The snapshot calendar is identical across CO2 levels — load one network to build
    # (and snap) the selection, then release it.
    ref = load_pypsa_network(co2_levels[0], n_nodes, use_sclopf=use_sclopf)
    hours = _snap_hours_to_grid(ref.snapshots, target_hours)
    logger.info(
        "Target hours {} snapped to grid hours {}.", tuple(target_hours), tuple(hours)
    )
    realizations = _build_snapshot_realizations(
        ref.snapshots, n_date_realizations, n_weekdays_per_season,
        n_sundays_per_season, hours, date_seed,
    )
    del ref

    # Save the random snapshot choices (the "separate config file").
    selection_record = {
        "realizations": realizations,
        "params": {
            "n_nodes": n_nodes,
            "n_date_realizations": n_date_realizations,
            "n_weekdays_per_season": n_weekdays_per_season,
            "n_sundays_per_season": n_sundays_per_season,
            "target_hours": tuple(target_hours),
            "grid_hours": tuple(hours),
            "seasons": _SEASONS,
            "date_seed": date_seed,
        },
    }
    with gzip.open(_snapshot_selection_path(n_nodes), "wb") as fh:
        pickle.dump(selection_record, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved snapshot selection to {}", _snapshot_selection_path(n_nodes))

    n_snaps = len(realizations[0])
    logger.info(
        "Campaign: {} CO2 level(s) x {} snapshot(s) x {} date realization(s) x {} "
        "power-factor seed(s) = {} AC solves.",
        len(co2_levels), n_snaps, n_date_realizations, n_power_factor_seeds,
        len(co2_levels) * n_snaps * n_date_realizations * n_power_factor_seeds,
    )

    all_summaries = {}
    for r in range(n_date_realizations):
        snaps = [str(ts) for ts in realizations[r].snapshot]
        for pf in range(n_power_factor_seeds):
            tag = f"_date{r}_pf{pf}"
            logger.info(
                "Campaign run: date realization {}/{}, power-factor seed {}/{} (tag {}).",
                r + 1, n_date_realizations, pf + 1, n_power_factor_seeds, tag,
            )
            all_summaries[(r, pf)] = run_comparison_grid(
                co2_lvl_tup=tuple(co2_levels), n_nodes=n_nodes, snapshots=snaps,
                n_processes=n_processes, fraction_PV_nodes=fraction_PV_nodes,
                power_factor_mid=power_factor_mid, power_factor_range=power_factor_range,
                seed=pf, sub_network=sub_network, use_sclopf=use_sclopf, x_tol=x_tol,
                overwrite=overwrite, notify=False, tag=tag,
            )

    if notify and isinstance(mattermost_url, str):
        message_text = (
            f"Finished representative-scenario AC/DC campaign: {len(co2_levels)} CO2 "
            f"level(s), {n_date_realizations} date realization(s) x {n_power_factor_seeds} "
            f"power-factor seed(s), {n_snaps} snapshots/realization, n={n_nodes}, "
            f"sub_network={sub_network}, n_processes={n_processes}. "
            f"{len(co2_levels) * n_snaps * n_date_realizations * n_power_factor_seeds} "
            f"AC solves total."
        )
        send_mattermost_messages.post_message(message_text, mattermost_url)

    return all_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare DC and AC power flows for representative snapshots of all "
                    "CO2 levels (5 date realizations x 10 power-factor seeds)."
    )
    parser.add_argument(
        "--n_processes", type=int, default=1,
        help="Worker processes per grid run (1 = sequential).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs whose AC output already exists.",
    )
    args = parser.parse_args()

    run_representative_scenario_campaign(
        n_processes=args.n_processes,
        overwrite=not args.resume,
    )
