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
import pickle

import numpy as np
import pandas as pd
import pypsa

from tqdm import tqdm
from loguru import logger

from utils.data_handling import load_pypsa_network
from utils.config import path_to_ac_dc_comparison

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


def _config_path(co2lvl: float, n_nodes: int) -> str:
    return path_to_ac_dc_comparison + f"/ac_system_config_Co2L{co2lvl}_n{n_nodes}.pklz"


def _dc_results_path(co2lvl: float, n_nodes: int) -> str:
    return path_to_ac_dc_comparison + f"/dc_powerflow_Co2L{co2lvl}_n{n_nodes}.h5"


def _ac_results_path(co2lvl: float, n_nodes: int) -> str:
    return path_to_ac_dc_comparison + f"/ac_powerflow_Co2L{co2lvl}_n{n_nodes}.h5"


def save_ac_system_config(config: dict, co2lvl: float, n_nodes: int) -> str:
    """Save the AC system configuration (bus-type assignment + power factors).

    Args:
        config (dict): Configuration returned by ``setup_system_choosePVnPQ_distribtueQ``.
        co2lvl (float): CO2 constraint (used in the filename).
        n_nodes (int): Aggregation level of nodes (used in the filename).

    Returns:
        str: Path to the written file.
    """
    path = _config_path(co2lvl, n_nodes)
    with gzip.open(path, "wb") as fh:
        pickle.dump(config, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved AC system configuration to {}", path)
    return path


def load_ac_system_config(co2lvl: float, n_nodes: int) -> dict:
    """Load a previously saved AC system configuration.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.

    Returns:
        dict: The stored configuration.
    """
    path = _config_path(co2lvl, n_nodes)
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
        save_ac_system_config(config, co2lvl, n_nodes)

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
        path = _dc_results_path(co2lvl, n_nodes)
        with pd.HDFStore(path, mode="w") as store:
            store.put("p0", dc_p0)
        logger.info("Saved DC results to {}", path)

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
        path = _ac_results_path(co2lvl, n_nodes)
        with pd.HDFStore(path, mode="w") as store:
            store.put("p0", ac_p0)
            store.put("q0", ac_q0)
            store.put("v_mag_pu", v_mag)
            store.put("v_ang", v_ang)
            store.put("converged", converged_s.to_frame())
        logger.info("Saved AC results to {}", path)

    return results


# --------------------------------------------------------------------------- #
# Stage 4: comparison
# --------------------------------------------------------------------------- #
def compare_line_loadings(
    co2lvl: float = 0.2,
    n_nodes: int = 600,
    use_sclopf: bool = True,
) -> pd.DataFrame:
    """Compare DC vs AC line loadings from the saved power-flow results.

    DC loading uses ``|P| / s_nom``; AC loading uses the apparent power
    ``sqrt(P^2 + Q^2) / s_nom``. Only snapshots that converged in the AC run are
    considered.

    Args:
        co2lvl (float): CO2 constraint.
        n_nodes (int): Aggregation level of nodes.
        use_sclopf (bool): Passed to ``load_pypsa_network``.

    Returns:
        pd.DataFrame: Per-line summary with mean/max DC and AC loading and the
        number of snapshots each line is overloaded (>1) under DC and AC.
    """
    dc_p0 = pd.read_hdf(_dc_results_path(co2lvl, n_nodes), "p0")
    ac_p0 = pd.read_hdf(_ac_results_path(co2lvl, n_nodes), "p0")
    ac_q0 = pd.read_hdf(_ac_results_path(co2lvl, n_nodes), "q0")
    converged = pd.read_hdf(_ac_results_path(co2lvl, n_nodes), "converged")[
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
