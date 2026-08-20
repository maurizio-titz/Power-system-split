#!/usr/bin/env python
"""Normalization of split counts to a fraction of total contingencies.

A split count becomes a probability once divided by the number of contingencies that were
screened, ``weighted_trigger_count * total_snapshot_weight``: the number of initial double
line failures times the number of hours the snapshots stand for.

Each entry of ``n_2_failures`` covers ``weight`` distinct contingencies, so the trigger
weighting is summed rather than the entries counted.
"""

from __future__ import annotations

import functools
import gzip
import os
import pickle
import warnings

import networkx as nx
import pandas as pd
from loguru import logger

from utils import cascade_simulation, data_handling
from utils.config import path_to_grid_data


def n_2_failures_path(co2l: float) -> str:
    """Path of the cached double line failures of one CO2 level.

    Args:
        co2l (float): CO2 constraint.

    Returns:
        str: Path to the ``.pklz`` written by ``scripts/extract_grid_data.py``.
    """
    return os.path.join(path_to_grid_data, f"n_2_failures_co2lvl{float(co2l)}.pklz")


def get_n_2_failures(co2l: float,
                     network=None,
                     snet_index: str = "0",
                     recompute: bool = True) -> list[dict]:
    """Screened double line failures of one CO2 level.

    Given a ``network``, the failures are recomputed from it (bridges excluded, as in the
    cascade simulation) — the network is the source of truth and the cache is not consulted.
    Without a network, or with ``recompute=False``, they are read from the cache written by
    ``scripts/extract_grid_data.py``, which is logged because the result then depends on that
    file being up to date. Recomputing costs ~50 s per CO2 level, reading the cache ~2 s.

    Args:
        co2l (float): CO2 constraint.
        network (pypsa.Network, optional): Network to recompute the failures from.
        snet_index (str): Subnetwork to build the graph from ("0" is the Continental
            European AC grid).
        recompute (bool): If ``False``, read the cache even though a ``network`` was given —
            for callers that need the network only for its snapshot weightings. Has no effect
            without a network, since there is nothing to recompute from.

    Returns:
        list[dict]: Entries with a "failures" and a "weight" key.

    Raises:
        FileNotFoundError: If the cache is used and the cache file does not exist.
    """

    if network is not None and recompute:
        nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
        _, _, num_parallels, _ = data_handling.get_matrices_from_nx_graph(nx_graph)
        bridge_idxs = data_handling.nx_edges_to_matrix_indices(
            nx.bridges(nx_graph), nx_graph
        )

        return cascade_simulation.calc_possible_double_line_failures(
            num_parallels, ignored_idxs=bridge_idxs
        )

    fpath = n_2_failures_path(co2l)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"{fpath} does not exist and no network was given to recompute the double "
            "line failures from. Run scripts/extract_grid_data.py for this CO2 level."
        )

    with gzip.open(fpath, "rb") as fh:
        n_2_failures = pickle.load(fh)
    logger.info(f"Co2L={co2l}: loaded double line failures from {fpath}")

    return n_2_failures


def weighted_trigger_count(
    n_2_failures: list[dict],
    same_corridor: bool | None = None,
) -> float:
    """Number of contingencies covered by ``n_2_failures``, trigger weighting summed.

    Args:
        n_2_failures (list[dict]): Failures as returned by :func:`get_n_2_failures`.
        same_corridor (bool, optional): If ``True``, count only failures of two circuits of
            the same corridor; if ``False``, only failures of two different corridors; if
            ``None`` (default), count all of them.

    Returns:
        float: Summed trigger weighting.
    """
    
    if same_corridor is None:
        return float(sum(failure["weight"] for failure in n_2_failures))
    
    if same_corridor:
        n_2_failures_used = [
            failure
            for failure in n_2_failures
            if (failure["failures"][0][0] == failure["failures"][1][0])
        ]

    else:
        n_2_failures_used = [
                    failure
                    for failure in n_2_failures
                    if (failure["failures"][0][0] != failure["failures"][1][0])
                ]
        
    return float(sum(failure["weight"] for failure in n_2_failures_used))


def total_snapshot_weight(
    network=None,
    snapshot_weights: pd.Series | None = None,
    co2l: float | None = None,
) -> float:
    """Total number of hours the snapshots of one simulation stand for.

    Either ``network.snapshot_weightings.generators.sum()`` or, without a network, recovered
    from the per-snapshot ``snapshot_weighting`` of the split properties. The latter only sees
    snapshots that produced at least one split, so it is scaled up to the full snapshot count
    of the cached snapshot list when the weighting is uniform (3 h throughout for the
    2920-segment networks); if it is not, the observed sum is returned with a warning.

    Args:
        network (pypsa.Network, optional): Network whose snapshot weightings are summed.
        snapshot_weights (pd.Series, optional): ``snapshot_weighting`` per unique snapshot of
            the *unfiltered* split properties. Used when ``network`` is None.
        co2l (float, optional): CO2 constraint, needed to load the cached snapshot list that
            reveals snapshots without any split.

    Returns:
        float: Total snapshot weight in hours.

    Raises:
        ValueError: If neither ``network`` nor ``snapshot_weights`` is given.
    """
    if network is not None:
        return float(network.snapshot_weightings.generators.sum())

    if snapshot_weights is None:
        raise ValueError(
            "Either a network or the per-snapshot weightings of the split properties are "
            "needed to determine the total snapshot weight."
        )

    observed = float(snapshot_weights.sum())
    if co2l is None:
        return observed

    try:
        n_snapshots = len(data_handling.load_snapshot_list(float(co2l)))
    except (FileNotFoundError, OSError):
        return observed

    n_missing = n_snapshots - len(snapshot_weights)
    if n_missing <= 0:
        return observed

    unique_weights = pd.unique(snapshot_weights)
    if len(unique_weights) == 1:
        return float(unique_weights[0]) * n_snapshots

    warnings.warn(
        f"CO2 level {co2l}: {n_missing} of {n_snapshots} snapshots produced no split and "
        "the snapshot weighting is not uniform, so the total snapshot weight is summed over "
        "the snapshots present only and comes out too small.",
        stacklevel=2,
    )

    return observed


@functools.lru_cache(maxsize=None)
def _cached_contingency_count(co2l: float, same_corridor: bool | None) -> float:
    """Cache :func:`weighted_trigger_count` per CO2 level (the pickle is large)."""
    return weighted_trigger_count(get_n_2_failures(co2l), same_corridor=same_corridor)


def contingency_normalization(
    co2l: float,
    network=None,
    snapshot_weights: pd.Series | None = None,
    same_corridor: bool | None = None,
    recompute: bool = True,
) -> float:
    """Number of contingencies screened at one CO2 level — the split-count denominator.

    ``weighted_trigger_count * total_snapshot_weight``. Dividing a split count by this turns
    it into the fraction of total contingencies.

    Args:
        co2l (float): CO2 constraint.
        network (pypsa.Network, optional): Source of the snapshot weightings. If given, the
            double line failures are recomputed from it rather than read from their cache.
        snapshot_weights (pd.Series, optional): ``snapshot_weighting`` per unique snapshot of
            the *unfiltered* split properties, used when ``network`` is None.
        same_corridor (bool, optional): Restrict to same-corridor (``True``) or
            different-corridor (``False``) contingencies; ``None`` uses all of them.
        recompute (bool): If ``False``, take the failure count from the cache while still
            taking the snapshot weightings from ``network`` — the count is then memoized per
            ``(co2l, same_corridor)``. See :func:`get_n_2_failures`.

    Returns:
        float: The denominator.
    """
    if network is None or not recompute:
        n_contingencies = _cached_contingency_count(float(co2l), same_corridor)
    else:
        n_contingencies = weighted_trigger_count(
            get_n_2_failures(co2l, network=network, recompute=True),
            same_corridor=same_corridor,
        )

    return n_contingencies * total_snapshot_weight(
        network=network, snapshot_weights=snapshot_weights, co2l=co2l
    )
