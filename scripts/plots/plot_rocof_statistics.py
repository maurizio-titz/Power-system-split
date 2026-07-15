#!/usr/bin/env python

"""Plot the statistics of RoCoF for components and splits."""

import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.append("./")

from utils import data_handling
from utils.config import path_to_evaluation_results_sclopf, path_to_vis_results_sclopf


# Read-only data made available to the worker processes.
#
# On platforms with the "fork" start method (Linux/macOS) it is populated in the
# PARENT before the pool is forked, so workers inherit it via copy-on-write — the
# large objects (comp_props_all, snapshot_weightings) live as a single physical copy
# in RAM, never duplicated per worker, as long as nothing mutates them.
#
# On platforms without fork ("spawn", e.g. Windows) the parent's globals are NOT
# inherited, so the data is handed to each worker once via _init_split_worker below
# (one pickled copy per worker — unavoidable there).
_WORKER_STATE = {}


def _init_split_worker(state):
    """Pool initializer for the 'spawn' start method (Windows): receive the shared
    read-only data from the parent and store it in this worker process."""
    _WORKER_STATE.update(state)


def _compute_split_properties(task):
    """Process a single (co2_r, rocof_r) pair — the body of the old inner loop."""
    co2_r, rocof_r = task
    comp_props_all = _WORKER_STATE["comp_props_all"]
    snapshot_weightings = _WORKER_STATE["snapshot_weightings"]
    n_nodes = _WORKER_STATE["n_nodes"]
    overwrite = _WORKER_STATE["overwrite"]
    complib = _WORKER_STATE["complib"]
    complevel = _WORKER_STATE["complevel"]

    comp_props_co2l = comp_props_all[comp_props_all.co2l == co2_r]

    save_fpath = (
        path_to_vis_results_sclopf
        + f"/split_properties_Co2L{co2_r}_rocofThreshold{rocof_r}_n{n_nodes}.h5"
    )
    if os.path.exists(save_fpath) and not overwrite:
        logger.info(f"Skipping Co2L={co2_r}, rocof_r={rocof_r} — file exists.")
        return

    comp_r = comp_props_co2l.copy()
    comp_r["blackout_load_loss_share"] = (
        (comp_r["rocof"].abs() > rocof_r).astype(int) * comp_r["load_share"]
    )
    comp_r["total_load_loss_share"] = np.maximum(
        comp_r["shedding_load_loss_share"], comp_r["blackout_load_loss_share"]
    )

    split_groups = comp_r.groupby(["co2l", "time_stamp", "split_number"])

    split_props = pd.DataFrame(index=split_groups.groups.keys())
    split_props.index = split_props.index.rename(
        ["co2l", "time_stamp", "split_number_snapshot"]
    )

    split_props["init_failure_0"] = split_groups.init_failure_0.unique().apply(lambda x: x[0]).astype(int)
    split_props["init_failure_1"] = split_groups.init_failure_1.unique().apply(lambda x: x[0]).astype(int)
    split_props["num_par_failure_0"] = split_groups.num_par_failure_0.unique().apply(lambda x: x[0]).astype(float)
    split_props["num_par_failure_1"] = split_groups.num_par_failure_1.unique().apply(lambda x: x[0]).astype(float)
    split_props["trigger_weighting"] = split_groups.trigger_weighting.unique().apply(lambda x: x[0]).astype(int)
    split_props["lost_load_share_shedding"] = (
        split_groups.shedding_load_loss_share.sum().astype(float)
    )
    split_props["lost_load_share_blackout"] = (
        split_groups.blackout_load_loss_share.sum().astype(float)
    )
    split_props["lost_load_share_total"] = (
        split_groups.total_load_loss_share.sum().astype(float)
    )
    split_props["n_components"] = split_groups.size().astype(int)
    split_props["largest_component_load_share"] = split_groups.load_share.max().astype(float)
    split_props["load"] = split_groups.load.sum().astype(float)
    split_props["snapshot_weighting"] = snapshot_weightings.generators.loc[
        split_props.index.get_level_values("time_stamp")
    ].values.astype(int)
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    split_props["co2l"] = split_props.index.get_level_values("co2l")

    split_props.to_hdf(
        save_fpath, key="df", mode="w", complib=complib, complevel=complevel
    )
    logger.info(f"Saved Co2L={co2_r}, rocof_r={rocof_r} → {save_fpath}")


def generate_split_properties_different_rocof_thresholds(
    rocof_tup: tuple[float, ...] = (0.5, 1.0, 1.1, 1.5, 
                                    2.0, 2.5, 5, 10, 15),
    co2_lvl_tup: tuple[float, ...] = (0.6, 0.2, 0.0),
    n_nodes: int = 600,
    overwrite: bool = False,
    n_jobs: int = 1,
    complib: str | None = "blosc:zstd",
    complevel: int = 9,
):
    """Generate split properties df like the default one with rocof=1Hz/s as criteria,
    but for each rocof threshold in rocof_tup. Saves one HDF5 per (co2_level, rocof_r).

    Each (co2_level, rocof_r) pair is computed independently; set ``n_jobs`` > 1 to
    process them in parallel using Python's multiprocessing.

    Output files are compressed via PyTables using ``complib``/``complevel`` (pass
    ``complib=None`` to disable compression)."""

    components_fpath = path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    comp_props_all = pd.read_hdf(components_fpath)

    snapshot_weightings = data_handling.load_pypsa_network(
        co2lvl=0.0, n_nodes=n_nodes, use_sclopf=True
    ).snapshot_weightings

    # Fail fast (preserves the original ValueError) before spawning any workers.
    for co2_r in co2_lvl_tup:
        if len(comp_props_all[comp_props_all.co2l == co2_r]) == 0:
            raise ValueError(f"Co2 Level {co2_r} not in DataFrame")

    tasks = [(co2_r, rocof_r) for co2_r in co2_lvl_tup for rocof_r in rocof_tup]
    n_jobs = max(1, min(n_jobs, len(tasks)))

    # Fill the shared state in the parent BEFORE forking any worker. Forked workers
    # inherit this dict via copy-on-write, so the big read-only objects are shared,
    # not copied per worker. Nothing below mutates these objects.
    _WORKER_STATE.update(
        comp_props_all=comp_props_all,
        snapshot_weightings=snapshot_weightings,
        n_nodes=n_nodes,
        overwrite=overwrite,
        complib=complib,
        complevel=complevel,
    )

    if n_jobs == 1:
        # Serial path: no pool overhead, easy to debug.
        for task in tqdm(tasks):
            _compute_split_properties(task)
        return

    if "fork" in mp.get_all_start_methods():
        # Linux/macOS: fork so workers inherit _WORKER_STATE via copy-on-write. Do NOT
        # pass the data through initializer/initargs — that would pickle and copy the
        # large objects once per worker, defeating the sharing.
        ctx = mp.get_context("fork")
        pool_kwargs = {}
    else:
        # Windows (no fork): spawn fresh workers and hand them the data via the
        # initializer (one pickled copy per worker — the base objects cannot be shared).
        ctx = mp.get_context("spawn")
        pool_kwargs = dict(initializer=_init_split_worker, initargs=(dict(_WORKER_STATE),))

    with ctx.Pool(processes=n_jobs, **pool_kwargs) as pool:
        for _ in tqdm(
            pool.imap_unordered(_compute_split_properties, tasks),
            total=len(tasks),
        ):
            pass


def check_if_rocof_1_aggrees_with_standard():
    
    return

def plot_split_risk_different_rocof_thresholds():
    return


def plot_rocof_histogram_values_n_absolute(co2_lvls_ls=[0.6, 0.2, 0.0]):
    """Plot the distribution of RoCoF both for absolute values and the
    value in two panels for the different co2 levels.
    Highlight the chosen RoCoF threshold and put out which quantile they correspond to
    for the absolute case."""
    return