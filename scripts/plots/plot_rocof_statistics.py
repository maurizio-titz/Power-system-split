#!/usr/bin/env python

"""Plot the statistics of RoCoF for components and splits. This scripts 
specifically compares how the number of blackouts changes if a different
RoCoF threshold is assumed to be valid. In general, the limit of 1Hz/s 
is an industry standard for large scale simulations."""

import functools
import glob
import multiprocessing as mp
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.append("./")

from utils import data_handling, normalize_split_risk, send_mattermost_messages
from utils.config import (
    mattermost_url,
    path_to_figures_sclopf,
    path_to_plot_data,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_sclopf,
)
from utils.plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    SINGLE_PLOT_SIZE,
    TICK_LABEL_FONTSIZE,
    TWO_PANEL_SIZE,
    add_panel_label,
    get_co2_color,
    save_figure,
    set_axis_style,
    setup_matplotlib_style,
)


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


def _make_pool_context():
    """Return ``(ctx, pool_kwargs)`` for a worker pool that shares ``_WORKER_STATE``.

    On fork platforms (Linux/macOS) the pool is forked and workers inherit the
    already-populated ``_WORKER_STATE`` via copy-on-write, so nothing is passed through the
    initializer — that would pickle and copy the large objects once per worker, defeating
    the sharing. On spawn platforms (Windows) the current state snapshot is handed to each
    worker once via the initializer. Call this AFTER ``_WORKER_STATE`` holds everything the
    workers need, since the spawn path snapshots it here."""
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork"), {}
    return (
        mp.get_context("spawn"),
        dict(initializer=_init_split_worker, initargs=(dict(_WORKER_STATE),)),
    )


def _compute_split_properties(task):
    """Process a single (co2_r, rocof_r) pair — the body of the old inner loop.

    Returns ``(rocof_r, split_props)`` so the parent can group the per-CO2 frames by
    RoCoF threshold and append them into a single file per threshold. The worker itself
    no longer writes to disk."""
    co2_r, rocof_r = task
    comp_props_all = _WORKER_STATE["comp_props_all"]
    snapshot_weightings = _WORKER_STATE["snapshot_weightings"]

    comp_props_co2l = comp_props_all[comp_props_all.co2l == co2_r]

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

    # No unwrapping of the per-group arrays returned by unique(): astype() then raises if a
    # split carries more than one value for a trigger property, which is the desired hard
    # failure — the components of a split must agree about their own trigger.
    split_props["init_failure_0"] = split_groups.init_failure_0.unique().astype(int)
    split_props["init_failure_1"] = split_groups.init_failure_1.unique().astype(int)
    split_props["num_par_failure_0"] = split_groups.num_par_failure_0.unique().astype(float)
    split_props["num_par_failure_1"] = split_groups.num_par_failure_1.unique().astype(float)
    split_props["trigger_weighting"] = split_groups.trigger_weighting.unique().astype(int)
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

    logger.info(f"Computed Co2L={co2_r}, rocof_r={rocof_r}")
    return rocof_r, split_props


def _concat_and_save(task):
    """Concat all per-CO2 frames for one RoCoF threshold, sort, and write the combined
    HDF5 file — the body of the old final append loop, for a single threshold.

    Run in a worker so the heavy compressed write (``blosc:zstd`` level 9 dominates the
    wall time) happens in parallel across thresholds. The per-CO2 frames are read from the
    shared worker state (inherited via fork, or handed over for spawn) keyed by
    ``rocof_r``, so they are not re-pickled per call on fork platforms."""
    rocof_r, save_fpath, complib, complevel = task
    frames = _WORKER_STATE["results"][rocof_r]
    combined = pd.concat(frames).sort_index()
    combined.to_hdf(
        save_fpath, key="df", mode="w", complib=complib, complevel=complevel
    )
    logger.info(f"Saved rocof_r={rocof_r} ({len(frames)} CO2 levels) → {save_fpath}")
    return rocof_r, len(frames)


def generate_split_properties_different_rocof_thresholds(
    rocof_tup: tuple[float, ...] = (0.8,0.9,1.0, 1.1, 1.2),
    co2_lvl_tup: tuple[float, ...] | None = None,
    n_nodes: int = 600,
    overwrite: bool = False,
    n_jobs: int = 1,
    complib: str | None = "blosc:zstd",
    complevel: int = 9,
):
    """Generate split properties df like the default one with rocof=1Hz/s as criteria,
    but for each rocof threshold in rocof_tup. Saves one HDF5 per rocof threshold, each
    containing all CO2 levels stacked together
    (``split_properties_rocofThreshold{rocof_r:G}_n{n_nodes}.h5``).

    ``co2_lvl_tup`` defaults to ``None``, meaning every CO2 level present in the
    component-properties data is used; pass a tuple to restrict to a subset.

    Each (co2_level, rocof_r) pair is computed independently; set ``n_jobs`` > 1 to
    process them in parallel using Python's multiprocessing. The per-CO2 frames are only
    appended together in the final step, once all workers are done.

    Output files are compressed via PyTables using ``complib``/``complevel`` (pass
    ``complib=None`` to disable compression)."""

    components_fpath = path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    comp_props_all = pd.read_hdf(components_fpath)

    snapshot_weightings = data_handling.load_pypsa_network(
        co2lvl=0.0, n_nodes=n_nodes, use_sclopf=True
    ).snapshot_weightings

    # Resolve the CO2 levels: all available in the data by default, else the requested
    # subset (fail fast, preserving the original ValueError, before spawning workers).
    if co2_lvl_tup is None:
        co2_levels = sorted(comp_props_all.co2l.unique(), reverse=True)
    else:
        for co2_r in co2_lvl_tup:
            if len(comp_props_all[comp_props_all.co2l == co2_r]) == 0:
                raise ValueError(f"Co2 Level {co2_r} not in DataFrame")
        co2_levels = list(co2_lvl_tup)

    def _save_fpath(rocof_r):
        return (
            path_to_vis_results_sclopf
            + f"/split_properties_rocofThreshold{rocof_r:G}_n{n_nodes}.h5"
        )

    # Per-RoCoF skip logic: drop thresholds whose combined (all-CO2) file already exists.
    rocofs_to_run = []
    for rocof_r in rocof_tup:
        if os.path.exists(_save_fpath(rocof_r)) and not overwrite:
            logger.info(f"Skipping rocof_r={rocof_r} — file exists.")
        else:
            rocofs_to_run.append(rocof_r)

    if not rocofs_to_run:
        logger.info("Nothing to do — all requested rocof threshold files already exist.")
        return

    tasks = [(co2_r, rocof_r) for rocof_r in rocofs_to_run for co2_r in co2_levels]
    n_jobs = max(1, min(n_jobs, len(tasks)))

    # Fill the shared state in the parent BEFORE forking any worker. Forked workers
    # inherit this dict via copy-on-write, so the big read-only objects are shared,
    # not copied per worker. Nothing below mutates these objects.
    _WORKER_STATE.update(
        comp_props_all=comp_props_all,
        snapshot_weightings=snapshot_weightings,
    )

    # Collect the per-CO2 frames grouped by their RoCoF threshold; appended below.
    results = {rocof_r: [] for rocof_r in rocofs_to_run}

    if n_jobs == 1:
        # Serial path: no pool overhead, easy to debug.
        for task in tqdm(tasks):
            rocof_r, split_props = _compute_split_properties(task)
            results[rocof_r].append(split_props)
    else:
        ctx, pool_kwargs = _make_pool_context()
        with ctx.Pool(processes=n_jobs, **pool_kwargs) as pool:
            for rocof_r, split_props in tqdm(
                pool.imap_unordered(_compute_split_properties, tasks),
                total=len(tasks),
            ):
                results[rocof_r].append(split_props)

    # Final append step: one file per RoCoF threshold, all CO2 levels stacked together.
    # concat + sort + the compressed write are independent per threshold, so fan them out
    # the same way as the compute step. Expose the collected frames on _WORKER_STATE first
    # so fork workers inherit them via copy-on-write (and the spawn snapshot below includes
    # them).
    _WORKER_STATE["results"] = results
    save_tasks = [
        (rocof_r, _save_fpath(rocof_r), complib, complevel)
        for rocof_r in rocofs_to_run
    ]
    n_save_jobs = max(1, min(n_jobs, len(save_tasks)))

    if n_save_jobs == 1:
        for task in tqdm(save_tasks, desc="saving"):
            _concat_and_save(task)
    else:
        # A NEW pool is required here even on fork: the compute pool (if any) was forked
        # before `results` was populated, so its workers never saw the frames.
        ctx, pool_kwargs = _make_pool_context()
        with ctx.Pool(processes=n_save_jobs, **pool_kwargs) as pool:
            for _ in tqdm(
                pool.imap_unordered(_concat_and_save, save_tasks),
                total=len(save_tasks),
                desc="saving",
            ):
                pass

    if isinstance(mattermost_url, str):
        message_text = (
            f"Finished split properties for {len(rocofs_to_run)} RoCoF threshold(s) "
            f"{tuple(rocofs_to_run)} over {len(co2_levels)} CO2 level(s), n={n_nodes}."
        )
        send_mattermost_messages.post_message(message_text, mattermost_url)


def check_if_rocof_1_aggrees_with_standard(n_nodes: int = 600):
    """Sanity check: the split properties generated with a RoCoF threshold of 1 Hz/s
    must reproduce the standard ``split_properties_all`` DataFrame, which already uses
    1 Hz/s as its blackout criterion.

    Discovers every per-CO2 rocof-threshold file in ``path_to_vis_results_sclopf`` and
    keeps the ones whose threshold parses to 1 (robust to it being written as ``1``,
    ``1.0`` or the current ``{value:g}`` form), loads the standard df once, and compares
    them CO2 level by CO2 level. Raises ``AssertionError`` listing every mismatch, so all
    CO2 levels are checked before failing."""

    standard_fpath = (
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5"
    )
    standard = pd.read_hdf(standard_fpath)

    # Discover the rocof-threshold files in the folder and keep the one whose threshold
    # equals 1. Parsing the value (instead of matching a literal) is robust to the
    # different formats the threshold has been written in ("1", "1.0", or the current
    # "{value:G}", which renders 1.0 as "1"). Each file already contains all CO2 levels.
    file_pattern = re.compile(
        rf"split_properties_rocofThreshold(?P<rocof>[-0-9.]+)_n{n_nodes}\.h5$"
    )
    rocof1_fpath = None
    for fpath in sorted(
        glob.glob(
            path_to_vis_results_sclopf
            + f"/split_properties_rocofThreshold*_n{n_nodes}.h5"
        )
    ):
        m = file_pattern.search(os.path.basename(fpath))
        if m and float(m.group("rocof")) == 1.0:
            rocof1_fpath = fpath
            break

    if rocof1_fpath is None:
        raise FileNotFoundError(
            f"No 'rocofThreshold1' split properties file found in "
            f"{path_to_vis_results_sclopf} for n={n_nodes}."
        )

    rocof1 = pd.read_hdf(rocof1_fpath)

    mismatches = []
    for co2_r in sorted(rocof1.co2l.unique(), reverse=True):
        rocof1_co2 = rocof1[rocof1.co2l == co2_r]
        standard_co2 = standard[standard.co2l == co2_r]

        if len(standard_co2) == 0:
            mismatches.append(
                f"Co2L={co2_r}: not present in standard df {standard_fpath}"
            )
            continue

        try:
            pd.testing.assert_frame_equal(
                standard_co2.sort_index(),
                rocof1_co2[standard_co2.columns].sort_index(),
                check_like=True,
            )
        except AssertionError as exc:
            mismatches.append(f"Co2L={co2_r}: {exc}")
        else:
            logger.info(f"Co2L={co2_r}: rocof=1 agrees with standard ✓")

    if mismatches:
        raise AssertionError(
            "rocof=1 split properties disagree with the standard df:\n"
            + "\n".join(mismatches)
        )

    logger.info(
        f"rocof=1 Hz/s agrees with the standard df for all "
        f"{rocof1.co2l.nunique()} CO2 levels."
    )

def _split_risk_norm_fpath(n_nodes):
    """Path of the cached per-CO2-level split-risk normalization (in ``path_to_plot_data``).

    ``path_to_plot_data`` is resolved before the ``PLOT_STYLE``/``PLOT_OUTPUT_DIR`` override
    of ``path_to_figures_sclopf`` in ``utils.config``, so the cache is shared by all plot
    styles instead of being recomputed per style.

    The ``weightedTrigger`` marker pins the convention: the denominator sums the trigger
    weighting, so pre-fix caches are not reused."""
    return os.path.join(
        path_to_plot_data, f"split_risk_normalization_weightedTrigger_n{n_nodes}.pklz"
    )


def _compute_split_risk_norm(co2ls_desc, n_nodes):
    """Compute the normalization for every CO2 level in ``co2ls_desc``.

    Each level uses its own network (``...Co2L{co2l}-2920SEG.nc``); the denominator is
    ``weighted_trigger_count * total_snapshot_weight`` as defined by
    ``utils.normalize_split_risk``. Returns a ``pd.Series`` indexed by ``co2l``. The network is
    loaded only for its snapshot weightings — the double line failures come from the cache
    written by ``scripts/extract_grid_data.py``. This is still the expensive part (one PyPSA
    network load per level), so it only runs when the cache file is missing."""

    norm = pd.Series(index=pd.Index(co2ls_desc, name="co2l"), dtype=float, name="norm")
    for co2l in tqdm(co2ls_desc, desc="normalization per co2 level"):
        network = data_handling.load_pypsa_network_from_path(
            path_to_pypsa_network_sclopf
            + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l}-2920SEG.nc",
            True,
        )
        weighted_trigger_count = normalize_split_risk.weighted_trigger_count(
            normalize_split_risk.get_n_2_failures(co2l, network=network),
            same_corridor=None,
        )
        total_weight = normalize_split_risk.total_snapshot_weight(network=network)
        norm.loc[co2l] = weighted_trigger_count * total_weight
        logger.info(
            f"Co2L={co2l}: {weighted_trigger_count:g} weighted double line failures "
            f"× total snapshot weight {total_weight:g} → norm {norm.loc[co2l]:g}"
        )
    return norm


@functools.lru_cache(maxsize=None)
def _split_risk_normalization(n_nodes: int = 600, overwrite: bool = False):
    """Return ``(co2ls_desc, co2_x, norm)`` shared by the split-risk fraction plots.

    ``co2ls_desc`` are the CO2 levels sorted ascending (so the x-axis, once inverted, runs
    from high to low CO2), ``co2_x`` the matching actual CO2 percentages, and ``norm`` the
    per-CO2-level denominator ``weighted_trigger_count * total_snapshot_weight`` that turns
    a split count into a fraction of total contingencies — the same normalization used by
    the secondary axis of panel c in ``create_combined_flow_and_split_statistics_plot`` and
    by the ``numGSS_per_contingency_hour`` column of
    ``scripts/meteo/snapshot_vs_numGSS_plot.build_snapshot_gss_table``.

    ``norm`` is a ``pd.Series`` indexed by ``co2ls_desc``, computed from each CO2 level's own
    network rather than from a single hardcoded one. The levels are expected to agree — the
    networks share their topology, hence their number of double line failures, and the
    snapshot weightings are 3 throughout — so a deviation is logged as a warning; the
    per-level values are still returned and used, since they are the correct denominators
    either way.

    Computing ``norm`` costs one PyPSA network load per CO2 level (minutes), so it is cached
    on disk as a gzip-compressed pickle (``.pklz``) in ``path_to_plot_data`` and only
    recomputed when that file is missing, when it does not cover exactly the CO2 levels
    currently available (a level was added or removed), or when ``overwrite=True``. ``co2ls_desc``/``co2_x`` are cheap and
    always derived fresh. Additionally cached in-process per ``(n_nodes, overwrite)``, since
    all three split-risk plot functions call this; callers must not mutate the result."""

    co2ls = data_handling.get_co2_levels(n_nodes)
    co2ls_desc = co2ls[::-1]
    co2_x = data_handling.get_actual_co2_level(co2ls_desc, percent=True)

    fpath = _split_risk_norm_fpath(n_nodes)
    norm = None
    if os.path.exists(fpath) and not overwrite:
        # compression is passed explicitly: pandas infers it from the extension only for
        # ".gz"/".bz2"/".zip"/".xz"/".zst", so a bare ".pklz" would be read/written raw.
        cached = pd.read_pickle(fpath, compression="gzip")
        # Only usable if it covers exactly today's CO2 levels — otherwise it predates a
        # change in the available networks and is recomputed instead of silently reused.
        if np.array_equal(cached.index.to_numpy(), np.asarray(co2ls_desc, dtype=float)):
            norm = cached
            logger.info(f"Loaded split-risk normalization from {fpath}")
        else:
            logger.warning(
                f"Cached normalization in {fpath} covers CO2 levels "
                f"{list(cached.index)} but {list(co2ls_desc)} are available — recomputing."
            )

    if norm is None:
        norm = _compute_split_risk_norm(co2ls_desc, n_nodes)
        os.makedirs(path_to_plot_data, exist_ok=True)
        norm.to_pickle(fpath, compression="gzip")
        logger.info(f"Saved split-risk normalization to {fpath}")

    if norm.nunique() > 1:
        logger.warning(
            "Normalization differs across CO2 levels (expected identical — same topology "
            f"and snapshot weightings of 3 everywhere): {norm.to_dict()}"
        )

    return co2ls_desc, co2_x, norm


def _split_props_fpath(rocof_r, n_nodes):
    """Path to the per-threshold split properties file (``{rocof_r:G}`` in the name)."""
    
    return (
        path_to_vis_results_sclopf
        + f"/split_properties_rocofThreshold{rocof_r:G}_n{n_nodes}.h5"
    )


def _split_counts_per_co2(fpath, co2ls_desc, bins):
    """Histogram each CO2 level's ``lost_load_share_blackout`` (in %) into ``bins``.

    Values are clipped into the bin range and weighted by ``total_weighting``, matching the
    denominator they are divided by.

    Returns a DataFrame indexed by ``co2ls_desc`` with the bin centers as columns."""
    split_props = pd.read_hdf(fpath)
    lost = split_props.lost_load_share_blackout.astype(float)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data = []
    for co2l in co2ls_desc:
        mask_co2 = split_props.co2l == co2l
        vals = data_handling.clip_to_bin_range(lost[mask_co2].values * 100, bins)
        counts, _ = np.histogram(
            vals, bins=bins, weights=split_props.total_weighting[mask_co2]
        )
        data.append(counts)
    return pd.DataFrame(np.stack(data), index=co2ls_desc, columns=bin_centers)


def _resolve_panel_grid(n_panels, n_cols, layout_2x3):
    """Return ``(n_rows, n_cols)`` for laying out ``n_panels`` panels.

    ``layout_2x3`` forces 3 columns, giving 2 rows for the 5- and 6-panel split-risk
    figures (and overriding ``n_cols``); otherwise ``n_cols`` is used, capped to
    ``n_panels`` so no column is left fully empty."""
    n_cols = 3 if layout_2x3 else max(1, int(n_cols))
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _place_panel_legend(fig, axes_flat, n_panels, n_cols, handles, labels, title):
    """Place the panel legend and hide unused grid cells.

    Panels occupy ``axes_flat[:n_panels]`` (filled row-major). If the grid has a leftover
    cell, the legend is drawn inside the first one — enlarged and with a single-line title
    (the ``\n`` in ``title`` is collapsed to a space), since it has a whole panel to itself.
    Otherwise it is anchored just outside the top-right panel, hugging its top-right corner.
    Every cell beyond the panels is turned off. No legend is drawn when ``handles`` is
    ``None``."""
    for extra in axes_flat[n_panels:]:
        extra.axis("off")
    if handles is None:
        return
    if len(axes_flat) > n_panels:  # a leftover cell exists -> legend inside it, large
        big = LEGEND_FONTSIZE + 10
        leg = axes_flat[n_panels].legend(
            handles, labels, title=title.replace("\n", " "), loc="center",
            fontsize=big, title_fontsize=big,
        )
    else:  # full grid -> legend just right of the top-right panel, aligned to its top
        top_right = axes_flat[n_cols - 1]  # row 0, last column
        leg = fig.legend(
            handles, labels, title=title, loc="upper left",
            bbox_to_anchor=(1.02, 1.0), bbox_transform=top_right.transAxes,
            fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE,
        )
    leg.get_title().set_horizontalalignment("center")


def plot_split_risk_different_rocof_thresholds(
    rocof_thresholds: tuple[float, ...] = (0.8,0.9,1.0, 1.1, 1.2),
    n_nodes: int = 600,
    save_fig: bool = True,
    n_cols: int = 3,
    layout_2x3: bool = False,
):
    """Plot the fraction of system splits falling into each share-of-load-not-served
    category as a function of CO2 level, one panel per RoCoF blackout threshold.

    This reproduces "panel c" of
    ``create_combined_flow_and_split_statistics_plot`` (in
    ``plot_combined_flow_split_statistics.py``) — which is hardwired to the single
    ``|RoCoF| > 1 Hz/s`` criterion baked into ``split_properties_all`` — but shows it for
    several RoCoF thresholds side by side. Each panel reads the per-threshold split
    properties file produced by ``generate_split_properties_different_rocof_thresholds``
    (``split_properties_rocofThreshold{rocof_r:G}_n{n_nodes}.h5``).

    The y-axis is the fraction of total contingencies:
    ``count / (weighted_trigger_count * total_snapshot_weight)`` — the same normalization
    used by the secondary axis of panel c. The panels (one per RoCoF threshold) are laid
    out in a grid of ``ceil(len(rocof_thresholds) / n_cols)`` rows by ``n_cols`` columns;
    ``layout_2x3=True`` forces a 2-row × 3-column grid (overriding ``n_cols``), which fits
    the default six thresholds exactly. When the grid has a leftover cell the legend is
    drawn inside it, otherwise it sits outside on the right. Thresholds whose data file
    does not exist yet are drawn as empty "no data" placeholders. Returns ``(fig, axes)``."""

    setup_matplotlib_style()

    co2ls_desc, co2_x, norm = _split_risk_normalization(n_nodes)

    bins = np.linspace(0, 100, 6).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cmap = plt.get_cmap("inferno_r")
    markers = ["o", "s", "D", "^", "v", "."]

    n_panels = len(rocof_thresholds)
    n_rows, n_cols = _resolve_panel_grid(n_panels, n_cols, layout_2x3)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), sharey=True, squeeze=False
    )
    axes_flat = axes.flatten()

    legend_handles, legend_labels = None, None

    for idx, (ax, rocof_r) in enumerate(
        tqdm(list(zip(axes_flat, rocof_thresholds)), total=n_panels, desc="rocof panels")
    ):
        fpath = _split_props_fpath(rocof_r, n_nodes)

        if not os.path.exists(fpath):
            logger.warning(f"No split properties file for rocof={rocof_r} — placeholder.")
            ax.text(
                0.5, 0.5, "no data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE,
            )
        else:
            data = _split_counts_per_co2(fpath, co2ls_desc, bins)

            for i, bin_center in enumerate(bin_centers):
                color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
                ax.step(
                    co2_x,
                    data.loc[:, bin_center] / norm,
                    where="mid",
                    label=rf"{bins[i]}-{bins[i+1]}\%",
                    alpha=0.8,
                    color=color,
                    marker=markers[i % len(markers)],
                    markersize=5,
                )

            ax.set_yscale("log")
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Common axis styling (applied to populated and placeholder panels alike).
        ax.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xticks([10, 30, 50], minor=True)
        ax.grid(False)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.invert_xaxis()

        # ylabel only on the leftmost column (sharey hides the inner y tick labels).
        if idx % n_cols == 0:
            ax.set_ylabel(
                "Fraction of total contingencies", fontsize=AXIS_LABEL_FONTSIZE
            )

        # Threshold identity: an in-axes annotation rather than an axes title.
        ax.text(
            0.5, 1.02, rf"$|\mathrm{{RoCoF}}|_\mathrm{{crit}} = {rocof_r:g}$ Hz/s",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        add_panel_label(ax, idx, y_offset=0.1)

    _place_panel_legend(
        fig, axes_flat, n_panels, n_cols, legend_handles, legend_labels,
        title="Share of load\n not served",
    )

    fig.tight_layout()

    if save_fig:
        save_figure(
            fig,
            f"split_risk_fraction_contingencies_different_rocof_thresholds_n{n_nodes}",
            path_to_figures_sclopf,
        )

    return fig, axes


def plot_split_risk_categories_across_rocof_thresholds(
    rocof_thresholds: tuple[float, ...] = (0.8,0.9,1.0, 1.1, 1.2),
    n_nodes: int = 600,
    save_fig: bool = True,
    n_cols: int = 3,
    layout_2x3: bool = False,
):
    """Plot the fraction of system splits vs CO2 level, one panel per share-of-load-not-
    served category, with one line per RoCoF threshold inside each panel.

    This is the transpose of ``plot_split_risk_different_rocof_thresholds``: there each
    panel fixes a RoCoF threshold and draws a line per load-loss category; here each panel
    fixes a load-loss category and draws a line per RoCoF threshold, so the thresholds are
    compared directly within a category. Same quantity and normalization as that function
    (fraction of total contingencies ``count / (weighted_trigger_count *
    total_snapshot_weight)``, log y, inverted x).

    Each threshold reads its per-threshold split properties file
    (``split_properties_rocofThreshold{rocof_r:G}_n{n_nodes}.h5``); thresholds whose file
    does not exist are skipped with a warning. The RoCoF thresholds are colored with the
    qualitative ``Dark2`` colormap (they are nominal categories, so no sequential map is
    used) — deliberately distinct from the ``inferno_r`` used for the load-loss categories
    in ``plot_split_risk_different_rocof_thresholds`` — so the two figures are not confused.

    The category panels are laid out in a grid of ``ceil(n_panels / n_cols)`` rows by
    ``n_cols`` columns; ``layout_2x3=True`` forces a 2-row × 3-column grid (overriding
    ``n_cols``), i.e. the first three categories on the top row and the remaining two on the
    bottom row with the RoCoF legend filling the leftover cell. When the grid has no
    leftover cell the legend sits outside on the right. Returns ``(fig, axes)``."""

    setup_matplotlib_style()

    co2ls_desc, co2_x, norm = _split_risk_normalization(n_nodes)

    bins = np.linspace(0, 100, 6).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # A qualitative colormap for the RoCoF thresholds (the lines): they are nominal
    # categories whose ordering/spacing is not meant to be read off the colors, so a
    # sequential map (viridis/inferno/cividis/turbo) would be misleading here. Dark2's
    # colors are indexed by integer position, one distinct hue per threshold, and it is
    # kept clear of the inferno_r used for the load-loss categories.
    cmap = plt.get_cmap("Dark2")
    markers = ["o", "s", "D", "^", "v", "."]

    # Load each threshold's per-CO2 histogram once (reused across every category panel).
    n_thr = len(rocof_thresholds)
    counts_per_thr = {}
    for rocof_r in tqdm(rocof_thresholds, total=n_thr, desc="rocof thresholds"):
        fpath = _split_props_fpath(rocof_r, n_nodes)
        if not os.path.exists(fpath):
            logger.warning(f"No split properties file for rocof={rocof_r} — skipping.")
            continue
        counts_per_thr[rocof_r] = _split_counts_per_co2(fpath, co2ls_desc, bins)

    n_panels = len(bin_centers)
    n_rows, n_cols = _resolve_panel_grid(n_panels, n_cols, layout_2x3)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), sharey=True, squeeze=False
    )
    axes_flat = axes.flatten()

    legend_handles, legend_labels = None, None

    for idx, (ax, bin_center) in enumerate(zip(axes_flat, bin_centers)):
        if not counts_per_thr:
            ax.text(
                0.5, 0.5, "no data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE,
            )
        else:
            for i, rocof_r in enumerate(rocof_thresholds):
                if rocof_r not in counts_per_thr:
                    continue
                color = cmap(i)  # qualitative map: integer index -> distinct hue
                ax.step(
                    co2_x,
                    counts_per_thr[rocof_r].loc[:, bin_center] / norm,
                    where="mid",
                    label=rf"${rocof_r:g}$",
                    alpha=0.8,
                    color=color,
                    marker=markers[i % len(markers)],
                    markersize=5,
                )

            ax.set_yscale("log")
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Common axis styling (applied to populated and placeholder panels alike).
        ax.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xticks([10, 30, 50], minor=True)
        ax.grid(False)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.invert_xaxis()

        # ylabel only on the leftmost column (sharey hides the inner y tick labels).
        if idx % n_cols == 0:
            ax.set_ylabel(
                "Fraction of total contingencies", fontsize=AXIS_LABEL_FONTSIZE
            )

        # Category identity: an in-axes annotation rather than an axes title.
        ax.text(
            0.5, 1.06, rf"Load not served: {bins[idx]}-{bins[idx+1]}\%",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=AXIS_LABEL_FONTSIZE + 4,
        )
        add_panel_label(ax, idx, y_offset=0.1)

    _place_panel_legend(
        fig, axes_flat, n_panels, n_cols, legend_handles, legend_labels,
        title=r"$|\mathrm{RoCoF}|_\mathrm{crit}$" + "\n[Hz/s]",
    )

    fig.tight_layout()

    if save_fig:
        save_figure(
            fig,
            f"split_risk_fraction_contingencies_categories_across_rocof_thresholds_n{n_nodes}",
            path_to_figures_sclopf,
        )

    return fig, axes


def plot_split_risk_high_loss_across_rocof_thresholds(
    rocof_thresholds: tuple[float, ...] = (0.8,0.9,1.0, 1.1, 1.2),
    n_nodes: int = 600,
    save_fig: bool = True,
):
    """Compare only the highest share-of-load-not-served category (80–100%) across the
    provided RoCoF thresholds, overlaid in a single panel.

    Same quantity and normalization as
    ``plot_split_risk_different_rocof_thresholds`` (fraction of total contingencies vs CO2
    level, log y, inverted x), but instead of one panel per threshold with a line per
    load-loss bin, this draws one line per RoCoF threshold for the 80–100% bin only.
    Thresholds whose data file does not exist are skipped with a warning. Returns
    ``(fig, ax)``."""

    setup_matplotlib_style()

    co2ls_desc, co2_x, norm = _split_risk_normalization(n_nodes)

    bins = np.linspace(0, 100, 6).astype(int)
    high_bin_center = 0.5 * (bins[-2] + bins[-1])  # the 80–100% category
    high_bin_label = rf"{bins[-2]}-{bins[-1]}\%"

    # Qualitative map for the RoCoF thresholds (nominal categories, order not encoded by
    # the colors); integer-indexed, kept clear of the inferno_r used for load-loss bins.
    cmap = plt.get_cmap("Dark2")
    fig, ax = plt.subplots(figsize=SINGLE_PLOT_SIZE)

    n_thr = len(rocof_thresholds)
    for i, rocof_r in enumerate(
        tqdm(rocof_thresholds, total=n_thr, desc="rocof thresholds")
    ):
        fpath = _split_props_fpath(rocof_r, n_nodes)
        if not os.path.exists(fpath):
            logger.warning(f"No split properties file for rocof={rocof_r} — skipping.")
            continue

        data = _split_counts_per_co2(fpath, co2ls_desc, bins)
        color = cmap(i)  # qualitative map: integer index -> distinct hue
        ax.step(
            co2_x,
            data.loc[:, high_bin_center] / norm,
            where="mid",
            label=rf"${rocof_r:g}$",
            color=color,
            marker="o",
            markersize=5,
            alpha=0.8,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Fraction of total contingencies", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticks([10, 30, 50], minor=True)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.invert_xaxis()

    # In-axes note of which load-loss category is shown (not an axes title).
    ax.text(
        0.5, 1.02, rf"Share of load not served: {high_bin_label}",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    leg = ax.legend(
        title=r"$|\mathrm{RoCoF}|_\mathrm{crit}$ [Hz/s]",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    leg.get_title().set_horizontalalignment("center")

    fig.tight_layout()

    if save_fig:
        save_figure(
            fig,
            f"split_risk_fraction_contingencies_highLoss_across_rocof_thresholds_n{n_nodes}",
            path_to_figures_sclopf,
        )

    return fig, ax


def _co2_percent_label(co2_r, n_nodes):
    """Short legend label for a co2 level: just the percent of 1990 (the legend title
    already states it is a CO2 level), e.g. ``60\\%``."""
    return rf"{int(data_handling.get_actual_co2_level(co2_r, n_nodes, percent=True))}\%"


def plot_rocof_histogram_values_n_absolute(
    co2_lvls_tup: tuple[float, ...]=(0.6, 0.2, 0.0),
    rocof_thresholds: tuple[float, ...]=(0.8,0.9,1.0, 1.1, 1.2),
    n_nodes: int = 600,
    min_load_share: float = 0.1,
    save_fig: bool = True,
):
    """Plot the distribution of RoCoF both for absolute values and the
    value in two panels for the different co2 levels.
    Highlight the chosen RoCoF threshold and put out which quantile they correspond to
    for the absolute case.

    For each (co2 level, threshold) the empirical quantile the threshold corresponds to
    on the absolute distribution — i.e. ``P(|RoCoF| <= threshold)`` — is logged and shown
    in a summary box below the panels. ``min_load_share`` (a fraction in ``[0, 1]``) keeps
    only components carrying at least that share of the load (``load_share >= min_load_share``;
    use ``0`` to keep all); it is always appended to the save path as a
    ``_minLoadShare{value}`` suffix. Returns ``(fig, (ax_signed, ax_abs), quantiles)`` where
    ``quantiles`` maps ``(co2_level, threshold)`` to the quantile value."""

    # Load component properties_all datafamre
    components_fpath = path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    comp_props_all = pd.read_hdf(components_fpath)

    comp_props_all = comp_props_all[comp_props_all["load_share"] >= min_load_share]

    # Shared bins from the finite rocof over the selected co2 levels, so the panels are
    # directly comparable across levels.
    selected = comp_props_all[comp_props_all.co2l.isin(co2_lvls_tup)]
    all_finite = selected["rocof"].to_numpy()
    all_finite = all_finite[np.isfinite(all_finite)]
    max_abs = np.nanmax(np.abs(all_finite))
    signed_bins = np.linspace(-max_abs, max_abs, 61)
    abs_bins = np.linspace(0, max_abs, 61)

    fig, (ax_signed, ax_abs) = plt.subplots(1, 2, figsize=TWO_PANEL_SIZE)

    quantiles = {}
    for co2_r in co2_lvls_tup:
        vals = comp_props_all.loc[comp_props_all.co2l == co2_r, "rocof"].to_numpy()
        r = vals[np.isfinite(vals)]
        if len(r) == 0:
            logger.warning(f"Co2L={co2_r}: no finite RoCoF values — skipping.")
            continue
        color = get_co2_color(co2_r)
        label = _co2_percent_label(co2_r, n_nodes)

        ax_signed.hist(
            r, bins=signed_bins, density=False, histtype="step", lw=2.5,
            color=color, label=label
        )
        ax_abs.hist(
            np.abs(r), bins=abs_bins, density=False, histtype="step", lw=2.5,
            color=color, label=label
        )

        abs_r = np.abs(r)
        for thr in rocof_thresholds:
            q = float((abs_r > thr).mean())
            quantiles[(co2_r, thr)] = q
            logger.info(
                f"Co2L={co2_r}: P(|RoCoF| > {thr} Hz/s) = {q:.4f}"
            )

    ax_signed.set_yscale("log")
    ax_abs.set_yscale("log")

    set_axis_style(
        ax_signed, xlabel=r"$\mathrm{RoCoF}$ [\unit{\hertz\per\second}]", ylabel="Count",
        label_fontsize=22, tick_labelsize=18,
    )
    set_axis_style(
        ax_abs, xlabel=r"$|\mathrm{RoCoF}|$ [\unit{\hertz\per\second}]", ylabel="Count",
        label_fontsize=22, tick_labelsize=18,
    )
    ax_abs.set_xlim(left=0)
    add_panel_label(ax_signed, 0)
    add_panel_label(ax_abs, 1)
    leg = ax_signed.legend(
        title="CO$_2$ level \n[\\% of 1990]",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    leg.get_title().set_horizontalalignment("center")

    # Arrange the panels, leaving room at the bottom for the quantile table.
    fig.tight_layout(rect=(0, 0.2, 1, 1))

    # Summary table below the panels: the empirical quantile P(|RoCoF| <= threshold)
    # for every (co2 level, threshold). usetex is enabled globally, so this renders in
    # the LaTeX font, and matplotlib's table draws proper cell lines.
    present_co2 = [
        co2_r for co2_r in co2_lvls_tup
        if any((co2_r, thr) in quantiles for thr in rocof_thresholds)
    ]
    # Build the table manually so the top-left corner cell labels the first column as the
    # CO2 levels, and each threshold header carries its Hz/s unit.
    header_row = [r"CO$_2$ level"] + [
        rf"${thr:g}$ Hz/s" for thr in rocof_thresholds
    ]
    table_data = [header_row]
    for co2_r in present_co2:
        table_data.append(
            [_co2_percent_label(co2_r, n_nodes)]
            + [f"{quantiles[(co2_r, thr)]:.3f}" for thr in rocof_thresholds]
        )

    tbl_ax = fig.add_axes((0.2, 0.01, 0.6, 0.13))
    tbl_ax.axis("off")
    # Caption above the table (not inside it / not the table title).
    fig.text(
        0.5,
        0.185,
        r"Quantile $P(|\mathrm{RoCoF}| > \mathrm{threshold})$",
        ha="center",
        va="bottom",
        fontsize=16,
    )
    table = tbl_ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.6)
    # Size every column to its content so the first column fits its (wider) labels.
    table.auto_set_column_width(col=list(range(len(header_row))))
    # Shade the header row and the CO2-level column to set them apart.
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == 0:
            cell.set_facecolor("whitesmoke")

    if save_fig:
        fig_name = f"rocof_histogram_values_n_absolute_n{n_nodes}"
        fig_name += f"_minLoadShare{min_load_share:g}"
        save_figure(fig, fig_name, path_to_figures_sclopf)

    return fig, (ax_signed, ax_abs), quantiles


def plot_rocof_exceedance_probability(
    co2_lvls_tup: tuple[float, ...]=(0.6, 0.2, 0.0),
    n_nodes: int = 600,
    min_load_share: float = 0.1,
    save_fig: bool = True,
    zoom_rocof_tup: tuple[float, float] | None = (0, 15),
):
    """Plot the empirical exceedance probability of the absolute RoCoF per co2 level:
    ``|RoCoF|`` [Hz/s] on the x-axis against ``P(|RoCoF| > x)`` on the y-axis.

    This is the complementary CDF (survival function, ``1 - CDF``) of ``|RoCoF|`` — the
    continuous counterpart of the per-threshold quantiles in
    ``plot_rocof_histogram_values_n_absolute``. ``min_load_share`` (a fraction in
    ``[0, 1]``) keeps only components carrying at least that share of the load
    (``load_share >= min_load_share``; use ``0`` to keep all); it is always appended to the
    save path as a ``_minLoadShare{value}`` suffix.

    ``zoom_rocof_tup`` is an ``(xmin, xmax)`` window; pass ``None`` to apply no zoom (a
    single full-range panel). When a window is given, the figure has two panels — the
    left shows the full range (with the zoom window shaded), the right the zoom — and the
    save path gets an ``_xzoom{xmin}-{xmax}`` suffix so it does not overwrite the full
    one. Returns ``(fig, axes)`` where ``axes`` is the list of panel axes."""

    components_fpath = path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    comp_props_all = pd.read_hdf(components_fpath)

    comp_props_all = comp_props_all[comp_props_all["load_share"] >= min_load_share]

    # Compute each co2 level's empirical exceedance curve once; reused across panels.
    curves = []
    for co2_r in co2_lvls_tup:
        vals = comp_props_all.loc[comp_props_all.co2l == co2_r, "rocof"].to_numpy()
        r = vals[np.isfinite(vals)]
        if len(r) == 0:
            logger.warning(f"Co2L={co2_r}: no finite RoCoF values — skipping.")
            continue
        # Empirical exceedance probability P(|RoCoF| > x) at each observed value.
        abs_sorted = np.sort(np.abs(r))
        exceedance = 1.0 - np.arange(1, abs_sorted.size + 1) / abs_sorted.size
        curves.append(
            (abs_sorted, exceedance, get_co2_color(co2_r), _co2_percent_label(co2_r, n_nodes))
        )

    def _draw(target_ax):
        for abs_sorted, exceedance, color, label in curves:
            target_ax.plot(
                abs_sorted, exceedance, drawstyle="steps-post", lw=2.5,
                color=color, label=label,
            )
        target_ax.set_yscale("log")

    xlabel = r"$|\mathrm{RoCoF}|$ [\unit{\hertz\per\second}]"
    ylabel = r"$P(|\mathrm{RoCoF}| > x)$"

    if zoom_rocof_tup is None:
        fig, ax = plt.subplots(figsize=SINGLE_PLOT_SIZE)
        axes = [ax]
        _draw(ax)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, label_fontsize=22, tick_labelsize=18)
        ax.set_xlim(left=0)
        legend_ax = ax
    else:
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=TWO_PANEL_SIZE)
        axes = [ax_full, ax_zoom]
        _draw(ax_full)
        _draw(ax_zoom)
        # Left: full range (starting at 0) with the zoom window shaded. Right: the zoom.
        ax_full.axvspan(zoom_rocof_tup[0], zoom_rocof_tup[1], color="grey", alpha=0.2)
        ax_full.set_xlim(left=0)
        ax_zoom.set_xlim(zoom_rocof_tup)
        # Re-autoscale the zoom panel's y-axis to the data inside the zoom window only —
        # matplotlib otherwise keeps the full-data y-range after set_xlim.
        xmin, xmax = zoom_rocof_tup
        y_in_window = [
            exceedance[(abs_sorted >= xmin) & (abs_sorted <= xmax)]
            for abs_sorted, exceedance, _, _ in curves
        ]
        y_in_window = (
            np.concatenate(y_in_window) if y_in_window else np.array([])
        )
        y_in_window = y_in_window[y_in_window > 0]  # log axis: drop tail zeros
        if y_in_window.size:
            ax_zoom.set_ylim(y_in_window.min() * 0.8, y_in_window.max() * 1.25)
        # Independent y-limits (no sharey); only the left panel carries the y-label since
        # both panels show the same quantity.
        set_axis_style(ax_full, xlabel=xlabel, ylabel=ylabel, label_fontsize=22, tick_labelsize=18)
        set_axis_style(ax_zoom, xlabel=xlabel, label_fontsize=22, tick_labelsize=18)
        add_panel_label(ax_full, 0)
        add_panel_label(ax_zoom, 1)
        legend_ax = ax_full

    leg = legend_ax.legend(
        title="CO$_2$ level \n[\\% of 1990]",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    leg.get_title().set_horizontalalignment("center")

    fig.tight_layout()

    if save_fig:
        fig_name = f"rocof_exceedance_probability_n{n_nodes}"
        fig_name += f"_minLoadShare{min_load_share:g}"
        if zoom_rocof_tup is not None:
            fig_name += f"_xzoom{zoom_rocof_tup[0]:g}-{zoom_rocof_tup[1]:g}"
        save_figure(fig, fig_name, path_to_figures_sclopf)

    return fig, axes


def plot_all_rocof_figures(
    co2_lvls_ls: tuple[float, ...]=(0.6, 0.2, 0.0),
    n_nodes: int = 600,
    min_load_share_ls: list[float] = [0., 0.1, .8],
    rocof_thresholds: tuple[float, ...] = (0.8,0.9,1.0, 1.1, 1.2),
    save_fig: bool = True,
):
    """Produce all RoCoF statistics figures in one call.

    Plots:
    - the two-panel signed/absolute RoCoF histogram with the exceedance-probability
      table (``plot_rocof_histogram_values_n_absolute``),
    - the absolute RoCoF exceedance-probability curve
      (``plot_rocof_exceedance_probability``),
    - the per-threshold split-risk panels
      (``plot_split_risk_different_rocof_thresholds``),
    - the per-category split-risk panels comparing thresholds
      (``plot_split_risk_categories_across_rocof_thresholds``), and
    - the 80–100% load-loss comparison across thresholds
      (``plot_split_risk_high_loss_across_rocof_thresholds``).

    ``co2_lvls_ls``, ``n_nodes``, ``min_load_share``, ``save_fig`` are forwarded to the
    histogram/exceedance figures; ``rocof_thresholds``/``n_nodes``/``save_fig`` to the
    split-risk figures."""

    for min_load_share in min_load_share_ls:
        plot_rocof_histogram_values_n_absolute(
            co2_lvls_tup=co2_lvls_ls,
            n_nodes=n_nodes,
            min_load_share=min_load_share,
            save_fig=save_fig,
        )
        plot_rocof_exceedance_probability(
            co2_lvls_tup=co2_lvls_ls,
            n_nodes=n_nodes,
            min_load_share=min_load_share,
            save_fig=save_fig,
            zoom_rocof_tup=(0, 5)
    )

    # Split-risk figures across RoCoF thresholds (independent of min_load_share, so drawn
    # once rather than per min_load_share).
    plot_split_risk_different_rocof_thresholds(
        rocof_thresholds=rocof_thresholds,
        n_nodes=n_nodes,
        save_fig=save_fig,
    )
    plot_split_risk_categories_across_rocof_thresholds(
        rocof_thresholds=rocof_thresholds,
        n_nodes=n_nodes,
        save_fig=save_fig,
    )
    plot_split_risk_high_loss_across_rocof_thresholds(
        rocof_thresholds=rocof_thresholds,
        n_nodes=n_nodes,
        save_fig=save_fig,
    )

