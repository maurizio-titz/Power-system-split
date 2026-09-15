import os
from itertools import product
import sys


sys.path.append("./")
sys.path.append("./../")
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.config import path_to_ac_dc_comparison, path_to_figures_sclopf
from utils.plot_style import setup_matplotlib_style, AXIS_LABEL_FONTSIZE, get_co2_color, TWO_PANEL_SIZE,TICK_LABEL_FONTSIZE, add_panel_label
from utils.data_handling import get_co2_levels, load_pypsa_network

fig_dir = os.path.join(path_to_figures_sclopf, "ac_dc_comparison")
os.makedirs(fig_dir, exist_ok=True)

# %%
n_nodes = 600
co2lvls = get_co2_levels(n_nodes)
co2lvl = co2lvls[-1]

pf_idxs = range(10)
date_idxs = range(5)
tol = 1e-6  # absolute accuracy/tolerance

n = load_pypsa_network(co2lvl, n_nodes)
n_sub = n.sub_networks.obj[0]

# %%
# --- Collect AC and DC loading for every line, snapshot, repetition, per CO2 level ---
ALL_CO2_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ALL_REAL_PERCENTS = [0, 5, 10, 20, 30, 40, 50, 58]

loading_df_by_co2 = {}
for co2lvl in ALL_CO2_LEVELS:
    ratio_records = []
    ac_records = []
    dc_records = []

    s_nom = n_sub.branches().xs('Line').s_nom

    for date_idx, pf_idx in product(date_idxs, pf_idxs):
        tag = f'_date{date_idx}_pf{pf_idx}'
        filename_dc = f"/dc_powerflow_Co2L{co2lvl}_n{n_nodes}{tag}.h5"
        filename_ac = f"/ac_powerflow_Co2L{co2lvl}_n{n_nodes}{tag}.h5"

        dc_p0 = pd.read_hdf(path_to_ac_dc_comparison + filename_dc)
        ac_p0 = pd.read_hdf(path_to_ac_dc_comparison + filename_ac, key='p0')
        ac_q0 = pd.read_hdf(path_to_ac_dc_comparison + filename_ac, key='q0')

        dc_loading = dc_p0.fillna(0).abs().divide(s_nom.reindex(dc_p0.columns), axis=1)
        ac_calc = np.sqrt(ac_p0.fillna(0) ** 2 + ac_q0.fillna(0) ** 2)
        ac_loading = ac_calc.divide(s_nom.reindex(ac_p0.columns), axis=1)

        ratio = ac_loading / dc_loading.replace(0, np.nan)  # avoid div-by-zero

        ratio_records.append(ratio.stack().rename("ratio"))
        ac_records.append(ac_loading.stack().rename("ac_loading"))
        dc_records.append(dc_loading.stack().rename("dc_loading"))

    ratio_data = pd.concat(ratio_records)
    ac_loading_data = pd.concat(ac_records)
    dc_loading_data = pd.concat(dc_records)

    for s in (ratio_data, ac_loading_data, dc_loading_data):
        s.index.names = ["snapshot", "Line"]

    # combine into one frame, keyed identically -- avoids any alignment issues
    loading_df = pd.concat(
        [ac_loading_data, dc_loading_data, ratio_data], axis=1
    ).dropna()
    loading_df_by_co2[co2lvl] = loading_df


# %% [markdown]
# ## Function definitions

# %%
def plot_ac_dc_ratio_hist_with_median(
    loading_df, dc_threshold=0.2, n_bins=30, fig=None, ax=None,
    xscale="linear", xlim=(0, 2),
):
    """Histogram of AC/DC ratio with a reference line at 1 and the median annotated.

    ``xscale="linear"`` (default) matches the left panel used elsewhere in this
    file: fixed-range bins over ``xlim``, zero-ratio entries excluded.
    ``xscale="log"`` instead uses log-symmetric bins around ratio=1 and ignores
    ``xlim``.
    """
    filtered = loading_df[loading_df["dc_loading"] > dc_threshold]

    # drop zero/negative/non-finite ratios
    ratio_vals = filtered["ratio"]
    valid_mask = (ratio_vals > 0) & np.isfinite(ratio_vals)
    n_dropped = (~valid_mask).sum()
    filtered = filtered[valid_mask]

    if len(filtered) == 0:
        raise ValueError(f"No valid entries with dc_loading > {dc_threshold}")

    if n_dropped > 0:
        print(f"Dropped {n_dropped} entries with ratio <= 0 or non-finite "
              f"(likely dc_loading == 0)")

    ratio_median = filtered["ratio"].median()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    if xscale == "log":
        max_log = np.abs(np.log10(filtered["ratio"])).max()
        bins = np.logspace(-max_log, max_log, n_bins + 1)
    else:
        bins = np.linspace(xlim[0], xlim[1], n_bins + 1)

    ax.hist(filtered["ratio"], bins=bins, color="steelblue", edgecolor="white")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="AC = DC")
    ax.axvline(ratio_median, color="crimson", linestyle="-", linewidth=1.5,
               label=f"median ratio = {ratio_median:.2f}")

    if xscale == "log":
        ax.set_xscale("log")
        if np.abs(bins[0]) > np.abs(bins[-1]):
            lim = 1 / np.abs(bins[0])
        else:
            lim = np.abs(bins[-1])
        ax.set_xlim(1 / lim, lim)
    else:
        ax.set_xlim(*xlim)
    return fig, ax


def select_lines_by_loading(loading_df, snapshot, targets=(1.0, 0.8),
                             n_per_target=2, n_low=2, low_quantile=0.6):
    """
    For a single snapshot, pick lines closest to each target DC loading
    level, plus a set of lower-loaded lines.

    Args:
        loading_df: DataFrame with ['ac_loading','dc_loading','ratio'],
                    indexed by ['snapshot', 'Line']
        snapshot: the snapshot value to select on
        targets: DC loading levels to match (e.g. near 100%, near 80%)
        n_per_target: how many lines to pick per target level
        n_low: how many "lower loading" lines to pick
        low_quantile: pick low lines from below this quantile of DC loading
                      (excluding lines already picked for targets)

    Returns:
        dict: {label: [line_names]}, e.g.
              {"~100%": [...], "~80%": [...], "lower loading": [...]}
    """
    sub = loading_df.xs(snapshot, level="snapshot")
    mean_dc = sub.groupby("Line")["dc_loading"].mean().dropna()

    selected = {}
    used = set()

    for target in targets:
        remaining = mean_dc.drop(index=used, errors="ignore")
        closest = (remaining - target).abs().sort_values().index[:n_per_target]
        label = f"~{int(target * 100)}\%"
        selected[label] = list(closest)
        used.update(closest)
    return selected


def plot_snapshot_ratio_grid(loading_df, snapshot, line_groups,
                              dc_threshold=0.0, n_bins=15):
    """
    Grid of ratio histograms for a single snapshot, grouped by loading
    category (e.g. "~100%", "~80%", "lower loading").

    Args:
        loading_df: DataFrame with ['ac_loading','dc_loading','ratio']
        snapshot: snapshot to filter to
        line_groups: dict {label: [line_names]}, e.g. from select_lines_by_loading
        dc_threshold: optional additional DC loading filter within the snapshot
        n_bins: number of log-spaced bins per subplot
    """
    sub_snapshot = loading_df.xs(snapshot, level="snapshot")

    all_lines = [(label, line) for label, lines in line_groups.items() for line in lines]
    n_cols = 2
    n_rows = len(all_lines) // n_cols + (len(all_lines) % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (label, line) in enumerate(all_lines):
        ax = axes[i]
        line_data = sub_snapshot.loc[[line]]
        line_data = line_data[line_data["dc_loading"] > dc_threshold]

        ratio_vals = line_data["ratio"]
        valid = (ratio_vals > 0) & np.isfinite(ratio_vals)
        ratio_vals = ratio_vals[valid]

        if len(ratio_vals) == 0:
            ax.set_title(f"{label} — Line {line}\n(no valid data)", fontsize=9)
            ax.axis("off")
            continue

        dc_mean = line_data["dc_loading"][valid].mean()
        median = ratio_vals.median()
        max_log = np.abs(np.log10(ratio_vals)).max()
        bins = np.logspace(-max_log, max_log, n_bins + 1) if max_log > 0 else \
            np.linspace(ratio_vals.min() * 0.9, ratio_vals.max() * 1.1, n_bins + 1)

        ax.hist(ratio_vals, bins=bins, color="steelblue", edgecolor="white")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(median, color="crimson", linestyle="-", linewidth=1.5)

        if max_log > 0:
            ax.set_xscale("log")
            ax.minorticks_off()
            ticks = np.round(np.geomspace(bins[0], bins[-1], 5), 2)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

        ax.set_title(
            f"{label} — Line {line}\nDC={int(dc_mean*100)}\%, median ratio={median:.2f}, n={len(ratio_vals)}",
            fontsize=9,
        )
        ax.set_xlabel("AC / DC loading", fontsize=8)
        ax.set_ylabel("Counts", fontsize=8)
        ax.tick_params(labelsize=8)

    for j in range(len(all_lines), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"AC/DC ratio by loading level — snapshot {snapshot}", fontsize=12)
    plt.tight_layout()
    return fig, axes


def compute_ac_dc_metrics(loading_df: pd.DataFrame, dc_threshold: float = 0.0):
    """Compute the Q-spread metric from a single CO2 level's loading_df.

    Parameters
    ----------
    loading_df : pd.DataFrame
        MultiIndex (snapshot, Line) with columns ["ac_loading", "dc_loading", ...].
    dc_threshold : float
        Only keep (snapshot, Line) rows with DC loading above this value
        (same units as the ``dc_loading`` column, e.g. 0.2 for "20% loaded").
        Lightly-loaded lines are excluded because a tiny DC denominator can
        turn small absolute Q-driven swings into huge *relative* swings that
        are numerical noise rather than an operationally meaningful spread.

    Returns
    -------
    diff_ac_dc : pd.Series
        AC - DC loading [percentage points], one value per (snapshot, Line,
        power-factor sample) row -- i.e. same length/index as the filtered
        loading_df.
    spread_q : pd.Series
        max_pf(AC) - min_pf(AC) loading [percentage points], one value per
        unique (snapshot, Line) pair (duplicates across pf samples collapsed).
    """
    if dc_threshold > 0:
        loading_df = loading_df[loading_df["dc_loading"] > dc_threshold]

    grouped = loading_df.divide(loading_df["dc_loading"], axis=0).groupby(level=["snapshot", "Line"])["ac_loading"]
    ac_max = grouped.transform("max")
    ac_min = grouped.transform("min")

    diff_ac_dc = (loading_df["ac_loading"] - loading_df["dc_loading"]) / loading_df["dc_loading"]
    spread_q_full = (ac_max - ac_min)
    # one row per (snapshot, Line) -- the value is identical across the pf
    # duplicates by construction (it's a transform), so .first() is exact,
    # not a lossy aggregation.
    spread_q = spread_q_full.groupby(level=["snapshot", "Line"]).first()

    return diff_ac_dc, spread_q


def _format_co2_label(value: float) -> str:
    """Format a CO2 percentage for display, robust to either input convention.

    Accepts either a fraction (0-1, e.g. 0.58) or an already-scaled percentage
    (0-100, e.g. 58) and always renders "CO2 = NN%".
    """
    pct = value * 100 if abs(value) <= 1 else value
    return f"CO$_2$ = {pct:.0f}\%"


def _set_scientific_y(ax):
    """Force scientific-notation y-tick labels -- counts range from ~1e3 to ~1e5."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(TICK_LABEL_FONTSIZE)


def _fixed_range_clip_count(data, lo, hi):
    """Count how many values fall outside [lo, hi] -- used purely for the
    'N outliers outside view' annotation, not for computing the bin range."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n_clipped = int(((data < lo) | (data > hi)).sum())
    return n_clipped, len(data)


def plot_ac_dc_ratio_hist(
    loading_df: pd.DataFrame,
    dc_threshold: float = 0.0,
    n_bins: int = 100,
    fig=None,
    ax=None,
    color: str = "steelblue",
    xlim: tuple = (0, 2),
):
    """Histogram of AC/DC loading ratio with linear, fixed-range bins.

    Bin edges are ``np.linspace(xlim[0], xlim[1], n_bins + 1)``, identical
    for every caller so rows stay comparable. Values outside ``xlim`` are
    excluded from the histogram; the caller is responsible for reporting how
    many were dropped.

    Parameters
    ----------
    loading_df : pd.DataFrame
        Must contain columns "dc_loading" and "ratio" (= ac_loading / dc_loading).
    dc_threshold : float
        Only keep rows with DC loading above this value (see
        ``compute_ac_dc_metrics`` for rationale).
    n_bins : int
        Number of linearly-spaced bins.
    fig, ax : matplotlib Figure/Axes, optional
        If not given, a new figure/axes is created.
    color : str
        Histogram bar color.
    xlim : tuple
        Fixed (lo, hi) range the bins span.

    Returns
    -------
    fig, ax : the (possibly newly created) Figure and Axes.
    """
    filtered = loading_df
    if dc_threshold > 0:
        filtered = filtered[filtered["dc_loading"] > dc_threshold]

    ratio = filtered["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio != 0]  # exclude zero-ratio (ac_loading == 0) entries

    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(ratio, bins=bins, color=color, edgecolor="white")

    return fig, ax


def plot_ac_dc_vs_q_uncertainty(
    loading_df_by_co2: dict,
    co2_levels: list,
    real_percents: list,
    n_bins_left: int = 100,
    n_bins_right: int = 60,
    dc_threshold: float = 0.0,
    xlim_left: tuple = (0, 2),
    xlim_right: tuple = (0, 20),
    figsize: tuple | None = None,
):
    """3-row x 2-col figure: rows = CO2 levels, columns = [AC/DC ratio, Q spread].

    Left column uses :func:`plot_ac_dc_ratio_hist` (linear-x, fixed-range
    bins spanning ``xlim_left``, vertical reference line at ratio=1).

    Right column shows the Q-driven spread (``max_pf(AC) - min_pf(AC)``,
    percentage points). Both columns share IDENTICAL bin edges per column
    across rows: the bin range is simply ``xlim_left``/``xlim_right``, split
    into equal-width bins. Values outside the respective xlim are excluded
    from the histogram, and their count is annotated on each panel so
    nothing is silently hidden.

    ``dc_threshold`` restricts both columns to (snapshot, Line) pairs whose DC
    loading exceeds this value (e.g. 0.2 for "only lines loaded >20%").
    """
    n_rows = len(co2_levels)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=figsize or (11, 2.6 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # --- pre-compute spread_q for every CO2 level ---
    spread_q_by_co2 = {}
    diff_ac_dc_by_co2 = {}

    for co2lvl in co2_levels:
        diff_ac_dc, spread_q = compute_ac_dc_metrics(loading_df_by_co2[co2lvl], dc_threshold=dc_threshold)
        spread_q_by_co2[co2lvl] = spread_q
        diff_ac_dc_by_co2[co2lvl] = diff_ac_dc

    # bin range = xlim_right directly, split into n_bins_right equal bins --
    # identical for every row, so bin width = diff(xlim_right) / n_bins_right
    bins_right = np.linspace(xlim_right[0], xlim_right[1], n_bins_right + 1)

    for i, co2lvl in enumerate(co2_levels):
        loading_df = loading_df_by_co2[co2lvl]
        spread_q = spread_q_by_co2[co2lvl]

        ax_left, ax_right = axes[i]

        # --- left: AC/DC ratio histogram, shared fixed-range bins across all rows ---
        ratio_filtered = loading_df
        if dc_threshold > 0:
            ratio_filtered = ratio_filtered[ratio_filtered["dc_loading"] > dc_threshold]
        ratio_left = ratio_filtered["ratio"].replace([np.inf, -np.inf], np.nan)
        ratio_left = ratio_left[ratio_left != 0]
        n_clip_left, n_left = _fixed_range_clip_count(ratio_left, xlim_left[0], xlim_left[1])
        plot_ac_dc_ratio_hist(
            loading_df, dc_threshold=dc_threshold, n_bins=n_bins_left,
            fig=fig, ax=ax_left, color="steelblue", xlim=xlim_left,
        )
        ax_left.set_xlim(*xlim_left)
        ax_left.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)

        # --- right: Q-spread histogram, shared fixed-range bins across all rows ---
        n_clip_right, n_right = _fixed_range_clip_count(spread_q, xlim_right[0], xlim_right[1])
        ax_right.hist(spread_q, bins=bins_right, color="indianred", edgecolor="white")
        ax_right.set_xlim(*xlim_right)

        for ax, n_clip, n_total in (
            (ax_left, n_clip_left, n_left),
            (ax_right, n_clip_right, n_right),
        ):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)

        for ax in (ax_left, ax_right):
            ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
            _set_scientific_y(ax)
            ax.text(
                0.02, 0.92, _format_co2_label(real_percents[i]),
                transform=ax.transAxes, fontsize=AXIS_LABEL_FONTSIZE,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
            )

    axes[-1, 0].set_xlabel("AC loading / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
    axes[-1, 1].set_xlabel(
        r"$(\max_{Q}$ AC loading $-$ $\min_{Q}$ AC loading)" "\n" r"/ DC loading",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    return fig, axes


def plot_ac_dc_vs_q_uncertainty_combined(
    loading_df_by_co2: dict,
    co2_levels: list,
    real_percents: list,
    n_bins_left: int = 60,
    n_bins_right: int = 40,
    dc_threshold: float = 0.2,
    xlim_left: tuple = (0, 2),
    xlim_right: tuple = (0, 10),
    figsize: tuple = TWO_PANEL_SIZE,
):
    """2-panel version of :func:`plot_ac_dc_vs_q_uncertainty`: all CO2 levels
    overlaid in a single axes per column (left = AC/DC ratio, right = Q spread)
    instead of one row per level.

    Uses ``histtype='step'`` and :func:`get_co2_color` so levels stay
    distinguishable and colors match every other CO2-level figure. Both panels
    share the same y-limits so counts are directly comparable; if the step
    outlines end up mostly on top of each other this may be worth revisiting,
    but the CO2-level color mapping should stay as-is.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    bins_left = np.linspace(xlim_left[0], xlim_left[1], n_bins_left + 1)
    bins_right = np.linspace(xlim_right[0], xlim_right[1], n_bins_right + 1)

    for co2lvl, real_pct in zip(co2_levels, real_percents):
        loading_df = loading_df_by_co2[co2lvl]
        color = get_co2_color(co2lvl)
        label = _format_co2_label(real_pct)

        filtered = loading_df
        if dc_threshold > 0:
            filtered = filtered[filtered["dc_loading"] > dc_threshold]
        ratio = filtered["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio != 0]  # exclude zero-ratio (ac_loading == 0) entries

        _, spread_q = compute_ac_dc_metrics(loading_df, dc_threshold=dc_threshold)

        ax_left.hist(ratio, bins=bins_left, histtype="step", lw=2.5, color=color, label=label,alpha = .9)
        ax_right.hist(spread_q, bins=bins_right, histtype="step", lw=2.5, color=color, label=label,alpha = .9)

    ax_left.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)
    ax_left.set_xlim(*xlim_left)
    ax_right.set_xlim(*xlim_right)

    y_max = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
    # ax_left.set_ylim(0, y_max)
    # ax_right.set_ylim(0, y_max)

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
        _set_scientific_y(ax)

    ax_left.set_xlabel("AC loading / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
    ax_right.set_xlabel(
        r"$(\max_{Q}$ AC loading $-$ $\min_{Q}$ AC loading)" "\n" r"/ DC loading",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax_left.legend(title="CO$_2$ level", loc="upper right", frameon=True)
    ax_right.legend(title="CO$_2$ level", loc="upper right", frameon=True)

    fig.tight_layout()
    return fig, (ax_left, ax_right)


def plot_ac_dc_ratio_log_lin_spread_q_combined(
    loading_df_by_co2: dict,
    co2_levels: list,
    real_percents: list,
    n_bins: int = 60,
    dc_threshold: float = 0.2,
    xlim_log: tuple = (1e-1, 1e1),
    xlim_ratio: tuple = (0, 2),
    xlim_spread_q: tuple = (0, .1),
    figsize: tuple = (16, 5),
):
    """3-panel figure, all CO2 levels overlaid as step histograms per panel:

    left   = AC/DC ratio, log x-scale, fixed range ``xlim_log``
    middle = AC/DC ratio, linear x-scale, fixed range ``xlim_ratio``
    right  = (AC - DC) / DC loading ("delta"), linear x-scale, fixed range
             ``xlim_delta``

    Uses ``histtype='step'`` and :func:`get_co2_color` so levels stay
    distinguishable and colors match every other CO2-level figure.
    """
    fig, (ax_log, ax_lin, ax_spread_q) = plt.subplots(1, 3, figsize=figsize)

    ratios = {}
    spread_qs = {}
    for co2lvl in co2_levels:
        loading_df = loading_df_by_co2[co2lvl]
        filtered = loading_df
        if dc_threshold > 0:
            filtered = filtered[filtered["dc_loading"] > dc_threshold]
        ratio = filtered["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio != 0]  # exclude zero-ratio (ac_loading == 0) entries
        ratios[co2lvl] = ratio

        _, spread_q = compute_ac_dc_metrics(loading_df, dc_threshold=dc_threshold)
        spread_qs[co2lvl] = spread_q

    # shared log bins across all CO2 levels, fixed range xlim_log
    bins_log = np.logspace(np.log10(xlim_log[0]), np.log10(xlim_log[1]), n_bins + 1)
    bins_lin = np.linspace(xlim_ratio[0], xlim_ratio[1], n_bins + 1)
    bins_spread_q = np.linspace(xlim_spread_q[0], xlim_spread_q[1], n_bins + 1)

    for co2lvl, real_pct in zip(co2_levels, real_percents):
        color = get_co2_color(co2lvl)
        label = _format_co2_label(real_pct)

        ax_log.hist(ratios[co2lvl], bins=bins_log, histtype="step", lw=2.5, color=color, label=label, alpha=.9)
        ax_lin.hist(ratios[co2lvl], bins=bins_lin, histtype="step", lw=2.5, color=color, label=label, alpha=.9)
        ax_spread_q.hist(spread_qs[co2lvl], bins=bins_spread_q, histtype="step", lw=2.5, color=color, label=label, alpha=.9)

    ax_log.set_xscale("log")
    ax_log.set_xlim(bins_log[0], bins_log[-1])
    ax_lin.set_xlim(*xlim_ratio)
    ax_spread_q.set_xlim(*xlim_spread_q)

    ax_log.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)
    ax_lin.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)
    ax_spread_q.axvline(0.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)

    for ax in (ax_log, ax_lin, ax_spread_q):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
        _set_scientific_y(ax)
        ax.legend(title="CO$_2$ level", loc="upper right", frameon=True)

    ax_log.set_xlabel("AC / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
    ax_lin.set_xlabel("AC / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spread_q.set_xlabel(r"$(\max_{Q}$ AC loading $-$ $\min_{Q}$ AC loading)" "\n" r"/ DC loading",
    fontsize=AXIS_LABEL_FONTSIZE,)


    fig.tight_layout()
    return fig, (ax_log, ax_lin, ax_spread_q)


def plot_ac_dc_ratio_log_lin_combined(
    loading_df_by_co2: dict,
    co2_levels: list,
    real_percents: list,
    n_bins: int = 60,
    dc_threshold: float = 0.2,
    xlim_log: tuple = (1e-1, 1e1),
    xlim_ratio: tuple = (0, 2),
    figsize: tuple = (32 / 3, 5),
    legend_fontsize: int = 18,
):
    """2-panel version of :func:`plot_ac_dc_ratio_log_lin_spread_q_combined`,
    dropping the Q-spread panel:

    left  = AC/DC ratio, log x-scale, fixed range ``xlim_log``
    right = AC/DC ratio, linear x-scale, fixed range ``xlim_ratio``

    Both panels show the same quantity, so the two x-ranges are deliberately
    independent: the log panel is centred on ratio=1 to resolve the tails, the
    linear one keeps the conventional 0-2 window. All CO2 levels are overlaid
    as ``histtype='step'`` outlines coloured by :func:`get_co2_color`.

    Bins and x-limits match the first two panels of the 3-panel version, and
    the default ``figsize`` is two of its panels wide (16/3 in each), so the
    two figures can be compared directly. Each panel keeps its own autoscaled
    y-limit -- the log and linear bins have different widths, so their counts
    are not directly comparable anyway.
    """
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=figsize)

    ratios = {}
    for co2lvl in co2_levels:
        filtered = loading_df_by_co2[co2lvl]
        if dc_threshold > 0:
            filtered = filtered[filtered["dc_loading"] > dc_threshold]
        ratio = filtered["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ratios[co2lvl] = ratio[ratio != 0]  # exclude zero-ratio (ac_loading == 0) entries

    # shared bins across all CO2 levels, fixed ranges so panels stay comparable
    bins_log = np.logspace(np.log10(xlim_log[0]), np.log10(xlim_log[1]), n_bins + 1)
    bins_lin = np.linspace(xlim_ratio[0], xlim_ratio[1], n_bins + 1)

    for co2lvl, real_pct in zip(co2_levels, real_percents):
        color = get_co2_color(co2lvl)
        # bare percentage -- the shared legend title carries "[% of 1990]"
        label = f"{int(real_pct)}\%"

        ax_log.hist(ratios[co2lvl], bins=bins_log, histtype="step", lw=2.5, color=color, label=label, alpha=.9)
        ax_lin.hist(ratios[co2lvl], bins=bins_lin, histtype="step", lw=2.5, color=color, label=label, alpha=.9)

    ax_log.set_xscale("log")
    ax_log.set_xlim(bins_log[0], bins_log[-1])
    ax_lin.set_xlim(*xlim_ratio)

    for i, ax in enumerate((ax_log, ax_lin)):
        ax.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)
        ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("AC / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
        _set_scientific_y(ax)
        add_panel_label(ax, i)

    # one legend in the linear panel only -- the colors are shared, so
    # repeating it in the log panel would just eat plotting area
    leg = ax_lin.legend(
        title="CO$_2$ level \n[\\% of 1990]",
        loc="upper right", frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
    )
    # centre the two title lines against each other (default is left-aligned)
    leg.get_title().set_multialignment("center")

    fig.tight_layout()
    return fig, (ax_log, ax_lin)


# %% [markdown]
# ## Figures
def plot_dc_ac_comparison():
    # %%
    # --- Collective AC/DC ratio histograms, one row per CO2 level ---
    setup_matplotlib_style()

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8),
        constrained_layout=True,
    )
    axes = axes.flatten()
    Real_percents = [58,20,0]
    CO2_levels = [0.6,0.2,0.0]
    for i, ax in enumerate(axes):
        co2lvl = CO2_levels[i]
        loading_df = loading_df_by_co2[co2lvl]

        fig, ax = plot_ac_dc_ratio_hist_with_median(
            loading_df, dc_threshold=0.2, n_bins=40, fig=fig, ax=ax,
            xscale="linear", xlim=(0, 2),
        )

        # reference line at ratio = 1 (AC == DC loading)
        ax.axvline(1.0, color="0.3", linestyle="--", linewidth=1.5, zorder=5)

        ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)
        _set_scientific_y(ax)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(title=f'CO$_2$ = {(Real_percents[i])}\%', loc='upper right', frameon=True)
    axes[-1].set_xlabel("AC loading / DC loading", fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ac_dc_ratio_hist_by_co2_level.pdf"))

    # %%
    # --- AC/DC ratio vs. Q-uncertainty spread, one row per CO2 level ---
    dc_threshold = 0.2
    xlim_right = (0, 1e-1)
    n_bins_right = 40
    fig, axes = plot_ac_dc_vs_q_uncertainty(
        loading_df_by_co2, [0.6,0.2,0], [0.58,0.2,0],
        n_bins_left=n_bins_right, n_bins_right=n_bins_right, dc_threshold=dc_threshold,
        xlim_right=xlim_right,
    )

    fig.savefig(os.path.join(fig_dir, "ac_dc_ratio_vs_q_uncertainty_spread.pdf"))

    # %%
    # --- Same comparison, but 2 panels with all CO2 levels overlaid per panel ---
    fig, axes = plot_ac_dc_vs_q_uncertainty_combined(
        loading_df_by_co2, [0.6,0.2,0.0], [58,20,0],
        n_bins_left=n_bins_right, n_bins_right=n_bins_right, dc_threshold=dc_threshold,
        xlim_right=xlim_right,
    )

    fig.savefig(os.path.join(fig_dir, "ac_dc_ratio_vs_q_uncertainty_spread_combined.pdf"))

    # %%
    # --- 3-panel: AC/DC ratio (log), AC/DC ratio (linear), AC diff (linear) ---
    fig, axes = plot_ac_dc_ratio_log_lin_spread_q_combined(
        loading_df_by_co2, [0.6, 0.2, 0.0], [58, 20, 0],
        dc_threshold=dc_threshold,
    )

    fig.savefig(os.path.join(fig_dir, "ac_dc_ratio_log_lin_spread_q_combined.pdf"))

    # %%
    # --- 2-panel: AC/DC ratio (log) and AC/DC ratio (linear) only ---
    fig, axes = plot_ac_dc_ratio_log_lin_combined(
        loading_df_by_co2, [0.6, 0.2, 0.0], [58, 20, 0],
        dc_threshold=dc_threshold,
    )

    fig.savefig(os.path.join(fig_dir, "ac_dc_ratio_log_lin_combined.pdf"))


if __name__ == "__main__":
    plot_dc_ac_comparison()