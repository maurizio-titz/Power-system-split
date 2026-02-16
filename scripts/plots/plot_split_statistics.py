#!/usr/bin/env python3
"""
Split Statistics Plot - Figure 5: split statistics
Creates histograms of power imbalance, rotational energy, and loss of load share distributions.
"""

import gzip
import pickle
import sys
import copy
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cycler import cycler

sys.path.append("./")

from utils import config
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure,
)


def create_split_statistics_plot(
    load_normalization=False, show_blackout_stats=True, secondary_proba_axis=False
):
    """Create split statistics plot.

    Parameters:
    -----------
    load_normalization : bool, default False
        Whether to normalize by load share
    show_blackout_stats : bool, default True
        Whether to display panel c (blackout statistics)
    """

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load network
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )

    # Determine number of simulations
    bridge_idxs = data_handling.nx_edges_to_matrix_indices(
        nx.bridges(nx_graph), nx_graph
    )
    n_2_failures = cascade_simulation.calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    selected_co2ls = np.array([0.0, 0.2, 0.6])

    # Load split component properties and split properties
    component_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"component_properties_all_n{n_nodes}.h5"
    )
    component_props.time_stamp = pd.to_datetime(component_props.time_stamp)

    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )

    # Rename columns if necessary
    if "neglidgeable_blackout" in split_props.columns:
        split_props.rename(
            columns={"neglidgeable_blackout": "negligible"}, inplace=True
        )

    # Add snapshot weighting
    component_props["total_weighting"] = (
        component_props["snapshot_weighting"] * component_props["trigger_weighting"]
    )

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Adjust layout based on whether blackout stats are shown
    if show_blackout_stats:
        fig = plt.figure(figsize=(12, 5))
        n_cols = 4
        width_ratios = [1, 1, 1, 1]
    else:
        fig = plt.figure(figsize=(8, 5))
        n_cols = 2
        width_ratios = [1, 1]

    gs_vertical = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 0.1])

    # Panel setup
    if not show_blackout_stats:
        wspace = 0.3
    elif show_blackout_stats and secondary_proba_axis:
        wspace = 0.40
    else:
        wspace = 0.4
    gsTop = GridSpecFromSubplotSpec(
        1,
        n_cols,
        subplot_spec=gs_vertical[0, :],
        width_ratios=width_ratios,
        hspace=0,
        wspace=wspace,
    )

    # Conditionally create panel c and its legend
    if show_blackout_stats:
        ax3_num_splits = fig.add_subplot(gsTop[2])
        ax3_num_splits_legend = fig.add_subplot(gsTop[3])
    else:
        ax3_num_splits = None
        ax3_num_splits_legend = None

    # Panel c: split number and categories (loss of load share distribution)
    if show_blackout_stats:
        cmap = plt.get_cmap("inferno_r")

        bins = np.linspace(0, 100, 6).astype(int)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        data = []
        for co2l in co2ls[::-1]:
            vals = (
                split_props[split_props.co2l == co2l].lost_load_share_blackout.values
                * 100
            )
            counts, _ = np.histogram(vals, bins=bins)
            data.append(counts)

        data = np.stack(data)
        data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

        plt.sca(ax3_num_splits)
        markers = ["o", "s", "D", "^", "v", "."]
        for i, bin_center in enumerate(bin_centers):
            color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
            counts = data.loc[:, bin_center]
            plt.plot(
                get_actual_co2_level(co2ls[::-1], percent=True),
                counts,
                label=rf"{bins[i]}-{bins[i+1]}\%",
                alpha=0.8,
                color=color,
                marker=markers[i % len(markers)],
                markersize=5,
            )

        ax3_num_splits.set_xlabel(
            r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax3_num_splits.set_ylabel(
            "Number of System Splits", fontsize=AXIS_LABEL_FONTSIZE
        )
        x_ticks_major = [0, 20, 40, 60]
        x_ticks_minor = [10, 30, 50]
        ax3_num_splits.set_xticks(x_ticks_major)
        ax3_num_splits.set_xticks(x_ticks_minor, minor=True)
        ax3_num_splits.grid(False)
        ax3_num_splits.tick_params(
            axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE
        )
        ax3_num_splits.set_yscale("log")
        ax3_num_splits.invert_xaxis()

        if secondary_proba_axis:
            # add secondary axis for fraction of total simulations
            ax3_num_splits_secondary = ax3_num_splits.twinx()
            for i, bin_center in enumerate(bin_centers):
                color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
                counts = data.loc[:, bin_center]
                # plot invisible lines to set the ticks
                plt.plot(
                    get_actual_co2_level(co2ls[::-1], percent=True),
                    counts
                    / len(n_2_failures)
                    / network.snapshot_weightings.generators.sum(),
                    label=rf"{bins[i]}-{bins[i+1]}\%",
                    alpha=0.0,
                )
            ax3_num_splits_secondary.set_ylabel(
                "Fraction of total contingencies", fontsize=AXIS_LABEL_FONTSIZE
            )
            ax3_num_splits_secondary.set_yscale("log")
            ax3_num_splits_secondary.tick_params(
                axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE
            )

        # Legend for panel c
        h, l = ax3_num_splits.get_legend_handles_labels()
        if secondary_proba_axis:
            # Place legend below the axis when secondary axis is present
            ax3_num_splits_legend.legend(
                h,
                l,
                title="Share of load\n not served",
                loc="center left",
                bbox_to_anchor=(0.1, 0.5),
                ncols=1,
                columnspacing=0.5,
            )
        else:
            ax3_num_splits_legend.legend(
                h,
                l,
                title="Share of load\n not served",
                loc="center",
                ncols=1,
                columnspacing=0.5,
            )
        ax3_num_splits_legend.axis("off")

    # Filter out very small components
    if load_normalization:
        # component_props_filtered = component_props
        component_props_filtered = component_props[component_props.load_share > 0.1]
    else:
        component_props_filtered = component_props[component_props.load_share > 0.1]

    # Panel b: Inertia histograms
    cmap = plt.get_cmap("cividis")

    ax2_inertia = fig.add_subplot(gsTop[1])
    ax2_inertia_legend = fig.add_subplot(gs_vertical[1, :])
    rot_energy = component_props_filtered.rot_energy / 1000
    if load_normalization:
        rot_energy = component_props_filtered.rot_energy / component_props_filtered.load
    max_vals = [
        rot_energy[component_props_filtered.co2l == co2l].max()
        for co2l in selected_co2ls
    ]
    min_vals = [
        rot_energy[component_props_filtered.co2l == co2l].min()
        for co2l in selected_co2ls
    ]
    bins = np.linspace(min(min_vals), max(max_vals), 15)

    for co2l in selected_co2ls:
        vals = ax2_inertia.hist(
            rot_energy[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.total_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            label=rf"{get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)} \%",
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax2_inertia.set_yscale("log")
    if load_normalization:
        ax2_inertia.set_xlabel(
            "Norm. rotational energy [s]", fontsize=AXIS_LABEL_FONTSIZE
        )
    else:
        ax2_inertia.set_xlabel("Rotational energy [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_inertia.set_ylabel("Number of split components", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_inertia.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    # Panel a: power imbalance histograms
    ax1_imbalance = fig.add_subplot(gsTop[0])
    if load_normalization:
        power_imbalance = (
            component_props_filtered.power_imbalance / component_props_filtered.load
        )
    else:
        power_imbalance = component_props_filtered.power_imbalance / 1000
    bins = np.linspace(power_imbalance.min(), power_imbalance.max(), 15)

    for co2l in selected_co2ls:
        h = ax1_imbalance.hist(
            power_imbalance[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.total_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            linewidth=3,
            label=rf"{get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)} \%",
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax1_imbalance.set_yscale("log")
    ax1_imbalance.set_ylabel("Number of split components", fontsize=AXIS_LABEL_FONTSIZE)
    if load_normalization:
        ax1_imbalance.set_xlabel(
            "Norm. power imbalance [1]", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax1_imbalance.set_xticks(
            np.arange(
                np.ceil(power_imbalance.min()),
                np.floor(power_imbalance.max()) + 1,
                step=1,
            )
        )
    else:
        ax1_imbalance.set_xlabel("Power imbalance [GW]", fontsize=AXIS_LABEL_FONTSIZE)

    ax1_imbalance.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    # ax1_imbalance.set_xticks(np.arange(-50, 51, step=25))

    # Legend for panels a and b
    h, l = ax2_inertia.get_legend_handles_labels()
    # Place the legend title to the left of the labels by using a dummy handle and label
    handles = [Line2D([], [], color="none")] + h[::-1]
    labels = [r"CO$_2$ level [\% of 1990]"] + l[::-1]
    ax2_inertia_legend.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(-0.08, 0.5),
        ncols=4,
        columnspacing=1,
        handletextpad=0.5,
        # frameon=False,
    )
    ax2_inertia_legend.axis("off")

    # Add subplot labels
    axes_to_label = [ax1_imbalance, ax2_inertia]
    if show_blackout_stats and ax3_num_splits is not None:
        axes_to_label.append(ax3_num_splits)

    for idx, ax_loss_lvl in enumerate(axes_to_label):
        add_panel_label(ax_loss_lvl, idx, y_offset=0.07)

    plt.tight_layout()
    # Save figure with consistent style
    save_name = (
        "split_statistics_normalized" if load_normalization else "split_statistics"
    )
    if show_blackout_stats:
        save_name += "_with_blackout_stats"
    if secondary_proba_axis:
        save_name += "_with_secondary_proba_axis"

    save_figure(fig, save_path, save_name)
    plt.show()


def create_blackout_statistics_plot(
    save_path=path_to_figures_sclopf,
    same_corridor=None,
    normalize_by_total_splits=False,
):
    """
    Create a standalone plot showing only the blackout statistics (Number of System Splits).

    Parameters
    ----------
    save_path : str, default path_to_figures_sclopf
        Path where the plot will be saved
    same_corridor : bool or None, default None
        If True, only consider splits where both failed lines are in the same corridor. If False, only consider splits where failed lines are in different corridors.
        If None, consider all splits.
    """
    # Setup
    n_nodes = 600
    os.makedirs(save_path, exist_ok=True)

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)

    # Load split properties
    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    if same_corridor is not None:
        split_props["same_corridor"] = (
            split_props.init_failure_0 == split_props.init_failure_1
        )
        split_props = split_props[split_props.same_corridor == same_corridor]

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Create figure for standalone blackout statistics with legend axis
    fig = plt.figure(figsize=(5, 3.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.3], wspace=0.17)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # Panel: split number and categories (loss of load share distribution)
    cmap = plt.get_cmap("inferno_r")

    bins = np.linspace(0, 100, 6).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data = []
    for co2l in co2ls[::-1]:
        vals = (
            split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
        )
        counts, _ = np.histogram(
            vals,
            weights=split_props[split_props.co2l == co2l].total_weighting,
            bins=bins,
        )
        data.append(counts)
        if normalize_by_total_splits:
            file_path = (
                data_handling.path_to_grid_data + f"n_2_failures_co2lvl{co2l}.pklz"
            )
            with gzip.open(file_path, "rb") as fh:
                n_2_failures = pickle.load(fh)
            network = data_handling.load_pypsa_network(
                n_nodes=n_nodes,
                co2lvl=co2l,
                use_sclopf=True,
                lopt=config.use_extensions,
            )
            if same_corridor is not None:
                n_2_failures = [
                    n2_failure
                    for n2_failure in n_2_failures
                    if (n2_failure["failures"][0][0] == n2_failure["failures"][1][0])
                    is same_corridor
                ]
                weighted_trigger_count = sum(
                    [initial_failure["weight"] for initial_failure in n_2_failures]
                )
                # incorporating the weighting of the snapshots
                number_of_simulations = (
                    weighted_trigger_count
                    * network.snapshot_weightings.generators.sum()
                )
            else:
                weighted_trigger_count = sum(
                    [initial_failure["weight"] for initial_failure in n_2_failures]
                )
                # incorporating the weighting of the snapshots
                number_of_simulations = (
                    weighted_trigger_count
                    * network.snapshot_weightings.generators.sum()
                )
            total_counts = number_of_simulations
            if total_counts > 0:
                data[-1] = counts / total_counts

    data = np.stack(data)
    data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

    markers = ["o", "s", "D", "^", "v", "."]
    for i, bin_center in enumerate(bin_centers):
        color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
        counts = data.loc[:, bin_center]
        ax.plot(
            get_actual_co2_level(co2ls[::-1], percent=True),
            counts,
            label=rf"{bins[i]}-{bins[i+1]}\%",
            alpha=0.8,
            color=color,
            marker=markers[i % len(markers)],
            markersize=5,
        )

    ax.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of System Splits", fontsize=AXIS_LABEL_FONTSIZE)
    x_ticks_major = [0, 20, 40, 60]
    x_ticks_minor = [10, 30, 50]
    ax.set_xticks(x_ticks_major)
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_yscale("log")
    ax.invert_xaxis()

    # Add legend to separate axis
    h, l = ax.get_legend_handles_labels()
    ax_legend.legend(
        h,
        l,
        title="Share of load not served",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        ncols=1,
        columnspacing=0.5,
    )
    # ax_legend.legend(
    #     h,
    #     l,
    #     title="Share of load not served",
    #     loc="center",
    #     ncols=1,
    #     columnspacing=0.5,
    # )

    ax_legend.axis("off")

    plt.tight_layout()

    # Save figure
    save_name = "blackout_statistics"
    if same_corridor is True:
        save_name += "_same_corridor"
    elif same_corridor is False:
        save_name += "_different_corridor"
    if normalize_by_total_splits:
        save_name += "_normalized"
    save_figure(fig, save_path, save_name)
    plt.show()


def plot_blackout_size_histograms(
    save_path=path_to_figures_sclopf, log_scale=True, n_cols=1
):
    # load split properties
    n_nodes = 600
    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    # setup matplotlib styling
    setup_matplotlib_style()

    co2ls = get_co2_levels(n_nodes)
    # create histogram for each co2 level
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(co2ls) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8, 2 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    bins = np.linspace(0, 100, 101)

    # First pass: collect all histogram data to determine global y-limits
    all_counts = []
    for idx, co2l in enumerate(co2ls):
        vals = (
            split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
        )
        counts, _ = np.histogram(
            vals,
            weights=split_props[split_props.co2l == co2l].total_weighting,
            bins=bins,
        )
        all_counts.extend(counts)

    # Determine global y-limits
    if log_scale:
        non_zero_counts = [c for c in all_counts if c > 0]
        ymin = min(non_zero_counts) if non_zero_counts else 0.1
        ymax = max(all_counts) if all_counts else 1
    else:
        ymin = 0
        ymax = max(all_counts) if all_counts else 1

    # Second pass: plot with consistent y-limits
    for idx, co2l in enumerate(co2ls):
        ax = axes[idx]
        vals = (
            split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
        )
        ax.hist(
            vals,
            weights=split_props[split_props.co2l == co2l].total_weighting,
            bins=bins,
            alpha=0.7,
            edgecolor="black",
            log=log_scale,
            label=rf"CO$_2$: {get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)}\%",
        )
        if not log_scale:
            ax.set_ylim(ymin, ymax * 1.1)  # Add 10% padding at top
        else:
            ax.set_ylim(ymin, ymax * 10)  # Add padding in log scale
        col_idx = idx % n_cols
        if col_idx == 0:
            ax.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(
                axis="y",
                which="both",
                labelleft=False,
            )
            ax.tick_params(axis="x", which="both", labelsize=TICK_LABEL_FONTSIZE)
        # ax.legend(fontsize=TICK_LABEL_FONTSIZE, loc='upper center')
        ax.text(
            0.5,
            0.85,
            rf"CO$_2$: {get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)}\%",
            transform=ax.transAxes,
            fontsize=TICK_LABEL_FONTSIZE,
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.2, alpha=1
            ),
        )

    # hide any unused axes
    for ax in axes[len(co2ls) :]:
        ax.axis("off")

    # set x-labels for bottom row axes only
    for ax in axes[(n_rows - 1) * n_cols : n_rows * n_cols]:
        if ax.get_visible():
            ax.set_xlabel(
                r"Share of load not served [\%]", fontsize=AXIS_LABEL_FONTSIZE
            )
    # add suptitle
    plt.suptitle("Blackout Size Distribution", fontsize=PANEL_LABEL_FONTSIZE + 2)

    plt.tight_layout()
    save_figure(fig, save_path, "blackout_size_histograms")
    plt.show()


def plot_component_number_vs_blackout_size(
    split_properties=None, save_dir=path_to_figures_sclopf
):
    if split_properties is None:
        split_properties = pd.read_hdf(
            path_to_vis_results_sclopf + f"split_properties_all_n600.h5", index_col=0
        )
    lls = split_properties["lost_load_share_blackout"].to_numpy()
    n_comp = split_properties["n_components"].to_numpy()

    # bins
    lls_bins = np.linspace(0, 1, 11)
    n_comp_min = int(np.nanmin(n_comp))
    n_comp_max = int(np.nanmax(n_comp))
    n_comp_bins = np.logspace(
        np.log10(np.nanmin(n_comp)), np.log10(np.nanmax(n_comp)), 11
    )

    # histogram and column normalization (per lost_load_share_blackout bin)
    hist2d, x_edges, y_edges = np.histogram2d(lls, n_comp, bins=[lls_bins, n_comp_bins])
    col_sums = hist2d.sum(axis=1, keepdims=True)
    hist2d_norm = np.divide(
        hist2d, col_sums, out=np.zeros_like(hist2d), where=col_sums != 0
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist2d_norm.T,
        cmap="viridis",
        shading="auto",
    )
    ax.set_yscale("log")
    import matplotlib.ticker as mticker

    yticks = np.unique(
        np.round(
            np.logspace(
                np.log10(n_comp_min),
                np.log10(n_comp_max),
                num=6,
            )
        ).astype(int)
    )
    yticks = yticks[(yticks >= n_comp_min) & (yticks <= n_comp_max)]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    fig.colorbar(mesh, ax=ax, label="Relative Frequency")
    ax.set_xlabel("Blackout size")
    ax.set_ylabel("Number of Components")
    ax.set_title("Column-Normalized 2D Histogram")
    f_name = "lls_vs_n_components_colnorm"
    save_figure(fig, save_dir, f_name)


if __name__ == "__main__":
    # create_split_statistics_plot()
    # for load_norm in [False]:
    #     for blackout_stat in [True]:
    #         create_split_statistics_plot(
    #             load_normalization=load_norm,
    #             show_blackout_stats=blackout_stat,
    #             secondary_proba_axis=True,
    #         )

    # Create standalone blackout statistics plot
    # plot_blackout_size_histograms(log_scale=True, n_cols=2)

    # # # create_split_statistics_plot(load_normalization=True, show_blackout_stats=False)
    # for same_corridor_option in [None, True, False]:
    #     for normalize_option in [False, True]:
    #         create_blackout_statistics_plot(
    #             same_corridor=same_corridor_option,
    #             normalize_by_total_splits=normalize_option,
    #         )
    plot_component_number_vs_blackout_size()
