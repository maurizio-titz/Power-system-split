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

# Logging
from tqdm import tqdm
from loguru import logger

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
    load_normalization=False, 
    show_blackout_stats=True, 
    secondary_proba_axis=False
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
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index='0')
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
        path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    )
    component_props.time_stamp = pd.to_datetime(component_props.time_stamp)

    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5", index_col=0
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
        width_ratios = [1, 1, 1.6, 1]
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

    save_figure(fig, save_name, save_path)
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
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5", index_col=0
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
                data_handling.path_to_grid_data + f"/n_2_failures_co2lvl{co2l}.pklz"
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
    save_figure(fig, save_name, save_path)
    plt.show()


def ax_blackout_size_histogram(ax, 
                               split_props_df: pd.DataFrame, 
                               co2lvl: float, 
                               n_bins: int = 100,
                               bin_edges: tuple[float, float] | None = None,
                               use_total_lost_load: bool = False, 
                               xscale_log: bool = False,
                               yscale_log: bool = False, 
                               color="C0",
                               linewidth: float = 2., 
                               label_str: str | None = None):
    """Draw the histogram of the blackout sizes on the axis ax"""
    
    mask_co2lvl = split_props_df.co2l == co2lvl
    
    needed_cols = ["lost_load_share_blackout",
                   "load", 
                   "snapshot_weighting", 
                   "trigger_weighting"]
    
    dtype_dict = {xx: float for xx in needed_cols}
    
    split_props_df_cut = split_props_df.loc[mask_co2lvl, needed_cols]
    split_props_df_cut = split_props_df_cut.astype(dtype_dict)
    if "total_weighting" not in split_props_df_cut.columns:
        split_props_df_cut["total_weighting"] = split_props_df_cut["snapshot_weighting"] * \
            split_props_df_cut["trigger_weighting"]
    
    
    lost_load_val = split_props_df_cut["lost_load_share_blackout"].values
    
    if use_total_lost_load:
        lost_load_val = (split_props_df_cut["lost_load_share_blackout"] * split_props_df_cut["load"]).values

        if bin_edges is None:
            if xscale_log:
                bin_edges = (1, lost_load_val.max())
            else:
                bin_edges = (0, lost_load_val.max())
    else: 
        if bin_edges is None:
            if xscale_log:
                bin_edges = (1, 0)
            else:
                bin_edges = (0, 1)
       
    if xscale_log:
        bin_arr = np.logspace(np.log10(min(bin_edges)), 
                              np.log10(max(bin_edges)), 
                              n_bins)
        
    else: 
        bin_arr = np.linspace(min(bin_edges), 
                              max(bin_edges), n_bins)
            
    # Plot it
    ax.hist(lost_load_val, 
            weights=split_props_df_cut.total_weighting,
            bins=bin_arr, histtype="step", color=color,
            linewidth=linewidth,
            label=label_str)
    
    # Aesthetics
    if xscale_log:
        ax.set_xscale("log")
    
    if yscale_log:
        ax.set_yscale("log")
    
    return
    

def plot_blackout_size_histograms(
    save_path: str = path_to_figures_sclopf, 
    yscale_log: bool = True, 
    xscale_log: bool = False,
    n_cols: int = 1,
    n_bins: int = 101, 
    n_nodes: int = 600, 
    use_total_load_values: bool = False,
    fname_suffix: str = "",
    xlims: tuple[float, float] | None = None
):
    # load split properties
    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5", index_col=0
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

    
    if not use_total_load_values:
        if xscale_log:
            if xlims is None:
                    xlims = (1e-2,  1e2)
            bins = np.logspace(np.log10(min(xlims)), np.log10(max(xlims)),
                               n_bins)
        else:
            if xlims is None:
                xlims = (0, 100)
            bins = np.linspace(min(xlims), max(xlims), n_bins)

    # First pass: collect all histogram data to determine global y-limits
    all_counts = []
    for idx, co2l in enumerate(co2ls):
        if use_total_load_values:
            ll_share_blackout = split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
            load = split_props[split_props.co2l == co2l].load
            vals = ll_share_blackout * load
            
            if xscale_log:
                if xlims is None:
                    xlims = (1e-2,  max(vals))
                bins = np.logspace(np.log10(min(xlims)), np.log10(max(xlims)), 
                                   n_bins)
            else:
                if xlims is None:
                    xlims = (0, max(vals))
                bins = np.linspace(min(xlims), max(xlims), n_bins)
        else:
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
    if yscale_log:
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
            log=yscale_log,
            label=rf"CO$_2$: {get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)}\%",
        )
        if yscale_log:
            ax.set_ylim(ymin, ymax * 10)  # Add padding in log scale
            
        else:
            ax.set_ylim(ymin, ymax * 1.1)  # Add 10% padding at top
            
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
        
        if xscale_log:
            ax.set_xscale('log')

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

    fname_fig = "blackout_size_histograms" + fname_suffix
    
    plt.tight_layout()
    save_figure(fig, fname_fig, save_path)
    
    
def plot_blackout_size_distributions_with_zoom_in(
    split_props: pd.DataFrame,
    co2_lvls: tuple[float] = (.6, .2, .0),
    n_nodes: int = 600,
    cut_off_val: float | None = 2e4,
    use_abs_vals: bool = False):
    """Plot a X by 2 plot that shows the blackout size distributions that
    shows the different requested CO2 levels in the rows. First column shows 
    the entire distribtuion while the second column shows the distribution beyond 
    a chosen cut-off in log-log scale."""
    
    setup_matplotlib_style()
    
    available_co2_lvls = get_co2_levels(n_nodes=n_nodes)
    
    nr_chosen_co2_lvls = len(co2_lvls)
    
    figsize = (10, 4)
    
    fig, ax_arr = plt.subplots(1, 2, 
                           figsize=figsize, sharey='all',
                           )
    
    ax_overview = ax_arr[0]
    ax_zoom = ax_arr[1]
    
    cmap_co2 = plt.get_cmap('cividis').copy()
    
    for idx, co2_lvl_r in enumerate(sorted(co2_lvls)[::-1]):
        diff_ls = abs(np.array(available_co2_lvls) - co2_lvl_r)
        idx_co2_ls = np.argmin(diff_ls)
        co2_lvl_from_list_r = available_co2_lvls[idx_co2_ls]
        
        if diff_ls[idx_co2_ls] > 1e-6 or np.count_nonzero(diff_ls < 1e-6) > 1:
            raise ValueError("co2 value either not in list or duplicate in list.")
        
        actual_co2_lvl = get_actual_co2_level(co2_lvl_from_list_r, 
                                              n_nodes=n_nodes)
        color_r = cmap_co2(idx_co2_ls / (len(available_co2_lvls) -1))
        
        ax_blackout_size_histogram(ax_overview, split_props,
                                   co2_lvl_from_list_r,
                                   use_total_lost_load=True,
                                   color=color_r,
                                   yscale_log=True,
                                   xscale_log=True)
        
        ax_blackout_size_histogram(ax_zoom, split_props,
                                   co2_lvl_from_list_r,
                                   use_total_lost_load=True,
                                   color=color_r,
                                   yscale_log=True,
                                   xscale_log=True,
                                   label_str=f"{round(actual_co2_lvl*100)}\\%")
    
    # Aesthetics
    if cut_off_val is not None:
    
        ax_zoom.set_xlim(left=cut_off_val)
        ax_overview.axvline(x=cut_off_val, ls="--", lw=2.,
                         color='k')
            

    ax_arr[0].set_ylabel("Count", size=AXIS_LABEL_FONTSIZE)
    
    """handles, leg = ax_zoom.get_legend_handles_labels()
    leg_fig = fig.legend(handles, leg, title="CO$_2$ level [\\% of 1990]",
                         bbox_anchor"center")
    """
    ax_zoom.legend(title="CO$_2$ level [\\% of 1990]",
                   loc="lower left")
    for ax_r in ax_arr:
        ax_r.set_xlabel("Lost Load [MW]")
    
    fname = "blackout_sizes_three_lvls_w_zoom"
    
    save_figure(fig, fname, path_to_figures_sclopf, organize_plots=False)


def create_blackout_statistics_common_vs_different_corridor_plot(n_nodes: int = 600,
                                                                  use_normalized: bool = True,
                                                                  use_steps: bool = True,
                                                                  cmap_blackout_categories: str = "inferno_r",
                                                                  show_ratio: bool = False,
                                                                  show_number_and_normalized: bool = False):
    """Create the plot that shows both the blackout statistics (Number of System Splits)
    for both common corridor (left panel) and common corridor (right panel).
    This is using in essence the same approach as 'create_blackout_statistics_plot'"""
    
    co2_lvls_all = get_co2_levels(n_nodes)[::-1]
    actual_co2_lvl = get_actual_co2_level(co2_lvls_all, percent=True)
    
    # Load split properties
    split_props = pd.read_hdf(
        os.path.join(path_to_vis_results_sclopf, f"split_properties_all_n{n_nodes}.h5"), key="df"
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(float)
    
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    
    mask_same_corridor = split_props.init_failure_0 == split_props.init_failure_1
    
    # Create histograms
    bins = np.linspace(0, 100, 6, dtype=int)
    bin_centers = .5 * (bins[:-1] + bins[1:])
    
    # Find normalization factor if needed
    if use_normalized or show_number_and_normalized:
        for idx_co2, co2l_r in enumerate(co2_lvls_all):
            file_path = (
                        os.path.join(data_handling.path_to_grid_data, f"n_2_failures_co2lvl{co2l_r}.pklz")
                    )
            with gzip.open(file_path, "rb") as fh_n2:
                n_2_failures = pickle.load(fh_n2)
                
        
            network = data_handling.load_pypsa_network(n_nodes=n_nodes, co2lvl=co2l_r,
                                                    use_sclopf=True, lopt=False)
            
            n_2_failures_same = [n2_f for n2_f in n_2_failures 
                                if (n2_f["failures"][0][0] == n2_f["failures"][1][0])]
            n_2_failures_different = [n2_f for n2_f in n_2_failures 
                                     if (n2_f["failures"][0][0] != n2_f["failures"][1][0])]   
            
            # Trigger and Snapshot weights
            weighted_trigger_count_same = sum([init_failure["weight"] 
                                               for init_failure in n_2_failures_same])
            weighted_trigger_count_different = sum([init_failure["weight"] 
                                               for init_failure in n_2_failures_different])
            
            number_of_simulations_same = weighted_trigger_count_same * network.snapshot_weightings.generators.sum()
            number_of_simulations_different = weighted_trigger_count_different * network.snapshot_weightings.generators.sum()
            
            
    # Same + different corridor
    split_props_same = split_props[mask_same_corridor]
    split_props_different = split_props[~mask_same_corridor]
    
    data_same = list()
    data_different = list()
    if show_number_and_normalized:
        data_same_norm = list()
        data_diff_norm = list()
    
    for co2l_r in tqdm(co2_lvls_all, total=len(co2_lvls_all), desc="CO2 Lvls"):
        split_props_same_co2r = split_props_same[split_props_same.co2l == co2l_r]
        vals = split_props_same_co2r.lost_load_share_blackout.values * 100.
        
        counts_same, _ = np.histogram(vals, bins=bins, weights=split_props_same_co2r.total_weighting)
            
        split_props_different_co2_r = split_props_different[split_props_different.co2l == co2l_r]
        
        vals = split_props_different_co2_r.lost_load_share_blackout.values * 100.
        
        counts_different, _ = np.histogram(vals, bins=bins, 
                                           weights=split_props_different_co2_r.total_weighting)
        
        if show_number_and_normalized:
            data_same.append(counts_same)
            data_different.append(counts_different)
            
            data_same_norm.append(counts_same / number_of_simulations_same) 
            data_diff_norm.append(counts_different / number_of_simulations_different)
            
        elif use_normalized:
            data_same.append(counts_same / number_of_simulations_same) 
            data_different.append(counts_different / number_of_simulations_different)
        else:
            data_same.append(counts_same)
            data_different.append(counts_different)
            
        
        
    df_same = pd.DataFrame(np.stack(data_same), index=co2_lvls_all, columns=bin_centers)
    df_different = pd.DataFrame(np.stack(data_different), index=co2_lvls_all, columns=bin_centers)
    
    if show_number_and_normalized:
        df_same_norm = pd.DataFrame(np.stack(data_same_norm), 
                                    index=co2_lvls_all, columns=bin_centers)
        df_diff_norm = pd.DataFrame(np.stack(data_diff_norm), 
                                    index=co2_lvls_all, columns=bin_centers)
    
    # Plot it
    setup_matplotlib_style()
    markers = ["o", "s", "D", "^", "v", "."]
    if show_number_and_normalized:
        fig, ax_ls = plt.subplots(2, 2,figsize=(8, 6),
                                  sharex='all', sharey='row')
        [[ax_same, ax_different],
         [ax_same_norm, ax_diff_norm]] = ax_ls
        ax_right_most = ax_different
    
    
    elif show_ratio:
        fig,  ax_ls = plt.subplots(1, 3,figsize=(10, 3.5),
                                                sharex='all')

        [ax_same, ax_different, ax_ratio] = ax_ls
        ax_same.sharey(ax_different)
        ax_right_most = ax_ratio
    else:
        fig, ax_ls = plt.subplots(1, 2,figsize=(8, 3.5),
                                                sharex='all', sharey='all')
        [ax_same, ax_different] = ax_ls
        ax_right_most = ax_different
    
    cmap = plt.get_cmap(cmap_blackout_categories).copy()
    for ii, bin_c_r in enumerate(bin_centers):
        color_r = cmap((ii+1) / (len(bin_centers) + 1))
        
        counts_same_r = df_same.loc[:, bin_c_r]
        counts_diff_r = df_different.loc[:, bin_c_r]
        if use_steps:
            if not show_number_and_normalized:
                ax_same.step(
                actual_co2_lvl, counts_same_r,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_different.step(
                actual_co2_lvl, counts_diff_r,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
            
            else:
                counts_same_r_norm = df_same_norm.loc[:, bin_c_r]
                counts_diff_r_norm = df_diff_norm.loc[:, bin_c_r]
                ax_same.step(
                actual_co2_lvl, counts_same_r,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_different.step(
                actual_co2_lvl, counts_diff_r,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
                
                ax_same_norm.step(
                actual_co2_lvl, counts_same_r_norm,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_diff_norm.step(
                actual_co2_lvl, counts_diff_r_norm,
                where='mid',
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
            
        else:
            if not show_number_and_normalized:
                ax_same.plot(
                actual_co2_lvl, counts_same_r,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_different.plot(
                actual_co2_lvl, counts_diff_r,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
                
            else:
                counts_same_r_norm = df_same_norm.loc[:, bin_c_r]
                counts_diff_r_norm = df_diff_norm.loc[:, bin_c_r]
                ax_same.plot(
                actual_co2_lvl, counts_same_r,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_different.plot(
                actual_co2_lvl, counts_diff_r,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
                
                ax_same_norm.plot(
                actual_co2_lvl, counts_same_r_norm,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
        
                ax_different_norm.plot(
                actual_co2_lvl, counts_diff_r_norm,
                label=rf"{bins[ii]}-{bins[ii+1]}\%",
                alpha=0.8,
                color=color_r,
                marker=markers[ii % len(markers)],
                markersize=5,
                )
            
        if show_ratio and not show_number_and_normalized:
            ratio = counts_same_r / counts_diff_r
            if use_steps:
                ax_ratio.step(actual_co2_lvl, ratio,
                        label=rf"{bins[ii]}-{bins[ii+1]}\%",
                        where="mid",
                        alpha=0.8,
                        color=color_r,
                        marker=markers[ii % len(markers)],
                        markersize=5,)
            else:
                ax_ratio.plot(actual_co2_lvl, ratio,
                        label=rf"{bins[ii]}-{bins[ii+1]}\%", 
                        alpha=0.8,
                        color=color_r,
                        marker=markers[ii % len(markers)],
                        markersize=5,)
                
        
    # Aesthetics
    if show_number_and_normalized:
        ax_iter = ax_ls.flatten()
    else:
        ax_iter = ax_ls
    for ax_r in ax_iter:
        ax_r.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
        ax_r.set_xticks([0, 20, 40, 60])
        ax_r.set_xticks([10, 30, 50], minor=True)
        ax_r.grid(False)
        ax_r.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    
    for ax_r in [ax_same, ax_different]:
        ax_r.set_yscale('log')
    xlim = ax_same.get_xlim()
    ax_same.set_xlim(left=max(xlim), right=min(xlim))
    if show_number_and_normalized:
        ax_same_norm.set_xlim(left=max(xlim), right=min(xlim))
    
    fig.text(.5, 1.1, "Same Corridor", transform=ax_same.transAxes,
             fontsize=PANEL_LABEL_FONTSIZE, ha='center')
    fig.text(.5, 1.1, "Different Corridor", transform=ax_different.transAxes,
             fontsize=PANEL_LABEL_FONTSIZE, ha='center')
    
    if show_ratio and not show_number_and_normalized:
        ax_ratio.set_ylabel("Ratio Fraction Same / Different")
        ax_ratio.axhline(y=1., linestyle="--", lw=2., color="k")
        ax_ratio.set_yscale('log')
        
    if show_number_and_normalized:
        ax_same.set_ylabel("Number of System Splits", 
                           fontsize=AXIS_LABEL_FONTSIZE)
        ax_same_norm.set_ylabel("Fraction of total contingencies", 
                                fontsize=AXIS_LABEL_FONTSIZE)
        ax_same_norm.set_yscale("log")
        
    elif use_normalized:
        ax_same.set_ylabel("Fraction of total contingencies", 
                           fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax_same.set_ylabel("Number of System Splits", fontsize=AXIS_LABEL_FONTSIZE)
    
    fig.tight_layout()
    pos_ax_right = ax_right_most.get_position()
    handles, leg = ax_right_most.get_legend_handles_labels()
    leg_f = fig.legend(handles, leg, title="Share of load\n not served",
               bbox_to_anchor=(pos_ax_right.x1, pos_ax_right.y0 + pos_ax_right.height *.5),
               loc="center left",
               ncols=1)
    leg_f.get_title().set_multialignment("center")
    
    # Save figure
    fname_fig = "blackout_statistics_common_vs_different_corridor"
    if use_normalized:
        fname_fig += "_normalized"
    if show_ratio:
        fname_fig += "_ratio"
    if show_number_and_normalized:
        fname_fig += "_show_num_n_normalized"
        
    for idx_ax, ax_r in enumerate(ax_iter):
        add_panel_label(ax_r, idx_ax, y_offset=0.1, x_offset=-0.075)
    
    if show_number_and_normalized:
        fig.align_ylabels([ax_same, ax_same_norm])
        ax_same.set_xlabel("")
        ax_different.set_xlabel("")
        
    save_figure(fig, fname_fig, path_to_figures_sclopf, organize_plots=False)
    
    return
    

def plot_component_number_vs_blackout_size(
    split_properties=None, save_dir=path_to_figures_sclopf
):
    if split_properties is None:
        split_properties = pd.read_hdf(
            path_to_vis_results_sclopf + f"/split_properties_all_n600.h5", index_col=0
        )
    lls = split_properties["lost_load_share_blackout"].to_numpy() * 100
    n_comp = split_properties["n_components"].to_numpy()

    # bins
    lls_bins = np.linspace(0, 100, 11)
    n_comp_bins = np.unique(
        np.round(
            np.logspace(np.log10(np.nanmin(n_comp)), np.log10(np.nanmax(n_comp)), 11)
        ).astype(int)
    )
    # n_comp_bins = np.concatenate(
    #     [n_comp_bins[:-2], [n_comp_bins[-1]]]
    # )  # remove second to last bin edge to avoid tiny last bin

    # histogram and column normalization (per lost_load_share_blackout bin)
    hist2d, x_edges, y_edges = np.histogram2d(lls, n_comp, 
                                              bins=[lls_bins, n_comp_bins])
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

    yticks = n_comp_bins

    ax.set_yticks(yticks, labels=[str(int(tick)) for tick in yticks])
    ax.minorticks_off()
    fig.colorbar(mesh, ax=ax, label="Relative Frequency")
    ax.set_xlabel("Share of load not served [\%]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Components", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Column-Normalized 2D Histogram", fontsize=AXIS_LABEL_FONTSIZE)
    f_name = "lls_vs_n_components_colnorm"
    save_figure(fig, f_name, save_dir)


if __name__ == "__main__":
    create_split_statistics_plot()
    for load_norm in [False]:
        for blackout_stat in [True]:
            create_split_statistics_plot(
                load_normalization=load_norm,
                show_blackout_stats=blackout_stat,
                secondary_proba_axis=True,
            )

    # Create standalone blackout statistics plot
    plot_blackout_size_histograms(yscale_log=True, n_cols=2)
    plot_blackout_size_histograms(yscale_log=True, n_cols=2, n_bins=50,
                                  fname_suffix="_lessbins")
    
    plot_blackout_size_histograms(yscale_log=True, xscale_log=True, n_cols=2,
                                  xlims=(1e-2, 1e2), fname_suffix="_log")
    plot_blackout_size_histograms(yscale_log=True, xscale_log=True, n_cols=2,
                                  xlims=(1e-2, 1e2), n_bins=50, fname_suffix="_log_lessbins")

    # # create_split_statistics_plot(load_normalization=True, show_blackout_stats=False)
    for same_corridor_option in [None, True, False]:
        for normalize_option in [False, True]:
            create_blackout_statistics_plot(
                same_corridor=same_corridor_option,
                normalize_by_total_splits=normalize_option,
            )
    plot_component_number_vs_blackout_size()
    
    create_blackout_statistics_common_vs_different_corridor_plot(use_normalized=True)
