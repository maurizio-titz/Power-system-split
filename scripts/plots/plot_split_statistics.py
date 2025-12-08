#!/usr/bin/env python3
"""
Split Statistics Plot - Figure 5: split statistics
Creates histograms of power imbalance, rotational energy, and loss of load share distributions.
"""

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


def create_split_statistics_plot(load_normalization=False, show_blackout_stats=True):
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
        width_ratios = [1, 1, 1, 0.4]
    else:
        fig = plt.figure(figsize=(8, 5))
        n_cols = 2
        width_ratios = [1, 1]

    gs_vertical = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 0.1])

    # Panel setup
    wspace = 0.4 if show_blackout_stats else 0.3
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

        # Legend for panel c
        h, l = ax3_num_splits.get_legend_handles_labels()
        ax3_num_splits_legend.legend(
            h,
            l,
            title="Blackout size",
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

    save_figure(fig, save_path, save_name)
    plt.show()


def create_blackout_statistics_plot(save_path=path_to_figures_sclopf):
    """
    Create a standalone plot showing only the blackout statistics (Number of System Splits).

    Parameters
    ----------
    save_path : str, default path_to_figures_sclopf
        Path where the plot will be saved
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
        title="Blackout size",
        loc="center",
        ncols=1,
        columnspacing=0.5,
    )
    ax_legend.axis("off")

    plt.tight_layout()

    # Save figure
    save_name = "blackout_statistics"
    save_figure(fig, save_path, save_name)
    plt.show()


if __name__ == "__main__":
    create_split_statistics_plot()
    for load_norm in [True, False]:
        for blackout_stat in [True, False]:
            create_split_statistics_plot(
                load_normalization=load_norm, show_blackout_stats=blackout_stat
            )

    # Create standalone blackout statistics plot
    create_blackout_statistics_plot()

    # Example usage:
    # create_split_statistics_plot(load_normalization=True, show_blackout_stats=False)
    # create_blackout_statistics_plot()
