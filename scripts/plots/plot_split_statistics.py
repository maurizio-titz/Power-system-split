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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cycler import cycler

# Add root path and import utilities
root_path = "../"
os.chdir(root_path)
sys.path.append(root_path)

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling, cascade_simulation


def create_split_statistics_plot():
    """Create split statistics plot."""

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
    selected_co2ls = np.array([0.1, 0.3, 0.6, 0.8])

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
    component_props["snapshot_weighting"] = network.snapshot_weightings.generators[
        component_props.time_stamp
    ].values

    # Calculate weighted split properties
    weighted_split_props_dict = dict(
        {
            co2l: {
                category: split_props.loc[
                    (split_props.category == category)
                    & (split_props.index.get_level_values(0) == co2l),
                    "snapshot_weighting",
                ].sum()
                for category in split_props.category.unique()
            }
            for co2l in co2ls
        }
    )

    # Setup matplotlib
    mpl.style.use("default")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

    fig = plt.figure(figsize=(9, 3.3))
    gs_vertical = GridSpec(2, 1, figure=fig, hspace=0.6, height_ratios=[1, 0.1])

    # Panel setup
    n_cols = 3
    gsTop = GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=gs_vertical[0, :], hspace=0, wspace=0.4
    )
    gs_bottom = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_vertical[1, :])
    ax1 = fig.add_subplot(gsTop[2])
    ax1_legend = fig.add_subplot(gs_bottom[1:])

    # Panel c: split number and categories (loss of load share distribution)
    cmap = plt.get_cmap("inferno_r")

    bins = np.linspace(0, 100, 6).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data = []
    for co2l in co2ls[::-1]:
        vals = (
            split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
        )
        counts, _ = np.histogram(vals, bins=bins)
        data.append(counts)

    data = np.stack(data)
    data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

    plt.sca(ax1)
    for i, bin_center in enumerate(bin_centers):
        color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
        counts = data.loc[:, bin_center]
        plt.plot(
            get_actual_co2_level(co2ls[::-1]) * 100,
            counts,
            label=rf"{bins[i]}-{bins[i+1]}\%",
            alpha=0.8,
            color=color,
            marker=".",
            markersize=10,
        )

    ax1.set_xlabel(r"CO$_2$ level [\% of 1990]")
    ax1.set_ylabel("Number of System Splits")
    ax1.set_yscale("log")
    ax1.invert_xaxis()

    # Legend for panel c
    h, l = ax1.get_legend_handles_labels()
    ax1_legend.legend(
        h,
        l,
        title="Loss of load share",
        loc="center",
        ncols=len(bin_centers),
        columnspacing=0.5,
    )
    ax1_legend.axis("off")

    # Filter out very small components
    component_props_filtered = component_props

    # Panel b: Inertia histograms
    cmap = plt.get_cmap("cividis")

    ax2 = fig.add_subplot(gsTop[1])
    ax2_legend = fig.add_subplot(gs_bottom[0])
    rot_energy = component_props_filtered.rot_energy / 1000
    bins = np.linspace(rot_energy.min(), rot_energy.max(), 15)

    for co2l in selected_co2ls:
        vals = ax2.hist(
            rot_energy[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.snapshot_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            label=r"{} \%".format(int(100 * co2l)),
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax2.set_yscale("log")
    ax2.set_xlabel("Rotational energy [GWs]")
    ax2.set_ylabel("Number of split components")

    # Panel a: power imbalance histograms
    ax3 = fig.add_subplot(gsTop[0])
    power_imbalance = component_props_filtered.power_imbalance / 1000
    bins = np.linspace(power_imbalance.min(), power_imbalance.max(), 15)

    for co2l in selected_co2ls:
        h = ax3.hist(
            power_imbalance[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.snapshot_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            linewidth=3,
            label=r"{} \%".format(int(100 * co2l)),
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax3.set_yscale("log")
    ax3.set_ylabel("Number of split components")
    ax3.set_xlabel("Power imbalance [GW]")
    ax3.set_xticks(np.arange(-50, 51, step=25))

    # Legend for panels a and b
    h, l = ax2.get_legend_handles_labels()
    ax2_legend.legend(
        h[::-1],
        l[::-1],
        title=r"CO$_2$ level [\% of 1990]",
        loc="center",
        ncols=3,
        columnspacing=0.5,
    )
    ax2_legend.axis("off")

    # Add subplot labels
    for ax_loss_lvl, label in zip([ax1, ax2, ax3], ["c", "b", "a"]):
        ax_loss_lvl.text(
            0 - 0.25,
            1 + 0.1,
            label,
            fontsize=30,
            weight="bold",
            verticalalignment="center",
            transform=ax_loss_lvl.transAxes,
        )

    plt.savefig(save_path + "split_statistics.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_split_statistics_plot()
