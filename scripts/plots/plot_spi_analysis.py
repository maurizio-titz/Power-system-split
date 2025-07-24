#!/usr/bin/env python3
"""
SPI Analysis Plot - Figure 3: SPI (System Polarization Index)
Creates histograms and vector plots for flow, inertia, and SPI analysis.
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
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cartopy.crs as ccrs

sys.path.append("./")

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_pre_outage_sclopf,
)
from utils import data_handling, cascade_simulation


def create_flow_and_inertia_plot():
    """Create SPI analysis plot with flow, inertia histograms and SPI vectors."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load network and data
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels and networks
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True
        )
        for co2l in co2ls
    }

    # Load pre-calculated data
    inertia_time = (
        np.load(
            path_to_pre_outage_sclopf + f"inertia_time_series_all_co2ls_{n_nodes}.npy"
        )
        / 1000
    )

    # Setup matplotlib
    plt.style.use("default")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    # Create figure with two panels
    f = plt.figure(figsize=(16, 6))
    gs_horizontal = GridSpec(1, 2, figure=f, wspace=0.3)
    ax_flow = f.add_subplot(gs_horizontal[0])
    ax_inertia = f.add_subplot(gs_horizontal[1])

    # Panel a: total flow histograms
    cmap = plt.get_cmap("cividis_r")
    unit_factor = 1e6
    selected_co2ls_spi = sorted(np.array([0.6, 0.2, 0.0]))  # CO2 levels to plot

    flows_lvls = []
    for count, co2l in enumerate(selected_co2ls_spi):
        n = networks[co2l]
        flow_distance = n.lines_t.p0.abs().mul(network.lines.length) / unit_factor
        flows_lvls.append(flow_distance.sum(axis=1))

    bins = np.histogram(np.hstack(flows_lvls), bins=40)[1]

    for count, co2l in enumerate(selected_co2ls_spi):
        ind = np.where(np.round(co2ls, 2) == co2l)[0][0]
        ax_flow.hist(
            flows_lvls[count],
            weights=network.snapshot_weightings.generators,
            bins=bins,
            histtype="step",
            label=r"{}\%".format(int(round(100 * co2l))),
            linewidth=3,
            color=cmap(
                np.where(selected_co2ls_spi == co2l)[0][0]
                / (len(selected_co2ls_spi) - 1)
            ),
            alpha=1,
        )

    # Add legend to left panel and styling
    leg = ax_flow.legend(fontsize=18, title=r"CO$_2$ level [\% of 1990]")
    plt.setp(leg.get_title(), fontsize=20)
    ax_flow.tick_params(axis="both", which="both", labelsize=20)
    ax_flow.set_xlabel(r"Total power flow distance [TW$\cdot$km]", fontsize=22)
    ax_flow.set_ylabel(r"Hours", fontsize=22)
    ax_flow.grid(True)

    # Panel b: rotational energy (without legend)
    bins = np.linspace(0, inertia_time.max(), 40)

    for count, co2l in enumerate(selected_co2ls_spi):
        ind = np.where(np.round(co2ls, 2) == co2l)[0][0]
        ax_inertia.hist(
            inertia_time[ind],
            weights=network.snapshot_weightings.generators,
            bins=bins,
            histtype="step",
            linewidth=3,
            color=cmap(
                np.where(selected_co2ls_spi == co2l)[0][0]
                / (len(selected_co2ls_spi) - 1)
            ),
            alpha=0.8,
        )

    ax_inertia.tick_params(axis="both", which="both", labelsize=20)
    ax_inertia.set_xlabel("Rotational energy [GWs]", fontsize=22)
    ax_inertia.set_ylabel("Hours", fontsize=22)
    ax_inertia.grid(True)

    # Add panel labels
    ax_flow.text(
        -0.1,
        1.05,
        r"\textbf{A}",
        fontsize=24,
        weight="bold",
        verticalalignment="center",
        transform=ax_flow.transAxes,
    )
    ax_inertia.text(
        -0.1,
        1.05,
        r"\textbf{B}",
        fontsize=24,
        weight="bold",
        verticalalignment="center",
        transform=ax_inertia.transAxes,
    )

    plt.tight_layout()
    plt.savefig(save_path + "flow_and_inertia.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_flow_and_inertia_plot()
