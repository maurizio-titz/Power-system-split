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

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_pre_outage_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure,
)


def create_flow_and_inertia_plot(mean_distance=False):
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

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Create figure with two panels
    f = plt.figure(figsize=(7, 4.5))
    gs_vertical = GridSpec(2, 1, figure=f, hspace=0.6, height_ratios=[1, 0.1])
    gs_horizontal = GridSpecFromSubplotSpec(1, 2, gs_vertical[0], wspace=0.23)

    ax_flow = f.add_subplot(gs_horizontal[0])
    ax_inertia = f.add_subplot(gs_horizontal[1])
    ax_legend = f.add_subplot(gs_vertical[1])

    # Panel a: total flow histograms
    cmap = plt.get_cmap("cividis")
    unit_factor = 1e6
    selected_co2ls_spi = sorted(np.array([0.6, 0.2, 0.0]))  # CO2 levels to plot

    flows_lvls = []
    for count, co2l in enumerate(selected_co2ls_spi):
        n = networks[co2l]
        flow_distance = (n.lines_t.p0.abs().mul(network.lines.length)).sum(axis=1)
        if mean_distance:
            total_transmissed_power = (
                n.generators_t.p.mul(network.generators.sign).sum(axis=1)
            ) + n.storage_units_t.p[n.storage_units_t.p > 0].sum(axis=1)
            flow_distance = flow_distance / total_transmissed_power
        else:
            flow_distance = flow_distance / unit_factor
        flows_lvls.append(flow_distance)
    flows_lvls = np.array(flows_lvls)

    bins = np.linspace(flows_lvls.min(), flows_lvls.max(), 40)
    for count, co2l in enumerate(selected_co2ls_spi):
        density, bins_ = np.histogram(
            flows_lvls[count, :],
            bins=bins,
            weights=network.snapshot_weightings.generators
            / network.snapshot_weightings.generators.sum(),
        )
        # Set zero values to NaN to avoid plotting lines at zero frequency
        density_masked = density.copy()
        density_masked[density_masked == 0] = np.nan

        ax_flow.stairs(
            density_masked,
            bins,
            label=r"{}\%".format(
                get_actual_co2_level(co2l, percent=True),
            ),
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
        )

        assert np.isclose(sum(density), 1, atol=1e-3)

    # Add legend to separate panel
    # Legend for panels a and b
    h, l = ax_flow.get_legend_handles_labels()
    # Create patch handles to match histogram appearance
    patch_handles = [
        mpl.patches.Patch(
            facecolor="white", edgecolor=patch.get_edgecolor(), linewidth=3
        )
        for patch in h
    ]
    # Place the legend title to the left of the labels by using a dummy handle and label
    handles = [mpl.patches.Patch(facecolor="none", edgecolor="none")] + patch_handles[
        ::-1
    ]
    labels = [r"CO$_2$ level [\% of 1990]"] + l[::-1]
    ax_legend.legend(
        handles,
        labels,
        loc="center left",
        ncols=4,
        columnspacing=1,
        handletextpad=0.5,
        # frameon=False,
    )
    ax_legend.axis("off")
    # plt.setp(ax_legend.get_legend().get_title(), fontsize=LEGEND_FONTSIZE)
    # plt.setp(leg.get_title(), fontsize=LEGEND_FONTSIZE)
    # ax_flow.set_xlim(bottom=40)
    ax_flow.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    if mean_distance:
        ax_flow.set_xlabel(
            r"Mean transmission distance [km]", fontsize=AXIS_LABEL_FONTSIZE
        )
    else:
        ax_flow.set_xlabel(
            r"Total power flow distance [TW$\cdot$km]", fontsize=AXIS_LABEL_FONTSIZE
        )
    ax_flow.set_ylabel(r"Frequency", fontsize=AXIS_LABEL_FONTSIZE)
    ax_flow.grid(True)
    # ax_flow.legend(
    #     title=r"CO$_2$ level [\% of 1990]",
    #     title_fontsize=LEGEND_FONTSIZE,
    #     frameon=False,
    #     loc="upper right",
    # )
    # ax_flow.set_xlim(left=39)

    # Panel b: rotational energy (without legend)
    bins = np.linspace(0, inertia_time.max(), 40)

    for count, co2l in enumerate(selected_co2ls_spi):
        ind = np.where(np.round(co2ls, 2) == co2l)[0][0]
        density, bins_ = np.histogram(
            inertia_time[ind, :],
            bins=bins,
            weights=network.snapshot_weightings.generators
            / network.snapshot_weightings.generators.sum(),
        )
        assert np.all(bins == bins_)

        # Set zero values to NaN to avoid plotting lines at zero frequency
        density_masked = density.copy()
        density_masked[density_masked == 0] = np.nan

        ax_inertia.stairs(
            density_masked,
            bins,
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
        )
        assert np.isclose(sum(density), 1, atol=1e-3)

    ax_inertia.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_inertia.set_xlabel("Rotational energy [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    # ax_inertia.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
    ax_inertia.grid(True)

    # Add panel labels
    add_panel_label(ax_flow, 0, y_offset=0.08)
    add_panel_label(ax_inertia, 1, y_offset=0.08)

    plt.tight_layout()
    f_name = "flow_and_inertia"
    if mean_distance:
        f_name += "_mean_distance"
    save_figure(f, save_path, f_name)
    plt.show()


if __name__ == "__main__":
    create_flow_and_inertia_plot(mean_distance=False)
