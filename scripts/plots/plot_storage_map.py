#!/usr/bin/env python3
"""
Storage Map Plot - Storage capacity visualization
Creates maps showing storage capacity distribution and total capacity versus CO2 levels.
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

# Add root path and import utilities
root_path = "../"
os.chdir(root_path)
sys.path.append(root_path)

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import path_to_pypsa_network_sclopf, path_to_figures_sclopf
from utils import data_handling
from utils.clustering_visualisation import truncate_colormap


def create_storage_map():
    """Create storage capacity map plot."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Configuration
    plot_output_capacity = (
        True  # select whether to plot storage OUTPUT capacity or storage capacity
    )

    # Load network
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )

    # Get CO2 levels and networks
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True
        )
        for co2l in co2ls
    }

    # Setup matplotlib
    mpl.style.use("default")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

    target_levels = [0.0]
    unit = "GWh"
    unit_factor = 1e-3

    f = plt.figure(figsize=(15, 10))

    gs_vertical = GridSpec(2, 2, width_ratios=[2, 0.6], wspace=0.13)
    gs_maps = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_vertical[0], wspace=-0.0, hspace=-0.2
    )
    gs_lines = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_vertical[1], height_ratios=[0.8, 0.2], hspace=0.4
    )

    axs_maps = np.array(
        [f.add_subplot(gs_maps[i], projection=ccrs.PlateCarree()) for i in range(2)]
    )
    ax_line = f.add_subplot(gs_lines[0])

    # Panel a, b: maps
    buses = list(network.buses.index)
    storage_types = network.storage_units["carrier"].unique()
    multi_index = pd.MultiIndex.from_product(
        [buses, storage_types], names=["bus", "carrier"]
    )

    battery_color = "red"
    h2_color = "green"

    for i, target_level in enumerate(target_levels):
        for ii, (current_store_type, current_color) in enumerate(
            zip(["battery", "H2"], [battery_color, h2_color])
        ):
            n = networks[target_level]
            storage_capacities = n.storage_units.max_hours * n.storage_units.p_nom
            if plot_output_capacity:
                storage_capacities = (
                    storage_capacities * n.storage_units.efficiency_dispatch
                )
            storage_capacities_grouped = storage_capacities.groupby(
                [n.storage_units["bus"], n.storage_units["carrier"]]
            ).sum()

            max_node_size = 0.85
            max_capacity = storage_capacities_grouped[:, current_store_type].max()
            max_capacity_rounded = round(
                max_capacity, -int(round(np.log10(max_capacity), 0)) + 1
            )

            n.plot(
                bus_sizes=storage_capacities_grouped[:, current_store_type]
                / max_capacity_rounded
                * max_node_size,
                line_colors="black",
                bus_colors=current_color,
                link_widths=0.5,
                line_widths=0.5,
                ax=axs_maps[ii],
            )

            legend_relative_circle_size = np.array([0.25, 1])
            legend_circle_size = [
                size * max_node_size for size in legend_relative_circle_size
            ]
            legend_circle_size = [
                round(size, -int(np.floor(np.log10(size))))
                for size in legend_circle_size
            ]
            pypsa.plot.add_legend_circles(
                axs_maps[ii],
                sizes=legend_circle_size,
                labels=[
                    f"{size*max_capacity_rounded*unit_factor:2g} {unit}"
                    for size in legend_circle_size
                ],
                patch_kw={"color": current_color},
            )

            current_store_type_string = (
                current_store_type[0].upper() + current_store_type[1:]
            )
            axs_maps[ii].set_title(current_store_type_string, fontsize=16)

    # Add labels
    axs_maps[0].text(
        0 - 0.1,
        1 + 0.05,
        r"\textbf{a}",
        fontsize=24,
        weight="bold",
        verticalalignment="center",
        transform=axs_maps[0].transAxes,
    )

    axs_maps[0].text(
        0 - 0.15,
        0.8,
        rf"CO$_2$={get_actual_co2_level(target_levels[0], percent=True)}\% ",
        fontsize=24,
        fontweight="bold",
        verticalalignment="center",
        transform=axs_maps[0].transAxes,
    )

    # Panel c: total capacity line plot
    ax_line.text(
        0 - 0.1,
        1 + 0.05,
        r"\textbf{b}",
        fontsize=24,
        weight="bold",
        verticalalignment="center",
        transform=ax_line.transAxes,
    )

    capacity_by_type_by_lvl = pd.DataFrame(index=storage_types, columns=co2ls[::-1])

    for target_level in co2ls:
        n = networks[target_level]
        print(f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{target_level}-2920SEG.nc")
        storage_capacities_all_lvl = pd.Series(index=multi_index, data=0)
        storage_capacities_lvl = n.storage_units.max_hours * n.storage_units.p_nom
        if plot_output_capacity:
            storage_capacities_lvl = (
                storage_capacities_lvl * n.storage_units.efficiency_dispatch
            )
        storage_capacities_grouped_lvl = storage_capacities_lvl.groupby(
            [n.storage_units["carrier"]]
        ).sum()
        capacity_by_type_by_lvl.loc[
            storage_capacities_grouped_lvl.index, target_level
        ] = storage_capacities_grouped_lvl

    capacity_by_type_by_lvl_plotting = capacity_by_type_by_lvl.T.copy()
    try:
        capacity_by_type_by_lvl_plotting.drop(index=[0.7, 0.8], inplace=True)
    except:
        pass

    capacity_by_type_by_lvl_plotting.rename(
        columns={"battery": "Battery", "hydro": "Hydro", "PHS": "Pumped hydro"},
        inplace=True,
    )
    colors = ["blue", "skyblue", h2_color, battery_color]

    ax_line.plot(
        get_actual_co2_level(capacity_by_type_by_lvl.columns, percent=True),
        capacity_by_type_by_lvl_plotting.loc[
            :, ["Hydro", "Pumped hydro", "H2", "Battery"]
        ].values
        * unit_factor,
        label=["Hydro", "Pumped hydro", "H2", "Battery"],
    )

    for feat_count in range(capacity_by_type_by_lvl_plotting.shape[1]):
        if "ydro" in capacity_by_type_by_lvl_plotting.columns[feat_count]:
            continue
        plt.scatter(
            get_actual_co2_level(capacity_by_type_by_lvl.columns, percent=True),
            (
                capacity_by_type_by_lvl_plotting.loc[
                    :, ["Hydro", "Pumped hydro", "H2", "Battery"]
                ].values
                * unit_factor
            )[:, feat_count],
            color=colors[feat_count],
        )

    secax = ax_line.secondary_yaxis(
        "right", functions=(lambda x: x * (1 / 6), lambda x: x * 6)
    )
    secax.set_ylabel("Battery power [GW]")

    for i, ii in enumerate(ax_line.lines):
        ii.set_color(colors[i])

    ax_line.invert_xaxis()
    ax_line.grid(True)
    plt.yscale("log")

    if plot_output_capacity:
        plt.ylabel(f"Storage output capacity [{unit}]")
        file_name = "storage_output_capacity_vs_co2level"
    else:
        plt.ylabel(f"Storage capacity [{unit}]")
        file_name = "storage_capacity_vs_co2level"

    plt.xlabel(r"CO2 level [\% of 1990]")
    plt.legend()

    h, l = ax_line.get_legend_handles_labels()
    ax_line.get_legend().remove()
    ax_line_legend = f.add_subplot(gs_lines[1])
    ax_line_legend.legend(
        h, l, borderaxespad=0.2, fontsize=15, ncol=2, loc="upper left"
    )
    ax_line_legend.axis("off")

    plt.savefig(save_path + file_name + ".pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_storage_map()
