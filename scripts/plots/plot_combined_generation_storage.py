#!/usr/bin/env python3
"""
Storage Map Plot - Storage capacity visualization
Creates maps showing storage capacity distribution and total capacity versus CO2 levels.
"""

import sys
import copy
import os
import itertools
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
    use_extensions,
)
from utils import data_handling
from utils.clustering_visualisation import truncate_colormap
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    SUBTITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    MAP_XLIM,
    MAP_YLIM,
    MAP_LINE_WIDTH,
    MAP_LINK_WIDTH,
    add_panel_label,
    save_figure,
    setup_map_axes,
)

# Backwards compatibility constants
SUBLABEL_FONTSIZE = PANEL_LABEL_FONTSIZE
LABEL_FONTSIZE = AXIS_LABEL_FONTSIZE
TICK_LABELSIZE = TICK_LABEL_FONTSIZE
AXIS_LABELSIZE = AXIS_LABEL_FONTSIZE


# Helper function for aggregating carriers
def aggregate_carriers(current_gen, carrier_keyword_to_new_carrier, inplace=False):
    """
    Aggregate carriers in current_gen MultiIndex DataFrame by keywords.

    Parameters
    ----------
    current_gen : pd.Series or pd.DataFrame
        Indexed by (bus, carrier).
    carrier_keyword_to_new_carrier : dict
        Dictionary mapping keywords to new carrier names.

    Returns
    -------
    pd.Series or pd.DataFrame
        With carriers aggregated by keyword.
    """
    if inplace:
        current_gen = current_gen.copy()
    for keyword, new_carrier in carrier_keyword_to_new_carrier.items():
        mask = [carrier for bus, carrier in current_gen.index if keyword in carrier]
        summed = current_gen.loc[(slice(None), mask)].groupby(level="bus").sum()
        for bus in summed.index:
            current_gen.loc[(bus, new_carrier)] = summed[bus]
        current_gen = current_gen.drop(
            index=[
                (bus, carrier)
                for bus, carrier in current_gen.index
                if keyword in carrier and carrier != new_carrier
            ]
        )
    return current_gen


def create_combined_generation_storage_plot():
    """Create combined generation and storage map plot."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load network graph and node positions
    snet_index = 0
    network = data_handling.load_pypsa_network(
        n_nodes=n_nodes, co2lvl=0.0, use_sclopf=True, lopt=use_extensions
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
        )
        for co2l in co2ls
    }

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Map styling constants (use from plot_style)
    LINE_WIDTH = MAP_LINE_WIDTH
    LINK_WIDTH = MAP_LINK_WIDTH

    # Create combined figure with generation on top, storage below
    f = plt.figure(
        figsize=(24, 18)
    )  # Optimized height for combined plot with equal map sections

    # Main grid: generation on top, storage below
    # Generation needs 2x height since it has 2 rows of maps vs storage's 1 row
    gs_main = GridSpec(2, 1, figure=f, height_ratios=[2, 1], hspace=0.15)

    # === GENERATION SECTION (TOP) ===
    gs_generation = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], width_ratios=[2.2, 1], wspace=0.1
    )
    gs_gen_left = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_generation[0], wspace=-0.23, hspace=0.1
    )
    gs_gen_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_generation[1], height_ratios=[1.2, 0.5], hspace=0.12
    )

    # Generation map axes
    ax1_gen = [
        f.add_subplot(gs_gen_left[0, i], projection=ccrs.PlateCarree())
        for i in range(2)
    ]
    ax2_gen = [
        f.add_subplot(gs_gen_left[1, i], projection=ccrs.PlateCarree())
        for i in range(2)
    ]
    axs_primary_gen = [ax1_gen, ax2_gen]
    ax3_gen = f.add_subplot(gs_gen_right[0, 0])
    ax3_legend_gen = f.add_subplot(gs_gen_right[1, 0])

    # === STORAGE SECTION (BOTTOM) ===
    gs_storage = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[1], width_ratios=[2.2, 1], wspace=0.1
    )
    gs_storage_maps = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_storage[0], wspace=-0.23, hspace=0.1
    )
    gs_storage_lines = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_storage[1], height_ratios=[1, 0.5], hspace=0.2
    )

    # Storage map axes
    axs_maps_storage = np.array(
        [
            f.add_subplot(gs_storage_maps[i], projection=ccrs.PlateCarree())
            for i in range(2)
        ]
    )
    ax_line_storage = f.add_subplot(gs_storage_lines[0])
    ax_line_legend_storage = f.add_subplot(gs_storage_lines[1])

    panel_label_axis = [
        ax1_gen[0],
        ax2_gen[0],
        ax3_gen,
        axs_maps_storage[0],
        ax_line_storage,
    ]
    for idx, ax_loss_lvl in enumerate(panel_label_axis):
        add_panel_label(ax_loss_lvl, idx, x_offset=-0.15)

    # === GENERATION PLOT ===
    # Calculate generation by carrier and CO2 level
    generation_by_carrier_and_co2l = pd.DataFrame(
        index=networks[0.6].carriers.index, columns=co2ls
    )
    carriers = list(generation_by_carrier_and_co2l.index)

    objectives = np.zeros(len(co2ls))

    for i, level in enumerate(co2ls):
        network_lvl = networks[level]
        generation_by_carrier_and_co2l[level] = (
            network_lvl.generators_t.p.mul(
                network_lvl.snapshot_weightings.generators, axis="index"
            )
            .mul(network_lvl.generators.sign)
            .T.groupby(network_lvl.generators["carrier"])
            .sum()
            .sum(axis=1)
        )
        objectives[i] = network_lvl.objective
    generation_by_carrier_and_co2l.dropna(axis=0, inplace=True)

    # Filter carriers
    plot_generator_type_threshold = 0.01
    max_carrier_share = (
        generation_by_carrier_and_co2l / generation_by_carrier_and_co2l.sum(axis=0)
    ).max(axis=1)
    carrier_mask = max_carrier_share > plot_generator_type_threshold
    carrier_mask["OCGT"] = True  # keep gas carriers, add them together later
    dropped_carriers = carrier_mask[~carrier_mask].index
    print(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
    generation_by_carrier_and_co2l = generation_by_carrier_and_co2l[carrier_mask]

    target_levels = [0.6, 0.0]
    # labels_gen = [
    #     [r"\textbf{A}", r"", r"\textbf{C}"],
    #     [r"\textbf{B}", "", r"\textbf{D}", "", ""],
    # ]

    # Panel a and b - Generation Maps
    vmax = 3.2e2
    months = [7, 12]

    carrier_keyword_to_new_carrier = {
        "offwind": "offwind-dc",
        "solar": "solar",
    }

    for i, target_level in enumerate(target_levels):
        n = networks[target_level]
        for ii, month in enumerate(months):
            axs_primary_gen[i][ii].set_ylim([0, vmax])
            current_snapshots = n.snapshots[n.snapshots.month == month]
            current_gen = (
                n.generators_t.p.mul(
                    network.snapshot_weightings.generators, axis="index"
                )
                .loc[current_snapshots]
                .sum()
                .mul(network.generators.sign)
            )
            current_gen = current_gen.groupby(
                [n.generators["bus"], n.generators["carrier"]]
            ).sum()
            current_gen = aggregate_carriers(
                current_gen, carrier_keyword_to_new_carrier, inplace=False
            )
            assert set(current_gen.index.get_level_values("carrier").unique()).issubset(
                set(carriers)
            ), f"carrier is not in carriers: {set(current_gen.index.get_level_values('carrier').unique()) - set(carriers)}"

            n.plot(
                bus_sizes=current_gen / 7e6,
                line_colors="black",
                link_colors="black",
                link_widths=LINK_WIDTH,
                line_widths=LINE_WIDTH,
                ax=axs_primary_gen[i][ii],
            )

            # Set consistent map limits and aspect ratio
            axs_primary_gen[i][ii].set_xlim(MAP_XLIM)
            axs_primary_gen[i][ii].set_ylim(MAP_YLIM)
            axs_primary_gen[i][ii].set_aspect("equal")

            if i == 0:
                axs_primary_gen[i][ii].set_title(
                    current_snapshots.month_name()[0], fontsize=TITLE_FONTSIZE
                )

            # axs_primary_gen[i][ii].text(
            #     0 - 0.1,
            #     1 + 0.05,
            #     labels_gen[i][ii],
            #     fontsize=SUBLABEL_FONTSIZE,
            #     weight="bold",
            #     verticalalignment="center",
            #     transform=axs_primary_gen[i][ii].transAxes,
            # )

        axs_primary_gen[i][0].text(
            0 - 0.15,
            0.4,
            rf"CO$_2$={get_actual_co2_level(target_level, percent=True)}\% ",
            fontsize=LABEL_FONTSIZE,
            verticalalignment="center",
            # rotation=90,
            transform=axs_primary_gen[i][0].transAxes,
        )

    # Panel c - Generation vs CO2 level
    marker = itertools.cycle((",", "x", ".", "o", "*"))

    colors_gens = network.carriers.set_index("nice_name").color
    colors_gens["Open-Cycle Gas"] = "#cc0099"
    colors_gens["Gas"] = "#cc0099"
    colors_gens["Offshore Wind"] = n.carriers.color["offwind-dc"]

    co2s = generation_by_carrier_and_co2l.columns.astype("float")
    index = np.argsort(co2s)

    generation_by_carrier_and_co2l = generation_by_carrier_and_co2l.rename(
        index=network.carriers.nice_name
    )
    gas_gen = (
        generation_by_carrier_and_co2l.loc["Combined-Cycle Gas"]
        + generation_by_carrier_and_co2l.loc["Open-Cycle Gas"]
    )
    offwind_carriers = [
        carrier for carrier in generation_by_carrier_and_co2l.index if "Off" in carrier
    ]
    offwind_gen = generation_by_carrier_and_co2l.loc[offwind_carriers].sum(axis=0)
    solar_carriers = [
        carrier
        for carrier in generation_by_carrier_and_co2l.index
        if "solar" in carrier.lower()
    ]
    solar_gen = generation_by_carrier_and_co2l.loc[solar_carriers].sum(axis=0)

    generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l.copy()

    # Remove individual gas carriers and combine them
    for col in generation_by_carrier_and_co2l_plot.index:
        if "Gas" in col:
            generation_by_carrier_and_co2l_plot.drop(col, axis=0, inplace=True)
    generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l_plot.rename(
        index=network.carriers.nice_name
    )
    generation_by_carrier_and_co2l_plot.loc["Gas"] = gas_gen

    generation_by_carrier_and_co2l_plot.loc["Offshore Wind"] = offwind_gen
    generation_by_carrier_and_co2l_plot.drop(offwind_carriers, axis=0, inplace=True)

    generation_by_carrier_and_co2l_plot.loc["Solar"] = solar_gen
    generation_by_carrier_and_co2l_plot.drop(
        [carrier for carrier in solar_carriers if carrier != "Solar"],
        axis=0,
        inplace=True,
    )

    for ind, data in generation_by_carrier_and_co2l_plot.iterrows():
        jit = 0
        # plot_inds = (data.values)[index] > 0 # only plot non-zero points
        plot_inds = np.ones(len(data.values), dtype=bool)  # plot all points
        ax3_gen.scatter(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-6)[plot_inds],
            color=colors_gens.loc[ind],
            label=ind,
            s=100,
            marker=next(marker),
        )

        ax3_gen.plot(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-6)[plot_inds],
            color=colors_gens.loc[ind],
            alpha=0.8,
        )

    ax3_gen.invert_xaxis()
    ax3_gen.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)
    ax3_gen.set_xlabel(r"$\text{CO}_2$ level [\% of 1990]", fontsize=AXIS_LABELSIZE)
    ax3_gen.set_ylabel(r"Total annual generation [TWh]", fontsize=AXIS_LABELSIZE)
    ax3_gen.grid(True)
    ax3_gen.yaxis.offsetText.set_fontsize(TICK_LABELSIZE)
    # ax3_gen.text(
    #     0 - 0.19,
    #     1 + 0.05,
    #     labels_gen[0][2],
    #     fontsize=SUBLABEL_FONTSIZE,
    #     weight="bold",
    #     verticalalignment="center",
    #     transform=ax3_gen.transAxes,
    # )
    ax3_gen.set_ylim((-2e1, 12.4e2))

    # Generation Legend
    h, l = ax3_gen.get_legend_handles_labels()
    # Capitalize all legend labels
    l = [label.capitalize() for label in l]
    ax3_legend_gen.legend(
        h, l, borderaxespad=0.3, fontsize=LEGEND_FONTSIZE, ncol=2, loc="center"
    )
    ax3_legend_gen.axis("off")

    # === STORAGE PLOT ===
    target_levels_storage = [0.0]
    unit = "GWh"
    unit_factor = 1e-3
    plot_output_capacity = True

    # Panel d, e: storage maps
    buses = list(network.buses.index)
    storage_types = network.storage_units["carrier"].unique()
    multi_index = pd.MultiIndex.from_product(
        [buses, storage_types], names=["bus", "carrier"]
    )

    battery_color = "tab:olive"
    h2_color = "green"

    for i, target_level in enumerate(target_levels_storage):
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
                link_widths=LINK_WIDTH,
                line_widths=LINE_WIDTH,
                ax=axs_maps_storage[ii],
            )

            # Set consistent map limits and aspect ratio (same as generation maps)
            axs_maps_storage[ii].set_xlim(MAP_XLIM)
            axs_maps_storage[ii].set_ylim(MAP_YLIM)
            axs_maps_storage[ii].set_aspect("equal")

            legend_relative_circle_size = np.array([0.25, 1])
            legend_circle_size = [
                size * max_node_size for size in legend_relative_circle_size
            ]
            legend_circle_size = [
                round(size, -int(np.floor(np.log10(size))))
                for size in legend_circle_size
            ]
            pypsa.plot.add_legend_circles(
                axs_maps_storage[ii],
                sizes=legend_circle_size,
                labels=[
                    f"{size*max_capacity_rounded*unit_factor:2g} {unit}"
                    for size in legend_circle_size
                ],
                patch_kw={"color": current_color},
            )

            current_store_type_string = (
                current_store_type[0].upper() + current_store_type[1:]
            ).replace("H2", "Hydrogen")
            axs_maps_storage[ii].set_title(
                current_store_type_string, fontsize=TITLE_FONTSIZE
            )

    # # Add storage labels
    # axs_maps_storage[0].text(
    #     0 - 0.1,
    #     1 + 0.05,
    #     r"\textbf{D}",
    #     fontsize=SUBLABEL_FONTSIZE,
    #     weight="bold",
    #     verticalalignment="center",
    #     transform=axs_maps_storage[0].transAxes,
    # )

    axs_maps_storage[0].text(
        0 - 0.15,
        0.4,
        rf"CO$_2$={get_actual_co2_level(target_levels_storage[0], percent=True)}\% ",
        fontsize=LABEL_FONTSIZE,
        weight="bold",
        verticalalignment="center",
        transform=axs_maps_storage[0].transAxes,
    )

    # # Panel f: total storage capacity line plot
    # ax_line_storage.text(
    #     0 - 0.19,
    #     1 + 0.05,
    #     r"\textbf{E}",
    #     fontsize=SUBLABEL_FONTSIZE,
    #     weight="bold",
    #     verticalalignment="center",
    #     transform=ax_line_storage.transAxes,
    # )

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

    ax_line_storage.plot(
        get_actual_co2_level(capacity_by_type_by_lvl.columns, percent=True),
        capacity_by_type_by_lvl_plotting.loc[
            :, ["Hydro", "Pumped hydro", "H2", "Battery"]
        ].values
        * unit_factor,
        label=["Hydro", "Pumped hydro", "Hydrogen", "Battery"],
    )

    for feat_count in range(capacity_by_type_by_lvl_plotting.shape[1]):
        if "ydro" in capacity_by_type_by_lvl_plotting.columns[feat_count]:
            continue
        ax_line_storage.scatter(
            get_actual_co2_level(capacity_by_type_by_lvl.columns, percent=True),
            (
                capacity_by_type_by_lvl_plotting.loc[
                    :, ["Hydro", "Pumped hydro", "H2", "Battery"]
                ].values
                * unit_factor
            )[:, feat_count],
            color=colors[feat_count],
        )

    secax = ax_line_storage.secondary_yaxis(
        "right", functions=(lambda x: x * (1 / 6), lambda x: x * 6)
    )
    secax.set_ylabel("Battery power [GW]", fontsize=AXIS_LABELSIZE)
    secax.tick_params(labelsize=TICK_LABELSIZE)

    for i, ii in enumerate(ax_line_storage.lines):
        ii.set_color(colors[i])

    ax_line_storage.invert_xaxis()
    ax_line_storage.grid(True)
    ax_line_storage.set_yscale("log")
    ax_line_storage.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)

    if plot_output_capacity:
        ax_line_storage.set_ylabel(
            f"Storage output capacity [{unit}]", fontsize=AXIS_LABELSIZE
        )
        file_name = "combined_generation_storage_output_capacity"
    else:
        ax_line_storage.set_ylabel(
            f"Storage capacity [{unit}]", fontsize=AXIS_LABELSIZE
        )
        file_name = "combined_generation_storage_capacity"

    ax_line_storage.set_xlabel(r"CO2 level [\% of 1990]", fontsize=AXIS_LABELSIZE)
    ax_line_storage.legend()

    h, l = ax_line_storage.get_legend_handles_labels()
    ax_line_storage.get_legend().remove()
    ax_line_legend_storage.legend(
        h, l, borderaxespad=0.3, fontsize=LEGEND_FONTSIZE, ncol=2, loc="center"
    )
    ax_line_legend_storage.axis("off")

    save_figure(f, save_path, file_name)
    plt.show()


if __name__ == "__main__":
    create_combined_generation_storage_plot()
