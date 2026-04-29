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

# Logging
from loguru import logger

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
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


_COUNTRY_3_TO_2 = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "CHE": "CH",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
}

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


def _country_from_bus_name(bus_name: str) -> str:
    bus_str = str(bus_name)
    if len(bus_str) >= 2 and bus_str[:2].isalpha():
        code = bus_str[:2].upper()
        if code == "UK":
            return "GB"
        return code
    if len(bus_str) >= 3 and bus_str[:3].isalpha():
        code3 = bus_str[:3].upper()
        return _COUNTRY_3_TO_2.get(code3, code3)
    return "UNK"


def _get_bus_country_map(network):
    if "country" in network.buses.columns:
        countries = network.buses["country"].fillna("UNK")
        return {bus: str(country) for bus, country in countries.items()}
    return {bus: _country_from_bus_name(bus) for bus in network.buses.index}


def _mean_generation_by_country(network) -> pd.Series:
    weights = network.snapshot_weightings.generators
    total_weight = weights.sum()
    gen_weighted = network.generators_t.p.mul(weights, axis="index").mul(
        network.generators.sign
    )
    mean_gen_by_generator = gen_weighted.sum(axis=0) / total_weight
    mean_gen_by_bus = mean_gen_by_generator.groupby(network.generators["bus"]).sum()
    bus_country_map = _get_bus_country_map(network)
    country_vals = {}
    for bus, value in mean_gen_by_bus.items():
        country = bus_country_map.get(bus, "UNK")
        country_vals[country] = country_vals.get(country, 0.0) + value
    return pd.Series(country_vals).sort_values(ascending=False)


def _aggregate_countries_by_region(df: pd.DataFrame) -> pd.DataFrame:
    region_to_countries = {
        # "Iberian Peninsula": ["ES", "PT"],
        "Balkans": ["BG", "GR", "HR", "RO", "SI", "RS", "AL", "BA", "MK", "ME", "XK"],
        "DE LU": ["DE", "LU"],
        # "NL+Belgium": ["NL", "BE"],
        # "Great Britain": ["GB", "IE"],
        "Scandinavia": ["NO", "SE", "FI"],
        "Balticum": ["LT", "LV", "EE"],
    }
    #     further agg:
    # - CH and AT
    # - CZ, HU, SK
    # - balticum
    # -  IE, GB

    df_agg = df.copy()
    for region, countries in region_to_countries.items():
        existing = [c for c in countries if c in df_agg.index]
        if not existing:
            continue
        df_agg.loc[region] = df_agg.loc[existing].sum()
        df_agg = df_agg.drop(index=existing)
    return df_agg


def create_generation_by_country_stacked_bar_plot(n_nodes: int = 600):
    """Create stacked bar plot of mean generation by country across CO2 levels."""

    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    setup_matplotlib_style()

    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
        )
        for co2l in co2ls
    }

    country_by_co2 = {}
    for co2l, network in networks.items():
        country_by_co2[co2l] = _mean_generation_by_country(network)

    generation_by_country = pd.DataFrame(country_by_co2).fillna(0.0)
    generation_by_country = _aggregate_countries_by_region(generation_by_country)
    plot_order = [
        "PT",
        "ES",
        "FR",
        "NL",
        "BE",
        "DE LU",
        "DK",
        "CH",
        "AT",
        "IT",
        "CZ",
        "PL",
        "SK",
        "HU",
        "SI",
        "RO",
        "Balkans",
        "GB",
        "IE",
        "Scandinavia",
    ]
    ordered_index = [x for x in plot_order if x in generation_by_country.index]
    ordered_index += [x for x in generation_by_country.index if x not in ordered_index]
    generation_by_country = generation_by_country.loc[ordered_index] / 1000

    co2_order = np.array(sorted(co2ls))
    x_labels = [f"{get_actual_co2_level(c, percent=True):g}%" for c in co2_order]

    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = mpl.colormaps.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(generation_by_country.index))]

    generation_by_country[co2_order].T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.8,
        edgecolor="none",
    )

    ax.set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Mean generation [GW]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.invert_xaxis()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="Country",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
    )

    fig.tight_layout()
    save_figure(fig, "generation_by_country_stacked_bar", save_path)
    plt.show()


def create_combined_generation_storage_plot(plot_battery_power=False):
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
    logger.info(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
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
        logger.info(f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{target_level}-2920SEG.nc")
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
        # if "ydro" in capacity_by_type_by_lvl_plotting.columns[feat_count]:
        #     continue
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

    if plot_battery_power:
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

    save_figure(f, file_name, save_path)
    plt.show()


def create_generation_capacity_plot(show_pie_charts=False):
    """Create generation capacity map plot showing installed capacity distribution across CO2 levels.

    Parameters
    ----------
    show_pie_charts : bool, optional
        If True, adds a second row with pie charts showing total capacity by carrier for each CO2 level.
        Default is False.
    """
    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load networks
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
        )
        for co2l in co2ls
    }

    setup_matplotlib_style()

    # Select three CO2 levels to display
    target_levels = [0.6, 0.3, 0.0]  # High, medium, low CO2

    # Get reference network for carrier info
    ref_network = networks[0.6]

    # Calculate capacity by carrier and CO2 level (for filtering)
    capacity_by_carrier_and_co2l = pd.DataFrame(
        index=ref_network.carriers.index, columns=co2ls
    )

    for i, level in enumerate(co2ls):
        network_lvl = networks[level]
        capacity_by_carrier_and_co2l[level] = network_lvl.generators.p_nom.groupby(
            network_lvl.generators["carrier"]
        ).sum()
    capacity_by_carrier_and_co2l.dropna(axis=0, inplace=True)
    # filter load carriers as well, since we want to focus on generation capacity here
    capacity_by_carrier_and_co2l.drop(
        index=capacity_by_carrier_and_co2l.index[
            capacity_by_carrier_and_co2l.index.str.contains("load", case=False)
        ],
        inplace=True,
    )
    # Filter carriers using same threshold as example
    plot_generator_type_threshold = 0.01
    max_carrier_share = (
        capacity_by_carrier_and_co2l / capacity_by_carrier_and_co2l.sum(axis=0)
    ).max(axis=1)
    carrier_mask = max_carrier_share > plot_generator_type_threshold
    # mask load carriers as well, even if they are above the threshold, since we want to focus on generation capacity here
    carrier_mask[carrier_mask.index.str.contains("load", case=False)] = False
    carrier_mask["OCGT"] = True  # keep gas carriers, add them together later
    dropped_carriers = carrier_mask[~carrier_mask].index
    logger.info(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
    filtered_carriers = capacity_by_carrier_and_co2l[carrier_mask].index.tolist()

    # Setup colors same as example
    colors_gens = ref_network.carriers.set_index("nice_name").color
    colors_gens["Open-Cycle Gas"] = "#cc0099"
    colors_gens["Gas"] = "#cc0099"
    colors_gens["Offshore Wind"] = ref_network.carriers.color["offwind-dc"]

    # Create figure
    if show_pie_charts:
        fig = plt.figure(figsize=(24, 16))
        gs_main = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.05)
        gs_maps = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.025)
        gs_pies = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.2)
    else:
        fig = plt.figure(figsize=(24, 8))
        gs_maps = GridSpec(1, 3, figure=fig, wspace=0.05)

    # Create map axes
    axs_maps = [
        fig.add_subplot(gs_maps[i], projection=ccrs.PlateCarree()) for i in range(3)
    ]

    # Create pie chart axes if needed
    if show_pie_charts:
        axs_pies = [fig.add_subplot(gs_pies[i]) for i in range(3)]

    # Constants
    LINE_WIDTH = MAP_LINE_WIDTH
    LINK_WIDTH = MAP_LINK_WIDTH

    carrier_keyword_to_new_carrier = {
        "offwind": "offwind-dc",
        "solar": "solar",
    }

    # Prepare data structures for pie charts
    if show_pie_charts:
        capacity_by_carrier = {}

    # Plot maps
    for i, target_level in enumerate(target_levels):
        n = networks[target_level]

        # Get generation capacity by bus and carrier (only filtered carriers)
        capacity = n.generators.p_nom.copy()
        # Filter to only include carriers that passed the threshold
        generator_mask = n.generators["carrier"].isin(filtered_carriers)
        capacity_filtered = capacity[generator_mask]
        generators_filtered = n.generators[generator_mask]

        capacity_by_bus_carrier = capacity_filtered.groupby(
            [generators_filtered["bus"], generators_filtered["carrier"]]
        ).sum()

        # Aggregate carriers
        capacity_by_bus_carrier = aggregate_carriers(
            capacity_by_bus_carrier, carrier_keyword_to_new_carrier, inplace=False
        )

        # Plot
        n.plot.map(
            bus_sizes=capacity_by_bus_carrier / 5e4,  # Adjust scaling factor
            line_colors="black",
            link_colors="black",
            link_widths=LINK_WIDTH,
            line_widths=LINE_WIDTH,
            ax=axs_maps[i],
        )

        # Set map limits
        axs_maps[i].set_xlim(MAP_XLIM)
        axs_maps[i].set_ylim(MAP_YLIM)
        axs_maps[i].set_aspect("equal")

        # Set title
        axs_maps[i].set_title(
            rf"CO$_2$ = {get_actual_co2_level(target_level, percent=True)}\%",
            fontsize=TITLE_FONTSIZE,
        )

        # Add panel label
        if i == 0:
            add_panel_label(axs_maps[i], 0, x_offset=-0.15)

        # Prepare pie chart data
        if show_pie_charts:
            # Get total capacity by carrier (filtered)
            total_capacity_by_carrier = capacity_filtered.groupby(
                generators_filtered["carrier"]
            ).sum()

            # Rename to nice names first
            total_capacity_renamed = total_capacity_by_carrier.rename(
                index=n.carriers.nice_name
            )

            # Aggregate Gas carriers
            gas_capacity = 0
            gas_carriers_found = []
            for carrier in total_capacity_renamed.index:
                if "Gas" in carrier:
                    gas_capacity += total_capacity_renamed[carrier]
                    gas_carriers_found.append(carrier)

            # Aggregate offshore wind carriers
            offwind_carriers = [
                carrier for carrier in total_capacity_renamed.index if "Off" in carrier
            ]
            offwind_capacity = (
                total_capacity_renamed.loc[offwind_carriers].sum()
                if offwind_carriers
                else 0
            )

            # Aggregate solar carriers
            solar_carriers = [
                carrier
                for carrier in total_capacity_renamed.index
                if "solar" in carrier.lower()
            ]
            solar_capacity = (
                total_capacity_renamed.loc[solar_carriers].sum()
                if solar_carriers
                else 0
            )

            # Build aggregated series
            capacity_series = total_capacity_renamed.copy()

            # Remove and replace gas carriers
            for carrier in gas_carriers_found:
                capacity_series = capacity_series.drop(carrier)
            if gas_capacity > 0:
                capacity_series["Gas"] = gas_capacity

            # Remove and replace offshore wind
            for carrier in offwind_carriers:
                if carrier in capacity_series.index:
                    capacity_series = capacity_series.drop(carrier)
            if offwind_capacity > 0:
                capacity_series["Offshore Wind"] = offwind_capacity

            # Remove and replace solar (keep one)
            solar_to_drop = [c for c in solar_carriers if c != "Solar"]
            for carrier in solar_to_drop:
                if carrier in capacity_series.index:
                    capacity_series = capacity_series.drop(carrier)
            if "Solar" in capacity_series.index:
                capacity_series["Solar"] = solar_capacity

            capacity_by_carrier[target_level] = capacity_series

    # Add legend to maps
    # Collect unique carriers from the last level processed
    legend_carriers = []
    legend_colors = []

    # Use capacity_series from pie chart data if available, otherwise create from maps
    if show_pie_charts:
        # Use carriers from the processed capacity_by_carrier
        sample_carriers = capacity_by_carrier[target_levels[0]]
        for carrier in sample_carriers.index:
            if carrier in colors_gens.index:
                legend_carriers.append(carrier)
                legend_colors.append(colors_gens.loc[carrier])
    else:
        # Build legend from the capacity data
        n = networks[target_levels[0]]
        capacity = n.generators.p_nom.copy()
        generator_mask = n.generators["carrier"].isin(filtered_carriers)
        capacity_filtered = capacity[generator_mask]
        generators_filtered = n.generators[generator_mask]

        total_capacity_by_carrier = capacity_filtered.groupby(
            generators_filtered["carrier"]
        ).sum()

        # Rename to nice names
        total_capacity_renamed = total_capacity_by_carrier.rename(
            index=n.carriers.nice_name
        )

        # Aggregate carriers same as pie charts
        gas_capacity = 0
        gas_carriers_found = []
        for carrier in total_capacity_renamed.index:
            if "Gas" in carrier:
                gas_capacity += total_capacity_renamed[carrier]
                gas_carriers_found.append(carrier)

        offwind_carriers_leg = [
            carrier for carrier in total_capacity_renamed.index if "Off" in carrier
        ]
        offwind_capacity = (
            total_capacity_renamed.loc[offwind_carriers_leg].sum()
            if offwind_carriers_leg
            else 0
        )

        solar_carriers_leg = [
            carrier
            for carrier in total_capacity_renamed.index
            if "solar" in carrier.lower()
        ]
        solar_capacity = (
            total_capacity_renamed.loc[solar_carriers_leg].sum()
            if solar_carriers_leg
            else 0
        )

        capacity_series_leg = total_capacity_renamed.copy()

        for carrier in gas_carriers_found:
            capacity_series_leg = capacity_series_leg.drop(carrier)
        if gas_capacity > 0:
            capacity_series_leg["Gas"] = gas_capacity

        for carrier in offwind_carriers_leg:
            if carrier in capacity_series_leg.index:
                capacity_series_leg = capacity_series_leg.drop(carrier)
        if offwind_capacity > 0:
            capacity_series_leg["Offshore Wind"] = offwind_capacity

        solar_to_drop_leg = [c for c in solar_carriers_leg if c != "Solar"]
        for carrier in solar_to_drop_leg:
            if carrier in capacity_series_leg.index:
                capacity_series_leg = capacity_series_leg.drop(carrier)
        if "Solar" in capacity_series_leg.index:
            capacity_series_leg["Solar"] = solar_capacity

        for carrier in capacity_series_leg.index:
            if carrier in colors_gens.index:
                legend_carriers.append(carrier)
                legend_colors.append(colors_gens.loc[carrier])

    # Sort legend carriers alphabetically
    sorted_legend = sorted(zip(legend_carriers, legend_colors), key=lambda x: x[0])
    legend_carriers = [item[0] for item in sorted_legend]
    legend_colors = [item[1] for item in sorted_legend]

    # Create legend patches
    legend_patches = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in zip(legend_carriers, legend_colors)
    ]

    # Add legend to the last map axes or create a separate legend
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.6),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Plot pie charts
    if show_pie_charts:
        for i, target_level in enumerate(target_levels):
            capacity_data = capacity_by_carrier[target_level]

            # Sort alphabetically
            capacity_data = capacity_data.sort_index()

            # Get colors and labels
            colors = []
            labels = []
            for carrier in capacity_data.index:
                if carrier in colors_gens.index:
                    colors.append(colors_gens.loc[carrier])
                    labels.append(carrier)
                else:
                    # Fallback for carriers not in color map
                    colors.append("gray")
                    labels.append(carrier)

            # Create pie chart
            wedges, texts, autotexts = axs_pies[i].pie(
                capacity_data.values,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": LEGEND_FONTSIZE},
            )

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_weight("bold")
                autotext.set_fontsize(LEGEND_FONTSIZE)

            axs_pies[i].set_title(
                rf"CO$_2$ = {get_actual_co2_level(target_level, percent=True)}\%",
                fontsize=TITLE_FONTSIZE,
            )

            # Add panel label for first pie chart
            if i == 0:
                add_panel_label(axs_pies[i], 1, x_offset=-0.15)

    # Save figure
    file_name = "generation_capacity_maps" + ("_with_pies" if show_pie_charts else "")
    save_figure(fig, file_name, save_path)
    plt.show()


if __name__ == "__main__":
    # create_combined_generation_storage_plot()
    # create_generation_by_country_stacked_bar_plot()

    # Create capacity maps with and without pie charts
    create_generation_capacity_plot(show_pie_charts=False)
    create_generation_capacity_plot(show_pie_charts=True)
