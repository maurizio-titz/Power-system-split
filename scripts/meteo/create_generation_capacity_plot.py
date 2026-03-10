from scripts.plots.plot_combined_generation_storage import aggregate_carriers
from utils import data_handling
from utils.config import path_to_figures_sclopf, use_extensions
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.plot_style import (
    LEGEND_FONTSIZE,
    MAP_LINE_WIDTH,
    MAP_LINK_WIDTH,
    MAP_XLIM,
    MAP_YLIM,
    TITLE_FONTSIZE,
    add_panel_label,
    save_figure,
    setup_matplotlib_style,
)


import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch


import os


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
    print(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
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
        n.plot(
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
