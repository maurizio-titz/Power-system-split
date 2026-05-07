#!/usr/bin/env python3
"""
Line Failure Probabilities Plot - Figure 7: line failures
Visualizes primary and secondary line failure probabilities across CO2 levels.
"""

import sys
import gzip
import pickle
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

# Logging
from tqdm import tqdm
from loguru import logger
import logging
import pypsa
pypsa.network.io.logger.setLevel(logging.ERROR)

sys.path.append("./")

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling
from utils.calculate_line_loadings import calculate_line_loadings_for_all_snapshots
from scripts.plots import plot_split_statistics
from utils.plot_style import (
    setup_matplotlib_style,
    PANEL_LABEL_FONTSIZE,
    COLORBAR_LABEL_FONTSIZE,
    COLORBAR_TICK_FONTSIZE,
    SUBTITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    PRIMARY_COLORS,
    add_panel_label,
    setup_colormap_scientific_notation,
    save_figure,
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


def _get_bus_country_map(network) -> dict:
    if "country" in network.buses.columns:
        countries = network.buses["country"].fillna("UNK")
        return {bus: str(country) for bus, country in countries.items()}
    return {bus: _country_from_bus_name(bus) for bus in network.buses.index}


def _edge_is_cross_border(u: str, v: str, bus_country_map: dict) -> bool:
    cu = bus_country_map.get(u, "UNK")
    cv = bus_country_map.get(v, "UNK")
    return cu != "UNK" and cv != "UNK" and cu != cv


def save_failure_probabilities_to_csv(
    edge_likelihoods,
    nx_graph,
    selected_co2ls,
    n_nodes,
    save_path,
    filename,
):
    """Save line failure probabilities to CSV.

    Args:
        edge_likelihoods: dict keyed by CO2 level with (u, v) -> prob mapping.
        nx_graph: networkx graph with edge list used for row order.
        selected_co2ls: iterable of CO2 levels (scenarios) to include as columns.
        n_nodes: number of nodes (used for actual CO2 level label).
        save_path: directory to save the CSV.
        filename: output CSV filename.
    """

    os.makedirs(save_path, exist_ok=True)
    edges = list(nx_graph.edges())
    row_index = [f"{u}-{v}" for u, v in edges]

    data = {}
    for co2l in selected_co2ls:
        level = np.round(co2l, 2)
        col_label = f"CO2_{int(get_actual_co2_level(co2l, n_nodes, percent=True))}%"
        data[col_label] = [edge_likelihoods[level][(u, v)] for u, v in edges]

    df = pd.DataFrame(data, index=row_index)
    df.index.name = "line"
    df.to_csv(os.path.join(save_path, filename))


def create_line_failure_plot_linear():
    """Create line failure probabilities plot with linear colorscale."""

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
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    selected_co2ls = np.array([0.0, 0.2, 0.6])

    # Load edge likelihoods
    edge_likelihoods_primary = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"/edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )
    edge_likelihoods_secondary = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"/edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )

    # Setup figure with individual colorbars for each subplot
    f = plt.figure(figsize=(24, 10))
    gs_main = GridSpec(
        2,
        3,
        figure=f,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.15,
        hspace=0.25,
    )

    # Create subplots with space for individual colorbars
    axs_primary = []
    axs_secondary = []
    axs_colorbars_primary = []
    axs_colorbars_secondary = []

    for i in range(3):
        # Create subplot with colorbar space
        gs_sub_primary = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[0, i], width_ratios=[1, 0.05], wspace=0.05
        )
        gs_sub_secondary = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[1, i], width_ratios=[1, 0.05], wspace=0.05
        )

        axs_primary.append(f.add_subplot(gs_sub_primary[0]))
        axs_colorbars_primary.append(f.add_subplot(gs_sub_primary[1]))

        axs_secondary.append(f.add_subplot(gs_sub_secondary[0]))
        axs_colorbars_secondary.append(f.add_subplot(gs_sub_secondary[1]))

    # Setup matplotlib styling
    setup_matplotlib_style()

    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]

    # Primary failures
    cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))

    # import cmcrameri.cm as cmc  # pip install cmcrameri
    # cmap = cmc.batlow
    cmap.set_under("gainsboro", 1.0)

    for count, co2l in enumerate(selected_co2ls[::-1]):
        level = np.round(co2l, 2)

        # Load likelihoods as dictionary and transform into array
        c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
        c_H_p_array = np.array(c_H_p)

        # Calculate individual vmin and vmax for this subplot
        valid_probs = c_H_p_array[c_H_p_array > 1e-12]
        if len(valid_probs) > 0:
            vmin_individual = np.min(valid_probs)
            vmax_individual = np.max(valid_probs)
        else:
            vmin_individual = 1e-6
            vmax_individual = 1e-3

        print(
            f"CO2 level {co2l}: Min val prim: {vmin_individual:.2e}, Max val prim: {vmax_individual:.2e}"
        )

        nodes = nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=axs_primary[count], node_color="black", node_size=0
        )

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=axs_primary[count],
            edge_color=c_H_p,
            width=3.5,
            edge_cmap=cmap,
            edge_vmin=vmin_individual,
            edge_vmax=vmax_individual,
        )

        axs_primary[count].axis("off")
        axs_primary[count].set_aspect("equal")

        # Individual colorbar for this primary subplot
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mplcolors.Normalize(vmin=vmin_individual, vmax=vmax_individual),
        )
        cb = f.colorbar(sm, cax=axs_colorbars_primary[count])
        setup_colormap_scientific_notation(cb)
        # Add title to all colorbars
        axs_colorbars_primary[count].set_title(
            r"$\langle p_{\ell}^{\text{p}}\rangle$",
            fontsize=COLORBAR_LABEL_FONTSIZE,
            weight="bold",
            verticalalignment="center",
            pad=15,
        )

    # Secondary failures
    cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
    cmap.set_under("gainsboro", 1.0)

    for count, co2l in enumerate(selected_co2ls[::-1]):
        level = np.round(co2l, 2)

        # Load likelihoods as dictionary and transform into array
        c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
        c_H_s_array = np.array(c_H_s)

        # Calculate individual vmin and vmax for this subplot
        valid_probs = c_H_s_array[c_H_s_array > 1e-12]
        if len(valid_probs) > 0:
            vmin_individual = np.min(valid_probs)
            vmax_individual = np.max(valid_probs)
        else:
            vmin_individual = 1e-6
            vmax_individual = 1e-3

        print(
            f"CO2 level {co2l}: Min val sec: {vmin_individual:.2e}, Max val sec: {vmax_individual:.2e}"
        )

        nodes = nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=axs_secondary[count], node_color="black", node_size=0
        )

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=axs_secondary[count],
            edge_color=c_H_s,
            width=3.5,
            edge_cmap=cmap,
            edge_vmin=vmin_individual,
            edge_vmax=vmax_individual,
        )

        axs_secondary[count].axis("off")
        axs_secondary[count].set_aspect("equal")

        # Individual colorbar for this secondary subplot
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mplcolors.Normalize(vmin=vmin_individual, vmax=vmax_individual),
        )
        cb = f.colorbar(sm, cax=axs_colorbars_secondary[count])
        cb.ax.tick_params(labelsize=16, width=1.0, which="major")
        # Format ticks in scientific notation
        cb.ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        # Add title to all colorbars
        axs_colorbars_secondary[count].set_title(
            r"$\langle p_{\ell}^{\text{s}}\rangle$",
            fontsize=16,
            weight="bold",
            verticalalignment="center",
            pad=15,
        )

    # Add titles and labels to primary subplots
    for count, co2l in enumerate(selected_co2ls[::-1]):
        # Add subplot titles and labels
        actual_co2_level = get_actual_co2_level(co2l, n_nodes, percent=True)
        axs_primary[count].set_title(
            r"CO$_2 =$ " + "{}%".format(int(actual_co2_level)),
            fontsize=SUBTITLE_FONTSIZE,
        )
        add_panel_label(axs_primary[count], count)
    cmap_name = cmap if isinstance(cmap, str) else cmap.name
    save_figure(f, "line_failure_probs_linear_" + cmap_name, save_path)
    plt.show()


def create_secondary_line_failure_plot():
    """Create line failure probabilities plot showing only secondary failures."""

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
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    selected_co2ls = np.array([0.0, 0.2, 0.6])

    # Load edge likelihoods
    edge_likelihoods_secondary = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )

    # Setup figure with individual colorbars for each subplot (single row)
    f = plt.figure(figsize=(24, 5))
    gs_main = GridSpec(
        1,
        3,
        figure=f,
        width_ratios=[1, 1, 1],
        wspace=0.05,
    )

    # Create subplots with space for individual colorbars
    axs_secondary = []
    axs_colorbars_secondary = []

    for i in range(3):
        # Create subplot with colorbar space
        gs_sub_secondary = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[0, i], width_ratios=[1, 0.05], wspace=-0.05
        )

        axs_secondary.append(f.add_subplot(gs_sub_secondary[0]))
        axs_colorbars_secondary.append(f.add_subplot(gs_sub_secondary[1]))

    # Setup matplotlib styling
    setup_matplotlib_style()

    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}"]

    # Secondary failures
    cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
    # cmap = copy.copy(mpl.cm.get_cmap("copper_r"))
    # cmap = mplcolors.ListedColormap(cmap(np.linspace(0, 0.9, 256)))
    cmap.set_under("gainsboro", 1.0)

    for count, co2l in enumerate(selected_co2ls[::-1]):
        level = np.round(co2l, 2)

        # Load likelihoods as dictionary and transform into array
        c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
        c_H_s_array = np.array(c_H_s)

        # Calculate individual vmin and vmax for this subplot
        valid_probs = c_H_s_array[c_H_s_array > 1e-12]
        if len(valid_probs) > 0:
            vmin_individual = np.min(valid_probs)
            vmax_individual = np.max(valid_probs)
        else:
            vmin_individual = 1e-6
            vmax_individual = 1e-3

        print(
            f"CO2 level {co2l}: Min val sec: {vmin_individual:.2e}, Max val sec: {vmax_individual:.2e}"
        )

        nodes = nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=axs_secondary[count], node_color="black", node_size=0
        )

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=axs_secondary[count],
            edge_color=c_H_s,
            width=3.5,
            edge_cmap=cmap,
            edge_vmin=vmin_individual,
            edge_vmax=vmax_individual,
        )

        axs_secondary[count].axis("off")
        axs_secondary[count].set_aspect("equal")

        # Individual colorbar for this secondary subplot
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mplcolors.Normalize(vmin=vmin_individual, vmax=vmax_individual),
        )
        cb = f.colorbar(sm, cax=axs_colorbars_secondary[count])
        setup_colormap_scientific_notation(cb)
        # Add title to all colorbars
        axs_colorbars_secondary[count].set_title(
            r"$\langle p_{\ell}^{\text{s}}\rangle$",
            fontsize=COLORBAR_LABEL_FONTSIZE,
            weight="bold",
            verticalalignment="center",
            pad=15,
        )

        # Add subplot titles and labels
        actual_co2_level = get_actual_co2_level(co2l, n_nodes, percent=True)
        axs_secondary[count].set_title(
            r"CO$_2 =$ " + r"{}\%".format(int(actual_co2_level)),
            fontsize=SUBTITLE_FONTSIZE,
        )
        add_panel_label(axs_secondary[count], count, x_offset=0.1)

    save_figure(f, "line_failure_probs_secondary_only", save_path)
    plt.show()


def create_total_line_failure_plot(
    unified_colorbar: bool = False,
    cmap: str =" crameri:roma_r",
    powernorm: bool = False,
    scale_width: bool = False,
    width_scale_sqrt: bool = False,
    in_second_row: str | None = None,
    second_row_condition_on_split: bool = True,
    save_fig: bool = True,
    calc_line_loading_gain: bool = False
):
    """Create line failure probabilities plot showing only secondary failures.

    Args:
        unified_colorbar: If True, use the same colorscale for all subplots with a single colorbar.
                         If False, use individual colorscales and colorbars for each subplot.
        powernorm: If True and unified_colorbar is True, use PowerNorm for colorbar scaling.
        scale_width: If True, scale edge widths by failure probability values.
        width_scale_log: If True and scale_width is True, use logarithmic scaling for widths.
    """

    if in_second_row is not None and in_second_row not in ["line_loading", "blackout_sizes"]:
        raise ValueError("Only 'line_loading', 'black_sizes' or None (to only have one row)" +
                         "are availabel for argument 'in_second_row'.")

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
    # import cmcrameri.cm as cmc  # pip install cmcrameri
    # cmap = cmc.batlow_r
    from cmap import Colormap

    if isinstance(cmap, str):
        cmap = Colormap(
            cmap
        ).to_mpl()  # case insensitive, convert to matplotlib colormap
    # cmap = copy.copy(mpl.cm.get_cmap("copper_r"))
    # cmap = mplcolors.ListedColormap(cmap(np.linspace(0, 0.9, 256)))
    cmap.set_under("gainsboro", 1.0)

    # Load network
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index='0')
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    selected_co2ls = np.array([0.0, 0.2, 0.6])

    # Load edge likelihoods
    edge_likelihoods = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"/edge_likelihoods_total_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )

    # Setup figure layout based on colorbar option
    if unified_colorbar:
        if in_second_row is None:
            # Single colorbar on the right
            f = plt.figure(figsize=(16, 4))
            gs_main = GridSpec(
                1,
                2,
                figure=f,
                width_ratios=[1, 0.02],
                wspace=0.0,
            )
            gs_plots = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=-0.15)
            axs_secondary = [f.add_subplot(gs_plots[i]) for i in range(3)]
            # Create colorbar subplot with vertical padding to reduce its height
            gs_colorbar = GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs_main[1], height_ratios=[0.2, 1, 0.2], hspace=0
            )
            ax_colorbar_unified = f.add_subplot(gs_colorbar[1])
            axs_colorbars_secondary = None
            
        else:
            # Single colorbar on the right
            f = plt.figure(figsize=(16, 8))
            gs_main = GridSpec(
                2,
                2,
                figure=f,
                width_ratios=[1, 0.02],
                wspace=0.0,
            )
            gs_top_plots = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=-0.15)
            
            axs_secondary = [f.add_subplot(gs_top_plots[ii]) for ii in range(3)]
            if in_second_row == "line_loading":
                gs_bottom_plot = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[2])
                
            elif in_second_row == "blackout_sizes":
                gs_bottom_plot = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[2])
                axs_bottom = f.add_subplot(gs_bottom_plot[0])
                #axs_bottom = [f.add_subplot(gs_bottom_plot[0])]
                #axs_bottom.extend([f.add_subplot(gs_bottom_plot[ii], sharey=axs_bottom[0]) for ii in range(1, 3)])
            #axs_bottom.extend([f.add_subplot(gs_bottom_plot[ii+1], sharey=axs_bottom[0]) for ii in range(2)])
            # Create colorbar subplot with vertical padding to reduce its height
            gs_colorbar = GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs_main[1], height_ratios=[0.2, 1, 0.2], hspace=0
            )
            ax_colorbar_unified = f.add_subplot(gs_colorbar[1])
            axs_colorbars_secondary = None
            
    else:
        # Individual colorbars for each subplot
        f = plt.figure(figsize=(24, 5))
        gs_main = GridSpec(
            1,
            3,
            figure=f,
            width_ratios=[1, 1, 1],
            wspace=0.05,
        )

        # Create subplots with space for individual colorbars
        axs_secondary = []
        axs_colorbars_secondary = []

        for i in range(3):
            # Create subplot with colorbar space
            gs_sub_secondary = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs_main[0, i], width_ratios=[1, 0.05], wspace=-0.05
            )

            axs_secondary.append(f.add_subplot(gs_sub_secondary[0]))
            axs_colorbars_secondary.append(f.add_subplot(gs_sub_secondary[1]))

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Calculate global vmin and vmax if using unified colorbar
    if unified_colorbar:
        all_valid_probs = []
        for idx_co2, co2l in enumerate(selected_co2ls):
            level = np.round(co2l, 2)
            c_H_s = [edge_likelihoods[level][(u, v)] for u, v in nx_graph.edges()]
            c_H_s_array = np.array(c_H_s)
            valid_probs = c_H_s_array[c_H_s_array > 1e-12]
            all_valid_probs.extend(valid_probs)

        if len(all_valid_probs) > 0:
            vmin_global = np.min(all_valid_probs)
            vmax_global = np.max(all_valid_probs)
        else:
            vmin_global = 1e-6
            vmax_global = 1e-3
        vmin_global = 0
        print(
            f"Global colorscale: Min val: {vmin_global:.2e}, Max val: {vmax_global:.2e}"
        )
        
    cmap_co2 = plt.get_cmap("cividis").copy()
    for count, co2l in tqdm(enumerate(selected_co2ls[::-1]), total=len(selected_co2ls)):
        level = np.round(co2l, 2)

        # Load likelihoods as dictionary and transform into array
        c_H_s = [edge_likelihoods[level][(u, v)] for u, v in nx_graph.edges()]
        c_H_s_array = np.array(c_H_s)

        # Calculate vmin and vmax based on colorbar option
        if unified_colorbar:
            vmin_individual = vmin_global
            vmax_individual = vmax_global
        else:
            # Calculate individual vmin and vmax for this subplot
            valid_probs = c_H_s_array[c_H_s_array > 1e-12]
            if len(valid_probs) > 0:
                vmin_individual = np.min(valid_probs)
                vmax_individual = np.max(valid_probs)
            else:
                vmin_individual = 1e-6
                vmax_individual = 1e-3

            print(
                f"CO2 level {co2l}: Min val sec: {vmin_individual:.2e}, Max val sec: {vmax_individual:.2e}"
            )

        nodes = nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=axs_secondary[count], node_color="black", node_size=0
        )

        # Calculate edge widths
        if scale_width:
            if width_scale_sqrt:
                # Sqrt scaling
                widths = np.array(c_H_s)
                # Normalize to a reasonable range (e.g., 0.5 to 5)
                widths = 0.5 + 4.5 * np.sqrt(
                    (widths - widths.min()) / (widths.max() - widths.min())
                    if widths.max() > widths.min()
                    else widths * 0
                )
                edge_widths = widths
            else:
                # Linear scaling
                widths = np.array(c_H_s)
                # Normalize to a reasonable range (e.g., 0.5 to 5)
                widths = 0.5 + 4.5 * (
                    (widths - widths.min()) / (widths.max() - widths.min())
                    if widths.max() > widths.min()
                    else widths * 0
                )
            edge_widths = widths
        else:
            edge_widths = 3.5

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=axs_secondary[count],
            edge_color=c_H_s,
            width=edge_widths,
            edge_cmap=cmap,
            edge_vmin=vmin_individual,
            edge_vmax=vmax_individual,
        )

        axs_secondary[count].axis("off")
        axs_secondary[count].set_aspect("equal")

        # Add individual colorbar if not using unified colorbar
        if not unified_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=mplcolors.Normalize(vmin=vmin_individual, vmax=vmax_individual),
            )
            cb = plt.colorbar(sm, cax=axs_colorbars_secondary[count])
            setup_colormap_scientific_notation(cb)
            
            # Add title to all colorbars
            axs_colorbars_secondary[count].set_title(
                r"$\langle p_{\ell}\rangle$",
                fontsize=COLORBAR_LABEL_FONTSIZE,
                weight="bold",
                verticalalignment="center",
                pad=15,
            )

        # Add subplot titles and labels
        actual_co2_level = get_actual_co2_level(co2l, n_nodes, percent=True)
        axs_secondary[count].set_title(
            r"CO$_2 =$ " + r"{}\%".format(int(actual_co2_level)),
            fontsize=SUBTITLE_FONTSIZE,
            y=0.93,
        )
        add_panel_label(axs_secondary[count], count, x_offset=0.1, y_offset=-0.015)
        
        # Plot second row if requested
        if in_second_row is not None:
            
            color_r = cmap_co2(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1))
            # line loading
            if in_second_row == "line_loading":
                n_bins_line_loading = 100
                line_loading_bins = np.linspace(0.5, 1., 
                                                n_bins_line_loading, endpoint=True)
                diff_bins = line_loading_bins[1] - line_loading_bins[0]
                
                # TODO move this data to plot_data folder and mkdir if neccesary
                # Evaluate lineloadings for each snap_shot
                fname_ll = f"/line_loadings_for_snapshots_Co2L{co2l:.2f}_n{n_nodes}"
                if second_row_condition_on_split:
                    fname_ll += "_only_splits"
                fpath_to_line_loading = path_to_figures_sclopf + fname_ll + ".pklz"
                if not os.path.exists(fpath_to_line_loading) or calc_line_loading_gain:
                    flow_dict, line_loading_dict = calculate_line_loadings_for_all_snapshots(co2lvl=co2l, n_nodes=n_nodes,
                                                                                             condition_on_split=True,
                                                                                             verbose=False)
                    with gzip.open(fpath_to_line_loading, "wb") as fh_ll_out:
                        pickle.dump((flow_dict, line_loading_dict), fh_ll_out)
                    logger.info(f"Calc. LineLoadings Co2L={co2l} and saved")
                else:
                    with gzip.open(fpath_to_line_loading, "rb") as fh_ll_in:
                        flow_dict, line_loading_dict = pickle.load(fh_ll_in)
                        
                # Concat all line loadings
                all_line_loadings = [item for sublist in line_loading_dict.values() for item in sublist]
                
                axs_bottom.hist(all_line_loadings, bins=line_loading_bins,
                                histtype='step', density=True,
                                color=color_r,
                                linewidth=3., label=f"{co2l * 100}\\%")
                
                
            elif in_second_row == "blackout_sizes":
                
                path_to_split_props = path_to_vis_results_sclopf +  f"/split_properties_all_n{n_nodes}.h5"
                split_props = pd.read_hdf(path_to_split_props, index_col=0)
                
                plot_split_statistics.ax_blackout_size_histogram(axs_bottom,
                                               split_props, co2l, yscale_log=True,
                                               xscale_log=True, use_total_lost_load=True,
                                               color=color_r,
                                               label_str=f"{co2l * 100}\\%")
            
            else:
                raise ValueError("")

    # Add unified colorbar if requested
    if unified_colorbar:
        if powernorm:
            norm = mplcolors.PowerNorm(gamma=0.6, vmin=vmin_global, vmax=vmax_global)
        else:
            norm = mplcolors.Normalize(vmin=vmin_global, vmax=vmax_global)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm,
        )
        cb = plt.colorbar(sm, cax=ax_colorbar_unified)
        setup_colormap_scientific_notation(cb)
        ax_colorbar_unified.set_title(
            r"$\boldsymbol{\langle p_{\ell}\rangle}$",
            fontsize=COLORBAR_LABEL_FONTSIZE,
            verticalalignment="center",
            pad=15,
        )

    if in_second_row == "line_loading":
        axs_bottom.set_xlim(left=0.5, right=1.)
        axs_bottom.set_yscale('log')
        
        axs_bottom.legend(loc="upper right", title="CO$_2$ level [\\% of 1990]")
        
        axs_bottom.set_xlabel("Line Loading $\\ell_{\\text{load}}$")
        if second_row_condition_on_split:
            axs_bottom.set_ylabel("$P(\\ell_{\\text{load}} \\vert {\\text{split}})$")
        else:
            axs_bottom.set_ylabel("$P(\\ell_{\\text{load}})$")
            
        add_panel_label(axs_bottom, 3)
        
    elif in_second_row == "blackout_sizes":
        axs_bottom.set_xlim(left=5)
        
        axs_bottom.set_ylabel("Count", size=24)
        axs_bottom.set_xlabel("Load not served [MW]", size=24)
        
        axs_bottom.legend(loc="upper left", 
                          title="CO$_2$ level [\\% of 1990]",
                          fontsize=18, title_fontsize=18)
        
        axs_bottom.tick_params(axis="both", labelsize=20)
        
        add_panel_label(axs_bottom, 3)
    
    cmap_name = cmap if isinstance(cmap, str) else cmap.name
    suffix = "_unified" if unified_colorbar else ""
    
    if powernorm:
        suffix += "_powernorm"
        
    if scale_width:
        suffix += "_sqrt_width" if width_scale_sqrt else "_linear_width"
    # save_figure(f, save_path, "line_failure_probs_total")
    
    if in_second_row is not None:
        suffix += "_" + in_second_row
    
    if save_fig:
        save_figure(f, f"line_failure_probs_total_{cmap_name}{suffix}", save_path)
    else:
        plt.show()


def create_cross_border_vulnerability_plots(
    top_n=50,
    percentiles=np.arange(1, 21),
    selected_co2ls=np.array([0.0, 0.2, 0.6]),
):
    """Plot cross-border vs same-country distribution among vulnerable lines.

    Creates a ranked bar chart (top-N lines) and a share-vs-percentile plot.
    """

    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    edges = list(nx_graph.edges())

    bus_country_map = _get_bus_country_map(network)
    cross_border_flags = np.array(
        [_edge_is_cross_border(u, v, bus_country_map) for u, v in edges]
    )
    overall_cross_share = (
        cross_border_flags.sum() / len(cross_border_flags)
        if len(cross_border_flags) > 0
        else 0.0
    )

    edge_likelihoods = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"/edge_likelihoods_total_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )

    setup_matplotlib_style()
    co2ls_ordered = list(selected_co2ls[::-1])

    fig_bar, axes_bar = plt.subplots(
        1,
        len(co2ls_ordered),
        figsize=(6 * len(co2ls_ordered), 4.2),
        sharey=True,
    )
    if len(co2ls_ordered) == 1:
        axes_bar = np.array([axes_bar])

    fig_share, axes_share = plt.subplots(
        1,
        len(co2ls_ordered),
        figsize=(6 * len(co2ls_ordered), 4.2),
        sharey=True,
    )
    if len(co2ls_ordered) == 1:
        axes_share = np.array([axes_share])

    for col, co2l in enumerate(co2ls_ordered):
        level = np.round(co2l, 2)
        probs = np.array([edge_likelihoods[level][(u, v)] for u, v in edges])
        order = np.argsort(probs)[::-1]
        top_n_eff = min(top_n, len(order))
        top_idx = order[:top_n_eff]

        top_probs = probs[top_idx]
        top_cross = cross_border_flags[top_idx]

        bar_colors = [
            PRIMARY_COLORS["red"] if is_cross else PRIMARY_COLORS["blue"]
            for is_cross in top_cross
        ]
        ax_bar = axes_bar[col]
        ax_bar.bar(np.arange(top_n_eff), top_probs, color=bar_colors, width=0.9)
        ax_bar.set_yscale("log")
        ax_bar.set_title(
            rf"CO$_2$ = {int(get_actual_co2_level(co2l, n_nodes, percent=True))}\%",
            fontsize=SUBTITLE_FONTSIZE,
        )
        ax_bar.set_xlabel("Rank", fontsize=AXIS_LABEL_FONTSIZE)
        if col == 0:
            ax_bar.set_ylabel("Failure probability", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bar.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax_bar.grid(True, axis="y", alpha=0.3)

        ax_share = axes_share[col]
        shares = []
        for p in percentiles:
            k = max(1, int(np.ceil(len(order) * p / 100)))
            k = min(k, len(order))
            share = cross_border_flags[order[:k]].sum() / k
            shares.append(share * 100)

        ax_share.plot(percentiles, shares, color=PRIMARY_COLORS["purple"], lw=2)
        ax_share.axhline(
            overall_cross_share * 100,
            color=PRIMARY_COLORS["gray"],
            ls="--",
            lw=1.5,
            label="overall share",
        )
        ax_share.set_title(
            rf"CO$_2$ = {int(get_actual_co2_level(co2l, n_nodes, percent=True))}\%",
            fontsize=SUBTITLE_FONTSIZE,
        )
        ax_share.set_xlabel(
            "Top percentile of lines [\%]", fontsize=AXIS_LABEL_FONTSIZE
        )
        if col == 0:
            ax_share.set_ylabel("Cross-border share [\%]", fontsize=AXIS_LABEL_FONTSIZE)
        ax_share.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax_share.grid(True, axis="y", alpha=0.3)
        if col == 0:
            ax_share.legend(
                fontsize=TICK_LABEL_FONTSIZE, frameon=True, loc="upper right"
            )

    legend_handles = [
        plt.Line2D([0], [0], color=PRIMARY_COLORS["blue"], lw=6, label="same-country"),
        plt.Line2D([0], [0], color=PRIMARY_COLORS["red"], lw=6, label="cross-border"),
    ]
    axes_bar[0].legend(
        handles=legend_handles,
        fontsize=TICK_LABEL_FONTSIZE,
        frameon=True,
        loc="upper right",
    )

    fig_bar.tight_layout()
    save_figure(fig_bar, "cross_border_vulnerability_ranked", save_path)
    fig_share.tight_layout()
    save_figure(fig_share, "cross_border_vulnerability_share", save_path)
    plt.show()


if __name__ == "__main__":
    # create_line_failure_plot_linear()
    # create_secondary_line_failure_plot()

    save_csv = True
    cmap = "crameri:Batlow_r"
    powernorm = False
    scale_width = True
    width_scale_sqrt = True

    if save_csv:
        n_nodes = 600
        save_path = path_to_figures_sclopf
        network = data_handling.load_pypsa_network_from_path(
            path_to_pypsa_network_sclopf
            + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
            True,
        )
        nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
        selected_co2ls = np.array([0.0, 0.2, 0.6])
        edge_likelihoods_total = pickle.load(
            open(
                path_to_vis_results_sclopf
                + f"/edge_likelihoods_total_all_co2ls_n{n_nodes}.pickle",
                "rb",
            )
        )
        save_failure_probabilities_to_csv(
            edge_likelihoods_total,
            nx_graph,
            selected_co2ls,
            n_nodes,
            save_path,
            "line_failure_probs_total.csv",
        )

    create_total_line_failure_plot(
        unified_colorbar=True,
        powernorm=powernorm,
        scale_width=scale_width,
        width_scale_sqrt=width_scale_sqrt,
        cmap=cmap,
    )
    create_cross_border_vulnerability_plots()
