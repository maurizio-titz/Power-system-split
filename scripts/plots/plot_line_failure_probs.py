#!/usr/bin/env python3
"""
Line Failure Probabilities Plot - Figure 7: line failures
Visualizes primary and secondary line failure probabilities across CO2 levels.
"""

import sys
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

sys.path.append("./")

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling
from utils.plot_style import (
    setup_matplotlib_style,
    PANEL_LABEL_FONTSIZE,
    COLORBAR_LABEL_FONTSIZE,
    COLORBAR_TICK_FONTSIZE,
    SUBTITLE_FONTSIZE,
    add_panel_label,
    setup_colormap_scientific_notation,
    save_figure,
)


# def create_line_failure_plot():
#     """Create line failure probabilities plot."""

#     # Setup
#     n_nodes = 600
#     save_path = path_to_figures_sclopf
#     os.makedirs(save_path, exist_ok=True)

#     # Load network
#     network = data_handling.load_pypsa_network_from_path(
#         path_to_pypsa_network_sclopf
#         + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
#         True,
#     )
#     nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
#     pos = nx.get_node_attributes(nx_graph, "pos")

#     # Get CO2 levels
#     co2ls = get_co2_levels(n_nodes)
#     selected_co2ls = np.array([0.0, 0.2, 0.6])

#     # Load edge likelihoods
#     edge_likelihoods_primary = pickle.load(
#         open(
#             path_to_vis_results_sclopf
#             + f"edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle",
#             "rb",
#         )
#     )
#     edge_likelihoods_secondary = pickle.load(
#         open(
#             path_to_vis_results_sclopf
#             + f"edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle",
#             "rb",
#         )
#     )

#     # Setup figure
#     f = plt.figure(figsize=(21, 10))
#     gs_vertical = GridSpec(1, 2, figure=f, width_ratios=[3, 0.05], wspace=-0.02)
#     gs_maps = GridSpecFromSubplotSpec(
#         2, 3, subplot_spec=gs_vertical[0], wspace=-0.1, hspace=-0.1
#     )
#     axs_primary = [f.add_subplot(gs_maps[0, i]) for i in range(3)]
#     axs_secondary = [f.add_subplot(gs_maps[1, i]) for i in range(3)]
#     gs_colorbars = GridSpecFromSubplotSpec(
#         5, 1, subplot_spec=gs_vertical[1], height_ratios=[0.2, 1, 0.2, 1, 0.2]
#     )
#     axs_colorbars = [f.add_subplot(gs_colorbars[i]) for i in [1, 3]]

#     mpl.style.use("default")
#     plt.rc("text", usetex=True)
#     plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

#     labels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}"]
#     vmaxvals = []
#     vminvals = []

#     # Primary failures
#     cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
#     cmap.set_under("gainsboro", 1.0)

#     for count, co2l in enumerate(selected_co2ls[::-1]):
#         level = np.round(co2l, 2)

#         # Load likelihoods as dictionary and transform into array
#         c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
#         c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

#         print(
#             "Min val prim: {:e}".format(
#                 np.amin(np.array(c_H_p)[np.array(c_H_p) > 1e-12])
#             )
#         )
#         print(
#             "Max val prim: {:e}".format(
#                 np.amax(np.array(c_H_p)[np.array(c_H_p) > 1e-12])
#             )
#         )

#         vmax = 1e-3
#         vmin = 1e-6

#         nodes = nx.draw_networkx_nodes(
#             nx_graph, pos=pos, ax=axs_primary[count], node_color="black", node_size=0
#         )

#         edges = nx.draw_networkx_edges(
#             nx_graph,
#             pos=pos,
#             ax=axs_primary[count],
#             edge_color=c_H_p_log,
#             width=3.5,
#             edge_cmap=cmap,
#             edge_vmin=np.log10(vmin),
#             edge_vmax=np.log10(vmax),
#         )

#         axs_primary[count].axis("off")

#     # Primary failures colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
#     cb = f.colorbar(sm, cax=axs_colorbars[0])
#     cb.ax.tick_params(labelsize=26, width=1.0, which="both")
#     axs_colorbars[0].set_title(
#         r"$\langle p_{\ell}^{\text{p}}\rangle$",
#         fontsize=25,
#         weight="bold",
#         verticalalignment="center",
#         pad=20,
#     )

#     # Secondary failures
#     cmap = copy.copy(mpl.cm.get_cmap("viridis_r"))
#     cmap.set_under("gainsboro", 1.0)

#     for count, co2l in enumerate(selected_co2ls[::-1]):
#         level = np.round(co2l, 2)

#         # Load likelihoods as dictionary and transform into array
#         c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
#         c_H_s_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_s])

#         print(
#             "Min val sec: {:e}".format(
#                 np.amin(np.array(c_H_s)[np.array(c_H_s) > 1e-12])
#             )
#         )
#         print(
#             "Max val sec: {:e}".format(
#                 np.amax(np.array(c_H_s)[np.array(c_H_s) > 1e-12])
#             )
#         )

#         vmax = 1e-3
#         vmin = 1e-5
#         vmaxvals.append(vmax)
#         vminvals.append(vmin)

#         nodes = nx.draw_networkx_nodes(
#             nx_graph, pos=pos, ax=axs_secondary[count], node_color="black", node_size=0
#         )

#         edges = nx.draw_networkx_edges(
#             nx_graph,
#             pos=pos,
#             ax=axs_secondary[count],
#             edge_color=c_H_s_log,
#             width=3.5,
#             edge_cmap=cmap,
#             edge_vmin=np.log10(vmin),
#             edge_vmax=np.log10(vmax),
#         )

#         axs_secondary[count].axis("off")

#         axs_primary[count].set_title(
#             r"CO$_2 =$ " + "{} \%".format(int(round(co2l * 100))), fontsize=28
#         )
#         axs_primary[count].text(
#             0 + 0.1,
#             1 + 0.05,
#             labels[count],
#             fontsize=30,
#             weight="bold",
#             verticalalignment="center",
#             transform=axs_primary[count].transAxes,
#         )

#     # Secondary failures colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
#     cb = f.colorbar(sm, cax=axs_colorbars[1])
#     cb.set_ticks([1e-5, 1e-4, 1e-3], labels=[r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])
#     cb.ax.tick_params(labelsize=26, width=1.0, which="major")
#     axs_colorbars[1].set_title(
#         r"$\langle p_{\ell}^{\text{s}}\rangle$",
#         fontsize=25,
#         weight="bold",
#         verticalalignment="center",
#         pad=20,
#     )

#     print(np.max(vmaxvals), np.min(vminvals))

#     plt.savefig(save_path + "line_failure_probs.pdf", bbox_inches="tight")
#     plt.show()


def create_line_failure_plot_linear():
    """Create line failure probabilities plot with linear colorscale."""

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
    edge_likelihoods_primary = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle",
            "rb",
        )
    )
    edge_likelihoods_secondary = pickle.load(
        open(
            path_to_vis_results_sclopf
            + f"edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle",
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

    save_figure(f, save_path, "line_failure_probs_linear")
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

    save_figure(f, save_path, "line_failure_probs_secondary_only")
    plt.show()


if __name__ == "__main__":
    # create_line_failure_plot_linear()
    create_secondary_line_failure_plot()
