#!/usr/bin/env python3
"""
Line Extension Mitigation Plot
Shows line extension costs and mitigation effects for reducing system splits.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.append("./")

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
    path_to_line_extension_mitigation_sclopf,
    path_to_pre_outage_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.cascade_simulation import LOOKUP_TABLE_NP
from utils.plot_mitigation_strategies import (
    plot_map_inertia_placement_final,
    calc_inertia_placement_ref_loss,
    color1,
    color2,
)
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    SUBTITLE_FONTSIZE, 
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure
)

# Backwards compatibility
AXIS_LABELSIZE = AXIS_LABEL_FONTSIZE
TICK_LABELSIZE = TICK_LABEL_FONTSIZE
SUBLABEL_FONTSIZE = PANEL_LABEL_FONTSIZE


def create_combined_mitigation_plot(use_annualized_costs=False):
    """Create combined mitigation plot with inertia on top and line extension below."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(path_to_line_extension_mitigation_sclopf, exist_ok=True)

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
    n_lines = nx_graph.number_of_edges()

    # Get CO2 levels and split properties
    co2ls = get_co2_levels(n_nodes)
    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )

    # Reference levels and setup
    co2l_ref = 0.6
    co2_lvl_map = 0.1  # Level for detailed maps
    ref_loss_factors = [1]  # Only use ref_loss_factor=1

    # Setup matplotlib with consistent styling
    setup_matplotlib_style()

    # === INERTIA MITIGATION DATA PREPARATION ===
    inertia_time = (
        np.load(
            path_to_pre_outage_sclopf + f"inertia_time_series_all_co2ls_{n_nodes}.npy"
        )
        / 1000
    )

    # Calculate inertia needed by level
    inertia_needed_by_lvl = {}
    for ref_loss_factor in ref_loss_factors:
        inertia_at_ref_loss_by_lvl = calc_inertia_placement_ref_loss(
            n_nodes=600,
            delta_Erot=1000,
            co2_lvl_ref=co2l_ref,
            split_properties=split_properties,
            ref_loss_factor=ref_loss_factor,
            co2_lvls=co2ls,
        )
        inertia_needed_by_lvl[ref_loss_factor] = inertia_at_ref_loss_by_lvl

    # === LINE EXTENSION MITIGATION DATA PREPARATION ===
    # Calculate parallel line extensions lookup
    table_rounded = np.round(LOOKUP_TABLE_NP, 5)
    num_par_paths = {}
    for tup in table_rounded:
        current_tup = tup
        num_par_paths[tup[0]] = [current_tup[0] - current_tup[1]]
        while current_tup[1] != 0:
            current_tup = table_rounded[
                np.where((table_rounded[:, 0] == current_tup[1]))[0][0]
            ]
            num_par_paths[tup[0]].append(current_tup[0] - current_tup[1])
    max_num_par = {k: max(v) for k, v in num_par_paths.items()}

    split_properties_reference = split_properties[
        split_properties.co2l == co2l_ref
    ].copy()
    lost_load_reference_lvl = split_properties_reference.lost_load_share_blackout.sum()

    # Calculate costs for line extensions

    cost_to_reach_ref_loss = {}
    lines_to_reach_ref_loss = {}  # Store number of lines needed

    for co2l in co2ls:
        if co2l == 0.6:
            continue

        if use_annualized_costs:
            f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2l}_n{n_nodes}.pkl"
        else:
            f_name = f"heuristic_costMin_loss_mitigation_Co2L{co2l}_n{n_nodes}.pkl"

        try:
            reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(
                open(path_to_line_extension_mitigation_sclopf + f_name, "rb")
            )
        except FileNotFoundError as e:
            # Calculate mitigation if file doesn't exist (simplified version)
            cost_to_reach_ref_loss[co2l] = 0
            lines_to_reach_ref_loss[co2l] = 0
            raise e

        # Find number of lines needed
        num_lines_to_reach_ref_loss = np.where(
            np.array(loss_with_mitigation) < lost_load_reference_lvl
        )[0][0]

        cost_to_reach_ref_loss[co2l] = sum(cost[:num_lines_to_reach_ref_loss])
        lines_to_reach_ref_loss[co2l] = num_lines_to_reach_ref_loss

    # === CREATE COMBINED FIGURE ===
    f = plt.figure(figsize=(16, 12))

    # Main grid: inertia on top, line extension below
    gs_main = GridSpec(2, 1, figure=f, height_ratios=[1, 1], hspace=0.35)

    # === INERTIA MITIGATION SECTION (TOP) ===
    gs_inertia = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.15)
    ax_inertia_loss = f.add_subplot(gs_inertia[0])  # Loss reduction curve
    ax_inertia_map = f.add_subplot(gs_inertia[1])  # Map
    ax_inertia_all = f.add_subplot(gs_inertia[2])  # All CO2 levels

    # === LINE EXTENSION SECTION (BOTTOM) ===
    gs_line = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.15)
    ax_line_loss = f.add_subplot(gs_line[0])  # Loss reduction curve
    ax_line_map = f.add_subplot(gs_line[1])  # Map
    ax_line_all = f.add_subplot(gs_line[2])  # All CO2 levels

    # === PLOT INERTIA MITIGATION ===
    unit = "GWs"
    unit_factor = 1e-3
    # Use consistent colors across both mitigation types
    color_reference_loss = color1  # Same color for both loss reduction curves
    color_all_levels = color2  # Same color for both all-levels curves
    color_loss_curve = "black"

    # Inertia: All CO2 levels plot
    for i, ref_loss_factor in enumerate(ref_loss_factors):
        inertia_at_ref_loss_by_lvl = inertia_needed_by_lvl[ref_loss_factor]
        label = f"$R_{{\\textrm{{ref}}}}$"  # Simplified label for ref_loss_factor=1

        ax_inertia_all.plot(
            np.array(list(inertia_at_ref_loss_by_lvl.keys())) * 100,
            np.array(list(inertia_at_ref_loss_by_lvl.values())) * unit_factor,
            label=label,
            color=color_reference_loss,
            linewidth=2,
        )

    ax_inertia_all.invert_xaxis()
    ax_inertia_all.set_ylabel(f"Inertia placed [{unit}]", fontsize=AXIS_LABELSIZE)
    ax_inertia_all.set_xlabel("CO$_2$ level [\\% of 1990]", fontsize=AXIS_LABELSIZE)
    ax_inertia_all.set_title("Synthetic inertia needed", fontsize=TITLE_FONTSIZE)
    ax_inertia_all.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax_inertia_all.grid(True, alpha=0.3)

    # Inertia: Map and loss reduction
    for i, ref_loss_factor in enumerate(ref_loss_factors):
        plot_curve = i == 0
        plot_map_inertia_placement_final(
            axes=(ax_inertia_map, ax_inertia_loss),
            co2_lvl=co2_lvl_map,
            nn=600,
            max_iter=10000,
            max_node_size=100,
            edge_width=0.2,
            delta_Erot=1000,
            rocof_thres=-1,
            l_share=0.0,
            resolve_strategy="random",
            show_step_number=False,
            plot_split_number=False,
            save_fig=False,
            co2_lvl_ref=co2l_ref,
            ref_loss_factor=ref_loss_factor,
            unit=unit,
            color=color_reference_loss,
            plot_curve=plot_curve,
            split_properties=split_properties,
            line_color=color_loss_curve,
        )

    # Set consistent styling for inertia plots
    title = f"Synthetic inertia mitigation \n CO$_2$ level={int(get_actual_co2_level(co2_lvl_map)*100)}\\%"
    ax_inertia_loss.set_title(
        title,
        fontsize=TITLE_FONTSIZE,
    )
    ax_inertia_loss.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax_inertia_loss.set_xlabel("Inertia placed [GWs]", fontsize=AXIS_LABELSIZE)

    co2_ref_percent = float(np.atleast_1d(get_actual_co2_level(co2l_ref)).item()) * 100
    loss_axis_label = (
        rf"Loss of load [$1/\textrm{{R}}_{{{int(round(co2_ref_percent))}\%}}$]"
    )
    ax_inertia_loss.set_ylabel(loss_axis_label, fontsize=AXIS_LABELSIZE)
    ax_inertia_loss.grid(True, alpha=0.3)

    ax_inertia_map.set_title(
        rf"Synthetic inertia to reach $\textrm{{R}}_{{{int(round(co2_ref_percent))}\%}}$"
        + "\n"
        + rf"CO$_2$ level={int(get_actual_co2_level(co2_lvl_map)*100)}\%",
        fontsize=TITLE_FONTSIZE,
    )

    # === PLOT LINE EXTENSION MITIGATION ===
    # Load specific data for the map level
    if use_annualized_costs:
        f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2_lvl_map}_n{n_nodes}.pkl"
    else:
        f_name = f"heuristic_costMin_loss_mitigation_Co2L{co2_lvl_map}_n{n_nodes}.pkl"

    try:
        reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(
            open(path_to_line_extension_mitigation_sclopf + f_name, "rb")
        )

        num_lines_to_reach_ref_loss = np.where(
            np.array(loss_with_mitigation) < lost_load_reference_lvl
        )[0][0]
        cost_to_reach_ref_loss_lvl = sum(cost[:num_lines_to_reach_ref_loss])

        # Line extension: Loss reduction curve
        ax_line_loss.plot(
            np.arange(len(loss_with_mitigation)),
            np.array(loss_with_mitigation) / lost_load_reference_lvl,
            color=color_loss_curve,
            linewidth=2,
            label="Loss reduction",
        )

        right_xlim = np.where(
            np.array(loss_with_mitigation) / lost_load_reference_lvl < 0.5
        )[0][0]

        # Add second y axis for costs
        ax2_line_cost = ax_line_loss.twinx()
        ax2_line_cost.plot(
            np.arange(len(cost)),
            np.cumsum(cost) / 1e9,
            color=color_loss_curve,
            linestyle="dotted",
            linewidth=2,
            label="Cost",
        )
        if use_annualized_costs:
            cost_label = "Annualized cost [billion €]"
        else:
            cost_label = "Cost [billion €]"
        ax2_line_cost.set_ylabel(cost_label, fontsize=AXIS_LABELSIZE)
        ax2_line_cost.set_ylim((0, 1.1 * sum(cost[:right_xlim]) / 1e9))
        ax2_line_cost.tick_params(axis="y", which="major", labelsize=TICK_LABELSIZE)

        # Add reference line for line number
        ax_line_loss.plot(
            [0, num_lines_to_reach_ref_loss],
            [1, 1],
            "--",
            c=color_reference_loss,
        )
        ax_line_loss.plot(
            [num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss],
            [0, 1],
            "--",
            c=color_reference_loss,
        )
        ax_line_loss.text(
            num_lines_to_reach_ref_loss + right_xlim / 200 * 5,
            1,
            f"{num_lines_to_reach_ref_loss} lines",
            verticalalignment="bottom",
            horizontalalignment="left",
            zorder=np.inf,
            fontsize=TICK_LABELSIZE,
        )

        # Add reference line for cost
        ax2_line_cost.plot(
            [num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss],
            [0, cost_to_reach_ref_loss_lvl / 1e9],
            "--",
            c=color_reference_loss,
        )
        ax2_line_cost.text(
            num_lines_to_reach_ref_loss - right_xlim / 200 * 5,
            cost_to_reach_ref_loss_lvl / 1e9,
            f"{cost_to_reach_ref_loss_lvl/ 1e9:.1f} billion €",
            verticalalignment="bottom",
            horizontalalignment="right",
            zorder=np.inf,
            fontsize=TICK_LABELSIZE,
        )

        ax_line_loss.set_xlim(0, right_xlim)
        ax_line_loss.set_ylim(bottom=0)
        ax_line_loss.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax_line_loss.set_xlabel("Number of reinforced lines", fontsize=AXIS_LABELSIZE)
        ax_line_loss.set_ylabel(loss_axis_label, fontsize=AXIS_LABELSIZE)
        title = f"Grid extension mitigation \n CO$_2$ level={int(get_actual_co2_level(co2_lvl_map)*100)}\\%"
        ax_line_loss.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
        )
        ax_line_loss.grid(True, alpha=0.3)

        # Create shared legend for both axes
        lines1, labels1 = ax_line_loss.get_legend_handles_labels()
        lines2, labels2 = ax2_line_cost.get_legend_handles_labels()
        ax_line_loss.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            fontsize=LEGEND_FONTSIZE,
        )

        # Line extension: Map
        pos = nx.get_node_attributes(nx_graph, "pos")
        reinforced_lines_selected = reinforced_lines[:num_lines_to_reach_ref_loss]
        lines_not_extended = np.setdiff1d(np.arange(n_lines), reinforced_lines_selected)
        mitigated_loss = np.array(
            [
                loss_with_mitigation[i] - loss_with_mitigation[i + 1]
                for i in range(len(loss_with_mitigation) - 1)
            ]
        )
        mitigated_loss = np.concatenate(
            (mitigated_loss, -np.ones(n_lines - len(mitigated_loss)))
        )
        mitigated_loss_selected = -np.ones(n_lines)

        reinforced_lines_selected_int = np.array(reinforced_lines_selected).astype(int)
        mitigated_loss_selected[reinforced_lines_selected_int] = (
            mitigated_loss[:num_lines_to_reach_ref_loss] / lost_load_reference_lvl
        )

        width = 2 * (mitigated_loss_selected > 0).astype(int) + 1
        cmap = copy.copy(plt.cm.get_cmap("plasma_r"))
        cmap.set_under("gainsboro", 1.0)

        nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=ax_line_map, node_color="black", node_size=0
        )

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax_line_map,
            edge_color=mitigated_loss_selected,
            width=width,
            edge_cmap=cmap,
            edge_vmin=0,
        )

        # Add colorbar for line extension map
        bbox = ax_line_map.get_position()
        ax_line_map_legend = f.add_axes(
            (bbox.x0 + bbox.width * 0.1, bbox.y0 - 0.01, bbox.width * 0.8, 0.03)
        )
        y_label_line = "$\\Delta \\textrm{R}_{{\\ell}}/\\textrm{R}_{\\textrm{{co2_ref_str}}\\%}$".replace(
            "co2_ref_str", str(int(round(co2_ref_percent)))
        )
        cbar_line = plt.colorbar(
            edges,
            ax=ax_line_map,
            cax=ax_line_map_legend,
            label=y_label_line,
            shrink=0.5,
            orientation="horizontal",
        )
        cbar_line.ax.tick_params(labelsize=TICK_LABELSIZE)
        cbar_line.ax.set_xlabel(y_label_line, fontsize=AXIS_LABELSIZE)

        ax_line_map.axis("off")
        ax_line_map.set_title(
            rf"Grid extension to reach $\textrm{{R}}_{{{int(round(co2_ref_percent))}\%}}$"
            + "\n"
            + rf"CO$_2$ level={int(get_actual_co2_level(co2_lvl_map)*100)}\%",
            fontsize=TITLE_FONTSIZE,
        )

    except FileNotFoundError:
        ax_line_loss.text(
            0.5,
            0.5,
            "Data not available",
            ha="center",
            va="center",
            transform=ax_line_loss.transAxes,
        )
        ax_line_map.text(
            0.5,
            0.5,
            "Data not available",
            ha="center",
            va="center",
            transform=ax_line_map.transAxes,
        )

    # Line extension: All CO2 levels plot
    if cost_to_reach_ref_loss:
        cost_df = pd.DataFrame.from_dict(
            cost_to_reach_ref_loss, orient="index", columns=["cost_ref_loss"]
        )
        lines_df = pd.DataFrame.from_dict(
            lines_to_reach_ref_loss, orient="index", columns=["lines_needed"]
        )

        actual_co2_ref = get_actual_co2_level(co2l_ref, percent=True)

        # Plot number of lines as primary axis
        ax_line_all.plot(
            get_actual_co2_level(lines_df.index, percent=True),
            lines_df.lines_needed,
            label="Number of reinforced lines",
            color=color_reference_loss,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax_line_all.invert_xaxis()

        # Add secondary y-axis for cost
        ax_line_all_sec = ax_line_all.twinx()
        ax_line_all_sec.plot(
            get_actual_co2_level(cost_df.index, percent=True),
            cost_df.cost_ref_loss / 1e9,
            label=f"Cost",
            linestyle="dotted",
            color=color_reference_loss,
            linewidth=2,
        )

        ax_line_all.set_xlabel("CO$_2$ level [\\% of 1990]", fontsize=AXIS_LABELSIZE)
        ax_line_all.set_ylabel("Number of reinforced lines", fontsize=AXIS_LABELSIZE)
        if use_annualized_costs:
            cost_all_label = "Annualized cost [billion €]"
        else:
            cost_all_label = "Cost [billion €]"
        ax_line_all_sec.set_ylabel(cost_all_label, fontsize=AXIS_LABELSIZE)
        ax_line_all.invert_xaxis()
        ax_line_all_sec.invert_xaxis()
        ax_line_all.set_title("Grid extension needed", fontsize=TITLE_FONTSIZE)
        ax_line_all.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax_line_all_sec.tick_params(axis="y", which="major", labelsize=TICK_LABELSIZE)
        ax_line_all.grid(True, alpha=0.3)

        # Add combined legend
        lines1, labels1 = ax_line_all.get_legend_handles_labels()
        lines2, labels2 = ax_line_all_sec.get_legend_handles_labels()
        ax_line_all.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=LEGEND_FONTSIZE,
        )

    # === ADD SUBPLOT LABELS ===
    labels = [
        r"\textbf{A}",
        r"\textbf{B}",
        r"\textbf{C}",
        r"\textbf{D}",
        r"\textbf{E}",
        r"\textbf{F}",
    ]
    axes_all = [
        ax_inertia_loss,
        ax_inertia_map,
        ax_inertia_all,
        ax_line_loss,
        ax_line_map,
        ax_line_all,
    ]

    for ax, label in zip(axes_all, labels):
        ax.text(
            -0.15,
            1.05,
            label,
            fontsize=SUBLABEL_FONTSIZE,
            weight="bold",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    # === APPLY CONSISTENT STYLING TO ALL AXES ===
    # Apply consistent font styling to all text elements
    for ax in [ax_inertia_loss, ax_inertia_all, ax_line_loss, ax_line_all]:
        # Set consistent tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

        # Set consistent axis label sizes
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABELSIZE)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABELSIZE)

        # Set consistent title sizes
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)

        # Ensure grid styling is consistent
        ax.grid(True, alpha=0.3)

    # Apply consistent styling to map titles
    for ax in [ax_inertia_map, ax_line_map]:
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)

    plt.tight_layout()
    f_name = "combined_mitigation_plot.pdf"
    if use_annualized_costs:
        f_name = f_name.replace(".pdf", "_annualized.pdf")
    plt.savefig(save_path + f_name, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_combined_mitigation_plot(use_annualized_costs=False)
    create_combined_mitigation_plot(use_annualized_costs=True)
