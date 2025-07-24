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

# Add root path and import utilities
root_path = "../"
os.chdir(root_path)
sys.path.append(root_path)

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
    path_to_line_extension_mitigation_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.cascade_simulation import LOOKUP_TABLE_NP


def create_line_extension_mitigation_plot():
    """Create line extension mitigation plot."""

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

    # Reference level and costs
    co2l_ref = 0.6
    split_properties_reference = split_properties[
        split_properties.co2l == co2l_ref
    ].copy()
    lost_load_reference_lvl = split_properties_reference.lost_load_share_blackout.sum()
    extension_cost_per_MWkm = 445

    # Calculate costs for different scenarios
    for annualized_costs in [True, False]:

        cost_to_reach_ref_loss = {}
        cost_to_reach_double_ref_loss = {}

        for co2l in co2ls:
            if co2l == 0.6:
                continue

            try:
                if annualized_costs:
                    f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2l}_n{n_nodes}.pkl"
                else:
                    f_name = (
                        f"heuristic_costMin_loss_mitigation_Co2L{co2l}_n{n_nodes}.pkl"
                    )

                reinforced_lines, loss_with_mitigation, num_blackouts, cost = (
                    pickle.load(
                        open(path_to_line_extension_mitigation_sclopf + f_name, "rb")
                    )
                )

            except FileNotFoundError:
                # Calculate mitigation if file doesn't exist
                split_props_lvl = split_properties[split_properties.co2l == co2l].copy()
                split_props_lvl.lost_load_share_blackout = (
                    split_props_lvl.lost_load_share_blackout.astype(float)
                )
                remaining_splits = split_props_lvl[
                    split_props_lvl.lost_load_share_blackout > 0
                ]
                remaining_splits = remaining_splits.loc[
                    :, ["lost_load_share_blackout", "init_failure_0", "init_failure_1"]
                ]

                # Calculate costs
                costs_ = network.lines.loc[
                    :,
                    ["bus0", "bus1", "capital_cost", "s_nom", "num_parallel", "length"],
                ].copy()
                costs_["num_par_ext"] = costs_.num_parallel.apply(
                    lambda x: max_num_par[np.round(x, 5)] if x in max_num_par else 1
                )

                if annualized_costs:
                    costs_["extension_cost"] = (
                        costs_.capital_cost * costs_.s_nom * costs_.num_par_ext
                    )
                else:
                    costs_["extension_cost"] = (
                        extension_cost_per_MWkm
                        * costs_.s_nom
                        * costs_.num_par_ext
                        * costs_.length
                    )

                loss_with_mitigation = [split_props_lvl.lost_load_share_blackout.sum()]
                reinforced_lines = []
                cost = []
                num_blackouts = [remaining_splits.shape[0]]

                # Greedy optimization loop
                while remaining_splits.shape[0] > 0:
                    trigger0 = (
                        remaining_splits.loc[
                            :, ["lost_load_share_blackout", "init_failure_0"]
                        ]
                        .groupby("init_failure_0")
                        .sum()
                    )
                    trigger1 = (
                        remaining_splits.loc[
                            :, ["lost_load_share_blackout", "init_failure_1"]
                        ]
                        .groupby("init_failure_1")
                        .sum()
                    )
                    trigger1.rename_axis("trigger", inplace=True)
                    trigger0.rename_axis("trigger", inplace=True)
                    loss_by_trigger = pd.Series(index=list(range(n_lines)), data=0)
                    loss_by_trigger.rename_axis("trigger", inplace=True)
                    loss_by_trigger = loss_by_trigger.add(
                        trigger0.lost_load_share_blackout, fill_value=0
                    )
                    loss_by_trigger = loss_by_trigger.add(
                        trigger1.lost_load_share_blackout, fill_value=0
                    )
                    loss_per_dollar = loss_by_trigger / costs_[
                        "extension_cost"
                    ].reset_index(drop=True)

                    trigger = loss_per_dollar.idxmax()
                    cost.append(costs_["extension_cost"][int(trigger)])
                    remaining_splits = remaining_splits[
                        (remaining_splits.init_failure_1 != trigger)
                        & (remaining_splits.init_failure_0 != trigger)
                    ]
                    loss_with_mitigation.append(
                        remaining_splits.lost_load_share_blackout.sum()
                    )
                    reinforced_lines.append(trigger)
                    num_blackouts.append(remaining_splits.shape[0])

                with open(path_to_line_extension_mitigation_sclopf + f_name, "wb") as f:
                    pickle.dump(
                        (reinforced_lines, loss_with_mitigation, num_blackouts, cost), f
                    )

            # Find number of lines needed
            try:
                num_lines_to_reach_ref_loss = np.where(
                    np.array(loss_with_mitigation) < lost_load_reference_lvl
                )[0][0]
            except IndexError:
                num_lines_to_reach_ref_loss = len(cost)

            try:
                num_lines_to_reach_double_ref_loss = np.where(
                    np.array(loss_with_mitigation) < 2 * lost_load_reference_lvl
                )[0][0]
            except IndexError:
                num_lines_to_reach_double_ref_loss = len(cost)

            cost_to_reach_ref_loss[co2l] = sum(cost[:num_lines_to_reach_ref_loss])
            cost_to_reach_double_ref_loss[co2l] = sum(
                cost[:num_lines_to_reach_double_ref_loss]
            )

        cost_to_reach_ref_loss = pd.DataFrame.from_dict(
            cost_to_reach_ref_loss, orient="index", columns=["cost_ref_loss"]
        )
        cost_to_reach_ref_loss.index.name = "co2l"
        cost_to_reach_ref_loss["cost_double_ref_loss"] = pd.Series(
            cost_to_reach_double_ref_loss
        )

        fname = "cost_to_reach_ref_loss.csv"
        if annualized_costs:
            fname = "annualized_" + fname
        else:
            fname = "total_" + fname

        cost_to_reach_ref_loss.to_csv(save_path + fname)

    # Create plots for specific CO2 levels
    co2_ref_str = str(int(100 * get_actual_co2_level(co2l_ref))) + "\%"
    plot_lvls = [0.1]
    plot_cost = True

    for annualized_costs in [True, False]:

        fname = "cost_to_reach_ref_loss.csv"
        if annualized_costs:
            fname = "annualized_" + fname
        else:
            fname = "total_" + fname

        cost_to_reach_ref_loss = pd.read_csv(save_path + fname, index_col=0)

        for co2l in plot_lvls:

            if annualized_costs:
                f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2l}_n{n_nodes}.pkl"
            else:
                f_name = f"heuristic_costMin_loss_mitigation_Co2L{co2l}_n{n_nodes}.pkl"

            reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(
                open(path_to_line_extension_mitigation_sclopf + f_name, "rb")
            )

            # Setup matplotlib
            mpl.style.use("default")
            plt.rc("text", usetex=True)
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

            num_lines_to_reach_ref_loss = np.where(
                np.array(loss_with_mitigation) < lost_load_reference_lvl
            )[0][0]
            num_lines_to_reach_double_ref_loss = np.where(
                np.array(loss_with_mitigation) < 2 * lost_load_reference_lvl
            )[0][0]

            fig = plt.figure(figsize=(11, 4))

            if plot_cost:
                wspace = 0.02
            else:
                wspace = -0.02
            gs_vertical = GridSpec(
                1, 3, figure=fig, width_ratios=[1.2, 1.8, 1.2], wspace=wspace
            )
            ax_loss_all = fig.add_subplot(gs_vertical[0])
            ax_loss_lvl = fig.add_subplot(gs_vertical[2])
            ax_map = fig.add_subplot(gs_vertical[1])

            # Panel c: loss reduction curve
            ax_loss_lvl.plot(
                np.arange(len(loss_with_mitigation)),
                np.array(loss_with_mitigation) / lost_load_reference_lvl,
            )

            right_xlim = np.where(
                np.array(loss_with_mitigation) / lost_load_reference_lvl < 0.5
            )[0][0]

            if plot_cost:
                # Add second y axis for costs
                ax2 = ax_loss_lvl.twinx()
                ax2.plot(
                    np.arange(len(cost)),
                    np.cumsum(cost) / 1e9,
                    color="black",
                    linestyle="dotted",
                )
                if annualized_costs:
                    y_label_cost = "Annualized construction cost [billion €]"
                else:
                    y_label_cost = "Construction cost [billion €]"
                ax2.set_ylabel(y_label_cost, fontsize=14)
                ax2.set_ylim((0, 1.1 * sum(cost[:right_xlim]) / 1e9))

            # Add reference lines
            ax_loss_lvl.plot([0, num_lines_to_reach_ref_loss], [1, 1], "--", c="k")
            ax_loss_lvl.plot(
                [num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss],
                [0, 1],
                "--",
                c="k",
            )
            ax_loss_lvl.text(
                num_lines_to_reach_ref_loss + right_xlim / 200 * 5,
                1,
                f"{num_lines_to_reach_ref_loss} lines",
                verticalalignment="bottom",
                horizontalalignment="left",
                zorder=np.inf,
                fontsize=14,
            )

            ax_loss_lvl.plot(
                [0, num_lines_to_reach_double_ref_loss], [2, 2], "--", c="g"
            )
            ax_loss_lvl.plot(
                [
                    num_lines_to_reach_double_ref_loss,
                    num_lines_to_reach_double_ref_loss,
                ],
                [0, 2],
                "--",
                c="g",
            )
            ax_loss_lvl.text(
                num_lines_to_reach_double_ref_loss + right_xlim / 200 * 5,
                2,
                f"{num_lines_to_reach_double_ref_loss} lines",
                verticalalignment="bottom",
                horizontalalignment="left",
                zorder=np.inf,
                fontsize=14,
            )

            ax_loss_lvl.set_xlim(0, right_xlim)
            ax_loss_lvl.set_ylim(bottom=0)
            ax_loss_lvl.tick_params(axis="both", which="major", labelsize=14)
            ax_loss_lvl.set_xlabel("Number of reinforced lines")

            y_label = "$\\textrm{R}/\\textrm{R}_{\\textrm{co2_ref_str}}$"
            y_label = y_label.replace("co2_ref_str", co2_ref_str)
            if plot_cost:
                y_label += " (solid)"
            ax_loss_lvl.set_ylabel(y_label, fontsize=14)
            ax_loss_lvl.grid()
            ax_loss_lvl.xaxis.label.set_size(14)

            # Panel b: map of line extensions
            pos = nx.get_node_attributes(nx_graph, "pos")
            reinforced_lines_selected = reinforced_lines[:num_lines_to_reach_ref_loss]
            lines_not_extended = np.setdiff1d(
                np.arange(n_lines), reinforced_lines_selected
            )
            reinforced_lines_full = reinforced_lines_selected + list(lines_not_extended)
            mitigated_loss = np.array(
                [
                    loss_with_mitigation[i] - loss_with_mitigation[i + 1]
                    for i in range(len(loss_with_mitigation) - 1)
                ]
            )
            mitigated_loss = np.concatenate(
                (mitigated_loss, -np.ones(n_lines - len(mitigated_loss)))
            )
            mitigated_loss_sorted = mitigated_loss[np.argsort(reinforced_lines_full)]
            mitigated_loss_selected = -np.ones(n_lines)

            # Convert reinforced_lines_selected to integer indices
            reinforced_lines_selected_int = np.array(reinforced_lines_selected).astype(
                int
            )
            mitigated_loss_selected[reinforced_lines_selected_int] = (
                mitigated_loss[:num_lines_to_reach_ref_loss] / lost_load_reference_lvl
            )

            width = 2 * (mitigated_loss_selected > 0).astype(int) + 1
            cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
            cmap.set_under("gainsboro", 1.0)

            nodes = nx.draw_networkx_nodes(
                nx_graph, pos=pos, ax=ax_map, node_color="black", node_size=0
            )

            edges = nx.draw_networkx_edges(
                nx_graph,
                pos=pos,
                ax=ax_map,
                edge_color=mitigated_loss_selected,
                width=width,
                edge_cmap=cmap,
                edge_vmin=0,
            )

            # Add colorbar
            bbox = ax_map.get_position()
            ax_map_legend = fig.add_axes(
                [bbox.x0 + bbox.width * 0.1, bbox.y0 - 0.0, bbox.width * 0.8, 0.03]
            )
            y_label = (
                "$\\Delta \\textrm{R}_{{\\ell}}/\\textrm{R}_{\\textrm{co2refString}}$"
            )
            y_label = y_label.replace("co2refString", str(int(100 * co2l_ref)) + "\\%")
            cbar = plt.colorbar(
                edges,
                ax=ax_map,
                cax=ax_map_legend,
                label=y_label,
                shrink=0.5,
                orientation="horizontal",
            )
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.set_xlabel(y_label, fontsize=14)

            ax_map.axis("off")

            # Panel a: cost vs CO2 level
            actual_co2_ref = get_actual_co2_level(co2l_ref, percent=True)
            ax_loss_all.plot(
                get_actual_co2_level(cost_to_reach_ref_loss.index, percent=True),
                cost_to_reach_ref_loss.cost_ref_loss / 1e9,
                label="$R_{{{:.0f}\\%}}$".format(actual_co2_ref),
            )
            ax_loss_all.plot(
                get_actual_co2_level(cost_to_reach_ref_loss.index, percent=True),
                cost_to_reach_ref_loss.cost_double_ref_loss / 1e9,
                label="$2R_{{{:.0f}\\%}}$".format(actual_co2_ref),
            )
            ax_loss_all.set_xlabel("CO2 level [% of 1990]")
            if annualized_costs:
                ax_loss_all.set_ylabel("Annualized Cost [billion €]")
            else:
                ax_loss_all.set_ylabel("Cost [billion €]")
            ax_loss_all.legend()
            ax_loss_all.invert_xaxis()
            ax_loss_all.set_title("Grid extension \\n to reach reference loss")

            # Add subplot labels
            ax_loss_all.text(
                0 - 0.2,
                1 + 0.04,
                "a",
                fontsize=36,
                weight="bold",
                verticalalignment="center",
                transform=ax_loss_all.transAxes,
            )

            ax_loss_lvl.text(
                0 - 0.2,
                1 + 0.04,
                "c",
                fontsize=36,
                weight="bold",
                verticalalignment="center",
                transform=ax_loss_lvl.transAxes,
            )
            ax_loss_lvl.set_title(
                f"{int(round(get_actual_co2_level(co2l, percent=True)))}\% CO$_2$ level \\n Loss reduction"
            )

            ax_map.text(
                0 + 0.15,
                1 + 0.04,
                "b",
                fontsize=36,
                weight="bold",
                verticalalignment="center",
                transform=ax_map.transAxes,
            )
            title = "{:.0f}\\% CO$_2$ level \\n Grid extension to reach $R_{{co2_ref_str}}$".format(
                round(get_actual_co2_level(co2l, percent=True))
            )
            title = title.replace("co2_ref_str", co2_ref_str)
            ax_map.set_title(title)

            fname = "costOpt_loss_reduction_blackouts_vs_reinforced_lines_costMin"
            if annualized_costs:
                fname += "_annualized"
            fname += f"_CO2{co2l}.pdf"

            plt.savefig(save_path + fname, bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    create_line_extension_mitigation_plot()
