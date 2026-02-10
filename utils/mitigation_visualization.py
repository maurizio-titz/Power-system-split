#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the results that came out of the mitigation strategies.
The two mitigation strategies are line extension and inertia placement."""

import os
from datetime import datetime as dt

import matplotlib
import networkx
import numpy as np
import pandas as pd

from utils import config


matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "figure.titlesize": 16,
    }
)
matplotlib.rcParams["text.usetex"] = True
import gzip
import pickle
from glob import glob

from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.data_handling import get_actual_co2_level
from utils import data_handling
from utils.config import (
    path_to_cascade_results_sclopf,
    path_to_evaluation_results_sclopf,
    path_to_inertia_mitigation_results_sclopf,
    path_to_line_extension_mitigation_sclopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_sclopf,
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
    save_figure,
)

color1 = "#d95f02"
color2 = "#7570b3"
color3 = "#1b9e77"
color4 = "#e7298a"

color_ls = [color1, color2, color3, color4]

script_path = os.path.dirname(__file__)


def plot_inertia_loss_mitigation_curve_new(
    mitigation_curve,
    target_value,
    delta_Erot,
    ax_loss,
    ax_number=None,
    show_step_number=False,
    modified_comp_idx=None,
    inertia_placed_ls=None,
    unit="GWs",
    color=None,
    annualized_cost_per_GWs_max=None,
    cost_rescale_factor=0.06,
):
    # unit = "MWs"

    if ax_number is None:
        plot_number_of_splits = False

    ## Plot mitigated lost load and remaining lost splits over time
    total_dangerous_splits = len(modified_comp_idx)
    inertia_placed_arr = np.array(inertia_placed_ls)

    if show_step_number:
        x_vals = inertia_placed_arr[:, 0]
    else:
        cumulative_inertia_placed = np.cumsum(inertia_placed_arr[:, 2] * delta_Erot)
        x_vals = cumulative_inertia_placed

    loss_ref_multiple = (mitigation_curve) / target_value

    if unit == "GWs":
        unit_factor = 1e-3
    elif unit == "MWs":
        unit_factor = 1
    else:
        raise ValueError(f"Unit '{unit}' not known!")

    ax_loss.plot(
        x_vals * unit_factor,
        loss_ref_multiple,
        linestyle="-",
        label=f"Loss Reduction",
        color=color,
    )
    # adjust upper xlim
    x_vals_half_ref_loss = x_vals[np.where(loss_ref_multiple < 0.5)]
    x_val_half_ref_loss = x_vals_half_ref_loss[0]

    if plot_number_of_splits:
        ax_number.plot(
            x_vals * unit_factor,
            inertia_placed_arr[:, 4] / total_dangerous_splits,
            linestyle="--",
            label=f"Split Number",
        )

    ax_loss.set_xlim(left=0, right=x_val_half_ref_loss * unit_factor)
    ax_loss.set_ylim(bottom=0)

    if annualized_cost_per_GWs_max is not None:
        x_lim_mask = x_vals <= x_val_half_ref_loss
        x_vals = x_vals[x_lim_mask]  # limit x vals to half ref loss for cost plot
        cost_unit_factor = 1e-9  # to billion euros
        cumulative_annualized_cost = np.cumsum(
            inertia_placed_arr[:, 2]
            * delta_Erot
            * unit_factor
            * cost_unit_factor
            * annualized_cost_per_GWs_max
        )[x_lim_mask]
        cumulative_annualized_cost_rescaled = (
            cumulative_annualized_cost / cost_rescale_factor
        )

        ax_loss_twin = ax_loss.twinx()
        ax_loss_twin.plot(
            x_vals * unit_factor,
            cumulative_annualized_cost_rescaled,
            linestyle=":",
            color=color,
            label=f"Annualized Cost",
        )
        ax_loss_twin.set_ylabel("Annualized Cost [billion €]")
        ax_loss_twin.grid(False)

        ax_loss_twin.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)

        ticks_primary = ax_loss.get_yticks()
        ax_loss_twin.set_yticks(ticks_primary)[:-1]
        ticks_labels_secondary = np.round(ticks_primary * cost_rescale_factor, 2)
        if all(ticks_labels_secondary % 1 == 0):
            ticks_labels_secondary = ticks_labels_secondary.astype(int)
        ax_loss_twin.set_yticklabels(ticks_labels_secondary)

        primary_lims = ax_loss.get_ylim()
        ax_loss_twin.set_ylim(primary_lims)

        ax_loss_twin.set_ylim(bottom=0)

        # Combine legends from both axes
        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_loss_twin.get_legend_handles_labels()
        ax_loss.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            fontsize=12,
        )

    return x_val_half_ref_loss


def plot_map_inertia_placement_final(
    co2_lvl,
    nn=400,
    max_iter=10000,
    max_node_size=200,
    edge_width=0.2,
    delta_Erot=10,
    rocof_thres=1,
    l_share=0.0,
    resolve_strategy="random",
    show_step_number: bool = False,
    plot_split_number: bool = False,
    save_fig=False,
    co2_lvl_ref: float = 0.6,
    axes=None,
    unit="GWs",
    show_params=False,
    ref_loss_factor=1,
    color=color1,
    plot_curve=True,
    use_sclopf=True,
    split_properties=None,
    line_color=None,
    annualized_cost_per_GWs_max=None,
    blackoutthreshold=0.0,
):
    """Plot the results of the inertia placement"""

    if unit == "GWs":
        unit_factor = 1e-3
    elif unit == "MWs":
        unit_factor = 1
    else:
        raise ValueError(f"Unit '{unit}' not known!")

    # Load graph
    pypsa_net = data_handling.load_pypsa_network(co2_lvl, nn, use_sclopf)
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)
    pos_nodes = networkx.get_node_attributes(nx_graph, "pos")

    # Load synthetic inertia placement
    if use_sclopf:
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_sclopf
        path_to_evaluation_results = path_to_evaluation_results_sclopf
    else:
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_lopf
        path_to_evaluation_results = path_to_evaluation_results_lopf

    # os.mkdir(path_to_inertia_mitigation_results, exist_ok=True)

    fpath_in = (
        path_to_inertia_mitigation_results
        + f"synthetic_inertia_placement_Co2{co2_lvl:g}"
        + f"_N{nn}_deltarotE{delta_Erot:g}_rocofthres{rocof_thres:g}"
        + f"_lshare{l_share:g}_maxiter{max_iter}_{resolve_strategy}"
    )
    if blackoutthreshold > 0.0:
        fpath_in += f"_blackoutthres{blackoutthreshold:g}"
    fpath_in += ".pklz"

    if not os.path.exists(fpath_in):
        fnames = os.listdir(path_to_inertia_mitigation_results)
        fnames_lvl = [f for f in fnames if f"Co2{co2_lvl:g}_" in f]
        if blackoutthreshold > 0.0:
            fnames_lvl = [
                f for f in fnames_lvl if f"_blackoutthres{blackoutthreshold:g}" in f
            ]
        if len(fnames_lvl) == 1:
            fpath_in = path_to_inertia_mitigation_results + fnames_lvl[0]
        else:
            raise FileNotFoundError(
                f"No unique inertia placement results for CO2 level {co2_lvl} found. {len(fnames_lvl)} files found. Skipping."
            )
    with gzip.open(fpath_in) as fh_in:
        (
            modified_comp_idx,
            _,
            inertia_placed_ls,
            _,
            resolve_counter,
            still_random_counter,
        ) = pickle.load(fh_in)

    # correcting delta_Erot factor if saved with different one
    delta_Erot_saved = int(fpath_in.split("deltarotE")[1].split("_rocofthres")[0])
    if delta_Erot_saved != delta_Erot:
        print("Correcting delta_Erot factor from", delta_Erot, "to", delta_Erot_saved)
        delta_Erot = delta_Erot_saved

    inertia_placed_res_arr = np.array(inertia_placed_ls)
    # for each opitimization step holds: [idx_step, idx_node, delta_rot_energy_factor, max_change, count_beyond_threshold]

    # get split properties
    if split_properties is None:
        split_properties = data_handling.load_split_props(nn, co2_lvl, use_sclopf)
    else:
        split_properties = split_properties.copy()
        split_properties = split_properties[split_properties.co2l == co2_lvl]
    if blackoutthreshold > 0.0:
        split_properties = split_properties[
            split_properties.lost_load_share_blackout >= blackoutthreshold
        ]

    total_loss_share_rocof_lvl = (
        split_properties.lost_load_share_blackout * split_properties.total_weighting
    ).sum()
    split_properties = data_handling.load_split_props(nn, co2_lvl_ref, use_sclopf)
    total_loss_share_rocof_ref = (
        split_properties.lost_load_share_blackout * split_properties.total_weighting
    ).sum()

    total_loss_share_rocof_ref_multiple = total_loss_share_rocof_ref * ref_loss_factor

    idx_reached_target = np.where(
        (total_loss_share_rocof_lvl - np.cumsum(inertia_placed_res_arr[:, 3]))
        < total_loss_share_rocof_ref_multiple
    )[0][0]

    inertia_node_idx = inertia_placed_res_arr[:, 1:3]
    # for each opitimization step holds: [idx_node, delta_rot_energy_factor]
    inertia_placements_by_node = [0] * len(nx_graph)
    # stepwise sum up inertia placements to get total inertia placed per node
    for indi_idx_r, indi_count_r in inertia_node_idx[:idx_reached_target, :]:
        inertia_placements_by_node[int(indi_idx_r)] += indi_count_r
    inertia_in_map_plot = np.sum(inertia_placements_by_node) * delta_Erot
    print("inertia placed in map: ", inertia_in_map_plot)

    # Plot graph
    if axes is None:
        fig, [ax, ax2] = plt.subplots(1, 2, figsize=(18, 10))
    else:
        ax, ax2 = axes
        fig = ax.get_figure()
    if plot_split_number:
        ax_twin_number = ax2.twinx()
    else:
        ax_twin_number = None

    ## Normalize node_size
    max_inertia_placements_in_one_node = max(inertia_placements_by_node)
    # if max_node_size is None:
    #     max_node_size = 200 * max_inertia_placements_in_one_node / 10
    node_width_arr = (
        np.array(inertia_placements_by_node) / max_inertia_placements_in_one_node
    ) * max_node_size

    networkx.draw_networkx_edges(nx_graph, pos_nodes, width=edge_width, ax=ax)
    networkx.draw_networkx_nodes(
        nx_graph, pos_nodes, node_size=node_width_arr, node_color=color, ax=ax
    )

    ## Plot mitigated lost load and remaining lost splits over time
    if plot_curve:
        plot_inertia_loss_mitigation_curve(
            resolve_strategy,
            co2_lvl,
            nn,
            max_iter,
            total_loss_share_rocof_lvl,
            total_loss_share_rocof_ref_multiple,
            delta_Erot,
            ax2,
            ax_number=None,
            show_step_number=False,
            modified_comp_idx=modified_comp_idx,
            inertia_placed_ls=inertia_placed_ls,
            unit=unit,
            color=line_color,
            annualized_cost_per_GWs_max=annualized_cost_per_GWs_max,
        )
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    cost_map_plot = (
        inertia_in_map_plot * unit_factor * annualized_cost_per_GWs_max * 1e-9
    )
    ax2_twin = ax2.get_shared_x_axes().get_siblings(ax2)
    if len(ax2_twin) > 1:
        ax2_twin = [ax for ax in ax2_twin if ax != ax2][0]
        ax2_twin.plot(
            [inertia_in_map_plot * unit_factor, inertia_in_map_plot * unit_factor],
            [ax2_twin.get_ylim()[0], cost_map_plot / 0.3],  # rescale divider 0.3
            linestyle="--",
            c=color,
            lw=2,
        )
    else:
        ax2.plot(
            [inertia_in_map_plot * unit_factor, inertia_in_map_plot * unit_factor],
            [ax2.get_ylim()[0], ref_loss_factor],
            linestyle="--",
            c=color,
            lw=2,
        )
    ax2.plot(
        [0, inertia_in_map_plot * unit_factor],
        [ref_loss_factor, ref_loss_factor],
        linestyle="--",
        c=color,
        lw=2,
    )
    right_xlim = ax2.get_xlim()[1]
    ax2.text(
        inertia_in_map_plot * unit_factor + right_xlim / 200 * 5,
        0.2,
        f"{int(round(inertia_in_map_plot*unit_factor))} GWs",
        verticalalignment="bottom",
        horizontalalignment="left",
        zorder=np.inf,
        fontsize=TICK_LABEL_FONTSIZE,
        c=color,
    )
    ax2.text(
        inertia_in_map_plot * unit_factor + right_xlim / 200 * 5,
        1.2,
        f"{cost_map_plot:.2f} bn. €",
        verticalalignment="bottom",
        horizontalalignment="left",
        zorder=np.inf,
        fontsize=TICK_LABEL_FONTSIZE,
        c=color,
    )

    for fact in [1, 0.5]:
        ax.plot(
            [],
            [],
            marker="o",
            color=color,
            markersize=np.sqrt(  # sqrt beause plot and nx.draw scale differently
                max_node_size * fact
            ),
            label=f"{int(delta_Erot * max_inertia_placements_in_one_node*unit_factor*fact)} {unit}",
            lw=0,
        )
    ax.legend(
        labelspacing=0,
        handletextpad=0.1,
        loc="upper left",
        # bbox_to_anchor=(0.7, 0.975),
        frameon=False,
        fontsize=18,
    )

    ax.axis("off")
    if show_params:
        para_text = f"$N={nn}$, CO$_2$-Level $={co2_lvl}$, \n$\\Delta E_0 = {(delta_Erot*unit_factor):.2f}${unit}"
        fig.text(
            0.0,
            0.975,
            para_text,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=22,
            transform=ax.transAxes,
        )
    if resolve_strategy == "random":
        resolve_strategy_str = "Place inertia randomly"

    elif resolve_strategy == "concentrate":
        resolve_strategy_str = "Concentrate inertia where inertia was placed previously"

    elif resolve_strategy == "hindsight":
        resolve_strategy_str = "Place inertia where max. lost load in next step."

    elif resolve_strategy == "hindsight_concentrate":
        resolve_strategy_str = (
            "Place inertia where max. lost load in next step + concentrate inertia."
        )
    else:
        raise IOError(f"Resolve equality method '{resolve_strategy}' not known!")

    label_text = (
        f"{resolve_strategy_str}"
        + f"\n Resolve needed: {(resolve_counter *100/max_iter)}\\%"
    )
    if show_params:
        if resolve_counter > 0:
            label_text += f", Random Decisions: {(((still_random_counter/resolve_counter)*100)):.2f}\\%"
        fig.text(
            0.5,
            0.05,
            label_text,
            horizontalalignment="center",
            fontsize=16,
            transform=ax.transAxes,
        )
    if show_step_number:
        ax2.set_xlabel("$t_n$")
    else:
        ax2.set_xlabel(f"inertia placed [{unit}]")

    if plot_split_number:
        ax_twin_number.set_ylabel("$N_{\\textrm{splits mit}}/N_{\\textrm{splits}}$")

    y_label = "$R_{\\textrm{co2string,mit}}/R_{\\textrm{co2refString}}$"
    y_label = y_label.replace(
        "co2string", str(int(100 * get_actual_co2_level(co2_lvl))) + "\%"
    )
    y_label = y_label.replace(
        "co2refString", str(int(100 * get_actual_co2_level(co2_lvl_ref))) + "\%"
    )

    if plot_split_number:
        y_label = y_label + " (-)"
    ax2.set_ylabel(y_label)

    if save_fig:
        fig_path = f"{path_to_inertia_mitigation_results_sclopf}/syn_inertia_map_co2lvl{co2_lvl:g}_nn{nn}_deltaErot{delta_Erot:g}_{resolve_strategy}.png"

        fig.savefig(fig_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    else:
        # plt.show()
        return fig, ax, ax2, ax_twin_number

    return


def plot_map_inertia_placement_new(
    co2_lvl,
    idx_reached_target,
    res_tuple,
    mitigation_curve,
    target_value,
    nn=600,
    max_iter=10000,
    max_node_size=200,
    edge_width=0.2,
    delta_Erot=10,
    resolve_strategy="random",
    show_step_number: bool = False,
    plot_split_number: bool = False,
    save_fig=False,
    co2_lvl_ref: float = 0.6,
    axes=None,
    unit="GWs",
    show_params=False,
    ref_loss_factor=1,
    color=color1,
    plot_curve=True,
    use_sclopf=True,
    line_color=None,
    annualized_cost_per_GWs_max=None,
):
    """Plot the results of the inertia placement"""

    if unit == "GWs":
        unit_factor = 1e-3
    elif unit == "MWs":
        unit_factor = 1
    else:
        raise ValueError(f"Unit '{unit}' not known!")

    # Load graph
    pypsa_net = data_handling.load_pypsa_network(co2_lvl, nn, use_sclopf)
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)
    pos_nodes = networkx.get_node_attributes(nx_graph, "pos")

    inertia_placed = res_tuple[2]
    inertia_node_idx = np.array(inertia_placed)[:, 1:3]
    modified_comp_idx = res_tuple[0]
    resolve_counter = res_tuple[4]
    # for each opitimization step holds: [idx_node, delta_rot_energy_factor]
    inertia_placements_by_node = [0] * len(nx_graph)
    # stepwise sum up inertia placements to get total inertia placed per node
    for indi_idx_r, indi_count_r in inertia_node_idx[:idx_reached_target, :]:
        inertia_placements_by_node[int(indi_idx_r)] += indi_count_r
    inertia_in_map_plot = np.sum(inertia_placements_by_node) * delta_Erot
    print("inertia placed in map: ", inertia_in_map_plot)

    # Plot graph
    if axes is None:
        fig, [ax, ax2] = plt.subplots(1, 2, figsize=(18, 10))
    else:
        ax, ax2 = axes
        fig = ax.get_figure()
    if plot_split_number:
        ax_twin_number = ax2.twinx()
    else:
        ax_twin_number = None

    ## Normalize node_size
    max_inertia_placements_in_one_node = max(inertia_placements_by_node)
    # if max_node_size is None:
    #     max_node_size = 200 * max_inertia_placements_in_one_node / 10
    node_width_arr = (
        np.array(inertia_placements_by_node) / max_inertia_placements_in_one_node
    ) * max_node_size

    networkx.draw_networkx_edges(nx_graph, pos_nodes, width=edge_width, ax=ax)
    networkx.draw_networkx_nodes(
        nx_graph, pos_nodes, node_size=node_width_arr, node_color=color, ax=ax
    )

    ## Plot mitigated lost load and remaining lost splits over time
    cost_rescale_factor = 0.04
    if plot_curve:
        plot_inertia_loss_mitigation_curve_new(
            mitigation_curve,
            target_value,
            delta_Erot,
            ax2,
            ax_number=None,
            show_step_number=False,
            modified_comp_idx=modified_comp_idx,
            inertia_placed_ls=inertia_placed,
            unit=unit,
            color=line_color,
            annualized_cost_per_GWs_max=annualized_cost_per_GWs_max,
            cost_rescale_factor=cost_rescale_factor,
        )
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    cost_map_plot = (
        inertia_in_map_plot * unit_factor * annualized_cost_per_GWs_max * 1e-9
    )
    ax2_twin = ax2.get_shared_x_axes().get_siblings(ax2)
    if len(ax2_twin) > 1:
        ax2_twin = [ax for ax in ax2_twin if ax != ax2][0]
        ax2_twin.plot(
            [inertia_in_map_plot * unit_factor, inertia_in_map_plot * unit_factor],
            [
                ax2_twin.get_ylim()[0],
                cost_map_plot / cost_rescale_factor,
            ],
            linestyle="--",
            c=color,
            lw=2,
        )
    else:
        ax2.plot(
            [inertia_in_map_plot * unit_factor, inertia_in_map_plot * unit_factor],
            [ax2.get_ylim()[0], ref_loss_factor],
            linestyle="--",
            c=color,
            lw=2,
        )
    ax2.plot(
        [0, inertia_in_map_plot * unit_factor],
        [ref_loss_factor, ref_loss_factor],
        linestyle="--",
        c=color,
        lw=2,
    )
    right_xlim = ax2.get_xlim()[1]
    top_ylim = ax2.get_ylim()[1]
    ax2.text(
        inertia_in_map_plot * unit_factor + right_xlim / 200 * 5,
        1.1,
        f"{int(round(inertia_in_map_plot*unit_factor))} GWs",
        verticalalignment="bottom",
        horizontalalignment="left",
        zorder=np.inf,
        fontsize=TICK_LABEL_FONTSIZE,
        c=color,
    )
    ax2.text(
        inertia_in_map_plot * unit_factor + right_xlim / 200 * 5,
        cost_map_plot / cost_rescale_factor * 0.98,
        f"{cost_map_plot:.2f} bn. €",
        verticalalignment="center",
        horizontalalignment="left",
        zorder=np.inf,
        fontsize=TICK_LABEL_FONTSIZE,
        c=color,
    )

    for fact in [1, 0.5]:
        ax.plot(
            [],
            [],
            marker="o",
            color=color,
            markersize=np.sqrt(  # sqrt beause plot and nx.draw scale differently
                max_node_size * fact
            ),
            label=f"{int(delta_Erot * max_inertia_placements_in_one_node*unit_factor*fact)} {unit}",
            lw=0,
        )
    ax.legend(
        labelspacing=0,
        handletextpad=0.1,
        loc="upper left",
        # bbox_to_anchor=(0.7, 0.975),
        frameon=False,
        fontsize=18,
    )

    ax.axis("off")
    if show_params:
        para_text = f"$N={nn}$, CO$_2$-Level $={co2_lvl}$, \n$\\Delta E_0 = {(delta_Erot*unit_factor):.2f}${unit}"
        fig.text(
            0.0,
            0.975,
            para_text,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=22,
            transform=ax.transAxes,
        )
    if resolve_strategy == "random":
        resolve_strategy_str = "Place inertia randomly"

    elif resolve_strategy == "concentrate":
        resolve_strategy_str = "Concentrate inertia where inertia was placed previously"

    elif resolve_strategy == "hindsight":
        resolve_strategy_str = "Place inertia where max. lost load in next step."

    elif resolve_strategy == "hindsight_concentrate":
        resolve_strategy_str = (
            "Place inertia where max. lost load in next step + concentrate inertia."
        )
    else:
        raise IOError(f"Resolve equality method '{resolve_strategy}' not known!")

    label_text = (
        f"{resolve_strategy_str}"
        + f"\n Resolve needed: {(resolve_counter *100/max_iter)}\\%"
    )
    if show_params:
        if resolve_counter > 0:
            label_text += f", Random Decisions: {(((still_random_counter/resolve_counter)*100)):.2f}\\%"
        fig.text(
            0.5,
            0.05,
            label_text,
            horizontalalignment="center",
            fontsize=16,
            transform=ax.transAxes,
        )
    if show_step_number:
        ax2.set_xlabel("$t_n$")
    else:
        ax2.set_xlabel(f"inertia placed [{unit}]")

    if plot_split_number:
        ax_twin_number.set_ylabel("$N_{\\textrm{splits mit}}/N_{\\textrm{splits}}$")

    y_label = "$R_{\\textrm{co2string,mit}}/R_{\\textrm{co2refString}}$"
    y_label = y_label.replace(
        "co2string", str(int(100 * get_actual_co2_level(co2_lvl))) + "\%"
    )
    y_label = y_label.replace(
        "co2refString", str(int(100 * get_actual_co2_level(co2_lvl_ref))) + "\%"
    )

    if plot_split_number:
        y_label = y_label + " (-)"
    ax2.set_ylabel(y_label)

    if save_fig:
        fig_path = f"{path_to_inertia_mitigation_results_sclopf}/syn_inertia_map_co2lvl{co2_lvl:g}_nn{nn}_deltaErot{delta_Erot:g}_{resolve_strategy}.png"

        fig.savefig(fig_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    else:
        # plt.show()
        return fig, ax, ax2, ax_twin_number

    return


def plot_map_inertia_placement(
    co2_lvl,
    nn=400,
    max_iter=10000,
    max_node_size=800,
    edge_width=0.2,
    delta_Erot=10,
    resolve_strategy="random",
    show_step_number: bool = False,
    save_fig=False,
):
    """Plot the results of the inertia placement"""

    # Load graph
    fpath_pypsa = (
        path_to_pypsa_network_sclopf
        + "sclopf-elec_s_"
        + f"{nn}_ec_lv1.0_Co2L{co2_lvl}-2920SEG.nc"
    )
    pypsa_net = data_handling.load_pypsa_network_from_path(fpath_pypsa, use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)
    pos_nodes = networkx.get_node_attributes(nx_graph, "pos")

    # Load synthetic inertia placement
    fpath_in = (
        path_to_inertia_mitigation_results_sclopf
        + f"synthetic_inertia_placement_Co2{co2_lvl:g}"
        + f"_N{nn}_deltarotE{delta_Erot:g}_rocofthres-1.00"
        + f"_lshare0.00_maxiter{max_iter}_{resolve_strategy}.pklz"
    )
    with gzip.open(fpath_in) as fh_in:
        (
            modified_comp_idx,
            _,
            inertia_placed_ls,
            _,
            resolve_counter,
            still_random_counter,
        ) = pickle.load(fh_in)

    inertia_placed_res_arr = np.array(inertia_placed_ls)

    inertia_node_idx = inertia_placed_res_arr[:, 1:3]

    node_count_ls = [0] * len(nx_graph)
    for indi_idx_r, indi_count_r in inertia_node_idx:
        node_count_ls[int(indi_idx_r)] += indi_count_r

    # Plot graph
    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(18, 10))
    ax2_twin = ax2.twinx()
    ## Normalize node_size
    max_node_count = max(node_count_ls)
    node_width_arr = (np.array(node_count_ls) / max_node_count) * max_node_size

    networkx.draw_networkx_edges(nx_graph, pos_nodes, width=edge_width, ax=ax)
    networkx.draw_networkx_nodes(
        nx_graph, pos_nodes, node_size=node_width_arr, node_color=color1, ax=ax
    )

    ## Plot mitigated lost load and remaining lost splits over time
    total_dangerous_splits = len(modified_comp_idx)
    inertia_placed_arr = np.array(inertia_placed_ls)

    if show_step_number:
        x_vals = inertia_placed_arr[:, 0]
    else:
        cumulative_inertia_placed = np.cumsum(inertia_placed_arr[:, 2] * delta_Erot)
        x_vals = cumulative_inertia_placed

    ax2_twin.plot(
        x_vals,
        np.cumsum(inertia_placed_arr[:, 3]) / total_dangerous_splits,
        color=color2,
        linestyle="--",
    )
    ax2.plot(
        x_vals,
        (1 - inertia_placed_arr[:, 4] / total_dangerous_splits),
        color=color2,
        linestyle="-",
    )

    # Asthetics
    legend_small_node_size = max_node_size * 0.25
    legend_large_node_size = max_node_size * 0.75
    ls_legend = [legend_small_node_size, legend_large_node_size]
    for size_r in ls_legend:
        ax.plot(
            [],
            [],
            marker="o",
            color=color1,
            markersize=np.sqrt(size_r),
            label=f"{int(size_r*delta_Erot)} MWs",
            lw=0,
        )
    ax.legend(
        labelspacing=1,
        loc="center left",
        bbox_to_anchor=(0.7, 0.975),
        frameon=False,
        fontsize=22,
    )

    ax.axis("off")

    para_text = (
        f"$N={nn}$, CO$_2$-Level $={co2_lvl}$, \n$\\Delta E_0 = {delta_Erot:.2f}$MWs"
    )
    fig.text(
        0.0,
        0.975,
        para_text,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=22,
        transform=ax.transAxes,
    )
    if resolve_strategy == "random":
        resolve_strategy_str = "Place inertia randomly"

    elif resolve_strategy == "concentrate":
        resolve_strategy_str = "Concentrate inertia where inertia was placed previously"

    elif resolve_strategy == "hindsight":
        resolve_strategy_str = "Place inertia where max. lost load in next step."

    elif resolve_strategy == "hindsight_concentrate":
        resolve_strategy_str = (
            "Place inertia where max. lost load in next step + concentrate inertia."
        )
    else:
        raise IOError(f"Resolve equality method '{resolve_strategy}' not known!")

    label_text = (
        f"{resolve_strategy_str}"
        + f"\n Resolve needed: {(resolve_counter *100/max_iter)}\\%"
    )
    if resolve_counter > 0:
        label_text += f", Random Decisions: {(((still_random_counter/resolve_counter)*100)):.2f}\\%"
    fig.text(
        0.5,
        0.05,
        label_text,
        horizontalalignment="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    if show_step_number:
        ax2.set_xlabel("$t_n$")
    else:
        ax2.set_xlabel("inertia placed [MWs]")

    ax2.set_ylabel("$N_{\\textrm{splits mit}}/N_{\\textrm{splits}}$")
    ax2_twin.set_ylabel("$L_{\\textrm{mit, loss}}/N_{\\textrm{splits}}$ (--)")

    if save_fig:
        fig_path = f"syn_inertia_map_co2lvl{co2_lvl:.2f}_nn{nn}_deltaErot{delta_Erot:.2f}_{resolve_strategy}.png"

        fig.savefig(fig_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_all_syn_inertia_map(nn=400):
    """Plot all c02 lvls and rot E"""

    file_list = glob(f"results/sclopf/syn_inertia_mitigation/*N{nn}*.pklz")

    co2_lvl_ls = [float(xx.split("Co2")[1].split(f"_N{nn}")[0]) for xx in file_list]
    rotE_ls = [float(xx.split("rotE")[1].split("_rocof")[0]) for xx in file_list]
    method_ls = [xx.split("_")[-1].split(".pklz")[0] for xx in file_list]

    for co2_r, rotE_r, method_r in zip(co2_lvl_ls, rotE_ls, method_ls):
        plot_map_inertia_placement(
            co2_r, delta_Erot=rotE_r, resolve_strategy=method_r, save_fig=True
        )


def plot_mitigate_load_loss_n_splits_over_time_diff_E0(
    co2_lvl_ls=[0.1, 0.8],
    deltaE_ls=[1, 5, 20],
    nn=400,
    max_iter=10000,
    resolve_method="random",
    save_fig=True,
):
    """Plot the mitigate

    Args:
        co2_lvl (_type_): _description_
        nn (int, optional): _description_. Defaults to 400.
        save_fig (bool, optional): _description_. Defaults to True.
    """

    linestyle_ls = ["-", "--", ":", "-."]

    # Load synthetic inertia placement
    fig, ax = plt.subplots(3, 1, sharex="all", figsize=(10, 9))
    already_labeled = False
    for co2_idx, co2_lvl_r in enumerate(co2_lvl_ls):
        for E_idx, delta_Erot in enumerate(deltaE_ls):
            fpath_in = (
                path_to_inertia_mitigation_results_sclopf
                + f"synthetic_inertia_placement_Co2{co2_lvl_r:.2f}"
                + f"_N{nn}_deltarotE{delta_Erot:g}_rocofthres-1.00_lshare0.00"
                + f"_maxiter{max_iter}_{resolve_method}.pklz"
            )
            with gzip.open(fpath_in) as fh_in:
                modified_comp_idx, _, inertia_placed_ls, _, _, _ = pickle.load(fh_in)

            inertia_placed_arr = np.array(inertia_placed_ls)

            total_dangerous_splits = len(modified_comp_idx)
            inertia_placed_arr = np.array(inertia_placed_ls)
            label_str = f" = {delta_Erot} MWs"
            if not already_labeled:
                label_str = "$\\Delta E_{\\rm rot}$" + label_str
                already_labeled = True

            style_dict = dict(color=color_ls[E_idx], linestyle=linestyle_ls[co2_idx])
            if co2_idx == 0:
                style_dict["label"] = label_str

            ax[0].plot(
                inertia_placed_arr[:, 0],
                (1 - inertia_placed_arr[:, 4] / total_dangerous_splits),
                **style_dict,
            )

            ax[1].plot(
                inertia_placed_arr[:, 0],
                np.cumsum(inertia_placed_arr[:, 3]) / total_dangerous_splits,
                **style_dict,
            )

            ax[2].plot(
                inertia_placed_arr[:, 0],
                np.cumsum(inertia_placed_arr[:, 2] * delta_Erot),
                **style_dict,
            )

    [
        ax[0].plot(
            [], [], color="k", linestyle=linestyle_ls[idx], label=f"CO$_2$={co2_r}"
        )
        for idx, co2_r in enumerate(co2_lvl_ls)
    ]
    # Aesthetics
    ax[0].set_ylabel("$N_{\\rm sp, m}/N_{\\textrm{sp}}$")
    ax[1].set_ylabel("$L_{\\rm l, m}/N_{\\textrm{spy}}$")
    ax[2].set_ylabel("$E_{\\rm r, t}$/MWs")
    ax[-1].set_xlabel("$t_n$")

    ax[0].set_xlim(left=0, right=max_iter)
    ax[0].legend(loc="upper left", fontsize=14)

    fig.tight_layout()

    fig.text(0.99, 0.01, resolve_method, horizontalalignment="right", fontsize=20)

    if save_fig:
        fig_path = f"syn_inertia_co2lvl_over_time_{resolve_method}.png"
        fig.savefig(fig_path)
        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_all_syn_inertia_over_time():

    resolve_method = ["random", "concentrate", "hindsight"]
    [
        plot_mitigate_load_loss_n_splits_over_time_diff_E0(
            resolve_method=xx, save_fig=True
        )
        for xx in resolve_method
    ]

    return


def calc_inertia_placement_ref_loss(
    n_nodes=600,
    co2_lvl_ref=0.6,
    ref_loss_factor=1,
    target="total_loss",
    max_iter=10000,
    delta_Erot=5000,
    rocof_thres=1,
    l_share=0.0,
    resolve_strategy="random",
    co2_lvls=(0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
    use_sclopf=True,
    split_properties=None,
    blackoutthreshold=0.0,
):
    """calculates the synthetic inertia needed to reach the reference loss level for different CO2 levels.

    Args:
        n_nodes (int, optional): _description_. Defaults to 400.
        co2_lvl_ref (float, optional): _description_. Defaults to 0.6.
        ref_loss_factor (int, optional): _description_. Defaults to 1.
        max_iter (int, optional): _description_. Defaults to 10000.
        delta_Erot (int, optional): _description_. Defaults to 5000.
        rocof_thres (int, optional): _description_. Defaults to -1.
        l_share (float, optional): _description_. Defaults to 0.0.
        resolve_strategy (str, optional): _description_. Defaults to "random".
        co2_lvls (list, optional): _description_. Defaults to [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0].

    Returns:

    """

    if split_properties is None:
        split_properties = data_handling.load_split_props(n_nodes, None, use_sclopf)

    split_properties_ref = split_properties[split_properties.co2l == co2_lvl_ref]

    if target == "total_loss":
        target_value = (
            split_properties_ref.lost_load_share_blackout
            * split_properties_ref.total_weighting
        ).sum() * ref_loss_factor
    elif target == "num_GSS":
        target_value = (
            (split_properties_ref.lost_load_share_blackout > blackoutthreshold)
            * split_properties_ref.total_weighting
        ).sum() * ref_loss_factor
        component_properties = pd.read_hdf(
            config.path_to_vis_results_sclopf
            + f"component_properties_all_n{n_nodes}.h5",
            index_col=0,
        )
    elif target == "lost_load_GSS":
        target_value = (
            (
                split_properties_ref.lost_load_share_blackout[
                    split_properties_ref.lost_load_share_blackout > blackoutthreshold
                ]
            )
            * split_properties_ref.total_weighting.sum()
            * ref_loss_factor
        ).sum()
        component_properties = pd.read_hdf(
            config.path_to_vis_results_sclopf
            + f"component_properties_all_n{n_nodes}.h5",
            index_col=0,
        )
    else:
        raise ValueError(f"Target '{target}' not known!")

    inertia_to_reach_target_by_lvl = {}
    steps_to_reach_target_by_lvl = {}
    res_tuples_by_lvl = {}
    mitigation_curve_by_lvl = {}
    delta_Erot_by_lvl = {}

    path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_sclopf
    for co2_lvl in co2_lvls:
        if co2_lvl == co2_lvl_ref:
            continue
        # Load synthetic inertia placement
        delta_Erot_saved, res_tuple = load_inertia_placement_results(
            n_nodes,
            max_iter,
            delta_Erot,
            rocof_thres,
            l_share,
            resolve_strategy,
            blackoutthreshold,
            path_to_inertia_mitigation_results,
            co2_lvl,
        )

        (
            modified_comp_idx,
            _,
            inertia_placed_ls,
            componont_mitigated_step,
            resolve_counter,
            still_random_counter,
        ) = res_tuple

        res_tuples_by_lvl[co2_lvl] = res_tuple

        if delta_Erot_saved != delta_Erot:
            print(
                "Correcting delta_Erot factor from", delta_Erot, "to", delta_Erot_saved
            )
            delta_Erot = delta_Erot_saved

        if target == "total_loss":
            mitigation_curve, idx_reached_ref_loss = get_idx_ref_loss_reached(
                split_properties,
                blackoutthreshold,
                target_value,
                co2_lvl,
                inertia_placed_ls,
            )
        else:
            split_properties_lvl = split_properties[split_properties.co2l == co2_lvl]
            component_properties_lvl = component_properties[
                component_properties.co2l == co2_lvl
            ]
            mitigation_curve, idx_reached_ref_loss = get_idx_ref_reached_GSS(
                split_properties_lvl,
                component_properties_lvl,
                componont_mitigated_step,
                target_value=target_value,
                blackout_threshold=blackoutthreshold,
                target=target,
            )

        total_placed_inertia = np.sum(
            np.array(inertia_placed_ls)[:idx_reached_ref_loss, 2] * delta_Erot
        )

        inertia_to_reach_target_by_lvl[co2_lvl] = total_placed_inertia
        steps_to_reach_target_by_lvl[co2_lvl] = idx_reached_ref_loss
        mitigation_curve_by_lvl[co2_lvl] = mitigation_curve
        delta_Erot_by_lvl[co2_lvl] = delta_Erot

    return (
        res_tuples_by_lvl,
        mitigation_curve_by_lvl,
        inertia_to_reach_target_by_lvl,
        steps_to_reach_target_by_lvl,
        delta_Erot_by_lvl,
    )


def get_idx_ref_loss_reached(
    split_properties,
    blackoutthreshold,
    total_loss_share_rocof_ref,
    co2_lvl,
    inertia_placed_ls,
):
    inertia_placed_res_arr = np.array(inertia_placed_ls)

    split_properties_lvl = split_properties[split_properties.co2l == co2_lvl]
    if blackoutthreshold is not None and blackoutthreshold > 0.0:
        split_properties_lvl = split_properties_lvl[
            split_properties_lvl.lost_load_share_blackout >= blackoutthreshold
        ]
    total_loss_share_rocof_lvl = (
        split_properties_lvl.lost_load_share_blackout
        * split_properties_lvl.total_weighting
    ).sum()

    loss_mitigation_curve = total_loss_share_rocof_lvl - np.cumsum(
        inertia_placed_res_arr[:, 3]
    )
    idx_reached_ref_loss = np.where(
        (loss_mitigation_curve) < total_loss_share_rocof_ref
    )[0][0]

    return loss_mitigation_curve, idx_reached_ref_loss


def load_inertia_placement_results(
    n_nodes,
    max_iter,
    delta_Erot,
    rocof_thres,
    l_share,
    resolve_strategy,
    blackoutthreshold,
    path_to_inertia_mitigation_results,
    co2_lvl,
):
    fpath_in = (
        path_to_inertia_mitigation_results
        + f"synthetic_inertia_placement_Co2{co2_lvl:g}"
        + f"_N{n_nodes}_deltarotE{delta_Erot:g}_rocofthres{rocof_thres:g}"
        + f"_lshare{l_share:g}_maxiter{max_iter}_{resolve_strategy}.pklz"
    )
    if not os.path.exists(fpath_in):
        fnames = os.listdir(path_to_inertia_mitigation_results)
        fnames_lvl = [f for f in fnames if f"Co2{co2_lvl:g}_" in f]
        if blackoutthreshold is not None and blackoutthreshold > 0.0:
            fnames_lvl = [
                f for f in fnames_lvl if f"blackoutthres{blackoutthreshold:g}" in f
            ]
        else:
            fnames_lvl = [
                f for f in fnames_lvl if "blackoutthres" not in f
            ]  # select only files without blackout threshold
        if len(fnames_lvl) == 1:
            fpath_in = path_to_inertia_mitigation_results + fnames_lvl[0]

        else:
            raise FileNotFoundError(
                f"No unique inertia placement results for CO2 level {co2_lvl} found. Skipping."
            )

        # correcting delta_Erot factor if saved with different one
    delta_Erot_saved = int(fpath_in.split("deltarotE")[1].split("_rocofthres")[0])

    with gzip.open(fpath_in) as fh_in:
        res_tuple = pickle.load(fh_in)

    return delta_Erot_saved, res_tuple


def get_idx_ref_reached_GSS(
    split_properties_lvl,
    component_properties_lvl,
    componont_mitigated_step,
    target_value: float,
    blackout_threshold=0.8,
    target: str = "num_GSS",
):
    """calculates the number of steps needed to reach the target number of GSS (globally severe split) or total lost load in GSS.
    Args:
        split_properties_lvl (pd.DataFrame)
        component_properties_lvl (pd.DataFrame)
        componont_mitigated_step (pd.Series): Series with index matching component_properties_lvl index filtered for critical components. holds the step number in which the component was mitigated.
        target_value (float): target value to reach (number of GSS or total lost load)
        blackout_threshold (float, optional): threshold to consider a split as GSS. Defaults to 0.8.
        target (str, optional): "num_GSS" or "lost_load_GSS". Defaults to "num_GSS".
    Returns:
        tuple: (mitigation curve: weighted number of GSS or weighted lost load per step, int: number of steps needed to reach the target)
    """
    component_properties_lvl.reset_index(drop=True, inplace=True)
    # consider only components with rocof > 1 Hz/s
    component_properties_lvl = component_properties_lvl[
        component_properties_lvl.rocof.abs() > 1
    ]
    if not "split_props_idx" in component_properties_lvl.columns:
        component_properties_lvl["split_props_idx"] = component_properties_lvl[
            ["co2l", "time_stamp", "split_number"]
        ].apply(tuple, axis=1)
    component_properties_lvl["mitigated_in_step"] = componont_mitigated_step

    n_steps = componont_mitigated_step.max()
    split_properties_steps = split_properties_lvl[
        ["lost_load_share_blackout", "total_weighting"]
    ].copy()

    # Prepare all columns at once
    step_columns = {}
    step_columns[f"lost_load_share_blackout_step-1"] = split_properties_steps[
        "lost_load_share_blackout"
    ]
    step_names = component_properties_lvl.mitigated_in_step.unique()
    # sort values
    step_names = np.sort(
        step_names[step_names >= 0]
    )  # contains -1 if components are never safed
    for step, step_name in enumerate(step_names):
        comp_idxs_in_step = component_properties_lvl[
            component_properties_lvl.mitigated_in_step == step_name
        ].index
        if len(comp_idxs_in_step) == 0:
            raise ValueError(
                f"Warning: No components mitigated in step {step}. Check component mitigation steps."
            )
        split_props_idxs_in_step = component_properties_lvl.loc[
            comp_idxs_in_step, "split_props_idx"
        ]

        # Create new column based on previous step
        prev_col = step_columns[f"lost_load_share_blackout_step{step-1}"]
        new_col = prev_col.copy()
        new_col.loc[split_props_idxs_in_step] -= (
            component_properties_lvl.loc[comp_idxs_in_step, "load_share"]
        ).values
        step_columns[f"lost_load_share_blackout_step{step}"] = new_col

        if new_col.equals(prev_col):
            raise ValueError(
                f"Warning: No change in lost load share in step {step}. Check if any components were mitigated in this step."
            )
            # Warning(
            #     f"Warning: No change in lost load share in step {step}. Check if any components were mitigated in this step."
            # )

    # Remove the step-1 column and concatenate all at once
    del step_columns[f"lost_load_share_blackout_step-1"]
    split_properties_steps = pd.concat(
        [split_properties_steps, pd.DataFrame(step_columns)], axis=1
    )

    if target == "num_GSS":
        weighted_number_of_GSS = (
            split_properties_steps["total_weighting"].values
            @ (
                split_properties_steps[
                    [col for col in split_properties_steps.columns if "step" in col]
                ]
                > blackout_threshold
            ).values
        )
        steps_needed = np.argmax(weighted_number_of_GSS <= target_value)
        return weighted_number_of_GSS, steps_needed

    elif target == "lost_load_GSS":
        weighted_loss = (
            split_properties_steps["total_weighting"].values
            @ split_properties_steps[
                [col for col in split_properties_steps.columns if "step" in col]
            ].values
        )
        steps_needed = np.argmax(weighted_loss <= target_value)
        return weighted_loss, steps_needed
    else:
        raise ValueError("Target must be 'num_GSS' or 'lost_load_GSS'")


def calc_post_inertia_split_props(
    n_nodes=400,
    co2_lvl_ref=0.6,
    ref_loss_factor=1,
    max_iter=10000,
    delta_Erot=5000,
    rocof_thres=1,
    l_share=0.0,
    resolve_strategy="random",
    co2_lvls=(0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
    use_sclopf=True,
    split_properties=None,
):
    """calculates the synthetic inertia needed to reach the reference loss level for different CO2 levels.

    Args:
        n_nodes (int, optional): _description_. Defaults to 400.
        co2_lvl_ref (float, optional): _description_. Defaults to 0.6.
        ref_loss_factor (int, optional): _description_. Defaults to 1.
        max_iter (int, optional): _description_. Defaults to 10000.
        delta_Erot (int, optional): _description_. Defaults to 5000.
        rocof_thres (int, optional): _description_. Defaults to -1.
        l_share (float, optional): _description_. Defaults to 0.0.
        resolve_strategy (str, optional): _description_. Defaults to "random".
        co2_lvls (list, optional): _description_. Defaults to [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0].

    Returns:
        _type_: _description_
    """

    if split_properties is None:
        split_properties = data_handling.load_split_props(n_nodes, None, use_sclopf)
    if total_loss_share_rocof_ref is None:
        split_properties_ref = split_properties[split_properties.co2l == co2_lvl_ref]

        total_loss_share_rocof_ref = (
            split_properties_ref.lost_load_share_blackout
            * split_properties_ref.total_weighting
        ).sum() * ref_loss_factor
    if "component_props" not in locals():
        component_props = pd.read_hdf(
            path_to_vis_results_sclopf + f"component_properties_all_n{n_nodes}.h5",
            key="df",
        )

    inertia_at_ref_loss_by_lvl = {}

    if use_sclopf:
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_sclopf
    else:
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_lopf
    for co2_lvl in co2_lvls:
        if co2_lvl == co2_lvl_ref:
            continue
        # Load synthetic inertia placement
        fpath_in = (
            path_to_inertia_mitigation_results
            + f"synthetic_inertia_placement_Co2{co2_lvl:g}"
            + f"_N{n_nodes}_deltarotE{delta_Erot:g}_rocofthres{rocof_thres:g}"
            + f"_lshare{l_share:g}_maxiter{max_iter}_{resolve_strategy}.pklz"
        )
        if not os.path.exists(fpath_in):
            fnames = os.listdir(path_to_inertia_mitigation_results)
            fnames_lvl = [f for f in fnames if f"Co2{co2_lvl:g}_" in f]
            if len(fnames_lvl) == 1:
                fpath_in = path_to_inertia_mitigation_results + fnames_lvl[0]

            else:
                raise FileNotFoundError(
                    f"No unique inertia placement results for CO2 level {co2_lvl} found. Skipping."
                )

        with gzip.open(fpath_in) as fh_in:
            (
                modified_comp_idx,
                _,
                inertia_placed_ls,
                _,
                resolve_counter,
                still_random_counter,
            ) = pickle.load(fh_in)

        # correcting delta_Erot factor if saved with different one
        delta_Erot_saved = int(fpath_in.split("deltarotE")[1].split("_rocofthres")[0])
        if delta_Erot_saved != delta_Erot:
            print(
                "Correcting delta_Erot factor from", delta_Erot, "to", delta_Erot_saved
            )
            delta_Erot = delta_Erot_saved

        inertia_placed_res_arr = np.array(inertia_placed_ls)

        split_properties_lvl = split_properties[split_properties.co2l == co2_lvl]
        total_loss_share_rocof_lvl = (
            split_properties_lvl.lost_load_share_blackout
            * split_properties_lvl.total_weighting
        ).sum()

        idx_reached_ref_loss = np.where(
            (total_loss_share_rocof_lvl - np.cumsum(inertia_placed_res_arr[:, 3]))
            < total_loss_share_rocof_ref
        )[0][0]

        total_placed_inertia_ref_loss = np.sum(
            inertia_placed_res_arr[:idx_reached_ref_loss, 2] * delta_Erot
        )

        inertia_at_ref_loss_by_lvl[co2_lvl] = total_placed_inertia_ref_loss

    # plt.plot(np.array(co2_lvls) * 100, np.array(inertia_at_ref_loss_by_lvl) / 1000)
    # # invert x axis
    # plt.gca().invert_xaxis()
    # # plt.legend(title="$\\Delta E_0 [MWhs]$")
    # plt.ylabel("Inertia placed [GWs]")
    # plt.xlabel("CO2 level [\% of 1990]")

    return inertia_at_ref_loss_by_lvl


def aggregate_months(casc_dict, show_progress=True, to_month=12, stop_timestamp_str=""):

    keys_all = list(casc_dict.keys())
    if stop_timestamp_str != "":
        keys_all = [
            key
            for key in keys_all
            if dt.strptime(key, "%Y-%m-%d %H:00")
            <= dt.strptime(stop_timestamp_str, "%Y-%m-%d %H:00")
        ]
    month_str_ls = [f"{xx+1:02d}" for xx in range(to_month)]
    nr_splits_per_month = list()
    for month_r in tqdm(month_str_ls, disable=not show_progress):
        month_r_keys = [xx for xx in keys_all if xx[5:7] == month_r]

        if len(month_r_keys) > 0:
            nr_splits = 0
            for key_r in month_r_keys:
                nr_splits += len(casc_dict[key_r])
            nr_splits_per_month.append([int(month_r), nr_splits])

    return np.array(nr_splits_per_month)


def data_line_mitigation_diff_nnlines(
    delta_para=5,
    n_lines_added=(400,),
    save_res=True,
    stop_timestamp_str="2013-01-01 00:00",
):
    """"""

    # Load file
    meta_data_path = path_to_cascade_results_sclopf

    norm_path = meta_data_path + f"system_splits_Co2L0.1_n{nn}.pklz"
    with gzip.open(norm_path, "rb") as fh_in_norm:
        norm_dict = pickle.load(fh_in_norm)

    norm_splits_per_month = aggregate_months(
        norm_dict, stop_timestamp_str=stop_timestamp_str
    )

    mitigate_split_ls = list()
    for nn_lines in tqdm(n_lines_added):
        small_change_path = (
            meta_data_path
            + f"system_splits_Co2L0.1_n{nn}_"
            + f"lineextension_nnlines{nn_lines}"
            + f"_deltanumpara{delta_para:.4g}_"
            + f"stopped_{stop_timestamp_str}.pklz"
        )
        with gzip.open(small_change_path, "rb") as fh_in_r:
            _, vulnerable_edges_r, dict_r = pickle.load(fh_in_r)
        splits_per_month_r = aggregate_months(dict_r)

        res_out_r = (nn_lines, vulnerable_edges_r, splits_per_month_r)
        mitigate_split_ls.append(res_out_r)

    if save_res:
        path_out = (
            path_to_line_extension_mitigation_sclopf
            + f"/cascdes_par_months_deltanumpara{delta_para:.4g}_diffnn_lines.pklz"
        )
        with gzip.open(path_out, "wb") as fh_out:
            pickle.dump((norm_splits_per_month, mitigate_split_ls), fh_out)

    return norm_splits_per_month, mitigate_split_ls


def data_line_mitigation_diff_numpara(nn_lines=10, save_res=True):
    """"""

    # Load file
    meta_data_path = path_to_cascade_results_sclopf

    norm_path = meta_data_path + f"system_splits_Co2L0.1_n{nn}.pklz"
    with gzip.open(norm_path, "rb") as fh_in_norm:
        norm_dict = pickle.load(fh_in_norm)

    norm_splits_per_month = aggregate_months(norm_dict)

    mitigate_split_ls = list()
    for delta_para in tqdm([1, 5]):
        small_change_path = (
            meta_data_path
            + f"system_splits_Co2L0.1_n{nn}_"
            + f"lineextension_nnlines{nn_lines}"
            + f"_deltanumpara{delta_para:.4f}_"
            + "stopped_2013-03-01 00:00.pklz"
        )
        with gzip.open(small_change_path, "rb") as fh_in_r:
            _, vulnerable_edges_r, dict_r = pickle.load(fh_in_r)
        splits_per_month_r = aggregate_months(dict_r, to_month=2)

        res_out_r = (delta_para, vulnerable_edges_r, splits_per_month_r)
        mitigate_split_ls.append(res_out_r)

    if save_res:
        path_out = (
            path_to_line_extension_mitigation_sclopf
            + f"/cascdes_par_months_nnlines{nn_lines}_diffnumpara_lines.pklz"
        )
        print(path_out)
        with gzip.open(path_out, "wb") as fh_out:
            pickle.dump((norm_splits_per_month, mitigate_split_ls), fh_out)

    return norm_splits_per_month, mitigate_split_ls


def plot_most_likely_lines_on_map(savefig=True):

    # Load network
    pypsa_net = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf + "sclopf-elec_s_400_ec_lv1.0_Co2L0.1-2920SEG.nc",
        use_sclopf=True,
    )
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)

    # Load file
    path_out = (
        path_to_line_extension_mitigation_sclopf
        + f"/cascdes_par_months_deltanumpara{1:.4f}_diffnn_lines.pklz"
    )

    with gzip.open(path_out) as fh_in:
        norm_month_ls, diff_para_ls = pickle.load(fh_in)

    vuln_res_dict = dict()
    for ele_r in diff_para_ls[::-1]:
        nn_lines, vulnerable_ls, _ = ele_r
        vuln_res_dict[nn_lines] = vulnerable_ls

    # Plot it
    edge_color_ls = list()
    edge_width_ls = list()

    for ll in nx_graph.edges():

        if (
            ll in vuln_res_dict[10]
            and ll in vuln_res_dict[20]
            and ll in vuln_res_dict[40]
        ):
            edge_color_ls.append(color_ls[0])
            edge_width_ls.append(3)

        elif (
            ll in vuln_res_dict[40]
            and ll in vuln_res_dict[20]
            and ll not in vuln_res_dict[10]
        ):
            edge_color_ls.append(color_ls[1])
            edge_width_ls.append(3)

        elif (
            ll in vuln_res_dict[40]
            and ll not in vuln_res_dict[20]
            and ll not in vuln_res_dict[10]
        ):
            edge_color_ls.append(color_ls[2])
            edge_width_ls.append(3)

        else:
            edge_color_ls.append("gray")
            edge_width_ls.append(1.3)

    fig, ax = plt.subplots(figsize=(8, 6))

    pos_graph = networkx.get_node_attributes(nx_graph, "pos")
    neti = networkx.draw_networkx_edges(
        nx_graph, pos_graph, ax=ax, edge_color=edge_color_ls, width=edge_width_ls
    )

    plt.legend(
        [
            matplotlib.lines.Line2D([0, 1], [0, 1], color=color_ls[idx], lw=3)
            for idx in range(3)
        ],
        [10, 20, 40],
    )

    ax.axis("off")

    if savefig:
        fig_path = path_to_line_extension_mitigation_sclopf + "/lines_on_map.png"
        fig.savefig(fig_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)
    else:
        plt.show()

    return


def plot_bar_histograms_diff_nnlines(
    delta_paras, save_fig=False, log_scale=False, norm_month_diff_para_tuples=None
):
    """plot the number of cascades for different mitigation strategies compared to base case.

    Args:
        delta_paras (_type_): _description_
        save_fig (bool, optional): _description_. Defaults to False.
        log_scale (bool, optional): _description_. Defaults to False.
        norm_month_diff_para_tuples (_type_, optional): _description_. Defaults to None.
    """

    # Load data
    fig, ax = plt.subplots(len(delta_paras), 1, figsize=(10, 8))
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    for idx_para, delta_para in enumerate(delta_paras):
        if norm_month_diff_para_tuples is None:
            path_out = (
                path_to_line_extension_mitigation_sclopf
                + f"/cascdes_par_months_deltanumpara{delta_para:.4g}_diffnn_lines.pklz"
            )
            with gzip.open(path_out) as fh_in:
                norm_month_ls, diff_para_ls = pickle.load(fh_in)
        else:
            norm_month_ls, diff_para_ls = norm_month_diff_para_tuples[idx_para]

        width_bar = 1 / (len(diff_para_ls) + 1) - 0.05

        norm_month_arr = np.array(norm_month_ls)

        bars = ax[idx_para].bar(
            norm_month_arr[:, 0] - width_bar,
            norm_month_arr[:, 1] * 1e-3,
            width_bar,
            label="Normal",
            color=color_ls[0],
        )
        ax[idx_para].bar_label(bars)

        for idx, ele_r in enumerate(diff_para_ls):
            nn_lines, _, splits_per_month = ele_r

            bars = ax[idx_para].bar(
                splits_per_month[:, 0] + idx * width_bar,
                splits_per_month[:, 1] * 1e-3,
                width_bar,
                label="$N_{\\rm lines}=" + f"{nn_lines}" + "$",
                # color=color_ls[idx],
            )
            ax[idx_para].bar_label(bars)

        title_str = (
            "Diff. number of lines for $\\Delta C_{\\rm para}=" + f"{delta_para}$"
        )
        ax[idx_para].set_title(title_str)
        ax[idx_para].set_ylim(top=ax[idx_para].get_ylim()[1] * 1.2)
        if log_scale:
            ax[idx_para].set_yscale("log")

    # Aesthetics
    for ax_r in ax:
        ax_r.set_xlabel("Months", fontsize=18)
        ax_r.set_ylabel("\\# Splits $/ 10^3$", fontsize=18)

    ax[0].legend(ncols=1, fontsize=12, loc="lower left")

    plt.tight_layout()

    if save_fig:
        fig_path = (
            path_to_line_extension_mitigation_sclopf + "/nr_of_cascades_diffnnlines.png"
        )
        fig.savefig(fig_path, bbox_inches="tight")

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_bar_histograms_diff_numpara(save_fig=False):

    # Load data
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for idx_li, nn_lines in enumerate([10, 20, 40]):
        path_out = (
            path_to_line_extension_mitigation_sclopf
            + f"/cascdes_par_months_nnlines{nn_lines}_diffnumpara_lines.pklz"
        )

        with gzip.open(path_out) as fh_in:
            norm_month_ls, diff_para_ls = pickle.load(fh_in)

        width_bar = 1 / (len(diff_para_ls) + 1) - 0.05

        norm_month_arr = np.array(norm_month_ls)

        ax[idx_li].bar(
            norm_month_arr[:, 0] - width_bar,
            norm_month_arr[:, 1] * 1e-3,
            width_bar,
            label="Normal",
            color=color_ls[0],
        )

        for idx, ele_r in enumerate(diff_para_ls):
            delta_para, _, splits_per_month = ele_r

            ax[idx_li].bar(
                splits_per_month[:, 0] + idx * width_bar,
                splits_per_month[:, 1] * 1e-3,
                width_bar,
                label="$\\Delta {C_{\\rm para}=" + f"{delta_para:.1f}" + "}$",
                color=color_ls[idx + 1],
            )

        title_str = "Diff. capacities for $N_{\\rm lines}=" + f"{nn_lines}$"
        ax[idx_li].set_title(title_str)

    # Aesthetics
    for ax_r in ax:
        ax_r.set_xlabel("Months", fontsize=18)
        ax_r.set_ylabel("\\# Splits $/ 10^3$", fontsize=18)

    ax[0].legend(ncols=1, fontsize=12)

    plt.tight_layout()

    if save_fig:
        fig_path = (
            path_to_line_extension_mitigation_sclopf
            + f"/nr_of_cascades_diffnumpara.png"
        )
        fig.savefig(fig_path, bbox_inches="tight")

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_map_marked_edges(nx_graph, pos, vulnerable_edges_r, reinforced_but_triggering):
    fig, ax = plt.subplots(figsize=(10, 10))
    nodes = networkx.draw_networkx_nodes(
        nx_graph, pos=pos, ax=ax, node_color="black", node_size=1
    )

    edges = networkx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        # edgelist=[number_to_edge_dict[line] for line in pair],
        edge_color="grey",
        # style=style,
        width=1.5,
        # edge_cmap=cmap,
        # edge_vmin=np.log10(vmin),
        # edge_vmax=np.log10(vmax)
    )
    edges = networkx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        edgelist=vulnerable_edges_r,
        edge_color="red",
        # style=style,
        width=10,
        # edge_cmap=cmap,
        # edge_vmin=np.log10(vmin),
        # edge_vmax=np.log10(vmax)
    )
    edges = networkx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        edgelist=reinforced_but_triggering,
        edge_color="blue",
        # style=style,
        width=5,
        # edge_cmap=cmap,
        # edge_vmin=np.log10(vmin),
        # edge_vmax=np.log10(vmax)
    )
    return fig, ax
