#!/usr/bin/env python3
"""
Inertia Mitigation Plot - Inertia placement analysis
Shows inertia needed for mitigation and the spatial distribution of synthetic inertia placement.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Add root path and import utilities
sys.path.append("./")


from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling
from utils.plot_mitigation_strategies import (
    plot_map_inertia_placement_final,
    calc_inertia_placement_ref_loss,
    color1,
    color2,
)


def create_inertia_mitigation_plot():
    """Create inertia mitigation plot."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Get CO2 levels and split properties
    co2ls = get_co2_levels(n_nodes)
    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )

    # Calculate inertia needed by level
    co2_lvl_ref = 0.6
    inertia_needed_by_lvl = {}
    ref_loss_factors = [1]

    for ref_loss_factor in ref_loss_factors:
        inertia_at_ref_loss_by_lvl = calc_inertia_placement_ref_loss(
            n_nodes=600,
            delta_Erot=1000,
            co2_lvl_ref=co2_lvl_ref,
            split_properties=split_properties,
            ref_loss_factor=ref_loss_factor,
            co2_lvls=co2ls,
        )
        inertia_needed_by_lvl[ref_loss_factor] = inertia_at_ref_loss_by_lvl

    # Setup matplotlib
    mpl.style.use("default")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

    colors = color1, color2
    labels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}"]

    co2_lvl_map = 0.1

    f = plt.figure(figsize=(9, 6.2))
    gs_vertical = GridSpec(2, 1, figure=f, height_ratios=[1, 1.5], hspace=0.2)
    gs_lines = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vertical[0], wspace=0.4)
    gs_maps = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vertical[1], wspace=0.1)
    ax_loss_reduction = f.add_subplot(gs_lines[0])
    ax_inertia_needed = f.add_subplot(gs_lines[1])
    axs_maps = [f.add_subplot(gs_maps[i]) for i in range(2)]

    unit = "GWs"
    if unit == "GWs":
        unit_factor = 1e-3
    elif unit == "MWs":
        unit_factor = 1
    else:
        raise ValueError(f"Unit '{unit}' not known!")

    # Inertia needed by level plot
    for i, ref_loss_factor in enumerate(ref_loss_factors):
        inertia_at_ref_loss_by_lvl = inertia_needed_by_lvl[ref_loss_factor]
        label = "$R_{\\textrm{mit}}=ref_loss_factor R_{\\textrm{co2refString}}$"
        label = label.replace(
            "co2refString", str(int(100 * get_actual_co2_level(co2_lvl_ref))) + "\%"
        )
        if ref_loss_factor != 1:
            label = label.replace("ref_loss_factor", str(ref_loss_factor) + "\cdot")
        else:
            label = label.replace("ref_loss_factor", "")

        ax_inertia_needed.plot(
            np.array(list(inertia_at_ref_loss_by_lvl.keys())) * 100,
            np.array(list(inertia_at_ref_loss_by_lvl.values())) * unit_factor,
            label=label,
            color=colors[i],
        )

    ax_inertia_needed.invert_xaxis()
    ax_inertia_needed.legend(title="Reference loss factor")
    ax_inertia_needed.set_ylabel(f"Inertia placed [{unit}]")
    ax_inertia_needed.set_xlabel("CO2 level [\% of 1990]")
    ax_inertia_needed.set_title("Inertia needed to reach reference lost load share")

    ticks = np.array(list(inertia_at_ref_loss_by_lvl.keys())) * 100
    ax_inertia_needed.set_xticks(ticks, labels=[str(int(i)) for i in ticks])
    if ticks[0] < ticks[-1]:
        ax_inertia_needed.invert_xaxis()
    ax_inertia_needed.grid(True)

    # Plot maps and loss reduction curves
    for i, ref_loss_factor in enumerate(ref_loss_factors):
        plot_curve = i == 0
        plot_map_inertia_placement_final(
            axes=(axs_maps[1 - i], ax_loss_reduction),
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
            co2_lvl_ref=co2_lvl_ref,
            ref_loss_factor=ref_loss_factors[i],
            unit=unit,
            color=colors[i],
            plot_curve=plot_curve,
            split_properties=split_properties,
        )

    ax_loss_reduction.set_xlim(
        0, inertia_needed_by_lvl[1][co2_lvl_map] * unit_factor * 1.2
    )

    # Add labels
    for ax_loss_lvl, label in zip([ax_loss_reduction, ax_inertia_needed], labels):
        ax_loss_lvl.text(
            0 - 0.2,
            1 + 0.2,
            label,
            fontsize=36,
            weight="bold",
            verticalalignment="center",
            transform=ax_loss_lvl.transAxes,
        )

    axs_maps[0].text(
        0 - 0.13,
        1,
        labels[2],
        fontsize=36,
        weight="bold",
        verticalalignment="center",
        transform=axs_maps[0].transAxes,
    )

    axs_maps[1].text(
        0 - 0.05,
        1,
        labels[3],
        fontsize=36,
        weight="bold",
        verticalalignment="center",
        transform=axs_maps[1].transAxes,
    )

    for ax_loss_lvl in axs_maps:
        ax_loss_lvl.legend(loc="upper left")

    ax_loss_reduction.set_title(
        f"Loss reduction {int(round(get_actual_co2_level(co2_lvl_map)*100))}\% CO$_2$ level"
    )

    plt.savefig(
        save_path + f"syn_inertia_mitigation_{co2_lvl_map}.pdf", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    create_inertia_mitigation_plot()
