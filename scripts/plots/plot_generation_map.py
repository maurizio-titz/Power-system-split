#!/usr/bin/env python3
"""
Generation Map Plot - Figure 2: generation map
Creates maps showing generation mix and total annual generation versus CO2 levels.
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
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import itertools

# Add root path and import utilities
root_path = "../"
os.chdir(root_path)
sys.path.append(root_path)

from utils.visualization import get_actual_co2_level, get_co2_levels
from utils.config import path_to_pypsa_network_sclopf, path_to_figures_sclopf
from utils import data_handling


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


def create_generation_map():
    """Create generation map plot."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load network graph and node positions
    snet_index = 0
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True
        )
        for co2l in co2ls
    }

    # Calculate generation by carrier and CO2 level
    generation_by_carrier_and_co2l = pd.DataFrame(
        index=networks[0.6].carriers.index, columns=co2ls
    )
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

    # Setup matplotlib
    mpl.style.use("default")
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

    target_levels = [0.6, 0.0]

    # Create figure
    f = plt.figure(figsize=(24, 12))
    outer = GridSpec(1, 2, width_ratios=[2, 1])
    gs_left = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0], wspace=-0.0, hspace=0.05
    )
    gs_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[1, 0.6]
    )

    ax1 = [
        f.add_subplot(gs_left[0, i], projection=ccrs.PlateCarree()) for i in range(2)
    ]
    ax2 = [
        f.add_subplot(gs_left[1, i], projection=ccrs.PlateCarree()) for i in range(2)
    ]
    axs_primary = [ax1, ax2]
    ax3 = f.add_subplot(gs_right[0, 0])
    ax3_legend = f.add_subplot(gs_right[1, 0])

    labels = [
        [r"\textbf{a}", r"", r"\textbf{c}"],
        [r"\textbf{b}", "", r"\textbf{d}", "", ""],
    ]

    # Panel a and b - Maps
    vmax = 3.2e2
    months = [7, 12]

    for i, target_level in enumerate(target_levels):
        n = data_handling.load_pypsa_network(target_level, n_nodes, True)
        for ii, month in enumerate(months):
            axs_primary[i][ii].set_ylim([0, vmax])
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

            n.plot(
                bus_sizes=current_gen / 7e6,
                line_colors="black",
                link_colors="black",
                link_widths=0.5,
                line_widths=0.5,
                ax=axs_primary[i][ii],
            )

            if i == 0:
                axs_primary[i][ii].set_title(
                    current_snapshots.month_name()[0], fontsize=24
                )

            axs_primary[i][ii].text(
                0 - 0.1,
                1 + 0.05,
                labels[i][ii],
                fontsize=36,
                weight="bold",
                verticalalignment="center",
                transform=axs_primary[i][ii].transAxes,
            )

        axs_primary[i][0].text(
            0 - 0.15,
            0.4,
            rf"CO$_2$={get_actual_co2_level(target_level, percent=True)}\% ",
            fontsize=24,
            verticalalignment="center",
            transform=axs_primary[i][0].transAxes,
        )

    # Panel c - Generation vs CO2 level
    marker = itertools.cycle((",", "x", ".", "o", "*"))

    colors_gens = network.carriers.set_index("nice_name").color
    colors_gens["Open-Cycle Gas"] = "#cc0099"
    colors_gens["Gas"] = "#cc0099"

    co2s = generation_by_carrier_and_co2l.columns.astype("float")
    index = np.argsort(co2s)

    generation_by_carrier_and_co2l = generation_by_carrier_and_co2l.rename(
        index=network.carriers.nice_name
    )
    gas_gen = (
        generation_by_carrier_and_co2l.loc["Combined-Cycle Gas"]
        + generation_by_carrier_and_co2l.loc["Open-Cycle Gas"]
    )
    generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l.copy()

    # Remove individual gas carriers and combine them
    for col in generation_by_carrier_and_co2l_plot.index:
        if "Gas" in col:
            generation_by_carrier_and_co2l_plot.drop(col, axis=0, inplace=True)
    generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l_plot.rename(
        index=network.carriers.nice_name
    )
    generation_by_carrier_and_co2l_plot.loc["Gas"] = gas_gen

    for ind, data in generation_by_carrier_and_co2l_plot.iterrows():
        jit = 0
        plot_inds = (data.values)[index] > 0
        ax3.scatter(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-6)[plot_inds],
            color=colors_gens.loc[ind],
            label=ind,
            s=100,
            marker=next(marker),
        )

        ax3.plot(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-6)[plot_inds],
            color=colors_gens.loc[ind],
            alpha=0.8,
        )

    ax3.invert_xaxis()
    ax3.tick_params(axis="both", which="both", labelsize=24)
    ax3.set_xlabel(r"$\text{CO}_2$ level [\% of 1990]", fontsize=24)
    ax3.set_ylabel(r"Total annual generation [TWh]", fontsize=24)
    ax3.grid(True)
    ax3.yaxis.offsetText.set_fontsize(24)
    ax3.text(
        0 - 0.15,
        1 + 0.05,
        labels[0][2],
        fontsize=36,
        weight="bold",
        verticalalignment="center",
        transform=ax3.transAxes,
    )
    ax3.set_ylim([-2e1, 12.4e2])

    # Legend
    h, l = ax3.get_legend_handles_labels()
    ax3_legend.legend(h, l, borderaxespad=0, fontsize=20.5, ncol=2)
    ax3_legend.axis("off")

    plt.tight_layout()
    plt.savefig(save_path + "generation_mix_versus_co2level.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_generation_map()
