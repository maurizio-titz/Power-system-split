#!/usr/bin/env python3
"""
Mean Nodal Load Map
Plots mean nodal load per bus for a selected CO2 level.
"""

import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.append("./")

from utils import data_handling
from utils.config import path_to_figures_sclopf, use_extensions
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    MAP_XLIM,
    MAP_YLIM,
    MAP_LINE_WIDTH,
    MAP_LINK_WIDTH,
    save_figure,
)


def _get_snapshot_weights(network):
    if hasattr(network, "snapshot_weightings"):
        if "loads" in network.snapshot_weightings:
            return network.snapshot_weightings.loads
        if "generators" in network.snapshot_weightings:
            return network.snapshot_weightings.generators
    return None


def create_mean_nodal_load_map(n_nodes=600, co2l=0.0):
    setup_matplotlib_style()

    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    network = data_handling.load_pypsa_network(
        n_nodes=n_nodes,
        co2lvl=co2l,
        use_sclopf=True,
        lopt=use_extensions,
    )

    weights = _get_snapshot_weights(network)
    loads_t = network.loads_t.p

    if weights is not None:
        weighted = loads_t.mul(weights, axis="index")
        mean_load_by_load = weighted.sum(axis=0) / weights.sum()
    else:
        mean_load_by_load = loads_t.mean(axis=0)

    mean_load_by_bus = mean_load_by_load.groupby(network.loads.bus).sum()

    max_load = mean_load_by_bus.max()
    if max_load <= 0:
        max_load = 1.0

    max_node_size = 0.5
    bus_sizes = mean_load_by_bus / max_load * max_node_size

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    network.plot(
        bus_sizes=bus_sizes,
        bus_colors="tab:blue",
        line_colors="black",
        link_colors="black",
        line_widths=MAP_LINE_WIDTH,
        link_widths=MAP_LINK_WIDTH,
        ax=ax,
    )

    ax.set_extent([MAP_XLIM[0], MAP_XLIM[1], MAP_YLIM[0], MAP_YLIM[1]])
    ax.set_aspect("equal")
    ax.set_title("Mean nodal load", fontsize=TITLE_FONTSIZE)

    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    save_figure(fig, f"mean_nodal_load_map_co2l{co2l}", save_path)
    plt.show()


if __name__ == "__main__":
    create_mean_nodal_load_map()
