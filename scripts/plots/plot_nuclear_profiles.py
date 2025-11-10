#!/usr/bin/env python3
"""
Split Statistics Plot - Figure 5: split statistics
Creates histograms of power imbalance, rotational energy, and loss of load share distributions.
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
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cycler import cycler

sys.path.append("./")

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure,
)


setup_matplotlib_style()

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

co2_levels = [0.6, 0.4, 0.2, 0.1, 0.05, 0.0]
networks = {
    co2l: data_handling.load_pypsa_network(n_nodes=600, co2lvl=co2l, use_sclopf=True)
    for co2l in co2_levels
}

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# Create a color map for months - winter (blue), spring (green), summer (red), autumn (orange)
month_colors = [
    "#0066CC",
    "#3399FF",
    "#66BB6A",
    "#9CCC65",
    "#FFEB3B",
    "#FF9800",
    "#FF5722",
    "#E91E63",
    "#9C27B0",
    "#673AB7",
    "#3F51B5",
    "#1976D2",
]

# Markers for each month
month_markers = ["o", "s", "^", "v", "D", "p", "*", "h", "H", "<", ">", "X"]

for co2_idx, co2l in enumerate(co2_levels):
    if co2_idx >= len(axes):
        break

    ax = axes[co2_idx]
    n = networks[co2l]

    # Get nuclear generators
    nuclear_gens = n.generators[n.generators.carrier == "nuclear"]

    if len(nuclear_gens) > 0:
        for month_idx in range(12):
            # Filter snapshots for current month
            month_snapshots = n.snapshots[n.snapshots.month == (month_idx + 1)]

            # Calculate nuclear generation for current month
            nuclear_generation_month = (
                n.generators_t.p[nuclear_gens.index].loc[month_snapshots].sum(axis=1)
            )

            # Group by hour and calculate mean
            daily_profile = nuclear_generation_month.groupby(
                month_snapshots.hour
            ).mean()

            ax.plot(
                daily_profile.index,
                daily_profile.values / 1000,
                color=month_colors[month_idx],
                marker=month_markers[month_idx],
                label=months[month_idx],
                linewidth=1.5,
                alpha=0.8,
                markersize=4,
                markevery=1,
            )

    ax.set_title(f"CO₂ level {int(co2l*100)}%", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

    if co2_idx >= 3:  # Bottom row
        ax.set_xlabel("Hour of Day")
    if co2_idx % 3 == 0:  # Left column
        ax.set_ylabel("Nuclear Power [GW]")

# Add legend to the last subplot
axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, ncol=1)

plt.suptitle(
    "Nuclear Power Daily Profiles by CO₂ Level and Month",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
# plt.tight_layout()
plt.show()
save_figure(fig, "nuclear_daily_profiles.pdf", path_to_figures_sclopf)
