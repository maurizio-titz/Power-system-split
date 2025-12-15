#!/usr/bin/env python3
"""
Capacity Plot - Total generation capacity by carrier vs CO2 level
Creates plots showing:
1. Total capacity by carrier across different CO2 levels
2. Change in capacity relative to reference level (0.6)
"""

import itertools
import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("./")

from utils import data_handling
from utils.config import path_to_figures_sclopf
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    save_figure,
    setup_matplotlib_style,
)


def create_capacity_plots():
    """Create capacity plots: total capacity and change relative to reference level."""

    # Setup
    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)
    reference_level = 0.6

    # Load networks
    print("Loading networks...")
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True
        )
        for co2l in co2ls
    }

    # Get reference network for carrier info
    network_ref = networks[reference_level]

    # Calculate capacity by carrier and CO2 level
    print("Calculating capacities by carrier...")
    capacity_by_carrier_and_co2l = pd.DataFrame(
        index=network_ref.carriers.index, columns=co2ls
    )

    for level in co2ls:
        network_lvl = networks[level]
        # Sum nominal capacity (p_nom) by carrier
        capacity_by_carrier = network_lvl.generators.groupby("carrier")["p_nom"].sum()
        capacity_by_carrier = capacity_by_carrier[
            [c for c in capacity_by_carrier.index if not "load" in c]
        ]
        capacity_by_carrier_and_co2l[level] = capacity_by_carrier

    capacity_by_carrier_and_co2l.dropna(axis=0, how="all", inplace=True)

    # Filter carriers based on maximum capacity share
    plot_carrier_threshold = 0.01
    max_carrier_share = (
        capacity_by_carrier_and_co2l / capacity_by_carrier_and_co2l.sum(axis=0)
    ).max(axis=1)
    carrier_mask = max_carrier_share > plot_carrier_threshold
    carrier_mask["OCGT"] = True  # Keep gas carriers to combine them

    dropped_carriers = carrier_mask[~carrier_mask].index
    print(f"Dropping {len(dropped_carriers)} carriers: {list(dropped_carriers)}")
    capacity_by_carrier_and_co2l = capacity_by_carrier_and_co2l[carrier_mask]

    # Aggregate similar carriers
    capacity_by_carrier_and_co2l = capacity_by_carrier_and_co2l.rename(
        index=network_ref.carriers.nice_name
    )

    # Combine gas carriers
    gas_carriers = [c for c in capacity_by_carrier_and_co2l.index if "Gas" in c]
    if len(gas_carriers) > 0:
        gas_capacity = capacity_by_carrier_and_co2l.loc[gas_carriers].sum(axis=0)
        capacity_by_carrier_and_co2l = capacity_by_carrier_and_co2l.drop(gas_carriers)
        capacity_by_carrier_and_co2l.loc["Gas"] = gas_capacity

    # Combine offshore wind carriers
    offwind_carriers = [c for c in capacity_by_carrier_and_co2l.index if "Off" in c]
    if len(offwind_carriers) > 0:
        offwind_capacity = capacity_by_carrier_and_co2l.loc[offwind_carriers].sum(
            axis=0
        )
        capacity_by_carrier_and_co2l = capacity_by_carrier_and_co2l.drop(
            offwind_carriers
        )
        capacity_by_carrier_and_co2l.loc["Offshore Wind"] = offwind_capacity

    # Combine solar carriers
    solar_carriers = [
        c for c in capacity_by_carrier_and_co2l.index if "solar" in c.lower()
    ]
    if len(solar_carriers) > 1:
        solar_capacity = capacity_by_carrier_and_co2l.loc[solar_carriers].sum(axis=0)
        capacity_by_carrier_and_co2l = capacity_by_carrier_and_co2l.drop(solar_carriers)
        capacity_by_carrier_and_co2l.loc["Solar"] = solar_capacity

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Get colors for carriers
    colors_carriers = network_ref.carriers.set_index("nice_name").color
    colors_carriers["Gas"] = "#cc0099"
    colors_carriers["Offshore Wind"] = network_ref.carriers.color["offwind-dc"]

    # Sort CO2 levels for plotting
    co2s = capacity_by_carrier_and_co2l.columns.astype("float")
    index = np.argsort(co2s)

    marker = itertools.cycle((",", "x", ".", "o", "*", "s", "v", "^", "<", ">"))

    # === PLOT 1: Total Capacity ===
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    for carrier, data in capacity_by_carrier_and_co2l.iterrows():
        plot_inds = (data.values)[index] > 0
        co2_vals = (get_actual_co2_level(co2s[index]) * 100)[plot_inds]
        capacity_vals = ((data.values)[index] * 1e-3)[plot_inds]  # Convert to GW

        # Get color for this carrier
        color = colors_carriers.get(carrier, "gray")

        ax1.scatter(
            co2_vals,
            capacity_vals,
            color=color,
            label=carrier,
            s=100,
            marker=next(marker),
        )

        ax1.plot(
            co2_vals,
            capacity_vals,
            color=color,
            alpha=0.8,
        )

    ax1.invert_xaxis()
    ax1.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax1.set_xlabel(r"$\text{CO}_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel(r"Total capacity [GW]", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.grid(True)
    ax1.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    plt.tight_layout()
    save_figure(fig1, save_path, "total_capacity_by_carrier")

    # === PLOT 2: Change Relative to Reference Level ===
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    # Calculate change relative to reference level
    if reference_level in capacity_by_carrier_and_co2l.columns:
        reference_capacities = capacity_by_carrier_and_co2l[reference_level]
        capacity_change = capacity_by_carrier_and_co2l.subtract(
            reference_capacities, axis=0
        )

        marker = itertools.cycle((",", "x", ".", "o", "*", "s", "v", "^", "<", ">"))

        for carrier, data in capacity_change.iterrows():
            plot_inds = (capacity_by_carrier_and_co2l.loc[carrier].values)[index] > 0
            co2_vals = (get_actual_co2_level(co2s[index]) * 100)[plot_inds]
            change_vals = ((data.values)[index] * 1e-3)[plot_inds]  # Convert to GW

            # Get color for this carrier
            color = colors_carriers.get(carrier, "gray")

            ax2.scatter(
                co2_vals,
                change_vals,
                color=color,
                label=carrier,
                s=100,
                marker=next(marker),
            )

            ax2.plot(
                co2_vals,
                change_vals,
                color=color,
                alpha=0.8,
            )

        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax2.invert_xaxis()
        ax2.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        ax2.set_xlabel(
            r"$\text{CO}_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax2.set_ylabel(
            rf"Change in capacity relative to {get_actual_co2_level(reference_level, percent=True)}\% level [GW]",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax2.grid(True)
        ax2.legend(fontsize=LEGEND_FONTSIZE, loc="best")

        plt.tight_layout()
        save_figure(fig2, save_path, "capacity_change_vs_reference")
    else:
        print(f"Warning: Reference level {reference_level} not found in data")

    plt.show()
    print(f"\nPlots saved to {save_path}")


if __name__ == "__main__":
    create_capacity_plots()
