#!/usr/bin/env python3
"""
Installed Capacity vs CO2 level plot.
Creates a panel-c style plot but using installed capacities instead of total generation.
"""

import os
import itertools
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Logging
from loguru import logger
import logging
import pypsa
pypsa.network.io.logger.setLevel(logging.ERROR)

sys.path.append("./")
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import path_to_figures_sclopf
from utils import data_handling
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    save_figure,
)


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


def create_installed_capacity_plot(aggregate_wind=False, add_nuclear=False, save=True):
    """Create installed capacity vs CO2 level plot.

    Parameters
    ----------
    aggregate_wind : bool
        If True, combine onshore and offshore wind into a single "Wind" series.
        If False, plot "Onshore Wind" and "Offshore Wind" separately.
    add_nuclear : bool
        If True, include a "Nuclear" series in the plot when available.
    save : bool
        If True, save the figure to the specified path.
    """

    n_nodes = 600
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load networks for all CO2 levels
    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True, lopt=False
        )
        for co2l in co2ls
    }

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Compute installed capacity by carrier and CO2 level
    installed_capacity_by_carrier = pd.DataFrame(
        index=networks[co2ls[0]].carriers.index, columns=co2ls
    )
    carriers = list(installed_capacity_by_carrier.index)

    for level in co2ls:
        n = networks[level]
        if "p_nom_opt" in n.generators.columns:
            p_nom = n.generators.p_nom_opt
        else:
            p_nom = n.generators.p_nom

        current_cap = (
            p_nom.mul(n.generators.sign).groupby(n.generators["carrier"]).sum()
        )
        installed_capacity_by_carrier[level] = current_cap

    installed_capacity_by_carrier.dropna(axis=0, inplace=True)

    # Colors by nice_name
    n_ref = networks[co2ls[0]]
    colors_gens = n_ref.carriers.set_index("nice_name").color
    colors_gens["Offshore Wind"] = n_ref.carriers.color.get("offwind-dc", "#1f77b4")

    # Prepare plotting data
    co2s = installed_capacity_by_carrier.columns.astype("float")
    index = np.argsort(co2s)

    installed_capacity_by_carrier = installed_capacity_by_carrier.rename(
        index=n_ref.carriers.nice_name
    )

    wind_carriers = [
        carrier for carrier in installed_capacity_by_carrier.index if "Wind" in carrier
    ]
    onshore_wind_carriers = [
        carrier
        for carrier in wind_carriers
        if "onshore" in carrier.lower() or "onwind" in carrier.lower()
    ]
    offshore_wind_carriers = [
        carrier
        for carrier in wind_carriers
        if "offshore" in carrier.lower() or "offwind" in carrier.lower()
    ]
    solar_carriers = [
        carrier
        for carrier in installed_capacity_by_carrier.index
        if "solar" in carrier.lower()
    ]
    nuclear_carriers = [
        carrier
        for carrier in installed_capacity_by_carrier.index
        if "nuclear" in carrier.lower()
    ]

    wind_gen = (
        installed_capacity_by_carrier.loc[wind_carriers].sum(axis=0)
        if wind_carriers
        else pd.Series(0, index=installed_capacity_by_carrier.columns)
    )
    onshore_wind_gen = (
        installed_capacity_by_carrier.loc[onshore_wind_carriers].sum(axis=0)
        if onshore_wind_carriers
        else pd.Series(0, index=installed_capacity_by_carrier.columns)
    )
    offshore_wind_gen = (
        installed_capacity_by_carrier.loc[offshore_wind_carriers].sum(axis=0)
        if offshore_wind_carriers
        else pd.Series(0, index=installed_capacity_by_carrier.columns)
    )
    solar_gen = (
        installed_capacity_by_carrier.loc[solar_carriers].sum(axis=0)
        if solar_carriers
        else pd.Series(0, index=installed_capacity_by_carrier.columns)
    )
    nuclear_gen = (
        installed_capacity_by_carrier.loc[nuclear_carriers].sum(axis=0)
        if nuclear_carriers
        else pd.Series(0, index=installed_capacity_by_carrier.columns)
    )

    if aggregate_wind:
        installed_capacity_plot = pd.DataFrame({"Wind": wind_gen, "Solar": solar_gen}).T

        wind_color_source = wind_carriers[0] if wind_carriers else "Offshore Wind"
        solar_color_source = solar_carriers[0] if solar_carriers else "Solar"
        colors_gens.loc["Wind"] = colors_gens.get(wind_color_source, "#1f77b4")
        colors_gens.loc["Solar"] = colors_gens.get(solar_color_source, "#ff7f0e")
    else:
        installed_capacity_plot = pd.DataFrame(
            {
                "Onshore Wind": onshore_wind_gen,
                "Offshore Wind": offshore_wind_gen,
                "Solar": solar_gen,
            }
        ).T

        colors_gens.loc["Onshore Wind"] = colors_gens.get("Onshore Wind", "#1f77b4")
        colors_gens.loc["Offshore Wind"] = colors_gens.get("Offshore Wind", "#2ca02c")
        colors_gens.loc["Solar"] = colors_gens.get("Solar", "#ff7f0e")

    if add_nuclear:
        installed_capacity_plot.loc["Nuclear"] = nuclear_gen
        colors_gens.loc["Nuclear"] = colors_gens.get("Nuclear", "#7f7f7f")

    # Plot
    f, ax = plt.subplots(figsize=(5, 5))
    marker = itertools.cycle((",", "x", ".", "o", "*"))

    for ind, data in installed_capacity_plot.iterrows():
        jit = 0
        plot_inds = np.ones(len(data.values), dtype=bool)
        ax.scatter(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-3)[plot_inds],
            color=colors_gens.loc[ind],
            label=ind,
            s=100,
            marker=next(marker),
        )

        ax.plot(
            (get_actual_co2_level(co2s[index]) * 100 + jit)[plot_inds],
            ((data.values)[index] * 1e-3)[plot_inds],
            color=colors_gens.loc[ind],
            alpha=0.8,
        )

    ax.invert_xaxis()
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel(r"$\text{CO}_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Installed capacity [GW]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True)
    ax.yaxis.offsetText.set_fontsize(TICK_LABEL_FONTSIZE)

    # Legend
    h, l = ax.get_legend_handles_labels()
    l = [label.capitalize() for label in l]
    ax.legend(h, l, fontsize=LEGEND_FONTSIZE, ncol=2, loc="best")

    if save:
        file_name = "installed_capacity_vs_co2"
        if add_nuclear:
            file_name += "_with_nuclear"
        if aggregate_wind:
            file_name += "_aggregated_wind"
        save_figure(f, file_name, save_path)
    plt.show()


if __name__ == "__main__":
    create_installed_capacity_plot(aggregate_wind=False, add_nuclear=False, save=True)
