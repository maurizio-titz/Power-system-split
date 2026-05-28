#!/usr/bin/env python3
"""
Plot mean inertia contribution by carrier type (generators, storage, load)
across all snapshots for each CO2 scenario.
"""

import os
import sys
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("./")

from utils import data_handling, subgraph_evaluation
from utils.config import path_to_figures_sclopf
from utils.data_handling import get_co2_levels, get_actual_co2_level
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TECH_COLORS,
    save_figure,
)


def _compute_mean_inertia_by_type(
    network,
    snet_index=0,
    participation_threshold=0.05,
    load_inertia_constant=0.4,
):
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)

    if (
        hasattr(network, "snapshot_weightings")
        and "generators" in network.snapshot_weightings
    ):
        weights = network.snapshot_weightings["generators"].reindex(network.snapshots)
        weights = weights.fillna(1.0)
    else:
        weights = pd.Series(1.0, index=network.snapshots)

    inertia_weighted_sum = {}
    weight_total = 0.0

    for snapshot, weight in weights.items():
        current_generation = (
            network.generators_t.p.loc[snapshot].mul(network.generators.sign).copy()
        )
        current_storage = network.storage_units_t.p.loc[snapshot]
        current_load = network.loads_t.p.loc[snapshot]

        inertia_by_type = subgraph_evaluation.get_inertia_by_type_subgraph(
            nx_graph,
            network.generators,
            current_generation,
            network.storage_units,
            current_storage,
            network.loads,
            current_load,
            participation_threshold=participation_threshold,
            load_inertia_constant=load_inertia_constant,
        )

        for carrier, value in inertia_by_type.items():
            inertia_weighted_sum[carrier] = inertia_weighted_sum.get(carrier, 0.0) + (
                value * weight
            )
        weight_total += weight

    if weight_total <= 0:
        raise ValueError("Snapshot weights sum to zero; cannot compute mean inertia.")

    return pd.Series(inertia_weighted_sum) / weight_total


def _get_carrier_colors(carriers, network=None):
    tab_colors = list(mpl.colormaps["tab20"].colors)
    colors = dict(TECH_COLORS)
    if network is not None and hasattr(network, "carriers"):
        if "color" in network.carriers.columns:
            colors.update(network.carriers["color"].dropna().to_dict())
    colors.setdefault("load", "#7f7f7f")
    colors.setdefault("rest", "#c7c7c7")
    colors.setdefault("gas", colors.get("CCGT", "#990066"))
    colors["H2"] = "green"
    colors["PHS"]

    for idx, carrier in enumerate(carriers):
        if carrier not in colors:
            colors[carrier] = tab_colors[idx % len(tab_colors)]

    return colors


def _get_carrier_hatches(carriers):
    hatch_cycle = [
        "//",
        "\\\\",
        "..",
        "xx",
        "++",
        "--",
        "oo",
        "OO",
        "**",
        "||",
        "/",
        "\\",
    ]
    hatches = {}
    for idx, carrier in enumerate(carriers):
        hatches[carrier] = hatch_cycle[idx % len(hatch_cycle)]
    return hatches


def create_mean_inertia_by_type_plot(
    n_nodes=600,
    snet_index=0,
    participation_threshold=0.05,
    load_inertia_constant=0.4,
    use_cache=True,
    overwrite_cache=False,
):
    """Create stacked bar plot of mean inertia contribution by carrier type."""

    os.makedirs(path_to_figures_sclopf, exist_ok=True)
    setup_matplotlib_style()

    co2_levels = get_co2_levels(n_nodes)
    cache_dir = os.path.join(path_to_figures_sclopf, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = (
        f"mean_inertia_by_type_n{n_nodes}_snet{snet_index}"
        f"_pt{participation_threshold}_load{load_inertia_constant}_ext{False}.pklz"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    inertia_by_type_by_co2 = None

    if use_cache and (not overwrite_cache) and os.path.exists(cache_path):
        with gzip.open(cache_path, "rb") as fh:
            cache_data = pickle.load(fh)
        cached_levels = cache_data.get("co2_levels")
        if cached_levels is None or list(cached_levels) != list(co2_levels):
            inertia_by_type_by_co2 = None
        else:
            inertia_by_type_by_co2 = cache_data.get("inertia_by_type_by_co2")

    sample_network = None
    if inertia_by_type_by_co2 is None:
        inertia_by_type_by_co2 = {}
        for co2l in co2_levels:
            network = data_handling.load_pypsa_network(
                n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True, lopt=False
            )
            if sample_network is None:
                sample_network = network
            inertia_by_type_by_co2[co2l] = _compute_mean_inertia_by_type(
                network,
                snet_index=snet_index,
                participation_threshold=participation_threshold,
                load_inertia_constant=load_inertia_constant,
            )
    else:
        if co2_levels.size > 0:
            sample_network = data_handling.load_pypsa_network(
                n_nodes=n_nodes,
                co2lvl=co2_levels[0],
                use_sclopf=True,
                lopt=False,
            )

        if use_cache:
            cache_data = {
                "co2_levels": list(co2_levels),
                "inertia_by_type_by_co2": inertia_by_type_by_co2,
            }
            with gzip.open(cache_path, "wb") as fh:
                pickle.dump(cache_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    inertia_df = pd.DataFrame(inertia_by_type_by_co2).fillna(0.0)
    inertia_df = inertia_df.loc[inertia_df.abs().sum(axis=1) > 0]

    # Combine gas carriers
    gas_rows = [carrier for carrier in inertia_df.index if carrier in {"OCGT", "CCGT"}]
    if gas_rows:
        inertia_df.loc["gas"] = inertia_df.loc[gas_rows].sum(axis=0)
        inertia_df = inertia_df.drop(index=gas_rows)

    # Aggregate small contributors into "rest"
    total_by_scenario = inertia_df.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = inertia_df.div(total_by_scenario.replace(0, np.nan), axis=1)
    max_share = shares.max(axis=1).fillna(0.0)
    low_share_carriers = max_share[max_share <= 0.02].index.tolist()
    if low_share_carriers:
        inertia_df.loc["rest"] = inertia_df.loc[low_share_carriers].sum(axis=0)
        inertia_df = inertia_df.drop(index=low_share_carriers)

    inertia_df = inertia_df.loc[
        inertia_df.sum(axis=1).sort_values(ascending=False).index
    ]

    # Convert to GWs for plotting (MWs -> GWs)
    inertia_df_gws = inertia_df / 1000.0

    fig, ax = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(len(co2_levels))
    bottom = np.zeros(len(co2_levels))
    colors = _get_carrier_colors(inertia_df_gws.index, network=sample_network)
    # hatches = _get_carrier_hatches(inertia_df_gws.index)

    for carrier in inertia_df_gws.index:
        values = inertia_df_gws.loc[carrier, co2_levels].values
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=carrier,
            color=colors.get(carrier),
            # hatch=hatches.get(carrier, ""),
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += values

    co2_labels = [
        f"{get_actual_co2_level(level, percent=True)}%" for level in co2_levels
    ]
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(co2_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean inertia contribution [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Mean inertia contribution by type", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.55),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    save_figure(fig, "mean_inertia_by_type", path_to_figures_sclopf)


if __name__ == "__main__":
    create_mean_inertia_by_type_plot()
