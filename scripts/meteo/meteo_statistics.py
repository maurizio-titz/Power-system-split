#!/usr/bin/env python3
"""
Split Statistics Plot - Figure 5: split statistics
Creates histograms of power imbalance, rotational energy, and loss of load share distributions.
"""

import gzip
import pickle
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

from utils import config
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_meteo_figures,
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


def load_weather_regimes():
    """Load weather regime data for 2013. Index is datetime, column 'weather_regime' contains the lifecycle WR index."""
    weather_regimes = pd.read_csv(
        "/srv/data/mtitz/weather_regimes_2013.csv", index_col=0, parse_dates=True
    )
    return weather_regimes


def create_weather_regime_blackout_plot(
    n_nodes: int = 600,
    co2_levels=None,
    n_blackout_bins: int = 5,
    save_path=path_to_meteo_figures,
    save: bool = True,
):
    """Plot blackout-size frequency by weather regime.

    For each weather regime, plot the relative frequency of blackout sizes.
    """
    setup_matplotlib_style()
    os.makedirs(save_path, exist_ok=True)

    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5",
        index_col=0,
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )

    if "time_stamp" in split_props.index.names:
        timestamps = split_props.index.get_level_values("time_stamp")
    elif "snapshot" in split_props.index.names:
        timestamps = split_props.index.get_level_values("snapshot")
    elif "time_stamp" in split_props.columns:
        timestamps = split_props["time_stamp"]
    elif "snapshot" in split_props.columns:
        timestamps = split_props["snapshot"]
    else:
        raise ValueError(
            "Split properties must include 'time_stamp' or 'snapshot' for regime mapping."
        )

    weather_regimes = load_weather_regimes()
    if "weather_regime" not in weather_regimes.columns:
        raise ValueError("Weather regime file must include 'weather_regime' column.")

    split_props_indexed = split_props.copy()
    split_props_indexed.index = pd.to_datetime(timestamps)
    merged = split_props_indexed.join(
        weather_regimes[["weather_regime"]],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching timestamps between split data and regimes.")

    if co2_levels is None:
        co2_levels = get_co2_levels(n_nodes)

    available_co2_levels = set(get_co2_levels(n_nodes))
    selected_levels = [lvl for lvl in co2_levels if lvl in available_co2_levels]
    if len(selected_levels) != len(co2_levels):
        missing = [lvl for lvl in co2_levels if lvl not in available_co2_levels]
        raise ValueError(f"Missing CO2 levels: {missing}")

    blackout_bins = np.linspace(0, 100, n_blackout_bins + 1).astype(int)
    blackout_centers = 0.5 * (blackout_bins[:-1] + blackout_bins[1:])
    cmap = plt.get_cmap("inferno_r")
    markers = ["o", "s", "D", "^", "v", "."]

    regimes = pd.Categorical(merged["weather_regime"]).categories.tolist()
    n_cols = len(selected_levels)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3.4 * n_cols, 3.5),
        sharey=True,
    )
    if n_cols == 1:
        axes = np.array([axes])

    for col_idx, co2l in enumerate(selected_levels):
        ax = axes[col_idx]
        props = merged[merged.co2l == co2l]
        if props.empty:
            ax.set_axis_off()
            continue

        for i in range(len(blackout_centers)):
            low = blackout_bins[i]
            high = blackout_bins[i + 1]
            color = cmap(
                i / (len(blackout_centers) + 1) + (1 / (len(blackout_centers) + 1))
            )

            counts_by_regime = []
            for regime in regimes:
                subset = props[props.weather_regime == regime]
                if subset.empty:
                    counts_by_regime.append(np.nan)
                    continue
                total_weight = subset.total_weighting.sum()
                if total_weight <= 0:
                    counts_by_regime.append(np.nan)
                    continue
                mask = (subset.lost_load_share_blackout.values * 100 >= low) & (
                    subset.lost_load_share_blackout.values * 100 < high
                )
                bin_weight = subset.loc[mask, "total_weighting"].sum()
                value = bin_weight / total_weight
                counts_by_regime.append(value if value > 0 else np.nan)

            ax.plot(
                regimes,
                counts_by_regime,
                label=rf"{blackout_bins[i]}-{blackout_bins[i+1]}\%",
                alpha=0.85,
                color=color,
                marker=markers[i % len(markers)],
                markersize=4,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Weather regime", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(False)
        ax.set_title(
            f"{get_actual_co2_level(co2l, percent=True):g}%",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        if col_idx == 0:
            ax.set_ylabel("Relative blackout frequency", fontsize=AXIS_LABEL_FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=TICK_LABEL_FONTSIZE,
            frameon=False,
        )

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    if save:
        save_figure(fig, "weather_regime_blackout_frequency", save_path)
    plt.show()


def create_blackout_statistics_plot(
    save_path=path_to_meteo_figures,
    same_corridor=None,
    normalize_by_total_splits=False,
    n_bins=5,
):
    """
    Create a standalone plot showing only the blackout statistics (Number of System Splits).

    Parameters
    ----------
    save_path : str, default path_to_meteo_figures
        Path where the plot will be saved
    same_corridor : bool or None, default None
        If True, only consider splits where both failed lines are in the same corridor. If False, only consider splits where failed lines are in different corridors.
        If None, consider all splits.
    """
    # Setup
    n_nodes = 600
    os.makedirs(save_path, exist_ok=True)

    # Get CO2 levels
    co2ls = get_co2_levels(n_nodes)

    # Load split properties
    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    if same_corridor is not None:
        split_props["same_corridor"] = (
            split_props.init_failure_0 == split_props.init_failure_1
        )
        split_props = split_props[split_props.same_corridor == same_corridor]

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Create figure for standalone blackout statistics with legend axis
    fig = plt.figure(figsize=(8, 3.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.3], wspace=0.17)
    ax_solar = fig.add_subplot(gs[0])
    ax_wind = fig.add_subplot(gs[1], sharey=ax_solar)
    ax_legend = fig.add_subplot(gs[2])

    # Panel: split number and categories (loss of load share distribution)
    cmap = plt.get_cmap("inferno_r")

    bins = np.linspace(0, 100, n_bins + 1).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data = []
    solar_capacity_gw = {}
    wind_capacity_gw = {}
    for co2l in co2ls[::-1]:
        network = data_handling.load_pypsa_network(
            n_nodes=n_nodes,
            co2lvl=co2l,
            use_sclopf=True,
            lopt=config.use_extensions,
        )

        if "p_nom_opt" in network.generators.columns:
            p_nom = network.generators.p_nom_opt
        else:
            p_nom = network.generators.p_nom
        solar_mask = network.generators["carrier"].str.contains(
            "solar", case=False, na=False
        )
        wind_mask = network.generators["carrier"].str.contains(
            "wind", case=False, na=False
        )
        solar_capacity_gw[co2l] = p_nom.loc[solar_mask].sum() / 1000.0
        wind_capacity_gw[co2l] = p_nom.loc[wind_mask].sum() / 1000.0

        vals = (
            split_props[split_props.co2l == co2l].lost_load_share_blackout.values * 100
        )
        counts, _ = np.histogram(
            vals,
            weights=split_props[split_props.co2l == co2l].total_weighting,
            bins=bins,
        )
        data.append(counts)
        if normalize_by_total_splits:
            file_path = (
                data_handling.path_to_grid_data + f"n_2_failures_co2lvl{co2l}.pklz"
            )
            with gzip.open(file_path, "rb") as fh:
                n_2_failures = pickle.load(fh)
            network = data_handling.load_pypsa_network(
                n_nodes=n_nodes,
                co2lvl=co2l,
                use_sclopf=True,
                lopt=config.use_extensions,
            )
            if same_corridor is not None:
                n_2_failures = [
                    n2_failure
                    for n2_failure in n_2_failures
                    if (n2_failure["failures"][0][0] == n2_failure["failures"][1][0])
                    is same_corridor
                ]
                weighted_trigger_count = sum(
                    [initial_failure["weight"] for initial_failure in n_2_failures]
                )
                # incorporating the weighting of the snapshots
                number_of_simulations = (
                    weighted_trigger_count
                    * network.snapshot_weightings.generators.sum()
                )
            else:
                weighted_trigger_count = sum(
                    [initial_failure["weight"] for initial_failure in n_2_failures]
                )
                # incorporating the weighting of the snapshots
                number_of_simulations = (
                    weighted_trigger_count
                    * network.snapshot_weightings.generators.sum()
                )
            total_counts = number_of_simulations
            if total_counts > 0:
                data[-1] = counts / total_counts

    data = np.stack(data)
    data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

    markers = ["o", "s", "D", "^", "v", "."]
    for i, bin_center in enumerate(bin_centers):
        color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
        counts = data.loc[:, bin_center].values
        counts = np.where(counts > 0, counts, np.nan)
        ax_solar.plot(
            [solar_capacity_gw[co2l] for co2l in co2ls[::-1]],
            counts,
            label=rf"{bins[i]}-{bins[i+1]}\%",
            alpha=0.8,
            color=color,
            marker=markers[i % len(markers)],
            markersize=5,
        )
        ax_wind.plot(
            [wind_capacity_gw[co2l] for co2l in co2ls[::-1]],
            counts,
            label=rf"{bins[i]}-{bins[i+1]}\%",
            alpha=0.8,
            color=color,
            marker=markers[i % len(markers)],
            markersize=5,
        )

    ax_solar.set_xlabel("Total solar capacity [GW]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_solar.set_ylabel("Number of System Splits", fontsize=AXIS_LABEL_FONTSIZE)
    ax_solar.grid(False)
    ax_solar.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_solar.set_yscale("log")

    ax_wind.set_xlabel("Total wind capacity [GW]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_wind.grid(False)
    ax_wind.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    # Add legend to separate axis
    h, l = ax_solar.get_legend_handles_labels()
    ax_legend.legend(
        h,
        l,
        title="Share of load not served",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        ncols=1,
        columnspacing=0.5,
    )
    # ax_legend.legend(
    #     h,
    #     l,
    #     title="Share of load not served",
    #     loc="center",
    #     ncols=1,
    #     columnspacing=0.5,
    # )

    ax_legend.axis("off")

    plt.tight_layout()

    # Save figure
    save_name = "renewable_cap_blackout_statistics"
    if same_corridor is True:
        save_name += "_same_corridor"
    elif same_corridor is False:
        save_name += "_different_corridor"
    if normalize_by_total_splits:
        save_name += "_normalized"
    save_figure(fig, save_name, save_path)
    plt.show()


def _get_renewable_generation_series(network) -> dict:
    gen = network.generators_t.p.mul(network.generators.sign, axis=1)
    carriers = network.generators["carrier"].str.lower()
    solar_mask = carriers.str.contains("solar", case=False, na=False)
    wind_mask = carriers.str.contains("wind", case=False, na=False)

    if solar_mask.any():
        solar_gen = gen.loc[:, solar_mask].sum(axis=1)
    else:
        solar_gen = pd.Series(0.0, index=network.snapshots)
    if wind_mask.any():
        wind_gen = gen.loc[:, wind_mask].sum(axis=1)
    else:
        wind_gen = pd.Series(0.0, index=network.snapshots)

    return {"Solar": solar_gen, "Wind": wind_gen}


def create_capacity_factor_grid(
    n_nodes: int = 600,
    co2_levels=[0.6, 0.2, 0.0],
    n_cap_bins: int = 10,
    n_blackout_bins: int = 5,
    normalize_by_capacity_frequency: bool = False,
    save_path=path_to_meteo_figures,
    save: bool = True,
):
    if co2_levels is None:
        co2_levels = get_co2_levels(n_nodes)

    available_co2_levels = set(get_co2_levels(n_nodes))
    selected_levels = [lvl for lvl in co2_levels if lvl in available_co2_levels]
    if len(selected_levels) != len(co2_levels):
        missing = [lvl for lvl in co2_levels if lvl not in available_co2_levels]
        raise ValueError(f"Missing CO2 levels: {missing}")

    setup_matplotlib_style()
    os.makedirs(save_path, exist_ok=True)

    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5",
        index_col=0,
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )

    snapshot_column = "time_stamp"
    split_props[snapshot_column] = pd.to_datetime(
        split_props.index.get_level_values(snapshot_column)
    )

    types = ["Solar", "Wind"]
    n_rows = len(types)
    n_cols = len(selected_levels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 2.4 * n_rows),
        sharey="row",
        sharex="row",
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    cap_bins = np.linspace(0.0, 1.0, n_cap_bins + 1)
    cap_centers = 0.5 * (cap_bins[:-1] + cap_bins[1:])
    blackout_bins = np.linspace(0, 100, n_blackout_bins + 1).astype(int)
    blackout_centers = 0.5 * (blackout_bins[:-1] + blackout_bins[1:])
    cmap = plt.get_cmap("inferno_r")
    markers = ["o", "s", "D", "^", "v", "."]

    for col_idx, co2l in enumerate(selected_levels):
        network = data_handling.load_pypsa_network(
            n_nodes=n_nodes,
            co2lvl=co2l,
            use_sclopf=True,
            lopt=config.use_extensions,
        )
        gen_by_type = _get_renewable_generation_series(network)

        props = split_props[split_props.co2l == co2l].copy()
        if props.empty:
            for row_idx in range(n_rows):
                axes[row_idx, col_idx].set_axis_off()
            continue

        for row_idx, carrier in enumerate(types):
            ax = axes[row_idx, col_idx]
            series = gen_by_type[carrier]
            max_val = (
                np.nanmax(series.values) if np.isfinite(series.values).any() else 0.0
            )
            if max_val <= 0.0:
                ax.set_axis_off()
                continue

            capacity_factor_all = (series / max_val).reindex(network.snapshots)
            capacity_factor = capacity_factor_all.reindex(props[snapshot_column])
            props = props.assign(capacity_factor=capacity_factor.values)
            props = props[np.isfinite(props.capacity_factor)]
            if props.empty:
                ax.set_axis_off()
                continue

            blackout_vals = props.lost_load_share_blackout.values * 100
            weights = props.total_weighting.values

            freq_counts = None
            if normalize_by_capacity_frequency:
                freq_mask = np.isfinite(capacity_factor_all.values)
                freq_values = capacity_factor_all.values[freq_mask]
                freq_weights = network.snapshot_weightings.generators.values[freq_mask]
                freq_counts, _ = np.histogram(
                    freq_values,
                    bins=cap_bins,
                    weights=freq_weights,
                )
                freq_counts = np.where(freq_counts > 0, freq_counts, np.nan)

            for i in range(len(blackout_centers)):
                low = blackout_bins[i]
                high = blackout_bins[i + 1]
                mask = (blackout_vals >= low) & (blackout_vals < high)
                if not np.any(mask):
                    continue
                counts, _ = np.histogram(
                    props.capacity_factor.values[mask],
                    bins=cap_bins,
                    weights=weights[mask],
                )
                if freq_counts is not None:
                    counts = counts / freq_counts
                counts = np.where(counts > 0, counts, np.nan)
                color = cmap(
                    i / (len(blackout_centers) + 1) + (1 / (len(blackout_centers) + 1))
                )
                ax.plot(
                    cap_centers,
                    counts,
                    label=rf"{blackout_bins[i]}-{blackout_bins[i+1]}\%",
                    alpha=0.85,
                    color=color,
                    marker=markers[i % len(markers)],
                    markersize=4,
                )

            ax.set_xlim(0.0, 1.0)
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.set_yscale("log")

            if row_idx == 0:
                ax.set_title(
                    f"{get_actual_co2_level(co2l, percent=True):g}%",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            if col_idx == 0:
                ylabel = f"{carrier} blackout count"
                if normalize_by_capacity_frequency:
                    ylabel = f"{carrier} blackout count / freq"
                ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Capacity factor", fontsize=AXIS_LABEL_FONTSIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=TICK_LABEL_FONTSIZE,
            frameon=False,
        )

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    if save:
        file_name = "renewable_capacity_factor_grid"
        if normalize_by_capacity_frequency:
            file_name += "_normalized_by_frequency"
        save_figure(fig, file_name, save_path)
    plt.show()


if __name__ == "__main__":
    # # create_split_statistics_plot(load_normalization=True, show_blackout_stats=False)
    # for same_corridor_option in [None, True, False]:
    #     for normalize_option in [False, True]:
    #         create_blackout_statistics_plot(
    #             same_corridor=same_corridor_option,
    #             normalize_by_total_splits=normalize_option,
    #         )
    create_blackout_statistics_plot()
    create_capacity_factor_grid(normalize_by_capacity_frequency=True)
    create_weather_regime_blackout_plot()
