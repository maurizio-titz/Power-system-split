#!/usr/bin/env python3
"""Plot power increments by carrier.

Creates a grid of histograms where rows are carrier types (generation and
storage) and columns are CO2 levels. Each histogram shows the distribution
of snapshot-to-snapshot power increments.
"""

import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./")

from utils import data_handling
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    save_figure,
)


def _rename_carriers_to_nice_name(
    gen_by_carrier: pd.DataFrame, network
) -> pd.DataFrame:
    carrier_map = network.carriers.nice_name.to_dict()
    return gen_by_carrier.rename(columns=carrier_map)


def _escape_latex_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    return (
        label.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _aggregate_generation_carriers(
    gen_by_carrier: pd.DataFrame, aggregate_wind: bool
) -> pd.DataFrame:
    gen = gen_by_carrier.copy()

    gas_cols = [col for col in gen.columns if "gas" in col.lower()]
    if gas_cols:
        gas_sum = gen[gas_cols].sum(axis=1)
        gen = gen.drop(columns=gas_cols, errors="ignore")
        gen["Gas"] = gas_sum

    solar_cols = [col for col in gen.columns if "solar" in col.lower()]
    if solar_cols:
        solar_sum = gen[solar_cols].sum(axis=1)
        gen = gen.drop(columns=solar_cols, errors="ignore")
        gen["Solar"] = solar_sum

    wind_on_cols = [
        col
        for col in gen.columns
        if "onshore" in col.lower() or "onwind" in col.lower()
    ]
    wind_off_cols = [
        col
        for col in gen.columns
        if "offshore" in col.lower() or "offwind" in col.lower()
    ]
    if aggregate_wind:
        wind_cols = wind_on_cols + wind_off_cols
        if wind_cols:
            wind_sum = gen[wind_cols].sum(axis=1)
            gen = gen.drop(columns=wind_cols, errors="ignore")
            gen["Wind"] = wind_sum
    else:
        if wind_on_cols:
            wind_on_sum = gen[wind_on_cols].sum(axis=1)
            gen = gen.drop(columns=wind_on_cols, errors="ignore")
            gen["Onshore Wind"] = wind_on_sum
        if wind_off_cols:
            wind_off_sum = gen[wind_off_cols].sum(axis=1)
            gen = gen.drop(columns=wind_off_cols, errors="ignore")
            gen["Offshore Wind"] = wind_off_sum

    return gen


def _aggregate_generation_series(
    gen_by_carrier: pd.Series, aggregate_wind: bool
) -> pd.Series:
    gen = gen_by_carrier.copy()

    gas_cols = [idx for idx in gen.index if "gas" in idx.lower()]
    if gas_cols:
        gas_sum = gen.loc[gas_cols].sum()
        gen = gen.drop(index=gas_cols, errors="ignore")
        gen.loc["Gas"] = gas_sum

    solar_cols = [idx for idx in gen.index if "solar" in idx.lower()]
    if solar_cols:
        solar_sum = gen.loc[solar_cols].sum()
        gen = gen.drop(index=solar_cols, errors="ignore")
        gen.loc["Solar"] = solar_sum

    wind_on_cols = [
        idx for idx in gen.index if "onshore" in idx.lower() or "onwind" in idx.lower()
    ]
    wind_off_cols = [
        idx
        for idx in gen.index
        if "offshore" in idx.lower() or "offwind" in idx.lower()
    ]
    if aggregate_wind:
        wind_cols = wind_on_cols + wind_off_cols
        if wind_cols:
            wind_sum = gen.loc[wind_cols].sum()
            gen = gen.drop(index=wind_cols, errors="ignore")
            gen.loc["Wind"] = wind_sum
    else:
        if wind_on_cols:
            wind_on_sum = gen.loc[wind_on_cols].sum()
            gen = gen.drop(index=wind_on_cols, errors="ignore")
            gen.loc["Onshore Wind"] = wind_on_sum
        if wind_off_cols:
            wind_off_sum = gen.loc[wind_off_cols].sum()
            gen = gen.drop(index=wind_off_cols, errors="ignore")
            gen.loc["Offshore Wind"] = wind_off_sum

    return gen


def _get_generation_by_carrier_timeseries(
    network, aggregate_wind: bool
) -> pd.DataFrame:
    gen = network.generators_t.p.mul(network.generators.sign, axis=1)
    gen_by_carrier = gen.groupby(network.generators["carrier"], axis=1).sum()
    gen_by_carrier = _rename_carriers_to_nice_name(gen_by_carrier, network)
    gen_by_carrier = _aggregate_generation_carriers(gen_by_carrier, aggregate_wind)
    return gen_by_carrier


def _get_total_generation_by_carrier(network, aggregate_wind: bool) -> pd.Series:
    gen_weighted = network.generators_t.p.mul(
        network.snapshot_weightings.generators, axis=0
    ).mul(network.generators.sign, axis=1)
    gen_by_carrier = gen_weighted.groupby(network.generators["carrier"], axis=1).sum()
    gen_by_carrier = _rename_carriers_to_nice_name(gen_by_carrier, network)
    gen_by_carrier = _aggregate_generation_series(
        gen_by_carrier.sum(axis=0), aggregate_wind
    )
    return gen_by_carrier


def _get_storage_by_carrier_timeseries(network) -> pd.DataFrame:
    if network.storage_units.empty:
        return pd.DataFrame(index=network.snapshots)
    storage = network.storage_units_t.p
    storage_by_carrier = storage.groupby(network.storage_units["carrier"], axis=1).sum()
    storage_by_carrier = _rename_carriers_to_nice_name(storage_by_carrier, network)
    return storage_by_carrier


def _get_total_storage_by_carrier(network) -> pd.Series:
    if network.storage_units.empty:
        return pd.Series(dtype=float)
    storage_weighted = network.storage_units_t.p.mul(
        network.snapshot_weightings.generators, axis=0
    )
    storage_by_carrier = storage_weighted.groupby(
        network.storage_units["carrier"], axis=1
    ).sum()
    storage_by_carrier = _rename_carriers_to_nice_name(storage_by_carrier, network)
    return storage_by_carrier.sum(axis=0)


def _combine_power_frames(
    generation: pd.DataFrame, storage: pd.DataFrame
) -> pd.DataFrame:
    if generation.empty:
        return storage
    if storage.empty:
        return generation
    combined = generation.reindex(
        columns=generation.columns.union(storage.columns)
    ).fillna(0.0)
    combined = combined.add(
        storage.reindex(columns=combined.columns).fillna(0.0), fill_value=0.0
    )
    return combined


def create_generation_increment_histograms(
    n_nodes: int = 600,
    co2_levels=None,
    aggregate_wind: bool = True,
    save: bool = True,
):
    if co2_levels is None:
        co2_levels = [0.6, 0.4, 0.2, 0.1, 0.05, 0.0]

    available_co2_levels = set(get_co2_levels(n_nodes))
    selected_levels = [lvl for lvl in co2_levels if lvl in available_co2_levels]
    if len(selected_levels) != len(co2_levels):
        missing = [lvl for lvl in co2_levels if lvl not in available_co2_levels]
        raise ValueError(f"Missing CO2 levels: {missing}")

    setup_matplotlib_style()
    os.makedirs(path_to_figures_sclopf, exist_ok=True)

    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True, lopt=False
        )
        for co2l in selected_levels
    }

    total_generation = {}
    for co2l in selected_levels:
        total_gen = _get_total_generation_by_carrier(networks[co2l], aggregate_wind)
        total_storage = _get_total_storage_by_carrier(networks[co2l])
        total_generation[co2l] = total_gen.add(total_storage, fill_value=0.0)

    total_generation_df = pd.DataFrame(total_generation).fillna(0.0)

    max_share = (
        total_generation_df / total_generation_df.sum(axis=0).replace(0.0, np.nan)
    ).max(axis=1)
    carrier_mask = max_share > 0.01

    storage_carriers = set()
    for co2l in selected_levels:
        storage_carriers.update(
            _get_total_storage_by_carrier(networks[co2l]).index.tolist()
        )

    carriers = total_generation_df.sum(axis=1).sort_values(ascending=False).index
    carriers = [
        carrier
        for carrier in carriers
        if (carrier in storage_carriers) or (carrier_mask.get(carrier, False))
    ]

    carriers = [
        carrier
        for carrier in carriers
        if "coal" not in carrier.lower() and "lignite" not in carrier.lower()
    ]

    if not carriers:
        raise ValueError("No carriers passed the generation threshold.")

    increments_by_level = {}
    power_by_level = {}
    for co2l in selected_levels:
        gen_by_carrier = _get_generation_by_carrier_timeseries(
            networks[co2l], aggregate_wind
        )
        storage_by_carrier = _get_storage_by_carrier_timeseries(networks[co2l])
        power_by_carrier = _combine_power_frames(gen_by_carrier, storage_by_carrier)
        power_by_carrier = power_by_carrier.reindex(columns=carriers).fillna(0.0)
        power_by_level[co2l] = power_by_carrier
        increments_by_level[co2l] = power_by_carrier.diff().iloc[1:] / 1000.0

    n_rows = len(carriers)
    n_cols = len(selected_levels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 1.8 * n_rows),
        sharex=False,
        sharey="row",
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    color_map = networks[selected_levels[0]].carriers.set_index("nice_name").color
    color_map["Gas"] = "#cc0099"
    if "offwind-dc" in networks[selected_levels[0]].carriers.index:
        offwind_color = networks[selected_levels[0]].carriers.color["offwind-dc"]
        color_map["Offshore Wind"] = offwind_color
        color_map["Wind"] = offwind_color
    if "Wind" not in color_map.index:
        wind_keys = [key for key in color_map.index if "wind" in key.lower()]
        if wind_keys:
            color_map["Wind"] = color_map.loc[wind_keys[0]]
            color_map["Onshore Wind"] = color_map.loc[wind_keys[0]]
    if "Solar" not in color_map.index:
        solar_keys = [key for key in color_map.index if "solar" in key.lower()]
        if solar_keys:
            color_map["Solar"] = color_map.loc[solar_keys[0]]

    for row_idx, carrier in enumerate(carriers):
        combined_vals = np.concatenate(
            [increments_by_level[co2l][carrier].values for co2l in selected_levels]
        )
        combined_vals = combined_vals[np.isfinite(combined_vals)]
        max_abs = np.max(np.abs(combined_vals)) if combined_vals.size else 0.0
        if max_abs == 0.0:
            max_abs = 1.0
        bins = np.linspace(-max_abs, max_abs, 41)
        row_ymax = 0.0

        for col_idx, co2l in enumerate(selected_levels):
            ax = axes[row_idx, col_idx]
            series_vals = increments_by_level[co2l][carrier].values
            series_power = power_by_level[co2l][carrier].values
            if (
                not np.isfinite(series_power).any()
                or np.nanmax(np.abs(series_power)) == 0
            ):
                ax.set_axis_off()
                continue
            counts, _, _ = ax.hist(
                series_vals[np.isfinite(series_vals)],
                bins=bins,
                density=True,
                color=color_map.get(carrier, "#4c72b0"),
                alpha=0.8,
            )
            if counts.size:
                row_ymax = max(row_ymax, np.nanmax(counts))
            ax.axvline(0, color="black", linewidth=1.0, alpha=0.7)
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

            if row_idx == 0:
                ax.set_title(
                    f"{get_actual_co2_level(co2l, percent=True):g}%",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            if col_idx == 0:
                ax.set_ylabel(
                    _escape_latex_label(carrier),
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("Power increment [GW]", fontsize=AXIS_LABEL_FONTSIZE)

        if row_ymax > 0.0:
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if ax.axison:
                    ax.set_ylim(0, row_ymax * 1.05)

    fig.tight_layout()

    if save:
        save_figure(fig, "generation_increments_by_carrier", path_to_figures_sclopf)

    plt.show()


if __name__ == "__main__":
    create_generation_increment_histograms()
