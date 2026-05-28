#!/usr/bin/env python3
"""Plot nuclear and storage contribution during low renewable snapshots.

Creates histograms per carrier type and CO2 scenario. Each subplot shows
power output for the lowest 10% of aggregate solar+wind generation snapshots
compared against all snapshots.
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
from utils.config import path_to_figures_sclopf
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    save_figure,
)


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


def _rename_carriers_to_nice_name(values: pd.DataFrame, network) -> pd.DataFrame:
    carrier_map = network.carriers.nice_name.to_dict()
    return values.rename(columns=carrier_map)


def _get_renewable_generation(network) -> pd.Series:
    gen = network.generators_t.p.mul(network.generators.sign, axis=1)
    carriers = network.generators["carrier"].str.lower()
    renewable_mask = carriers.str.contains("solar") | carriers.str.contains("wind")
    if not renewable_mask.any():
        return pd.Series(0.0, index=network.snapshots)
    gen_renewable = gen.loc[:, renewable_mask].sum(axis=1)
    return gen_renewable


def _get_nuclear_series(network) -> pd.Series:
    gen = network.generators_t.p.mul(network.generators.sign, axis=1)
    carriers = network.generators["carrier"].str.lower()
    nuclear_mask = carriers.str.contains("nuclear")
    if not nuclear_mask.any():
        return pd.Series(0.0, index=network.snapshots)
    nuclear = gen.loc[:, nuclear_mask].sum(axis=1)
    return nuclear


def _get_storage_series_by_type(network) -> pd.DataFrame:
    if network.storage_units.empty:
        return pd.DataFrame(index=network.snapshots)
    storage = network.storage_units_t.p
    storage_by_carrier = storage.groupby(network.storage_units["carrier"], axis=1).sum()
    storage_by_carrier = _rename_carriers_to_nice_name(storage_by_carrier, network)
    return storage_by_carrier


def _collect_type_series(network) -> dict:
    series_by_type = {}
    series_by_type["Nuclear"] = _get_nuclear_series(network)
    storage_by_carrier = _get_storage_series_by_type(network)
    for carrier in storage_by_carrier.columns:
        series_by_type[carrier] = storage_by_carrier[carrier]
    return series_by_type


def create_low_renewable_histograms(
    n_nodes: int = 600,
    co2_levels=None,
    low_quantile: float = 0.1,
    remove_h2_except_zero: bool = True,
    sharey_rows: bool = False,
    save: bool = True,
):
    if co2_levels is None:
        co2_levels = [0.2, 0.1, 0.05, 0.0]

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

    types = []
    series_by_level = {}
    low_mask_by_level = {}

    for co2l in selected_levels:
        network = networks[co2l]
        renewable = _get_renewable_generation(network)
        threshold = renewable.quantile(low_quantile)
        low_mask_by_level[co2l] = renewable <= threshold

        series_by_type = _collect_type_series(network)
        series_by_level[co2l] = series_by_type
        if not types:
            types = list(series_by_type.keys())

    n_rows = len(types)
    n_cols = len(selected_levels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 1.9 * n_rows),
        sharey="row" if sharey_rows else False,
        sharex="row",
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    color_map = networks[selected_levels[0]].carriers.set_index("nice_name").color
    if "Nuclear" not in color_map.index:
        nuclear_keys = [key for key in color_map.index if "nuclear" in key.lower()]
        if nuclear_keys:
            color_map["Nuclear"] = color_map.loc[nuclear_keys[0]]

    for row_idx, carrier in enumerate(types):
        combined_vals = []
        for co2l in selected_levels:
            series = series_by_level[co2l].get(
                carrier, pd.Series(0.0, index=networks[co2l].snapshots)
            )
            series_vals = series.values / 1000.0
            if np.isfinite(series_vals).any():
                combined_vals.append(series_vals[np.isfinite(series_vals)])
        if combined_vals:
            combined_vals = np.concatenate(combined_vals)
            max_abs = np.max(np.abs(combined_vals))
        else:
            max_abs = 0.0
        if max_abs == 0.0:
            max_abs = 1.0
        bins = np.linspace(-max_abs, max_abs, 41)

        for col_idx, co2l in enumerate(selected_levels):
            ax = axes[row_idx, col_idx]
            carrier_lower = carrier.lower()
            if remove_h2_except_zero and co2l != 0.0:
                if "hydrogen" in carrier_lower or carrier_lower == "h2":
                    ax.set_axis_off()
                    continue
            series = series_by_level[co2l].get(
                carrier, pd.Series(0.0, index=networks[co2l].snapshots)
            )
            series_vals = series.values / 1000.0
            if (
                not np.isfinite(series_vals).any()
                or np.nanmax(np.abs(series_vals)) == 0
            ):
                ax.set_axis_off()
                continue

            low_mask = low_mask_by_level[co2l].values
            low_vals = series_vals[low_mask]
            all_vals = series_vals

            all_vals = all_vals[np.isfinite(all_vals)]
            low_vals = low_vals[np.isfinite(low_vals)]
            total_count = len(all_vals)
            if total_count == 0:
                ax.set_axis_off()
                continue

            all_weights = np.ones_like(all_vals) / total_count
            low_weights = np.ones_like(low_vals) / total_count

            ax.hist(
                all_vals,
                bins=bins,
                weights=all_weights,
                color="black",
                alpha=1,
                linewidth=2,
                label="All",
                histtype="step",
            )
            ax.hist(
                low_vals,
                bins=bins,
                weights=low_weights,
                color=color_map.get(carrier, "#4c72b0"),
                alpha=0.8,
                label="Lowest 10% renewables",
            )
            ax.set_xlim(-max_abs, max_abs)
            # ax.axvline(np.median(all_vals), color="black", linewidth=1.0, alpha=0.7)
            # ax.text(
            #     np.median(all_vals),
            #     0.9 * ax.get_ylim()[1],
            #     f"Median: {np.median(all_vals):.2f}",
            #     color="black",
            #     fontsize=TICK_LABEL_FONTSIZE,
            #     ha="center",
            # )
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
                ax.set_xlabel("Power [GW]", fontsize=AXIS_LABEL_FONTSIZE)

    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(
    #         handles,
    #         labels,
    #         loc="upper right",
    #         fontsize=LEGEND_FONTSIZE,
    #         frameon=False,
    #     )

    fig.tight_layout()

    if save:
        save_figure(fig, "nuclear_storage_low_renewables", path_to_figures_sclopf)

    plt.show()


if __name__ == "__main__":
    create_low_renewable_histograms()
