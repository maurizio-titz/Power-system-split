#!/usr/bin/env python3
"""
Plot total placed synthetic inertia by country to reach reference scenario.
"""

import gzip
import os
import pickle
import sys
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./")

from utils import data_handling
from utils.config import (
    path_to_figures_sclopf,
    path_to_inertia_mitigation_results_sclopf,
    path_to_vis_results_sclopf,
)
from utils.mitigation_visualization import (
    calc_inertia_placement_ref_loss,
    plot_map_inertia_placement_new,
)
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    save_figure,
    get_co2_color,
)


annualized_cost_per_MWs_max_DE = 888.5  # € per MWs/a of synthetic inertia
annualized_cost_per_GWs_max = annualized_cost_per_MWs_max_DE * 1e3  # € per GWs/a


_COUNTRY_3_TO_2 = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "CHE": "CH",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
}


def _country_from_bus_name(bus_name: str) -> str:
    bus_str = str(bus_name)
    if len(bus_str) >= 2 and bus_str[:2].isalpha():
        code = bus_str[:2].upper()
        if code == "UK":
            return "GB"
        return code
    if len(bus_str) >= 3 and bus_str[:3].isalpha():
        code3 = bus_str[:3].upper()
        return _COUNTRY_3_TO_2.get(code3, code3)
    return "UNK"


def _get_bus_country_map(network) -> Dict[str, str]:
    if "country" in network.buses.columns:
        countries = network.buses["country"].fillna("UNK")
        return {bus: str(country) for bus, country in countries.items()}

    return {bus: _country_from_bus_name(bus) for bus in network.buses.index}


def _aggregate_inertia_by_country(
    node_list: Iterable[str],
    inertia_by_node: np.ndarray,
    bus_country_map: Dict[str, str],
) -> pd.Series:
    country_vals = {}
    for idx, bus in enumerate(node_list):
        country = bus_country_map.get(bus, "UNK")
        country_vals[country] = country_vals.get(country, 0.0) + inertia_by_node[idx]
    return pd.Series(country_vals).sort_values(ascending=False)


def _calc_target_value(
    split_properties: pd.DataFrame,
    co2_lvl_ref: float,
    target: str,
    blackoutthreshold: float,
) -> float:
    split_properties_ref = split_properties[split_properties.co2l == co2_lvl_ref]
    if target == "total_loss":
        return (
            split_properties_ref.lost_load_share_blackout
            * split_properties_ref.total_weighting
        ).sum()
    if target == "num_GSS":
        return (
            (split_properties_ref.lost_load_share_blackout > blackoutthreshold)
            * split_properties_ref.total_weighting
        ).sum()
    if target == "lost_load_GSS":
        return (
            split_properties_ref.lost_load_share_blackout[
                split_properties_ref.lost_load_share_blackout > blackoutthreshold
            ]
            * split_properties_ref.total_weighting.sum()
        ).sum()
    raise ValueError(f"Target '{target}' not known!")


def _load_or_calc_inertia_ref_loss_results(
    *,
    n_nodes: int,
    co2_lvl_ref: float,
    target: str,
    blackoutthreshold: float,
    delta_Erot: float,
    max_iter: int,
    resolve_strategy: str,
    co2_lvls,
    split_properties: pd.DataFrame,
    ref_loss_factor: float,
):
    fname = (
        f"inertia_placement_results_refLoss_allCo2lvls_n{n_nodes}_{target}"
        f"_blackoutthres{blackoutthreshold}.pkl"
    )
    cache_dir = os.path.join(path_to_inertia_mitigation_results_sclopf, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = (
        f"inertia_refLoss_cache_n{n_nodes}_{target}"
        f"_blackout{blackoutthreshold}_ref{co2_lvl_ref}_deltaErot{delta_Erot}.pklz"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)

    legacy_path = path_to_inertia_mitigation_results_sclopf + fname
    if os.path.exists(legacy_path):
        with gzip.open(legacy_path, "rb") as f:
            return pickle.load(f)

    inertia_needed_by_ref_loss = {}
    mitigation_curves_by_ref_loss = {}
    steps_to_reach_target_by_ref_loss = {}

    (
        res_tuples_by_lvl,
        mitigation_curve,
        inertia_at_ref_loss_by_lvl,
        steps_to_reach_target_by_lvl,
        delta_Erot_by_lvl,
    ) = calc_inertia_placement_ref_loss(
        n_nodes=n_nodes,
        co2_lvl_ref=co2_lvl_ref,
        ref_loss_factor=ref_loss_factor,
        target=target,
        max_iter=max_iter,
        delta_Erot=delta_Erot,
        rocof_thres=1,
        l_share=0.0,
        resolve_strategy=resolve_strategy,
        co2_lvls=co2_lvls,
        use_sclopf=True,
        split_properties=split_properties,
        blackoutthreshold=blackoutthreshold,
    )

    inertia_needed_by_ref_loss[ref_loss_factor] = inertia_at_ref_loss_by_lvl
    mitigation_curves_by_ref_loss[ref_loss_factor] = mitigation_curve
    steps_to_reach_target_by_ref_loss[ref_loss_factor] = steps_to_reach_target_by_lvl

    results = (
        res_tuples_by_lvl,
        inertia_needed_by_ref_loss,
        mitigation_curves_by_ref_loss,
        steps_to_reach_target_by_ref_loss,
        delta_Erot_by_lvl,
    )

    with gzip.open(cache_path, "wb") as f:
        pickle.dump(results, f)
    with gzip.open(legacy_path, "wb") as f:
        pickle.dump(results, f)

    return results


def create_inertia_by_country_plot(
    co2_lvl_map: float = 0.2,
    co2_lvl_ref: float = 0.6,
    n_nodes: int = 600,
    target: str = "num_GSS",
    blackoutthreshold: float = 0.0,
    delta_Erot: float = 5000,
    resolve_strategy: str = "random",
    max_iter: int = 10000,
    plot_inertia_map: bool = False,
    ref_loss_factor: float = 1.0,
):
    """Create bar plot of total placed synthetic inertia by country to reach reference scenario."""
    if co2_lvl_map == co2_lvl_ref:
        raise ValueError(
            "co2_lvl_map must differ from co2_lvl_ref to compute mitigation."
        )

    os.makedirs(path_to_figures_sclopf, exist_ok=True)

    setup_matplotlib_style()

    network = data_handling.load_pypsa_network(co2_lvl_map, n_nodes, use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    node_list = list(nx_graph)
    bus_country_map = _get_bus_country_map(network)

    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5",
        index_col=0,
    )

    co2_lvls = get_co2_levels(n_nodes)
    (
        res_tuples_by_lvl,
        _inertia_needed_by_ref_loss,
        mitigation_curves_by_ref_loss,
        steps_to_reach_target_by_ref_loss,
        delta_Erot_by_lvl,
    ) = _load_or_calc_inertia_ref_loss_results(
        n_nodes=n_nodes,
        co2_lvl_ref=co2_lvl_ref,
        target=target,
        blackoutthreshold=blackoutthreshold,
        delta_Erot=delta_Erot,
        max_iter=max_iter,
        resolve_strategy=resolve_strategy,
        co2_lvls=co2_lvls,
        split_properties=split_properties,
        ref_loss_factor=ref_loss_factor,
    )

    if co2_lvl_map not in res_tuples_by_lvl:
        raise ValueError(
            f"No inertia placement results found for co2 level {co2_lvl_map}."
        )

    res_tuple = res_tuples_by_lvl[co2_lvl_map]
    inertia_placed_ls = res_tuple[2]
    idx_reached_target = steps_to_reach_target_by_ref_loss[ref_loss_factor][co2_lvl_map]
    delta_Erot_used = delta_Erot_by_lvl[co2_lvl_map]

    inertia_node_idx = np.array(inertia_placed_ls)[:idx_reached_target, 1:3]
    inertia_by_node = np.zeros(len(node_list))
    for node_idx, delta_factor in inertia_node_idx:
        inertia_by_node[int(node_idx)] += delta_factor

    inertia_by_node = inertia_by_node * delta_Erot_used  # MWs

    inertia_by_country = _aggregate_inertia_by_country(
        node_list, inertia_by_node, bus_country_map
    )

    inertia_by_country_gws = (inertia_by_country / 1e3).sort_index()

    if plot_inertia_map:
        fig, (ax_bar, ax_map) = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        ax_curve = fig.add_axes([0, 0, 0.01, 0.01])
        ax_curve.set_axis_off()
    else:
        fig, ax_bar = plt.subplots(figsize=(12, 6))

    inertia_by_country_gws.plot(
        kind="bar",
        ax=ax_bar,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )

    co2_pct = int(round(get_actual_co2_level(co2_lvl_map, percent=True)))
    ref_pct = int(round(get_actual_co2_level(co2_lvl_ref, percent=True)))
    ax_bar.set_title(
        f"Synthetic inertia to reach reference scenario\nCO$_2$ level={co2_pct}\\% → {ref_pct}\\%",
        fontsize=TITLE_FONTSIZE,
    )
    ax_bar.set_ylabel(
        "Total placed synthetic inertia [GWs]", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax_bar.set_xlabel("Country", fontsize=AXIS_LABEL_FONTSIZE)
    ax_bar.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax_bar.grid(True, axis="y", alpha=0.3)

    if plot_inertia_map:
        target_value = _calc_target_value(
            split_properties, co2_lvl_ref, target, blackoutthreshold
        )
        plot_map_inertia_placement_new(
            co2_lvl=co2_lvl_map,
            idx_reached_target=idx_reached_target,
            res_tuple=res_tuple,
            mitigation_curve=mitigation_curves_by_ref_loss[ref_loss_factor][
                co2_lvl_map
            ],
            target_value=target_value,
            nn=n_nodes,
            max_iter=max_iter,
            delta_Erot=delta_Erot_used,
            resolve_strategy=resolve_strategy,
            axes=(ax_map, ax_curve),
            plot_curve=False,
            annualized_cost_per_GWs_max=annualized_cost_per_GWs_max,
        )
        ax_curve.set_visible(False)
        ax_map.set_title(
            f"Inertia placement map\nCO$_2$ level={co2_pct}\\%",
            fontsize=TITLE_FONTSIZE,
        )
        ax_map.set_aspect("equal")

    fig.tight_layout()

    fname = f"inertia_by_country_Co2L{co2_lvl_map}_ref{co2_lvl_ref}_{target}"
    if blackoutthreshold is not None and blackoutthreshold > 0.0:
        fname += f"_blackoutThres{blackoutthreshold}"
    save_figure(fig, path_to_figures_sclopf, fname)
    plt.show()


def create_inertia_by_country_all_co2_plot(
    co2_lvl_ref: float = 0.6,
    n_nodes: int = 600,
    target: str = "num_GSS",
    blackoutthreshold: float = 0.0,
    delta_Erot: float = 5000,
    resolve_strategy: str = "random",
    max_iter: int = 10000,
    ref_loss_factor: float = 1.0,
    co2_lvls: Iterable[float] = None,
):
    """Plot total placed synthetic inertia per country across all CO2 levels."""
    os.makedirs(path_to_figures_sclopf, exist_ok=True)

    setup_matplotlib_style()

    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5",
        index_col=0,
    )
    if co2_lvls is None:
        co2_lvls = list(get_co2_levels(n_nodes))
    (
        res_tuples_by_lvl,
        _inertia_needed_by_ref_loss,
        _mitigation_curves_by_ref_loss,
        steps_to_reach_target_by_ref_loss,
        delta_Erot_by_lvl,
    ) = _load_or_calc_inertia_ref_loss_results(
        n_nodes=n_nodes,
        co2_lvl_ref=co2_lvl_ref,
        target=target,
        blackoutthreshold=blackoutthreshold,
        delta_Erot=delta_Erot,
        max_iter=max_iter,
        resolve_strategy=resolve_strategy,
        co2_lvls=co2_lvls,
        split_properties=split_properties,
        ref_loss_factor=ref_loss_factor,
    )

    co2_lvls_plot = [lvl for lvl in co2_lvls if lvl != co2_lvl_ref]
    co2_lvls_plot = sorted(co2_lvls_plot, reverse=True)

    inertia_by_country_df = None

    for co2_lvl in co2_lvls_plot:
        if co2_lvl not in res_tuples_by_lvl:
            raise ValueError(
                f"No inertia placement results found for co2 level {co2_lvl}."
            )

        network = data_handling.load_pypsa_network(co2_lvl, n_nodes, use_sclopf=True)
        nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
        node_list = list(nx_graph)
        bus_country_map = _get_bus_country_map(network)

        res_tuple = res_tuples_by_lvl[co2_lvl]
        inertia_placed_ls = res_tuple[2]
        idx_reached_target = steps_to_reach_target_by_ref_loss[ref_loss_factor][co2_lvl]
        delta_Erot_used = delta_Erot_by_lvl[co2_lvl]

        inertia_node_idx = np.array(inertia_placed_ls)[:idx_reached_target, 1:3]
        inertia_by_node = np.zeros(len(node_list))
        for node_idx, delta_factor in inertia_node_idx:
            inertia_by_node[int(node_idx)] += delta_factor

        inertia_by_node = inertia_by_node * delta_Erot_used  # MWs
        inertia_by_country = _aggregate_inertia_by_country(
            node_list, inertia_by_node, bus_country_map
        )
        inertia_by_country_gws = (inertia_by_country / 1e3).sort_index()

        col_name = int(round(get_actual_co2_level(co2_lvl, percent=True)))
        series = inertia_by_country_gws.rename(col_name)
        if inertia_by_country_df is None:
            inertia_by_country_df = series.to_frame()
        else:
            inertia_by_country_df = inertia_by_country_df.join(series, how="outer")

    if inertia_by_country_df is None or inertia_by_country_df.empty:
        raise ValueError("No inertia placement results available for CO2 levels.")

    inertia_by_country_df = inertia_by_country_df.fillna(0.0)
    inertia_by_country_df = inertia_by_country_df.sort_index()
    inertia_by_country_df = inertia_by_country_df.reindex(
        sorted(inertia_by_country_df.columns, reverse=True), axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x_positions = np.arange(len(inertia_by_country_df.index))
    for col in inertia_by_country_df.columns:
        ax.plot(
            x_positions,
            inertia_by_country_df[col].values,
            label=f"{int(col)}\\%",
            color=get_co2_color(col / 100),  # Normalize col to [0, 1] for colormap
            linewidth=2,
            alpha=0.8,
        )

    ref_pct = int(round(get_actual_co2_level(co2_lvl_ref, percent=True)))
    ax.set_title(
        f"Synthetic inertia to reduce number of GSS \nto reference scenario ({ref_pct}\\%)",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_xlabel("Country", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Total placed synthetic inertia [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(inertia_by_country_df.index, rotation=45, ha="right")
    ax.legend(
        title="CO$_2$ level [\\% of 1990]",
        ncol=3,
        fontsize=TICK_LABEL_FONTSIZE - 2,
        frameon=True,
        fancybox=True,
    )

    fig.tight_layout()

    fname = f"inertia_by_country_allCo2_ref{co2_lvl_ref}_{target}"
    if blackoutthreshold is not None and blackoutthreshold > 0.0:
        fname += f"_blackoutThres{blackoutthreshold}"
    save_figure(fig, path_to_figures_sclopf, fname)
    plt.show()


if __name__ == "__main__":
    create_inertia_by_country_all_co2_plot(
        co2_lvl_ref=0.6,
        co2_lvls=[0.4, 0.2, 0.0],
        target="num_GSS",
        blackoutthreshold=0.8,
    )
    # for co2l_map in [0.2, 0.0]:
    #     create_inertia_by_country_plot(
    #         co2_lvl_map=co2l_map,
    #         co2_lvl_ref=0.6,
    #         n_nodes=600,
    #         target="num_GSS",
    #         blackoutthreshold=0.8,
    #         delta_Erot=5000,
    #         resolve_strategy="random",
    #         max_iter=10000,
    #         plot_inertia_map=True,
    #         ref_loss_factor=1.0,
    #     )
    # for co2l in [0.2, 0.0]:
    #     create_inertia_by_country_plot(
    #         co2_lvl_map=co2l,
    #         co2_lvl_ref=0.6,
    #         target="num_GSS",
    #         blackoutthreshold=0.8,
    #         plot_inertia_map=True,
    #     )
