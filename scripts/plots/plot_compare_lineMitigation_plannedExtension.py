#!/usr/bin/env python3
"""
This script extracts the line extensions needed to reach the stability metric
value of the reference scenario. It plots the line extensions for the
different scenarios in one figure and saves lines extensions (node pairs) to
disk.
"""

import os
import pickle
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append("./")

from utils import data_handling
from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_figures_sclopf,
    path_to_line_extension_mitigation_sclopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_sclopf,
)
from utils.plot_style import (
    MAP_LINE_COLOR,
    MAP_LINE_WIDTH,
    MAP_MULTI_SIZE,
    TITLE_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    setup_matplotlib_style,
    save_figure,
)
from scripts.plots.plot_mitigation import create_gridExt_mitigation_filename


def _normalize_blackoutthreshold(blackoutthreshold: float) -> float:
    return 0.0 if blackoutthreshold is None else blackoutthreshold


def _get_reference_target_value(
    split_properties: pd.DataFrame,
    target: str,
    blackoutthreshold: float,
    co2l_ref: float,
) -> float:
    split_properties_reference = split_properties[
        split_properties.co2l == co2l_ref
    ].copy()
    if target == "total_loss":
        return split_properties_reference.lost_load_share_blackout.sum()
    if target == "num_GSS":
        return (
            split_properties_reference["total_weighting"]
            * (split_properties_reference.lost_load_share_blackout > blackoutthreshold)
        ).sum()
    if target == "lost_load_GSS":
        return (
            split_properties_reference["total_weighting"]
            * split_properties_reference.lost_load_share_blackout
        ).sum()
    raise ValueError(
        "target must be one of 'num_GSS', 'total_loss', or 'lost_load_GSS'"
    )


def _load_line_mitigation_results(
    n_nodes: int,
    co2l: float,
    target: str,
    blackoutthreshold: float,
    build_380kV_only: bool,
    annualized_costs: bool,
):
    f_name = create_gridExt_mitigation_filename(
        n_nodes,
        build_380kV_only,
        target,
        blackoutthreshold,
        annualized_costs,
        co2l,
    )
    full_path = os.path.join(path_to_line_extension_mitigation_sclopf, f_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Missing mitigation file: {full_path}. Run line extension mitigation first."
        )
    with open(full_path, "rb") as fh:
        return pickle.load(fh)


def _line_index_to_nodes(nx_graph: nx.Graph) -> Dict[int, Tuple[str, str]]:
    return {idx: edge for idx, edge in enumerate(nx_graph.edges())}


def extract_planned_line_extensions(
    n_nodes: int = 600,
    co2_levels: Iterable[float] = None,
    co2l_ref: float = 0.6,
    target: str = "num_GSS",
    blackoutthreshold: float = 0.0,
    build_380kV_only: bool = False,
    annualized_costs: bool = False,
) -> Tuple[
    Dict[float, List[int]], Dict[float, List[Tuple[str, str]]], Dict[float, List[float]]
]:
    """Extract line extensions to reach reference stability metric.

    Returns
    -------
    selected_line_indices : dict
            Dictionary mapping co2 levels to selected line indices.
    selected_node_pairs : dict
            Dictionary mapping co2 levels to selected node pairs.
    selected_line_lengths : dict
            Dictionary mapping co2 levels to selected line lengths (km).
    """

    blackoutthreshold = _normalize_blackoutthreshold(blackoutthreshold)

    if co2_levels is None:
        co2_levels = get_co2_levels(n_nodes, ignore_lvls=(co2l_ref,))
    else:
        co2_levels = [lvl for lvl in co2_levels if lvl != co2l_ref]

    os.makedirs(path_to_line_extension_mitigation_sclopf, exist_ok=True)

    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5",
        index_col=0,
    )
    split_properties["lost_load_share_blackout"] = split_properties[
        "lost_load_share_blackout"
    ].astype(float)

    target_value = _get_reference_target_value(
        split_properties,
        target=target,
        blackoutthreshold=blackoutthreshold,
        co2l_ref=co2l_ref,
    )

    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l_ref}-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    line_index_to_nodes = _line_index_to_nodes(nx_graph)

    selected_line_indices: Dict[float, List[int]] = {}
    selected_node_pairs: Dict[float, List[Tuple[str, str]]] = {}
    selected_line_lengths: Dict[float, List[float]] = {}

    for co2l in co2_levels:
        reinforced_lines, loss_with_mitigation, num_blackouts, _ = (
            _load_line_mitigation_results(
                n_nodes=n_nodes,
                co2l=co2l,
                target=target,
                blackoutthreshold=blackoutthreshold,
                build_380kV_only=build_380kV_only,
                annualized_costs=annualized_costs,
            )
        )

        if target == "num_GSS":
            post_mitigation_value = num_blackouts
        else:
            post_mitigation_value = loss_with_mitigation

        num_lines_to_reach_ref = np.where(
            np.array(post_mitigation_value) < target_value
        )[0][0]
        selected_lines = reinforced_lines[:num_lines_to_reach_ref]

        selected_line_indices[co2l] = list(map(int, selected_lines))
        selected_node_pairs[co2l] = [
            line_index_to_nodes[int(idx)] for idx in selected_lines
        ]

        # Extract line lengths from the network
        line_indices_list = [network.lines.index[int(idx)] for idx in selected_lines]
        selected_line_lengths[co2l] = network.lines.loc[
            line_indices_list, "length"
        ].tolist()

    return selected_line_indices, selected_node_pairs, selected_line_lengths


def save_line_extension_node_pairs(
    selected_node_pairs: Dict[float, List[Tuple[str, str]]],
    selected_line_lengths: Dict[float, List[float]],
    n_nodes: int,
    target: str,
    blackoutthreshold: float,
    build_380kV_only: bool,
    annualized_costs: bool,
) -> str:
    """Save planned line extensions (node pairs) to disk."""
    blackoutthreshold = _normalize_blackoutthreshold(blackoutthreshold)
    rows = []
    for co2l, pairs in selected_node_pairs.items():
        lengths = selected_line_lengths[co2l]
        for rank, ((bus0, bus1), length) in enumerate(zip(pairs, lengths), start=1):
            rows.append(
                {
                    "co2l": co2l,
                    "rank": rank,
                    "bus0": bus0,
                    "bus1": bus1,
                    "length_km": length,
                }
            )

    df = pd.DataFrame(rows)
    suffix_parts = [f"n{n_nodes}", f"target_{target}"]
    if annualized_costs:
        suffix_parts.append("annualized")
    if build_380kV_only:
        suffix_parts.append("380kVonly")
    if blackoutthreshold > 0.0:
        suffix_parts.append(f"blackoutthres{blackoutthreshold}")
    suffix = "_".join(suffix_parts)
    out_path = os.path.join(
        path_to_line_extension_mitigation_sclopf,
        f"planned_line_extensions_{suffix}.csv",
    )
    df.to_csv(out_path, index=False)
    return out_path


def plot_planned_line_extensions(
    selected_line_indices: Dict[float, List[int]],
    n_nodes: int = 600,
    co2l_ref: float = 0.6,
    figure_name: str = "planned_line_extensions_compare",
    n_cols: int = 3,
):
    """Plot selected line extensions on maps for all scenarios."""

    os.makedirs(path_to_figures_sclopf, exist_ok=True)
    setup_matplotlib_style()

    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l_ref}-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    pos = nx.get_node_attributes(nx_graph, "pos")
    all_edges = list(nx_graph.edges())

    co2_levels = list(selected_line_indices.keys())
    n_plots = len(co2_levels)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=MAP_MULTI_SIZE, squeeze=False)

    for idx, co2l in enumerate(co2_levels):
        ax = axes[idx // n_cols][idx % n_cols]
        selected_edges = [all_edges[i] for i in selected_line_indices[co2l]]

        nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=ax, node_color="black", node_size=0
        )
        nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edge_color="lightgray",
            width=MAP_LINE_WIDTH,
        )
        nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edgelist=selected_edges,
            edge_color=MAP_LINE_COLOR,
            width=MAP_LINE_WIDTH * 3,
        )

        ax.set_aspect("equal")
        ax.axis("off")
        co2_percent = get_actual_co2_level(co2l, percent=True)
        ax.set_title(
            rf"CO$_2$ level={int(co2_percent)}\%",
            fontsize=TITLE_FONTSIZE,
        )
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    for ax in axes.flatten()[n_plots:]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, figure_name, path_to_figures_sclopf)
    plt.close(fig)


def run_planned_extension_comparison(
    n_nodes: int = 600,
    target: str = "num_GSS",
    blackoutthreshold: float = 0.0,
    build_380kV_only: bool = False,
    annualized_costs: bool = False,
):
    """Run extraction, save node pairs, and create the comparison plot."""

    selected_line_indices, selected_node_pairs, selected_line_lengths = (
        extract_planned_line_extensions(
            n_nodes=n_nodes,
            target=target,
            blackoutthreshold=blackoutthreshold,
            build_380kV_only=build_380kV_only,
            annualized_costs=annualized_costs,
        )
    )

    save_line_extension_node_pairs(
        selected_node_pairs=selected_node_pairs,
        selected_line_lengths=selected_line_lengths,
        n_nodes=n_nodes,
        target=target,
        blackoutthreshold=blackoutthreshold,
        build_380kV_only=build_380kV_only,
        annualized_costs=annualized_costs,
    )

    plot_planned_line_extensions(
        selected_line_indices=selected_line_indices,
        n_nodes=n_nodes,
        figure_name=f"planned_line_extensions_compare_{target}",
    )


if __name__ == "__main__":
    run_planned_extension_comparison(
        n_nodes=600,
        target="num_GSS",
        blackoutthreshold=0.8,
        build_380kV_only=True,
        annualized_costs=True,
    )
