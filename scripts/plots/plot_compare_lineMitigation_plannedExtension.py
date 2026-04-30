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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    MAP_XLIM,
    MAP_YLIM,
    TITLE_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    setup_matplotlib_style,
    setup_map_axes,
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
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5",
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
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l_ref}-2920SEG.nc",
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
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l_ref}-2920SEG.nc",
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


def load_planned_line_extensions_csv(csv_path: str, co2l: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["co2l"] == co2l].copy()
    if df.empty:
        raise ValueError(f"No entries for co2l={co2l} in {csv_path}")
    return df.sort_values("rank")


def _edge_from_pair(nx_graph: nx.Graph, bus0: str, bus1: str) -> Tuple[str, str]:
    if nx_graph.has_edge(bus0, bus1):
        return (bus0, bus1)
    if nx_graph.has_edge(bus1, bus0):
        return (bus1, bus0)
    raise ValueError(f"Edge not found in graph: {bus0}-{bus1}")


def plot_planned_line_extensions_single_co2(
    csv_path: str,
    co2l: float,
    n_nodes: int = 600,
    co2l_ref: float = 0.6,
    figure_name: str | None = None,
    pad_deg: float = 1.5,
    red_lines_rank: list = [],
):
    """Plot planned line extensions for a single CO2 level.

    Reads selected lines from CSV and zooms map to selected lines.
    """

    setup_matplotlib_style()
    os.makedirs(path_to_figures_sclopf, exist_ok=True)

    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l_ref}-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index=0)
    pos = nx.get_node_attributes(nx_graph, "pos")

    df = load_planned_line_extensions_csv(csv_path, co2l)
    selected_edges = []
    red_lines = []
    for _, row in df.iterrows():
        if row["rank"] in red_lines_rank:
            red_lines.append(_edge_from_pair(nx_graph, row["bus0"], row["bus1"]))
        selected_edges.append(_edge_from_pair(nx_graph, row["bus0"], row["bus1"]))

    selected_nodes = {n for edge in selected_edges for n in edge}

    coords = np.array([pos[node] for node in selected_nodes if node in pos])
    if coords.size > 0:
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        xlim = (min_x - pad_deg, max_x + pad_deg)
        ylim = (min_y - pad_deg, max_y + 2 * pad_deg)
    else:
        xlim = MAP_XLIM
        ylim = MAP_YLIM

    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    nx.draw_networkx_nodes(nx_graph, pos=pos, ax=ax, node_color="black", node_size=0)
    nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        edge_color="lightgray",
        width=MAP_LINE_WIDTH,
    )
    if red_lines:
        nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edgelist=red_lines,
            edge_color="red",
            width=MAP_LINE_WIDTH * 5,
        )
    blue_lines = [edge for edge in selected_edges if edge not in red_lines]
    nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        edgelist=blue_lines,
        edge_color="blue",
        width=MAP_LINE_WIDTH * 5,
    )

    for _, row in df.iterrows():
        edge = _edge_from_pair(nx_graph, row["bus0"], row["bus1"])
        p0 = pos[edge[0]]
        p1 = pos[edge[1]]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = np.hypot(dx, dy)
        if length == 0:
            offset = (0.0, 0.0)
        else:
            nx_off = -dy / length
            ny_off = dx / length
            offset = (0.3 * nx_off, 0.3 * ny_off)
        mid = ((p0[0] + p1[0]) / 2.0 + offset[0], (p0[1] + p1[1]) / 2.0 + offset[1])
        ax.text(
            mid[0],
            mid[1],
            str(int(row["rank"])),
            fontsize=14,
            ha="center",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
            },
        )

    setup_map_axes(ax, xlim=xlim, ylim=ylim)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")

    co2_percent = get_actual_co2_level(co2l, percent=True)
    ax.set_title(
        rf"Grid extension mitigation at CO$_2$ level={int(co2_percent)}\%",
        fontsize=TITLE_FONTSIZE,
    )

    fig.tight_layout()
    if figure_name is None:
        figure_name = f"planned_line_extensions_co2l{co2l}"
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
    # run_planned_extension_comparison(
    #     n_nodes=600,
    #     target="num_GSS",
    #     blackoutthreshold=0.8,
    #     build_380kV_only=True,
    #     annualized_costs=True,
    # )
    plot_planned_line_extensions_single_co2(
        csv_path=os.path.join(
            path_to_line_extension_mitigation_sclopf,
            "planned_line_extensions_n600_target_num_GSS_annualized_380kVonly_blackoutthres0.8.csv",
        ),
        co2l=0.2,
        figure_name="planned_line_extensions_co2l0.2_blackoutthres0.8",
        red_lines_rank=[11],
        pad_deg=0.75,
    )
