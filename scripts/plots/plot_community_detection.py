#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Community detection on the grid graph using line limits as edge strength."""

import argparse
import ast
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append("./")

from utils import data_handling
from utils.config import path_to_figures_sclopf
from utils.plot_style import (
    MAP_LINE_COLOR,
    MAP_LINE_WIDTH,
    PRIMARY_COLORS,
    MAP_XLIM,
    MAP_YLIM,
    get_plot_config,
    save_figure,
    setup_map_axes,
    setup_matplotlib_style,
)

MAP_SIZE = (8, 5)  # Single map size in inches


def _coerce_pos(value) -> Tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return float(parsed[0]), float(parsed[1])
        except (ValueError, SyntaxError):
            return None
    return None


def _get_positions(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    pos_attr = nx.get_node_attributes(graph, "pos")
    if pos_attr:
        pos = {}
        for node, value in pos_attr.items():
            coerced = _coerce_pos(value)
            if coerced is not None:
                pos[node] = coerced
        if pos:
            return pos

    x_attr = nx.get_node_attributes(graph, "x")
    y_attr = nx.get_node_attributes(graph, "y")
    if x_attr and y_attr:
        return {node: (float(x_attr[node]), float(y_attr[node])) for node in x_attr}

    return nx.spring_layout(graph, seed=42, weight="line_limit")


def _set_weights_F(graph: nx.Graph) -> None:
    for u, v, attrs in graph.edges(data=True):
        line_limit = attrs.get("s_nom", None)
        if line_limit is None:
            raise ValueError(
                f"Edge ({u}, {v}) is missing 's_nom' attribute for line limit."
            )
        attrs["community_weight"] = line_limit


def _set_weights_bxF(graph: nx.Graph) -> None:
    for u, v, attrs in graph.edges(data=True):
        line_limit = attrs.get("s_nom", None)
        b = np.abs(attrs.get("weight", None))
        if line_limit is None:
            raise ValueError(
                f"Edge ({u}, {v}) is missing 's_nom' attribute for line limit."
            )
        if b is None:
            raise ValueError(
                f"Edge ({u}, {v}) is missing 'weight' attribute for b/F calculation."
            )
        attrs["community_weight"] = line_limit * b


def _detect_communities(graph: nx.Graph, seed: int):
    try:
        from networkx.algorithms.community import leiden_communities

        communities = leiden_communities(graph, weight="community_weight", seed=seed)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(
            greedy_modularity_communities(graph, weight="community_weight")
        )
    return communities


def _communities_to_node_map(communities) -> Dict[str, int]:
    node_to_comm = {}
    for idx, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = idx
    return node_to_comm


def _plot_communities(graph: nx.Graph, node_to_comm: Dict[str, int], output_path: str):
    setup_matplotlib_style()
    pos = _get_positions(graph)
    communities = list(set(node_to_comm.values()))
    cmap = plt.get_cmap("tab20", max(len(communities), 1))

    node_colors = [cmap(node_to_comm[node]) for node in graph.nodes()]

    community_weights = np.array(
        [graph.edges[edge].get("community_weight", 1.0) for edge in graph.edges()]
    )
    if community_weights.size == 0:
        widths = MAP_LINE_WIDTH
    else:
        lo, hi = np.percentile(community_weights, [5, 95])
        scaled = (community_weights - lo) / (hi - lo + 1e-9)
        widths = MAP_LINE_WIDTH + 2.5 * np.clip(scaled, 0, 1)

    fig, ax = plt.subplots(
        figsize=MAP_SIZE, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color=MAP_LINE_COLOR,
        alpha=0.35,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=40,
        linewidths=0.0,
        alpha=1,
        ax=ax,
    )

    # setup_map_axes(ax, xlim=MAP_XLIM, ylim=MAP_YLIM)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
    ax.axis("off")
    fig.tight_layout()
    save_dir = os.path.dirname(output_path) or None
    filename = os.path.basename(output_path)
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pdf"
    save_figure(fig, base, save_path=save_dir, formats=[ext.lstrip(".")])
    plt.close(fig)


def _country_from_bus_name(bus_name: str) -> str:
    bus_str = str(bus_name)
    if len(bus_str) >= 2 and bus_str[:2].isalpha():
        code = bus_str[:2].upper()
        if code == "UK":
            return "GB"
        return code
    if len(bus_str) >= 3 and bus_str[:3].isalpha():
        code3 = bus_str[:3].upper()
        country_3_to_2 = {
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
        return country_3_to_2.get(code3, code3)
    return "UNK"


def _get_bus_country_map(
    graph: nx.Graph, co2lvl: float, n_nodes: int
) -> Dict[str, str]:
    node_country_attr = nx.get_node_attributes(graph, "country")
    if node_country_attr:
        return {node: str(country) for node, country in node_country_attr.items()}

    try:
        network = data_handling.load_pypsa_network(
            co2lvl=co2lvl, n_nodes=n_nodes, use_sclopf=True
        )
        if "country" in network.buses.columns:
            countries = network.buses["country"].fillna("UNK")
            return {bus: str(country) for bus, country in countries.items()}
        return {bus: _country_from_bus_name(bus) for bus in network.buses.index}
    except Exception:
        return {bus: _country_from_bus_name(bus) for bus in graph.nodes()}


def _classify_cross_country_edges(
    graph: nx.Graph, bus_country_map: Dict[str, str]
) -> Dict[Tuple[str, str], bool]:
    cross_country = {}
    for u, v in graph.edges():
        cu = bus_country_map.get(u, "UNK")
        cv = bus_country_map.get(v, "UNK")
        cross_country[(u, v)] = cu != "UNK" and cv != "UNK" and cu != cv
    return cross_country


def _plot_intercommunity_edges(
    graph: nx.Graph,
    inter_edges: list[Tuple[str, str]],
    cross_border_edges: list[Tuple[str, str]],
    output_path: str,
):
    setup_matplotlib_style()
    pos = _get_positions(graph)
    fig, ax = plt.subplots(
        figsize=MAP_SIZE, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=MAP_LINE_WIDTH,
        edge_color=MAP_LINE_COLOR,
        alpha=0.15,
        ax=ax,
    )

    if inter_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=inter_edges,
            width=MAP_LINE_WIDTH * 3.5,
            edge_color=PRIMARY_COLORS["blue"],
            alpha=1,
            ax=ax,
        )

    if cross_border_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=cross_border_edges,
            width=MAP_LINE_WIDTH * 3.5,
            edge_color=PRIMARY_COLORS["red"],
            alpha=1,
            ax=ax,
        )

    legend_handles = []
    if inter_edges:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=PRIMARY_COLORS["blue"],
                lw=MAP_LINE_WIDTH * 3.5,
                label="same-country",
            )
        )
    if cross_border_edges:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=PRIMARY_COLORS["red"],
                lw=MAP_LINE_WIDTH * 3.5,
                label="cross-border",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            title="Inter-community edges",
            framealpha=1.0,
            facecolor="white",
        )

    # setup_map_axes(ax, xlim=MAP_XLIM, ylim=MAP_YLIM)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
    ax.axis("off")
    fig.tight_layout()
    save_dir = os.path.dirname(output_path) or None
    filename = os.path.basename(output_path)
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pdf"
    save_figure(fig, base, save_path=save_dir, formats=[ext.lstrip(".")])
    plt.close(fig)


def main(weight_type: str = "line_limit"):
    parser = argparse.ArgumentParser(
        description="Community detection on grid graph using line limits as edge strength."
    )
    parser.add_argument("--snet-index", type=int, default=0)
    parser.add_argument("--co2lvl", type=float, default=0.0)
    parser.add_argument("--n-nodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path for plot. Defaults to figures directory.",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["line_limit", "boverF"],
        default="line_limit",
        help="Type of edge weight to use for community detection.",
    )

    args = parser.parse_args()

    graph = data_handling.load_networkx_graph(
        snet_index=args.snet_index, co2lvl=args.co2lvl
    )
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        graph
    )  # Ensure graph is fully loaded and attributes are accessible
    if weight_type == "line_limit":
        _set_weights_F(graph)
    elif weight_type == "boverF":
        _set_weights_bxF(graph)

    communities = _detect_communities(graph, seed=args.seed)
    node_to_comm = _communities_to_node_map(communities)

    inter_edges = [
        (u, v) for u, v in graph.edges() if node_to_comm.get(u) != node_to_comm.get(v)
    ]

    bus_country_map = _get_bus_country_map(
        graph, co2lvl=args.co2lvl, n_nodes=args.n_nodes
    )
    cross_country_map = _classify_cross_country_edges(graph, bus_country_map)
    cross_border_edges = [edge for edge in inter_edges if cross_country_map[edge]]
    inter_edges_non_cross = [
        edge for edge in inter_edges if not cross_country_map[edge]
    ]

    if args.output:
        output_path = args.output
    else:
        config = get_plot_config()
        output_dir = config.get("output_dir") or ""
        if not output_dir:
            plot_style = os.getenv("PLOT_STYLE", "").strip()
            if plot_style:
                output_dir = os.path.join(path_to_figures_sclopf, plot_style)
            else:
                output_dir = path_to_figures_sclopf

        output_path = os.path.join(
            output_dir,
            f"community_detection_{weight_type}_snet{args.snet_index}_co2lvl{args.co2lvl}.pdf",
        )

    output_inter_path = os.path.splitext(output_path)[0] + "_intercommunity.pdf"

    _plot_communities(graph, node_to_comm, output_path)
    _plot_intercommunity_edges(
        graph,
        inter_edges_non_cross,
        cross_border_edges,
        output_inter_path,
    )

    total_edges = graph.number_of_edges()
    total_cross = sum(1 for edge in graph.edges() if cross_country_map[edge])
    inter_edges_count = len(inter_edges)
    inter_cross_count = len(cross_border_edges)

    inter_cross_pct = (
        100.0 * inter_cross_count / inter_edges_count if inter_edges_count else 0.0
    )
    total_cross_pct = 100.0 * total_cross / total_edges if total_edges else 0.0

    print(f"Detected {len(communities)} communities.")
    print(f"Saved plot: {output_path}")
    print(f"Saved inter-community plot: {output_inter_path}")
    print(
        "Inter-community cross-border edges: "
        f"{inter_cross_count}/{inter_edges_count} "
        f"({inter_cross_pct:.2f}%)"
    )
    print(
        "All cross-border edges: "
        f"{total_cross}/{total_edges} "
        f"({total_cross_pct:.2f}%)"
    )


if __name__ == "__main__":
    main(weight_type="boverF")
    main(weight_type="line_limit")
