#!/usr/bin/env python3
"""
PCA of nodal injections across CO2 scenarios.

For each available CO2 level, this script:
1) loads nodal effective injections from disk,
2) performs PCA,
3) plots cumulative explained variance vs. number of components.
"""

import os
import sys
from ast import literal_eval

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.decomposition import PCA

sys.path.append("./")

from utils.config import path_to_figures_sclopf
from utils.config import use_extensions
from utils import data_handling
from utils.data_handling import (
    get_actual_co2_level,
    get_co2_levels,
    load_effective_injections,
    load_networkx_graph,
)
from utils.plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    save_figure,
    setup_matplotlib_style,
)


def _build_injection_matrix(effective_injections):
    """Convert effective injection dict (snapshot -> vector) to 2D matrix.

    Returns:
            np.ndarray: shape (n_snapshots, n_nodes)
    """

    snapshots = sorted(effective_injections.keys())
    rows = [
        np.asarray(effective_injections[snapshot]).ravel() for snapshot in snapshots
    ]
    matrix = np.vstack(rows)
    return matrix


def _build_generation_matrix_from_network(network, node_order=None):
    """Build snapshot x node matrix for nodal generation (+storage dispatch)."""

    generation_by_generator = network.generators_t.p.mul(
        network.generators.sign, axis=1
    )
    generation_by_bus = (
        generation_by_generator.T.groupby(network.generators["bus"]).sum().T
    )

    if hasattr(network, "storage_units") and len(network.storage_units) > 0:
        storage_by_bus = (
            network.storage_units_t.p.T.groupby(network.storage_units["bus"]).sum().T
        )
        generation_by_bus = generation_by_bus.add(storage_by_bus, fill_value=0.0)
        print("Included storage dispatch in generation matrix.")

    if node_order is None:
        node_order = list(generation_by_bus.columns)
    generation_by_bus = generation_by_bus.reindex(columns=node_order, fill_value=0.0)
    return generation_by_bus.to_numpy()


def _get_sorted_snapshots(effective_injections):
    return pd.to_datetime(sorted(effective_injections.keys()))


def _parse_node_position(pos_value):
    """Parse node position from gml-loaded value into (x, y)."""

    if isinstance(pos_value, str):
        parsed = literal_eval(pos_value)
    else:
        parsed = pos_value

    if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) == 2:
        return float(parsed[0]), float(parsed[1])

    raise ValueError(f"Unsupported node position format: {type(pos_value)}")


def _get_daily_and_yearly_profiles(score_series, rolling_days=14):
    """Return hourly mean profile and rolling-smoothed yearly profile."""

    daily_profile = score_series.groupby(score_series.index.hour).mean().sort_index()

    if len(score_series.index) < 2:
        window = 1
    else:
        delta = score_series.index.to_series().diff().dropna().median()
        hours_per_step = max(delta / pd.Timedelta(hours=1), 1e-9)
        samples_per_day = max(int(round(24 / hours_per_step)), 1)
        window = max(samples_per_day * rolling_days, 1)

    yearly_smoothed = score_series.rolling(
        window=window, min_periods=1, center=True
    ).mean()
    return daily_profile, yearly_smoothed


def _pca_cumulative_explained_variance(data_matrix):
    """Compute cumulative explained variance ratio using sklearn PCA.

    Args:
            data_matrix (np.ndarray): shape (n_samples, n_features)

    Returns:
            np.ndarray: cumulative explained variance ratio, shape (n_components,)
    """

    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be a 2D array")
    if data_matrix.shape[0] < 2:
        raise ValueError("Need at least 2 samples for PCA")

    pca = PCA(svd_solver="full")
    pca.fit(data_matrix)
    explained_variance_ratio = np.nan_to_num(pca.explained_variance_ratio_)
    return np.cumsum(explained_variance_ratio)


def _get_node_matrix_for_co2lvl(co2lvl, node_data, n_nodes=600, node_order=None):
    """Load node-level snapshot matrix for a CO2 level.

    Args:
        co2lvl (float): CO2 level
        node_data (str): "effective_injections" or "generation"
        n_nodes (int): network aggregation level
        node_order (list[str] | None): desired node order for columns

    Returns:
        tuple[pd.DatetimeIndex, np.ndarray]: snapshots and matrix (n_snapshots, n_nodes)
    """

    if node_data == "effective_injections":
        effective_injections = load_effective_injections(co2lvl=co2lvl)
        snapshots = _get_sorted_snapshots(effective_injections)
        matrix = _build_injection_matrix(effective_injections)
        return snapshots, matrix

    if node_data == "generation":
        network: pypsa.Network = data_handling.load_pypsa_network(
            n_nodes=n_nodes,
            co2lvl=co2lvl,
            use_sclopf=True,
            lopt=use_extensions,
        )
        snapshots = pd.to_datetime(network.snapshots)
        matrix = _build_generation_matrix_from_network(network, node_order)
        return snapshots, matrix

    raise ValueError("node_data must be either 'effective_injections' or 'generation'")


def create_injection_pca_plot(n_nodes=600, node_data="effective_injections"):
    """Create one figure with PCA cumulative explained variance for each CO2 level."""

    os.makedirs(path_to_figures_sclopf, exist_ok=True)
    setup_matplotlib_style()

    co2_levels = get_co2_levels(n_nodes)
    if len(co2_levels) == 0:
        raise RuntimeError(f"No CO2 levels available for n_nodes={n_nodes}")

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")

    for idx, co2lvl in enumerate(co2_levels):
        _, node_matrix = _get_node_matrix_for_co2lvl(
            co2lvl=co2lvl,
            node_data=node_data,
            n_nodes=n_nodes,
        )
        cumulative_ev = _pca_cumulative_explained_variance(node_matrix)

        n_components = np.arange(1, cumulative_ev.size + 1)

        actual_co2_pct = np.asarray(get_actual_co2_level(co2lvl, percent=True)).item()
        label = rf"CO$_2$={int(actual_co2_pct)}\%"

        color = cmap(idx / max(len(co2_levels) - 1, 1))
        ax.plot(
            n_components,
            cumulative_ev,
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax.set_xlabel("Number of principal components", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative explained variance", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        f"PCA of nodal {node_data.replace('_', ' ')} across CO$_2$ levels",
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, 30)
    # ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2, frameon=False)

    save_figure(
        fig,
        f"injection_pca_explained_variance_{node_data}",
        path_to_figures_sclopf,
    )
    return fig, ax


def create_injection_pca_component_profiles_plot(
    n_nodes=600,
    n_components=5,
    snet_index=0,
    rolling_days=14,
    node_data="effective_injections",
):
    """Create grid plot with one row per CO2 level and one column per PCA component.

    Each column contains:
    - left: spatial map of PCA component loadings,
    - right/top: mean daily profile of PCA score,
    - right/bottom: yearly profile of PCA score (rolling mean).
    """

    os.makedirs(path_to_figures_sclopf, exist_ok=True)
    setup_matplotlib_style()

    co2_levels = list(get_co2_levels(n_nodes))
    if len(co2_levels) == 0:
        raise RuntimeError(f"No CO2 levels available for n_nodes={n_nodes}")

    n_cols = n_components
    n_rows = len(co2_levels)

    fig = plt.figure(figsize=(4.9 * n_cols, 3.0 * n_rows))
    outer = GridSpec(n_rows, n_cols, figure=fig, wspace=0.14, hspace=0.32)

    for row_idx, co2lvl in enumerate(co2_levels):
        nx_graph = load_networkx_graph(snet_index=snet_index, co2lvl=co2lvl)
        node_order = list(nx_graph.nodes())
        snapshots, node_matrix = _get_node_matrix_for_co2lvl(
            co2lvl=co2lvl,
            node_data=node_data,
            n_nodes=n_nodes,
            node_order=node_order,
        )

        pca = PCA(
            n_components=min(n_components, node_matrix.shape[0], node_matrix.shape[1]),
            svd_solver="full",
        )
        scores = pca.fit_transform(node_matrix)
        explained = pca.explained_variance_ratio_

        pos_dict = nx_graph.nodes(data="pos")
        x_coords = np.zeros(len(node_order))
        y_coords = np.zeros(len(node_order))
        pos_lookup = {}
        for node_idx, node in enumerate(node_order):
            x_node, y_node = _parse_node_position(dict(pos_dict)[node])
            x_coords[node_idx] = x_node
            y_coords[node_idx] = y_node
            pos_lookup[node] = (x_node, y_node)

        edge_segments = [
            [pos_lookup[u], pos_lookup[v]]
            for u, v in nx_graph.edges()
            if u in pos_lookup and v in pos_lookup
        ]

        co2_pct = int(np.asarray(get_actual_co2_level(co2lvl, percent=True)).item())

        n_available_components = min(n_components, pca.components_.shape[0])
        component_profiles = []
        for comp_idx in range(n_available_components):
            component_scores = pd.Series(scores[:, comp_idx], index=snapshots)
            daily_profile, yearly_smoothed = _get_daily_and_yearly_profiles(
                component_scores, rolling_days=rolling_days
            )
            component_profiles.append((daily_profile, yearly_smoothed))

        daily_y_abs = max(
            [
                np.nanmax(np.abs(daily_profile.to_numpy()))
                for daily_profile, _ in component_profiles
            ]
            + [1e-12]
        )
        yearly_y_abs = max(
            [
                np.nanmax(np.abs(yearly_smoothed.to_numpy()))
                for _, yearly_smoothed in component_profiles
            ]
            + [1e-12]
        )

        day_x_min = int(
            min(daily_profile.index.min() for daily_profile, _ in component_profiles)
        )
        day_x_max = int(
            max(daily_profile.index.max() for daily_profile, _ in component_profiles)
        )
        year_x_min = snapshots.min()
        year_x_max = snapshots.max()

        for comp_idx in range(n_available_components):
            component_spec = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=outer[row_idx, comp_idx],
                width_ratios=[2.1, 0.75],
                wspace=0.08,
            )
            right_spec = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=component_spec[0, 1],
                height_ratios=[1, 1],
                hspace=0.06,
            )

            ax_map = fig.add_subplot(component_spec[0, 0])
            ax_daily = fig.add_subplot(right_spec[0, 0])
            ax_year = fig.add_subplot(right_spec[1, 0])

            component_weights = pca.components_[comp_idx]
            max_abs_weight = np.nanmax(np.abs(component_weights))
            max_abs_weight = max(max_abs_weight, 1e-12)
            marker_sizes = 10 * (np.abs(component_weights) / max_abs_weight)

            if edge_segments:
                edge_collection = LineCollection(
                    edge_segments,
                    colors="black",
                    linewidths=0.15,
                    alpha=0.45,
                    zorder=0,
                )
                ax_map.add_collection(edge_collection)

            cmap = "RdBu_r"
            ax_map.scatter(
                x_coords,
                y_coords,
                c=component_weights,
                cmap=cmap,
                vmin=-max_abs_weight,
                vmax=max_abs_weight,
                s=marker_sizes,
                edgecolors="black",
                linewidths=0.02,
            )
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.set_aspect("equal")
            ax_map.set_xlim(float(np.nanmin(x_coords)), float(np.nanmax(x_coords)))
            ax_map.set_ylim(float(np.nanmin(y_coords)), float(np.nanmax(y_coords)))

            ax_map.set_title(
                rf"PC{comp_idx + 1} ({explained[comp_idx] * 100:.1f}\%)",
                fontsize=11,
            )
            if comp_idx == 0:
                ax_map.set_ylabel(rf"CO$_2$={co2_pct}\%", fontsize=11)

            daily_profile, yearly_smoothed = component_profiles[comp_idx]

            ax_daily.plot(
                daily_profile.index.to_numpy(),
                daily_profile.to_numpy(),
                color="black",
                lw=1.2,
            )
            ax_daily.set_xlim(day_x_min, day_x_max)
            ax_daily.set_xticks(daily_profile.index.to_numpy())
            ax_daily.set_ylim(-daily_y_abs, daily_y_abs)
            ax_daily.margins(x=0)
            if row_idx < n_rows - 1:
                ax_daily.set_xticklabels([])
            ax_daily.tick_params(axis="both", labelsize=8)
            ax_daily.set_title("Daily", fontsize=9)

            year_x = mdates.date2num(yearly_smoothed.index.to_pydatetime())
            ax_year.plot(
                year_x,
                yearly_smoothed.to_numpy(),
                color="tab:blue",
                lw=1.2,
            )
            ax_year.set_xlim(
                float(mdates.date2num(year_x_min.to_pydatetime())),
                float(mdates.date2num(year_x_max.to_pydatetime())),
            )
            ax_year.set_ylim(-yearly_y_abs, yearly_y_abs)
            ax_year.margins(x=0)
            if row_idx < n_rows - 1:
                ax_year.set_xticklabels([])
            ax_year.tick_params(axis="both", labelsize=8)
            ax_year.set_title("Yearly (roll.)", fontsize=9)

    fig.suptitle(
        f"Leading PCA components of nodal {node_data.replace('_', ' ')}: maps, daily and yearly profiles",
        fontsize=TITLE_FONTSIZE,
        y=0.995,
    )

    save_figure(
        fig,
        f"injection_pca_component_profiles_grid_{node_data}",
        path_to_figures_sclopf,
    )
    return fig


if __name__ == "__main__":
    # create_injection_pca_plot(n_nodes=600)
    # create_injection_pca_component_profiles_plot(
    #     n_nodes=600,
    #     n_components=5,
    #     node_data="effective_injections",
    # )
    create_injection_pca_component_profiles_plot(
        n_nodes=600,
        n_components=5,
        node_data="generation",
    )
