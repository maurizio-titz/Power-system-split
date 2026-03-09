# %%
import gzip
import pickle
import numpy as np
import sys
import os

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.append("../..")  # To import from parent directory
sys.path.append("./")  # To import from parent directory
sys.path.append("../")  # To import from parent directory

from scripts.meteo.meteo_statistics import load_weather_regimes
from utils import data_handling
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from utils.data_handling import get_co2_levels
from utils import config
from utils import plot_style

# %load_ext autoreload
# %autoreload 2

plot_style.setup_matplotlib_style()

sort_by = "n_samples"
# sort_by = "weighted_lost_load"

fpath = "/srv/data/jlange/power-system-split/no_extensions/results/sclopf/clustering/rocof_blackout_Co2L0.6_0.5_0.4_0.3_0.2_0.1_0.05_0.0_n600_lls0.1/clustering_results/agg_params_01bf408e.pklz"
with gzip.open(fpath, "rb") as f:
    res = pickle.load(f)

with gzip.open(fpath.replace(".pklz", "_centroid_res.pklz"), "rb") as f:
    cluster_res = pickle.load(f)
# %% load clustering results
n_nodes = 600
co2l_list = list(get_co2_levels(n_nodes))

n_jobs_distance = 32
n_jobs_clustering = 2
distance_metric_kwargs = {
    "name": "composite",
    "metrics": [
        {
            "name": "hamming",
            "n_jobs": n_jobs_distance,
        },
        {
            "name": "cosine_distance",
            "preprocessing": {
                "method": "boundary_field",
                "target": "classes",  # Transform classes
                "alpha": 0.4,
                "steps": 3,
            },
            "n_jobs": n_jobs_distance,
        },
    ],
    "combiner": geometric_mean,
    "cache_components": True,
}

n_clusters_list = np.arange(32, 1025, step=32).tolist()
clustering_params = {
    "agg": {
        "n_iter": 1e4,
        "n_jobs": n_jobs_clustering,
        "HPs": {
            "n_clusters": n_clusters_list,
            "linkage": [
                "average",
                "complete",
                # "single",
            ],  # "single" is not used as it leads to bad results
            "metric": ["precomputed"],
        },
        "calc_silhouette": True,
    },
}
# Initialize clustering object
cl = Clustering(
    n_nodes,
    co2l_list,
    indicator_type="rocof",
    transformation="blackout",
    blackout_size_threshold=0.1,
    distance_metric_kwargs=distance_metric_kwargs,
    clustering_params=clustering_params,
    distance_matrix_dtype=np.float16,
)

split_properties = pd.read_hdf(
    f"{cl.data_dir}/data_filtered.h5",
    key="split_props",
)
split_properties["cluster_idx"] = -1

net = data_handling.load_pypsa_network(
    n_nodes=n_nodes,
    co2lvl=0.0,
    use_sclopf=True,
    lopt=config.use_extensions,
)
weather_regimes = load_weather_regimes()
if "weather_regime" not in weather_regimes.columns:
    raise ValueError("Weather regime file must include 'weather_regime' column.")
weather_regimes["weighting"] = net.snapshot_weightings.generators.reindex(
    weather_regimes.index
)

if "time_stamp" in split_properties.index.names:
    timestamps = split_properties.index.get_level_values("time_stamp")
elif "snapshot" in split_properties.index.names:
    timestamps = split_properties.index.get_level_values("snapshot")
elif "time_stamp" in split_properties.columns:
    timestamps = split_properties["time_stamp"]
elif "snapshot" in split_properties.columns:
    timestamps = split_properties["snapshot"]
else:
    raise ValueError(
        "Split properties must include 'time_stamp' or 'snapshot' for regime mapping."
    )

split_props_indexed = split_properties.copy()
split_props_indexed.index = pd.to_datetime(timestamps)
split_properties = split_props_indexed.join(
    weather_regimes[["weather_regime"]],
    how="inner",
)
weighted_regime_frequencies = weather_regimes.groupby("weather_regime").weighting.sum()


# %%
def plot_cluster(ax, centroid, failed_edges_prob, nx_graph, pos, plot_edges=False):
    edge_log_scale = False
    if edge_log_scale:
        vmin_edge = 1e-3
        vmax_edge = 1.0
    else:
        vmin_edge = 0.0
        vmax_edge = 1.0
    vmin_edge_orig = vmin_edge
    vmax_edge_orig = vmax_edge
    from matplotlib import cm
    from utils.clustering_visualisation import truncate_colormap

    edge_cmap = cm.get_cmap("inferno_r")
    edge_cmap = truncate_colormap(edge_cmap, 0.1, 0.95, 1000)
    edge_cmap.set_under("gainsboro", 1.0)

    plot_order = np.argsort(centroid[:, 2])  # Sort by the value of the third class
    colors_classes = ["blue", "lightgray", "red"]
    nodes = np.array(list(nx_graph.nodes()))
    pos_arr = np.array(list(pos.values()))
    for i_node in plot_order:
        node = nodes[i_node]
        ax.pie(
            centroid[i_node, :],
            colors=colors_classes,
            radius=0.44,
            center=(pos[node][0], pos[node][1]),
        )

    ax.set_xlim(pos_arr[:, 0].min() - 1, pos_arr[:, 0].max() + 1)
    ax.set_ylim(pos_arr[:, 1].min() - 1, pos_arr[:, 1].max() + 1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])

    if plot_edges:
        if edge_log_scale:
            failed_edges_prob = np.array(
                [np.log10(x) if x > 1e-12 else -np.inf for x in failed_edges_prob]
            )
            vmin_edge = np.log10(vmin_edge)
            vmax_edge = np.log10(vmax_edge)
        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            # edge_color="black",
            width=6,
            edge_cmap=edge_cmap,
            edge_color=failed_edges_prob,
            edge_vmin=vmin_edge,
            edge_vmax=vmax_edge,
        )
        if isinstance(edges, list):
            for edge in edges:
                edge.set_zorder(0)
        else:
            edges.set_zorder(0)


selected_clusters = []
for cluster_number in np.arange(1, 13):
    cluster_idx = (
        cluster_res["centroids_df"]
        .sort_values(by=sort_by, ascending=False)
        .iloc[cluster_number - 1, :]
        .name
    )
    centroid = cluster_res["centroids"][cluster_idx]
    edge_centroids = cluster_res["edge_centroids"][cluster_idx]
    failed_edges_prob = edge_centroids.copy()
    selected_clusters.append(
        {
            "cluster_number": int(cluster_number),
            "centroid": centroid,
            "failed_edges_prob": failed_edges_prob,
        }
    )

    # nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
    cluster_mask = cluster_res["group_masks"][cluster_idx]
    split_properties.loc[cluster_mask, "cluster_idx"] = cluster_number

# %%
co2l = 0.0
plot_df = split_properties.loc[
    (split_properties["cluster_idx"] > 0) & (split_properties.co2l == co2l),
    ["weather_regime", "cluster_idx", "total_weighting"],
].copy()

if "total_weighting" not in plot_df.columns:
    plot_df["total_weighting"] = (
        split_properties.loc[plot_df.index, "snapshot_weighting"]
        * split_properties.loc[plot_df.index, "trigger_weighting"]
    )

cluster_ids = [entry["cluster_number"] for entry in selected_clusters]
weighted_freq = (
    plot_df.pivot_table(
        index="weather_regime",
        columns="cluster_idx",
        values="total_weighting",
        aggfunc="sum",
        fill_value=0.0,
    )
    .sort_index()
    .reindex(columns=cluster_ids, fill_value=0.0)
)
## probability of cluster occuring at given weather regime
conditional_probability_clusterRegime = weighted_freq.T.div(
    weighted_regime_frequencies
).T
normalized_conditional_probability_clusterRegime = (
    conditional_probability_clusterRegime.div(
        conditional_probability_clusterRegime.sum(axis=0), axis=1
    )
)


def _get_capacity_factor_series(network, carrier_name):
    gen = network.generators_t.p.mul(network.generators.sign, axis=1)
    carriers = network.generators["carrier"].str.lower()
    carrier_mask = carriers.str.contains(carrier_name, case=False, na=False)
    if not carrier_mask.any():
        return pd.Series(np.nan, index=pd.to_datetime(network.snapshots))
    total_gen = gen.loc[:, carrier_mask].sum(axis=1)
    max_gen = (
        np.nanmax(total_gen.values) if np.isfinite(total_gen.values).any() else 0.0
    )
    if max_gen <= 0.0:
        return pd.Series(np.nan, index=pd.to_datetime(network.snapshots))
    return pd.Series(
        total_gen.values / max_gen, index=pd.to_datetime(network.snapshots)
    )


def _relative_cluster_freq_by_capacity_bin(
    df,
    cf_column,
    cluster_columns,
    bins,
):
    tmp = df.copy()
    tmp["cf_bin"] = pd.cut(tmp[cf_column], bins=bins, include_lowest=True)
    weighted = tmp.pivot_table(
        index="cf_bin",
        columns="cluster_idx",
        values="total_weighting",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=cluster_columns, fill_value=0.0)
    interval_index = pd.IntervalIndex.from_breaks(bins, closed="right")
    weighted = weighted.reindex(interval_index, fill_value=0.0)
    relative = weighted.div(weighted.sum(axis=0), axis=1).fillna(0.0)
    centers = np.array([interval.mid for interval in relative.index])
    return centers, relative


network = data_handling.load_pypsa_network(
    n_nodes=n_nodes,
    co2lvl=0.0,
    use_sclopf=True,
    lopt=config.use_extensions,
)
solar_cf = _get_capacity_factor_series(network, "solar")
wind_cf = _get_capacity_factor_series(network, "wind")

plot_df["timestamp"] = pd.to_datetime(plot_df.index)
plot_df["solar_cf"] = solar_cf.reindex(plot_df["timestamp"]).values
plot_df["wind_cf"] = wind_cf.reindex(plot_df["timestamp"]).values

cf_bins = np.linspace(0.0, 1.0, 11)
solar_x, solar_relative_freq = _relative_cluster_freq_by_capacity_bin(
    plot_df[np.isfinite(plot_df["solar_cf"])],
    "solar_cf",
    cluster_ids,
    cf_bins,
)
wind_x, wind_relative_freq = _relative_cluster_freq_by_capacity_bin(
    plot_df[np.isfinite(plot_df["wind_cf"])],
    "wind_cf",
    cluster_ids,
    cf_bins,
)

n_clusters = len(normalized_conditional_probability_clusterRegime.columns)
clusters_per_row = 3
n_cluster_rows = int(np.ceil(n_clusters / clusters_per_row))

fig = plt.figure(figsize=(22, 7.0 * n_cluster_rows))
outer_gs = GridSpec(
    n_cluster_rows,
    clusters_per_row,
    figure=fig,
    hspace=0.4,
    wspace=0.25,
)

nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
pos = nx.get_node_attributes(nx_graph, "pos")

weather_labels = normalized_conditional_probability_clusterRegime.index.astype(str)
weather_x = np.arange(len(weather_labels))
cf_bar_width = (cf_bins[1] - cf_bins[0]) * 0.9

for i_cluster, cluster_idx in enumerate(
    normalized_conditional_probability_clusterRegime.columns
):
    grid_row = i_cluster // clusters_per_row
    grid_col = i_cluster % clusters_per_row

    cluster_data = selected_clusters[i_cluster]

    # 2×2 grid: map upper-left, weather upper-right, solar lower-left, wind lower-right
    panel_gs = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer_gs[grid_row, grid_col],
        hspace=0.45,
        wspace=0.35,
    )

    ax_map = fig.add_subplot(panel_gs[0, 0])
    plot_cluster(
        ax=ax_map,
        centroid=cluster_data["centroid"],
        failed_edges_prob=cluster_data["failed_edges_prob"],
        nx_graph=nx_graph,
        pos=pos,
    )

    # Title for the whole 2×2 panel
    fig.add_subplot(outer_gs[grid_row, grid_col], frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.title(f"Cluster {int(cluster_idx)}", fontweight="bold", pad=12, y=1.05)

    ax_weather = fig.add_subplot(panel_gs[0, 1])
    ax_solar = fig.add_subplot(panel_gs[1, 0])
    ax_wind = fig.add_subplot(panel_gs[1, 1])

    ax_weather.bar(
        weather_x,
        normalized_conditional_probability_clusterRegime[cluster_idx],
        width=0.8,
        alpha=0.9,
    )
    ax_weather.set_xticks(weather_x)
    ax_weather.set_xticklabels(weather_labels, rotation=45)
    ax_weather.set_xlabel("Weather regime")
    ax_weather.set_ylabel("Norm. P(Cluster | Regime)")
    ax_weather.grid(False)
    ax_weather.set_ylim(bottom=0)

    ax_solar.bar(
        solar_x,
        solar_relative_freq[cluster_idx],
        width=cf_bar_width,
        alpha=0.9,
    )
    ax_solar.set_xlabel("Solar capacity factor")
    ax_solar.set_ylabel("Rel. freq.")
    ax_solar.grid(False)
    ax_solar.set_xlim(0.0, 1.0)
    ax_solar.set_ylim(bottom=0)

    ax_wind.bar(
        wind_x,
        wind_relative_freq[cluster_idx],
        width=cf_bar_width,
        alpha=0.9,
    )
    ax_wind.set_xlabel("Wind capacity factor")
    ax_wind.set_ylabel("Rel. freq.")
    ax_wind.grid(False)
    ax_wind.set_xlim(0.0, 1.0)
    ax_wind.set_ylim(bottom=0)

for i_cluster in range(n_clusters, n_cluster_rows * clusters_per_row):
    grid_row = i_cluster // clusters_per_row
    grid_col = i_cluster % clusters_per_row
    ax_empty = fig.add_subplot(outer_gs[grid_row, grid_col])
    ax_empty.axis("off")

fig.suptitle(
    f"Relative cluster frequency by weather regime and renewable capacity factor for CO2L = {co2l}%",
    y=0.95,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
plt.show()

f_name = f"weather_regime_cluster_comparison_co2l{co2l}.png"
from utils.config import path_to_meteo_figures

plot_style.save_figure(fig, f_name, path_to_meteo_figures)

# %%
