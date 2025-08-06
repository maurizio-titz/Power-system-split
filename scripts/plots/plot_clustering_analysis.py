#!/usr/bin/env python3
"""
Clustering Analysis Plot Generation Script

This script generates the clustering analysis visualization showing:
- Network topology with centroids and failure probabilities
- Histograms of loss distribution by CO2 levels
- Colorbars for both node blackout probability and line failure probability

The plot shows the top 4 clusters sorted by weighted lost load.
"""

import os
import sys
import warnings
import gzip
import pickle
import copy


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import silhouette_score

# Add project root to path
root_path = "./"
sys.path.append(root_path)

# Import project utilities
from utils.data_handling import get_co2_levels, get_actual_co2_level
from utils.config import path_to_clustering_results_sclopf, path_to_vis_results_sclopf
from utils.clustering_visualisation import *
from utils import data_handling
from utils.config import path_to_pypsa_network_sclopf, path_to_figures_sclopf
from utils import cascade_simulation
from utils.clustering import load_clustering, get_path_to_clustering_dir
from scripts.plots.plot_combined_generation_storage import LABEL_FONTSIZE
from utils.plot_style import *

# Import plot styling
setup_matplotlib_style()


def load_clustering_data(
    n_nodes=600, indicator_type="rocof", transformation="blackout"
):
    """Load clustering data and network setup."""
    # Load network graph and node positions
    snet_index = 0
    network = data_handling.load_pypsa_network(0.6, n_nodes, True)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Calculate failure statistics
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    bridge_idxs = data_handling.nx_edges_to_matrix_indices(
        nx.bridges(nx_graph), nx_graph
    )
    double_line_failures = cascade_simulation.calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )
    num_snapshots_weighted = network.snapshot_weightings.objective.sum()
    num_failures_weighted = len(double_line_failures) * num_snapshots_weighted

    # Get CO2 levels and setup paths
    co2l_list = get_co2_levels(n_nodes=n_nodes)
    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l=co2l_list,
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=None,
        lost_load_share=0.05,
    )

    return {
        "nx_graph": nx_graph,
        "pos": pos,
        "network": network,
        "co2l_list": co2l_list,
        "save_dir": save_dir,
        "num_failures_weighted": num_failures_weighted,
        "n_nodes": n_nodes,
    }


def load_processed_data(save_dir, n_nodes):
    """Load pre-processed clustering and split data."""
    # Load split properties
    split_properties_filtered = pd.read_hdf(
        save_dir + f"data_filtered_{n_nodes}.h5", index_col=0
    )

    # Load masks and weights
    with gzip.open(save_dir + f"/masks_dict.pklz", "rb") as f:
        masks_dict = pickle.load(f)
    with gzip.open(save_dir + f"/weights_filtered_dict.pklz", "rb") as f:
        weights_dict = pickle.load(f)

    # Load blackout vectors
    with gzip.open(
        save_dir + f"blackout_vectors_filtered_dict_{n_nodes}.pklz", "rb"
    ) as f:
        blackout_vectors_filtered_dict = pickle.load(f)

    # Load failed edges indicator
    failed_edges_indicator_vectors_filtered = load_masked_indicator_vectors(
        "failed_edges", masks_dict=masks_dict
    )

    # # Load distance matrix
    # distance_matrix_path = (
    #     save_dir + f"/distance_matrix_n{n_nodes}_bACC_katz_maxD3_decay1.npy"
    # )
    # distance_matrix = np.load(distance_matrix_path)

    return {
        "split_properties_filtered": split_properties_filtered,
        "masks_dict": masks_dict,
        "weights_dict": weights_dict,
        "blackout_vectors_filtered_dict": blackout_vectors_filtered_dict,
        "failed_edges_indicator_vectors_filtered": failed_edges_indicator_vectors_filtered,
        # "distance_matrix": distance_matrix,
    }


def setup_colormaps():
    """Setup colormaps for nodes and edges."""
    node_cmap = plt.get_cmap("plasma_r")
    node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
    node_cmap.set_under("gainsboro", 1.0)

    edge_cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
    edge_cmap.set_under("gainsboro", 1.0)

    return node_cmap, edge_cmap


def find_latest_clustering_file(save_dir):
    """Find the most recent clustering file."""
    clustering_files = [
        f for f in os.listdir(save_dir) if "fitted" in f and f.endswith(".pklz")
    ]
    if not clustering_files:
        raise FileNotFoundError(f"No clustering files found in {save_dir}")

    # Sort by modification time and take the most recent
    clustering_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True
    )
    return clustering_files[0]


def find_best_clustering_file(save_dir):
    """Find the best clustering file."""
    # read clustering results df
    clust_res = pd.read_csv(
        os.path.join(save_dir, "clustering_res_info.csv"), index_col=0
    )
    clust_res = clust_res.sort_values(by="silhouette_score", ascending=False)
    clustering_files = clust_res.index.tolist()
    return clustering_files[0]


def create_clustering_analysis_plot(fname=""):
    """Create the main clustering analysis plot."""

    # Load data
    print("Loading clustering data...")
    data = load_clustering_data()
    nx_graph = data["nx_graph"]
    pos = data["pos"]
    co2l_list = data["co2l_list"]
    save_dir = data["save_dir"]
    num_failures_weighted = data["num_failures_weighted"]
    n_nodes = data["n_nodes"]

    # Find and load clustering results
    print("Loading clustering results...")
    if not fname:
        fname = find_best_clustering_file(save_dir)
    print(f"Using clustering file: {fname}")
    clustering_res_path = os.path.join(save_dir, fname)

    print("Loading processed data...")
    processed_data = load_processed_data(save_dir, n_nodes)
    split_properties_filtered = processed_data["split_properties_filtered"]
    masks_dict = processed_data["masks_dict"]
    weights_dict = processed_data["weights_dict"]
    blackout_vectors_filtered_dict = processed_data["blackout_vectors_filtered_dict"]
    failed_edges_indicator_vectors_filtered = processed_data[
        "failed_edges_indicator_vectors_filtered"
    ]
    # distance_matrix = processed_data["distance_matrix"]

    # Setup colormaps
    node_cmap, edge_cmap = setup_colormaps()

    # Prepare data arrays
    weights = np.concatenate([weights_dict[co2l] for co2l in co2l_list])
    print("total weighted number of events:", weights.sum())
    blackout_vectors_filtered = np.concatenate(
        list(blackout_vectors_filtered_dict.values())
    )
    split_lost_load = split_properties_filtered.lost_load_share_blackout
    split_weighted_lost_load = split_lost_load * weights
    co2l_masks = [(split_properties_filtered.co2l == co2l).values for co2l in co2l_list]

    # Check required files exist
    labels_all_path = clustering_res_path.replace("fitted.pklz", "labels_all.npy")
    group_masks_path = clustering_res_path.replace("fitted.pklz", "group_masks.pklz")

    if not os.path.exists(labels_all_path):
        raise FileNotFoundError(f"Required file not found: {labels_all_path}")
    if not os.path.exists(group_masks_path):
        raise FileNotFoundError(f"Required file not found: {group_masks_path}")

    # Load clustering results
    with gzip.open(clustering_res_path, "rb") as f:
        clustering_res = pickle.load(f)

    labels_all = np.load(labels_all_path, allow_pickle=True)
    print(f"{clustering_res.labels_.shape[0]} unique blackouts.")
    exit()

    with gzip.open(group_masks_path, "rb") as f:
        group_masks = pickle.load(f)

    # Process clustering results
    labels, counts = np.unique(clustering_res.labels_, return_counts=True)
    n_clusters = len(labels)

    # Create centroids dataframe
    centroids_df = pd.DataFrame(index=labels)
    centroids_df["weighted_lost_load"] = np.array(
        [np.sum(split_weighted_lost_load[labels_all == i]) for i in labels]
    )
    centroids_df["lost_load_share"] = (
        centroids_df["weighted_lost_load"] / centroids_df["weighted_lost_load"].sum()
    )
    centroids_df["n_samples"] = np.array(
        [weights[group_masks[label]].sum() for label in labels]
    )

    # Calculate centroids
    centroids = {
        label: blackout_vectors_filtered[group_masks[label]].T
        @ weights[group_masks[label]]
        / group_masks[label].sum()
        for label in labels
    }

    edge_centroids = {
        label: failed_edges_indicator_vectors_filtered[group_masks[label]].T
        @ weights[group_masks[label]]
        / group_masks[label].sum()
        for label in labels
    }

    # Sort by weighted lost load
    centroids_df = centroids_df.sort_values(by="weighted_lost_load", ascending=False)

    # === CREATE THE PLOT ===
    print("Creating visualization...")

    # Plot parameters
    n_subplots = 12  # Show top 12 clusters
    ncols = 3
    n_rows = int(np.ceil(n_subplots / ncols))
    fig_scaling = 4

    # Color scale parameters
    vmax = 1.0
    vmin = 0.001
    vmin_edge = 1e-3
    vmax_edge = 1.0
    node_cbar_label = "blackout probability"

    # CO2 levels for histograms
    if len(co2l_list) > 3:
        co2_lvls_hist = [0.6, 0.2, 0.0]
    else:
        co2_lvls_hist = co2l_list
    co2l_inds_hist = [
        np.where(np.array(co2l_list) == co2l)[0][0] for co2l in co2_lvls_hist
    ]

    # Create figure
    fig = plt.figure(figsize=(ncols * fig_scaling, n_rows * fig_scaling * 1.3))

    gs = GridSpec(2, 1, figure=fig, hspace=0.15, height_ratios=[0.05, n_rows][::-1])
    gs_legend_colorax = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0])
    gs_group = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs[0, 0], hspace=0.2)

    plot_count = 0
    df_iter = centroids_df.iterrows()

    # Plot clusters
    for row in range(n_rows):
        if plot_count >= n_subplots:
            break

        gs_row = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_group[row, 0], hspace=-0.05, height_ratios=[3, 1]
        )
        gs_row_map = GridSpecFromSubplotSpec(
            1, ncols, subplot_spec=gs_row[0, 0], wspace=0.0
        )
        gs_row_hist = GridSpecFromSubplotSpec(
            1, ncols, subplot_spec=gs_row[1, 0], wspace=0.18, hspace=0.5
        )

        for column in range(ncols):
            if plot_count >= n_subplots:
                break

            label, centroid_row = next(df_iter, (None, None))
            if centroid_row is None or label is None:
                break

            plot_count += 1

            centroid = centroids[label]
            failed_edges_prob = edge_centroids[label]

            # Create map subplot
            ax = fig.add_subplot(gs_row_map[0, column])

            plot_centroid_with_failures(
                nx_graph,
                pos,
                edge_centroids,
                node_cmap,
                edge_cmap,
                vmax,
                vmin,
                vmin_edge,
                vmax_edge,
                centroid,
                failed_edges_prob,
                ax,
            )

            ax.axis("off")

            # Add cluster information
            subtitle = f"={round(centroid_row.lost_load_share*100, ndigits=1)}\%\nn={round(centroid_row.n_samples/centroids_df.n_samples.sum()*100)}\%"
            subtitle = r"$\bar{R}$" + subtitle
            ax.set_title(
                plot_count,
                y=0.90,
                x=-0.05,
                fontsize=TITLE_FONTSIZE,
                loc="left",
            )
            ax.text(
                -15,
                52.5,
                subtitle,
                fontsize=LABEL_FONTSIZE,
            )

            # Create histogram subplot
            ax_hist = fig.add_subplot(gs_row_hist[0, column])
            plot_group_lost_load_hist_by_co2_single(
                group_masks[label],
                split_lost_load,
                co2l_masks,
                co2l_inds_hist,
                co2_lvls_hist,
                weights,
                n_failures_weighted=num_failures_weighted,
                ax=ax_hist,
            )
            ax_hist.grid(True)

            if column != 0:
                ax_hist.set_ylabel("")
            ax_hist.tick_params(axis="y", which="major", pad=0)
            ax_hist.set_xticks(np.arange(0, 101, step=20))
            ax_hist.set_xlabel("Blackout size [\%]", rotation=0, labelpad=0)
            h, l = ax_hist.get_legend_handles_labels()
            ax_hist.legend().set_visible(False)

    # Add legend and colorbars
    ax_hist_legend = fig.add_subplot(gs_legend_colorax[0])
    ax_hist_legend.axis("off")
    if "h" in locals() and "l" in locals():
        ax_hist_legend.legend(
            h,
            l,
            loc="center",
            fontsize=LEGEND_FONTSIZE,
            ncols=1,
            title=r"CO$_2$ level [\% of 1990]",
        )

    # Node colorbar
    cbar_ax_node = fig.add_subplot(gs_legend_colorax[1])
    sm_node = plt.cm.ScalarMappable(
        cmap=node_cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax)
    )
    cb_node = fig.colorbar(sm_node, cax=cbar_ax_node, orientation="horizontal")
    cb_node.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, width=1.0, which="both")
    cb_node.ax.set_xlabel(node_cbar_label, fontsize=COLORBAR_LABEL_FONTSIZE, rotation=0)
    cb_node.ax.xaxis.set_label_position("top")

    # Edge colorbar
    cbar_ax_edge = fig.add_subplot(gs_legend_colorax[2])
    sm_edge = plt.cm.ScalarMappable(
        cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
    )
    cb_edge = fig.colorbar(sm_edge, cax=cbar_ax_edge, orientation="horizontal")
    cb_edge.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, width=1.0, which="both")
    cb_edge.ax.set_xlabel(
        "line failure probability", fontsize=COLORBAR_LABEL_FONTSIZE, rotation=0
    )
    cb_edge.ax.xaxis.set_label_position("top")

    # Save the plot
    save_figure(fig, path_to_figures_sclopf, f"clustering_analysis_{n_subplots}")
    print(
        f"Plot saved to: {os.path.join(path_to_figures_sclopf, f'clustering_analysis_{n_subplots}.pdf')}"
    )

    return fig


if __name__ == "__main__":
    try:
        fig = create_clustering_analysis_plot()
        print("Clustering analysis plot generated successfully!")
    except Exception as e:
        print(f"Error generating clustering analysis plot: {e}")
        raise
