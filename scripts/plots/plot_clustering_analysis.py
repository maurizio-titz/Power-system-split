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

from tqdm import tqdm


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
from utils.clustering.data_handling import get_path_to_clustering_dir, load_clustering
from utils.data_handling import get_co2_levels, get_actual_co2_level
from utils.config import path_to_clustering_results_sclopf, path_to_vis_results_sclopf
from utils.clustering_visualisation import *
from utils import data_handling
from utils.config import path_to_pypsa_network_sclopf, path_to_figures_sclopf
from utils import cascade_simulation
from scripts.plots.plot_combined_generation_storage import LABEL_FONTSIZE
from utils.plot_style import *

# Import plot styling
setup_matplotlib_style()


def load_clustering_data(
    n_nodes=600,
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
    try:
        file_path = data_handling.path_to_grid_data + f"n_2_failures_co2lvl{0.0}.pklz"
        with gzip.open(file_path, "rb") as fh:
            n_2_failures = pickle.load(fh)
    except FileNotFoundError:
        bridge_idxs = data_handling.nx_edges_to_matrix_indices(
            nx.bridges(nx_graph), nx_graph
        )
        n_2_failures = cascade_simulation.calc_possible_double_line_failures(
            num_parallels, ignored_idxs=bridge_idxs
        )
    weighted_trigger_count = sum(
        [initial_failure["weight"] for initial_failure in n_2_failures]
    )
    # incorporating the weighting of the snapshots
    num_failures_weighted = (
        weighted_trigger_count * network.snapshot_weightings.generators
    ).sum()

    return {
        "nx_graph": nx_graph,
        "pos": pos,
        "network": network,
        "num_failures_weighted": num_failures_weighted,
    }


def load_processed_data(save_dir, n_nodes):
    """Load pre-processed clustering and split data."""
    # Load split properties
    split_properties_filtered = pd.read_hdf(save_dir + f"data_filtered.h5", index_col=0)

    # Load masks and weights
    with gzip.open(save_dir + f"/masks_dict.pklz", "rb") as f:
        masks_dict = pickle.load(f)
    # with gzip.open(save_dir + f"/weights_filtered_dict.pklz", "rb") as f:
    #     weights_dict = pickle.load(f)

    # Load blackout vectors
    with gzip.open(save_dir + f"blackout_vectors_filtered_dict.pklz", "rb") as f:
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
        # "weights_dict": weights_dict,
        "blackout_vectors_filtered_dict": blackout_vectors_filtered_dict,
        "failed_edges_indicator_vectors_filtered": failed_edges_indicator_vectors_filtered,
        # "distance_matrix": distance_matrix,
    }


def setup_colormaps():
    """Setup colormaps for nodes and edges."""
    node_cmap = plt.get_cmap("plasma_r")
    node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
    node_cmap.set_under("gainsboro", 1.0)

    edge_cmap = copy.copy(mpl.cm.get_cmap("inferno_r"))
    edge_cmap = truncate_colormap(edge_cmap, 0.1, 0.95, 1000)
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


def find_best_clustering_file(save_dir, alg_filter=None):
    """Find the best clustering file."""
    # read clustering results df
    clust_res = pd.read_csv(
        os.path.join(save_dir, "clustering_res_info.csv"), index_col=0
    )
    clust_res = clust_res.sort_values(by="silhouette_score", ascending=False)
    clustering_files = clust_res.filename.tolist()
    if isinstance(alg_filter, str):
        alg_filter = [alg_filter]
    if alg_filter:
        clustering_files = [
            f for f in clustering_files if all(alg in f for alg in alg_filter)
        ]
    return clustering_files[0]


def classes_to_nonNeg(class_vecs):
    if class_vecs.dtype != np.integer:
        raise ValueError("class_vecs must be of integer type.")
    min_class = class_vecs.min()
    if min_class < 0:
        # assume classes are min, min+1 ... max
        class_counts = 0
        max_class = class_vecs.max()
        for class_val in range(min_class, max_class + 1):
            class_counts += np.sum(class_vecs == class_val)
        if class_counts != class_vecs.size:
            raise ValueError(
                "class_vecs contain non-consecutive class labels with negative values."
            )
        classes = np.arange(min_class, max_class + 1)
        class_vecs_nonNeg = np.zeros_like(class_vecs)
        if len(classes) > 3:
            raise ValueError("More than 3 classes found in class_vecs.")
        classes = np.sort(classes)
        old_class_to_new = {old_class: i for i, old_class in enumerate(classes)}
        for i, class_val in enumerate(classes):
            class_vecs_nonNeg[class_vecs == class_val] = i
        return class_vecs_nonNeg, old_class_to_new
    else:
        return class_vecs, None


def create_clustering_analysis_plot(
    clustering_res_full_path="",
    n_subplots=12,
    alg_filter=None,
    average_over_classes=False,
    colors_classes=["blue", "lightgray", "red"],
    edge_log_scale=True,
    plot_dir=path_to_figures_sclopf,
    n_nodes=600,
    co2l_list=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    sort_by="accumulative_lost_load",
):
    """Create the main clustering analysis plot.
    Parameters
    sort_by: "accumulative_lost_load" or "frequency"
    """

    # Load data
    print("Loading clustering data...")
    data = load_clustering_data(n_nodes=n_nodes)
    nx_graph = data["nx_graph"]
    pos = data["pos"]
    num_failures_weighted = data["num_failures_weighted"]
    cluster_res_dir = os.path.dirname(clustering_res_full_path) + "/"
    fname = os.path.basename(clustering_res_full_path)

    # Find and load clustering results
    print("Loading clustering results...")
    if not fname:
        fname = find_best_clustering_file(cluster_res_dir, alg_filter)
    print(f"Using clustering file: {fname}")
    clustering_res_path = os.path.join(cluster_res_dir, fname)

    print("Loading processed data...")
    processed_data = load_processed_data(cluster_res_dir + "../data/", n_nodes)
    split_properties_filtered = processed_data["split_properties_filtered"]
    weights_filtered = split_properties_filtered.total_weighting
    blackout_vectors_filtered_dict = processed_data["blackout_vectors_filtered_dict"]
    failed_edges_indicator_vectors_filtered = processed_data[
        "failed_edges_indicator_vectors_filtered"
    ]
    # distance_matrix = processed_data["distance_matrix"]

    # Setup colormaps
    node_cmap, edge_cmap = setup_colormaps()

    # Prepare data arrays
    try:
        with gzip.open(
            clustering_res_path.replace(".pklz", "_centroid_res.pklz"), "rb"
        ) as f:
            centroid_res = pickle.load(f)
        weights = centroid_res["weights"]
        split_lost_load = centroid_res["split_lost_load"]
        co2l_masks = centroid_res["co2l_masks"]
        group_masks = centroid_res["group_masks"]
        centroids_df = centroid_res["centroids_df"]
        centroids = centroid_res["centroids"]
        edge_centroids = centroid_res["edge_centroids"]
        print("Loaded pre-computed centroids from disk.")
    except FileNotFoundError:
        print("Computing centroids from clusters...")
        (
            weights,
            split_lost_load,
            co2l_masks,
            group_masks,
            centroids_df,
            centroids,
            edge_centroids,
        ) = compute_centroids_from_clusters(
            average_over_classes,
            co2l_list,
            clustering_res_path,
            split_properties_filtered,
            weights_filtered,
            blackout_vectors_filtered_dict,
            failed_edges_indicator_vectors_filtered,
        )
    if sort_by == "frequency":
        centroids_df = centroids_df.sort_values(by="n_samples", ascending=False)
    elif sort_by == "accumulative_lost_load":
        centroids_df = centroids_df.sort_values(
            by="weighted_lost_load", ascending=False
        )
    else:
        raise ValueError("sort_by must be 'frequency' or 'accumulative_lost_load'")

    # === CREATE THE PLOT ===
    print("Creating visualization...")

    # Plot parameters
    ncols = 4
    n_rows = int(np.ceil(n_subplots / ncols))
    fig_scaling = 4

    # Color scale parameters
    vmax = 1.0
    vmin = 0.001
    if edge_log_scale:
        vmin_edge = 1e-3
        vmax_edge = 1.0
    else:
        vmin_edge = 0.0
        vmax_edge = 1.0
    node_cbar_label = "blackout probability"

    # CO2 levels for histograms
    co2_lvls_hist = [0.6, 0.2, 0.0]  # default levels
    if set(co2l_list) >= set(co2_lvls_hist):
        pass
    else:  # select first, last and middle levels available
        if len(co2l_list) >= 3:
            co2_lvls_hist = [
                co2l_list[0],
                co2l_list[len(co2l_list) // 2],
                co2l_list[-1],
            ]
        else:
            co2_lvls_hist = co2l_list

    co2l_inds_hist = [
        np.where(np.array(co2l_list) == co2l)[0][0] for co2l in co2_lvls_hist
    ]

    # Create figure
    fig = plt.figure(figsize=(ncols * fig_scaling, n_rows * fig_scaling * 1.3))

    gs = GridSpec(2, 1, figure=fig, hspace=0.15, height_ratios=[0.05, n_rows][::-1])

    gs_clusters = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs[0, 0], hspace=0.2)

    plot_count = 0
    df_iter = centroids_df.iterrows()

    print(f"Plotting top {n_subplots} clusters...")
    # Plot clusters
    for row in range(n_rows):
        if plot_count >= n_subplots:
            break

        gs_row = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_clusters[row, 0],
            hspace=-0.05,
            height_ratios=[3, 1],
            wspace=-0.01,
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
                node_cmap,
                edge_cmap,
                vmax,
                vmin,
                vmin_edge,
                vmax_edge,
                centroid,
                failed_edges_prob,
                ax,
                colors_classes=colors_classes,
                radius=0.4,
                edge_log_scale=edge_log_scale,
            )

            ax.axis("off")

            # Add cluster information
            cluster_label = (
                f"{plot_count}"
                rf"\\"
                rf"$R={round(centroid_row.lost_load_share*100, ndigits=1)}\%$"
                rf"\\"
                rf"$\beta={round(centroid_row.n_samples/centroids_df.n_samples.sum()*100)}\%$"
            )
            # cluster_label = f"{plot_count}," rf"\\" f"test line2," rf"\\" f"test line3"
            ax.set_title(
                cluster_label,
                y=0.8,
                x=-0,
                fontsize=LABEL_FONTSIZE,
                loc="left",
                horizontalalignment="left",
            )
            # ax.text(
            #     -15,
            #     52.5,
            #     subtitle,
            #     fontsize=LABEL_FONTSIZE,
            # )

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

            ax_hist.tick_params(axis="y", which="major", pad=0)
            ax_hist.set_xticks(np.arange(0, 101, step=20))
            ax_hist.set_xlabel("Blackout size [\%]", rotation=0, labelpad=0)
            if column != 0:
                ax_hist.set_ylabel("")
                ax_hist.set_yticklabels([])

            h, l = ax_hist.get_legend_handles_labels()
            ax_hist.legend().set_visible(False)

    if centroid.ndim == 1:
        gs_legend_colorax = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 0.5, 0.5]
        )
    else:
        gs_legend_colorax = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 1, 0.5]
        )
    # Add legend and colorbars
    ax_hist_legend = fig.add_subplot(gs_legend_colorax[0])
    ax_hist_legend.axis("off")
    if "h" in locals() and "l" in locals():
        l = [
            label if label.endswith("\%") else r"{}".format(label.replace("%", "\%"))
            for label in l
        ]
        ax_hist_legend.legend(
            h,
            l,
            loc="center",
            fontsize=LEGEND_FONTSIZE,
            ncols=3,
            title=r"CO$_2$ level [\% of 1990]",
            handletextpad=0.2,
            columnspacing=0.7,
        )

    # Node colorbar
    if centroid.ndim == 1:
        cbar_ax_node = fig.add_subplot(gs_legend_colorax[1])
        sm_node = plt.cm.ScalarMappable(
            cmap=node_cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax)
        )
        cb_node = fig.colorbar(sm_node, cax=cbar_ax_node, orientation="horizontal")
        cb_node.ax.tick_params(
            labelsize=COLORBAR_TICK_FONTSIZE, width=1.0, which="both"
        )
        cb_node.ax.set_xlabel(
            node_cbar_label, fontsize=COLORBAR_LABEL_FONTSIZE, rotation=0
        )
        cb_node.ax.xaxis.set_label_position("top")
    else:
        cbar_ax_node = fig.add_subplot(gs_legend_colorax[1])

        class_labels = [
            r"RoCoF$<-1$",
            "stable",
            r"RoCoF$>1$",
        ]
        legend_handles = []
        for i, color in enumerate(colors_classes):
            # use a rectangular patch for the legend handle
            handle = mpl.patches.Rectangle(
                (0, 0),
                width=1.0,
                height=0.6,
                facecolor=color,
                edgecolor="black",
                label=class_labels[i] if i < len(class_labels) else f"Class {i}",
            )
            legend_handles.append(handle)

        # Create the legend
        cbar_ax_node.legend(
            handles=legend_handles,
            loc="center",
            fontsize=COLORBAR_LABEL_FONTSIZE,
            title="Node Classes",
            title_fontsize=COLORBAR_LABEL_FONTSIZE,
            ncol=len(colors_classes),
            handletextpad=0.2,
            columnspacing=0.7,
        )
        cbar_ax_node.axis("off")

    # Edge colorbar
    cbar_ax_edge = fig.add_subplot(gs_legend_colorax[2])
    if edge_log_scale:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
        )
    else:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.Normalize(vmin=vmin_edge, vmax=vmax_edge)
        )
    cb_edge = fig.colorbar(sm_edge, cax=cbar_ax_edge, orientation="horizontal")
    cb_edge.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, width=1.0, which="both")
    cb_edge.ax.set_xlabel(
        "line failure probability", fontsize=COLORBAR_LABEL_FONTSIZE, rotation=0
    )
    cb_edge.ax.xaxis.set_label_position("top")

    # Save the plot
    save_name = f"clustering_analysis_{ncols}cols_{n_subplots}"
    if sort_by:
        save_name += f"_sortedBy{sort_by.replace('_', ' ').title().replace(' ', '')[0].lower() + sort_by.replace('_', ' ').title().replace(' ', '')[1:]}"
    if fname:
        save_name += f"_{fname.replace('.pklz','')}"
    save_figure(fig, plot_dir, save_name)
    print(f"Plot saved to: {os.path.join(plot_dir, f'{save_name}.pdf')}")

    return fig


def compute_centroids_from_clusters(
    average_over_classes,
    co2l_list,
    clustering_res_path,
    split_properties_filtered,
    weights_filtered,
    blackout_vectors_filtered_dict,
    failed_edges_indicator_vectors_filtered,
):
    weights = weights_filtered.values
    print("total weighted number of events:", weights.sum())
    blackout_vectors_filtered = np.concatenate(
        [
            blackout_vectors_filtered_dict[co2l]
            for co2l in split_properties_filtered.co2l.unique()
        ]
    )
    split_lost_load = split_properties_filtered.lost_load_share_blackout
    split_weighted_lost_load = split_lost_load * weights
    co2l_masks = [(split_properties_filtered.co2l == co2l).values for co2l in co2l_list]

    # Check required files exist
    labels_all_path = clustering_res_path.replace(".pklz", "_labels_all.npy")
    group_masks_path = clustering_res_path.replace(".pklz", "_group_masks.pklz")

    if not os.path.exists(labels_all_path):
        raise FileNotFoundError(f"Required file not found: {labels_all_path}")
    if not os.path.exists(group_masks_path):
        raise FileNotFoundError(f"Required file not found: {group_masks_path}")

    print("Processing clustering results...")
    # Load clustering results
    with gzip.open(clustering_res_path, "rb") as f:
        clustering_res = pickle.load(f)["model"]

    labels_all = np.load(labels_all_path, allow_pickle=True)
    print(f"{clustering_res.labels_.shape[0]} unique blackouts.")

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
    classes = np.unique(blackout_vectors_filtered)
    n_classes = len(classes)
    if average_over_classes:
        centroids = {
            label: blackout_vectors_filtered[group_masks[label]].T
            @ weights[group_masks[label]]
            / weights[group_masks[label]].sum()
            for label in labels
        }
    else:
        # map classes to non-negative
        centroids = {}
        blackout_vectors_filtered_nonNeg, old_class_to_new = classes_to_nonNeg(
            blackout_vectors_filtered
        )
        for label in tqdm(labels, desc="Calculating centroids"):
            weighted_node_class_probs = (
                np.array(
                    [
                        np.bincount(
                            outcomes_node,
                            minlength=n_classes,
                            weights=weights[group_masks[label]],
                        )
                        for outcomes_node in blackout_vectors_filtered_nonNeg[
                            group_masks[label]
                        ].T
                    ],
                )
                / weights[group_masks[label]].sum()
            )  # n_nodes x n_classes
            centroids[label] = weighted_node_class_probs

    edge_centroids = {
        label: failed_edges_indicator_vectors_filtered[group_masks[label]].T
        @ weights[group_masks[label]]
        / weights[
            group_masks[label]
        ].sum()  #! chech if replacing below line with this is correct!
        # / group_masks[label].sum()
        for label in labels
    }

    # Sort by weighted lost load
    centroids_df = centroids_df.sort_values(by="weighted_lost_load", ascending=False)

    # save all variables to disk
    save_path = clustering_res_path.replace(".pklz", "_centroid_res.pklz")
    with gzip.GzipFile(save_path, "wb") as f:
        pickle.dump(
            {
                "weights": weights,
                "split_lost_load": split_lost_load,
                "co2l_masks": co2l_masks,
                "group_masks": group_masks,
                "centroids_df": centroids_df,
                "centroids": centroids,
                "edge_centroids": edge_centroids,
            },
            f,
        )

    return (
        weights,
        split_lost_load,
        co2l_masks,
        group_masks,
        centroids_df,
        centroids,
        edge_centroids,
    )


# def analyse_clusters_temporal_occurence_patterns(
#     fname="", alg_filter=None, n_clusters_to_analyze=12
# ):
#     """
#     Create individual histograms for each cluster showing occurrence by hour and month.
#     For each cluster, creates plots for CO2 levels 0%, 20%, and 60%.

#     Parameters
#     ----------
#     fname : str
#         Clustering filename to use
#     alg_filter : str, optional
#         Algorithm filter for clustering file selection
#     n_clusters_to_analyze : int, default 4
#         Number of top clusters to analyze
#     """

#     # Load data
#     print("Loading clustering data...")
#     data = load_clustering_data()
#     nx_graph = data["nx_graph"]
#     pos = data["pos"]
#     co2l_list = data["co2l_list"]
#     save_dir = data["save_dir"]
#     num_failures_weighted = data["num_failures_weighted"]
#     n_nodes = data["n_nodes"]

#     # Find and load clustering results
#     print("Loading clustering results...")
#     if not fname:
#         fname = find_best_clustering_file(save_dir, alg_filter)
#     print(f"Using clustering file: {fname}")
#     clustering_res_path = os.path.join(save_dir, fname)

#     print("Loading processed data...")
#     processed_data = load_processed_data(save_dir, n_nodes)
#     split_properties_filtered = processed_data["split_properties_filtered"]
#     masks_dict = processed_data["masks_dict"]
#     weights_dict = processed_data["weights_dict"]
#     blackout_vectors_filtered_dict = processed_data["blackout_vectors_filtered_dict"]
#     failed_edges_indicator_vectors_filtered = processed_data[
#         "failed_edges_indicator_vectors_filtered"
#     ]
#     # distance_matrix = processed_data["distance_matrix"]

#     # Setup colormaps
#     node_cmap, edge_cmap = setup_colormaps()

#     # Prepare data arrays
#     weights = np.concatenate([weights_dict[co2l] for co2l in co2l_list])
#     print("total weighted number of events:", weights.sum())
#     blackout_vectors_filtered = np.concatenate(
#         list(blackout_vectors_filtered_dict.values())
#     )
#     split_lost_load = split_properties_filtered.lost_load_share_blackout
#     split_weighted_lost_load = split_lost_load * weights
#     co2l_masks = [(split_properties_filtered.co2l == co2l).values for co2l in co2l_list]

#     # Check required files exist
#     labels_all_path = clustering_res_path.replace(".pklz", "_labels_all.npy")
#     group_masks_path = clustering_res_path.replace(".pklz", "_group_masks.pklz")

#     if not os.path.exists(labels_all_path):
#         raise FileNotFoundError(f"Required file not found: {labels_all_path}")
#     if not os.path.exists(group_masks_path):
#         raise FileNotFoundError(f"Required file not found: {group_masks_path}")

#     print("Processing clustering results...")
#     # Load clustering results
#     with gzip.open(clustering_res_path, "rb") as f:
#         clustering_res = pickle.load(f)

#     labels_all = np.load(labels_all_path, allow_pickle=True)
#     print(f"{clustering_res.labels_.shape[0]} unique blackouts.")

#     with gzip.open(group_masks_path, "rb") as f:
#         group_masks = pickle.load(f)

#     # Process clustering results
#     labels, counts = np.unique(clustering_res.labels_, return_counts=True)
#     n_clusters = len(labels)

#     # Create centroids dataframe
#     centroids_df = pd.DataFrame(index=labels)
#     centroids_df["weighted_lost_load"] = np.array(
#         [np.sum(split_weighted_lost_load[labels_all == i]) for i in labels]
#     )
#     centroids_df["lost_load_share"] = (
#         centroids_df["weighted_lost_load"] / centroids_df["weighted_lost_load"].sum()
#     )
#     centroids_df["n_samples"] = np.array(
#         [weights[group_masks[label]].sum() for label in labels]
#     )

#     # Calculate centroids
#     centroids = {
#         label: blackout_vectors_filtered[group_masks[label]].T
#         @ weights[group_masks[label]]
#         / weights[group_masks[label]].sum()
#         for label in labels
#     }

#     edge_centroids = {
#         label: failed_edges_indicator_vectors_filtered[group_masks[label]].T
#         @ weights[group_masks[label]]
#         / group_masks[label].sum()
#         for label in labels
#     }

#     # Sort by weighted lost load
#     centroids_df = centroids_df.sort_values(by="weighted_lost_load", ascending=False)

#     # Analyze temporal patterns
#     print("Analyzing temporal occurrence patterns...")

#     # Get time information from split properties
#     split_properties_filtered["time_stamp"] = pd.to_datetime(
#         split_properties_filtered.index.get_level_values("time_stamp")
#     )
#     split_properties_filtered["hour"] = split_properties_filtered["time_stamp"].dt.hour
#     split_properties_filtered["month"] = split_properties_filtered[
#         "time_stamp"
#     ].dt.month
#     split_properties_filtered["day_of_year"] = split_properties_filtered[
#         "time_stamp"
#     ].dt.dayofyear

#     # Select top clusters to analyze (e.g., top 6)
#     top_clusters = centroids_df.head(n_clusters_to_analyze).index.tolist()

#     # CO2 levels to analyze
#     co2_levels_to_plot = [0.0, 0.2, 0.6]  # 0%, 20%, 60%
#     co2_level_names = ["0%", "20%", "60%"]

#     # Create individual plots for each cluster
#     for cluster_idx, label in enumerate(top_clusters):
#         print(f"Creating plots for Cluster {cluster_idx + 1}")

#         # Create figure with 2 rows (hour, month) and 3 columns (CO2 levels)
#         fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#         fig.suptitle(
#             f"Cluster {cluster_idx + 1} - Temporal Occurrence Patterns", fontsize=16
#         )

#         cluster_mask = labels_all == label
#         cluster_data = split_properties_filtered[cluster_mask]
#         cluster_weights = weights[cluster_mask]

#         # Create histograms for each CO2 level
#         for co2_idx, co2_level in enumerate(co2_levels_to_plot):
#             # Filter data for this CO2 level
#             co2_mask = cluster_data["co2l"] == co2_level
#             co2_cluster_data = cluster_data[co2_mask]
#             co2_cluster_weights = cluster_weights[co2_mask]

#             if len(co2_cluster_data) == 0:
#                 # No data for this CO2 level, create empty plots
#                 axes[0, co2_idx].text(
#                     0.5,
#                     0.5,
#                     "No data",
#                     ha="center",
#                     va="center",
#                     transform=axes[0, co2_idx].transAxes,
#                 )
#                 axes[1, co2_idx].text(
#                     0.5,
#                     0.5,
#                     "No data",
#                     ha="center",
#                     va="center",
#                     transform=axes[1, co2_idx].transAxes,
#                 )
#                 axes[0, co2_idx].set_title(
#                     f"CO$_2$ Level: {co2_level_names[co2_idx]} - Hourly Distribution"
#                 )
#                 axes[1, co2_idx].set_title(
#                     f"CO$_2$ Level: {co2_level_names[co2_idx]} - Monthly Distribution"
#                 )
#                 continue

#             # Hourly distribution (top row)
#             ax_hour = axes[0, co2_idx]
#             hourly_counts = np.zeros(24)
#             for hour in range(24):
#                 hour_mask = co2_cluster_data["hour"] == hour
#                 if hour_mask.any():
#                     hourly_counts[hour] = co2_cluster_weights[hour_mask].sum()

#             # Normalize to percentage
#             if hourly_counts.sum() > 0:
#                 hourly_counts = hourly_counts / hourly_counts.sum() * 100

#             ax_hour.bar(range(24), hourly_counts, alpha=0.7, color=f"C{cluster_idx}")
#             ax_hour.set_title(
#                 f"CO$_2$ Level: {co2_level_names[co2_idx]} - Hourly Distribution"
#             )
#             ax_hour.set_xlabel("Hour of Day")
#             ax_hour.set_ylabel("Occurrence [%]")
#             ax_hour.set_xticks(range(0, 24, 4))
#             ax_hour.grid(True, alpha=0.3)

#             # Monthly distribution (bottom row)
#             ax_month = axes[1, co2_idx]
#             monthly_counts = np.zeros(12)
#             for month in range(1, 13):
#                 month_mask = co2_cluster_data["month"] == month
#                 if month_mask.any():
#                     monthly_counts[month - 1] = co2_cluster_weights[month_mask].sum()

#             # Normalize to percentage
#             if monthly_counts.sum() > 0:
#                 monthly_counts = monthly_counts / monthly_counts.sum() * 100

#             ax_month.bar(
#                 range(1, 13), monthly_counts, alpha=0.7, color=f"C{cluster_idx}"
#             )
#             ax_month.set_title(
#                 f"CO$_2$ Level: {co2_level_names[co2_idx]} - Monthly Distribution"
#             )
#             ax_month.set_xlabel("Month")
#             ax_month.set_ylabel("Occurrence [%]")
#             ax_month.set_xticks(range(1, 13))
#             ax_month.set_xticklabels(
#                 ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
#             )
#             ax_month.grid(True, alpha=0.3)

#         plt.tight_layout()

#         # Save individual cluster plot
#         save_name = f"temporal_patterns_cluster_{cluster_idx + 1}"
#         if alg_filter:
#             save_name += f"_{alg_filter}"
#         if fname:
#             save_name += f"_{fname.replace('.pklz','')}"
#         save_figure(fig, path_to_figures_sclopf, save_name)
#         print(
#             f"Cluster {cluster_idx + 1} plot saved to: {os.path.join(path_to_figures_sclopf, f'{save_name}.pdf')}"
#         )

#         plt.show()

#     print(f"Temporal analysis completed for {n_clusters_to_analyze} clusters!")
#     return None
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle("Temporal Occurrence Patterns of Top Clusters", fontsize=16)

#     # Color palette for clusters
#     colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(top_clusters)))

#     # Plot 1: Hourly distribution
#     ax1 = axes[0, 0]
#     for i, label in enumerate(top_clusters):
#         cluster_mask = labels_all == label
#         cluster_data = split_properties_filtered[cluster_mask]
#         cluster_weights = weights[cluster_mask]

#         # Calculate weighted hourly distribution
#         hourly_counts = np.zeros(24)
#         for hour in range(24):
#             hour_mask = cluster_data["hour"] == hour
#             hourly_counts[hour] = cluster_weights[hour_mask].sum()

#         # Normalize to percentage
#         hourly_counts = hourly_counts / hourly_counts.sum() * 100

#         ax1.plot(
#             range(24),
#             hourly_counts,
#             "o-",
#             color=colors[i],
#             label=f"Cluster {i+1}",
#             alpha=0.7,
#             linewidth=2,
#         )

#     ax1.set_xlabel("Hour of Day")
#     ax1.set_ylabel("Occurrence Probability [%]")
#     ax1.set_title("Hourly Distribution")
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#     ax1.set_xticks(range(0, 24, 4))

#     # Plot 2: Monthly distribution
#     ax2 = axes[0, 1]
#     for i, label in enumerate(top_clusters):
#         cluster_mask = labels_all == label
#         cluster_data = split_properties_filtered[cluster_mask]
#         cluster_weights = weights[cluster_mask]

#         # Calculate weighted monthly distribution
#         monthly_counts = np.zeros(12)
#         for month in range(1, 13):
#             month_mask = cluster_data["month"] == month
#             monthly_counts[month - 1] = cluster_weights[month_mask].sum()

#         # Normalize to percentage
#         monthly_counts = monthly_counts / monthly_counts.sum() * 100

#         ax2.plot(
#             range(1, 13), monthly_counts, "o-", color=colors[i], alpha=0.7, linewidth=2
#         )

#     ax2.set_xlabel("Month")
#     ax2.set_ylabel("Occurrence Probability [%]")
#     ax2.set_title("Monthly Distribution")
#     ax2.grid(True, alpha=0.3)
#     ax2.set_xticks(range(1, 13))
#     ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

#     # Plot 3: Day of year heatmap
#     ax3 = axes[0, 2]
#     day_cluster_matrix = np.zeros((len(top_clusters), 366))

#     for i, label in enumerate(top_clusters):
#         cluster_mask = labels_all == label
#         cluster_data = split_properties_filtered[cluster_mask]
#         cluster_weights = weights[cluster_mask]

#         for day in range(1, 367):
#             day_mask = cluster_data["day_of_year"] == day
#             if day_mask.any():
#                 day_cluster_matrix[i, day - 1] = cluster_weights[day_mask].sum()

#     # Normalize each row
#     for i in range(len(top_clusters)):
#         if day_cluster_matrix[i, :].sum() > 0:
#             day_cluster_matrix[i, :] = (
#                 day_cluster_matrix[i, :] / day_cluster_matrix[i, :].sum()
#             )

#     im = ax3.imshow(
#         day_cluster_matrix, aspect="auto", cmap="viridis", interpolation="nearest"
#     )
#     ax3.set_xlabel("Day of Year")
#     ax3.set_ylabel("Cluster")
#     ax3.set_title("Daily Occurrence Patterns")
#     ax3.set_yticks(range(len(top_clusters)))
#     ax3.set_yticklabels([f"C{i+1}" for i in range(len(top_clusters))])

#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax3)
#     cbar.set_label("Normalized Occurrence")

#     # Plot 4: Hour vs Month heatmap for top cluster
#     ax4 = axes[1, 0]
#     top_label = top_clusters[0]
#     cluster_mask = labels_all == top_label
#     cluster_data = split_properties_filtered[cluster_mask]
#     cluster_weights = weights[cluster_mask]

#     hour_month_matrix = np.zeros((24, 12))
#     for hour in range(24):
#         for month in range(1, 13):
#             mask = (cluster_data["hour"] == hour) & (cluster_data["month"] == month)
#             if mask.any():
#                 hour_month_matrix[hour, month - 1] = cluster_weights[mask].sum()

#     # Normalize
#     if hour_month_matrix.sum() > 0:
#         hour_month_matrix = hour_month_matrix / hour_month_matrix.sum()

#     im4 = ax4.imshow(hour_month_matrix, aspect="auto", cmap="plasma", origin="lower")
#     ax4.set_xlabel("Month")
#     ax4.set_ylabel("Hour of Day")
#     ax4.set_title(f"Top Cluster: Hour vs Month")
#     ax4.set_xticks(range(12))
#     ax4.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
#     ax4.set_yticks(range(0, 24, 4))

#     cbar4 = plt.colorbar(im4, ax=ax4)
#     cbar4.set_label("Normalized Occurrence")

#     # Plot 5: CO2 level vs temporal patterns
#     ax5 = axes[1, 1]
#     for co2l in co2l_list[:3]:  # Show top 3 CO2 levels
#         co2_mask = split_properties_filtered["co2l"] == co2l
#         co2_data = split_properties_filtered[co2_mask]
#         co2_weights = weights[co2_mask]

#         # Calculate hourly distribution for this CO2 level
#         hourly_counts = np.zeros(24)
#         for hour in range(24):
#             hour_mask = co2_data["hour"] == hour
#             if hour_mask.any():
#                 hourly_counts[hour] = co2_weights[hour_mask].sum()

#         # Normalize
#         if hourly_counts.sum() > 0:
#             hourly_counts = hourly_counts / hourly_counts.sum() * 100

#         ax5.plot(
#             range(24),
#             hourly_counts,
#             "o-",
#             label=f"CO$_2$: {int(co2l*100)}%",
#             alpha=0.7,
#             linewidth=2,
#         )

#     ax5.set_xlabel("Hour of Day")
#     ax5.set_ylabel("Occurrence Probability [%]")
#     ax5.set_title("Hourly Distribution by CO$_2$ Level")
#     ax5.grid(True, alpha=0.3)
#     ax5.legend()
#     ax5.set_xticks(range(0, 24, 4))

#     # Plot 6: Seasonal patterns
#     ax6 = axes[1, 2]
#     seasons = ["Winter", "Spring", "Summer", "Fall"]
#     season_months = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

#     for i, label in enumerate(top_clusters[:3]):  # Show top 3 clusters
#         cluster_mask = labels_all == label
#         cluster_data = split_properties_filtered[cluster_mask]
#         cluster_weights = weights[cluster_mask]

#         seasonal_counts = []
#         for season_month_list in season_months:
#             season_mask = cluster_data["month"].isin(season_month_list)
#             seasonal_counts.append(cluster_weights[season_mask].sum())

#         # Normalize
#         total = sum(seasonal_counts)
#         if total > 0:
#             seasonal_counts = [count / total * 100 for count in seasonal_counts]

#         ax6.bar(
#             np.arange(len(seasons)) + i * 0.25,
#             seasonal_counts,
#             width=0.25,
#             label=f"Cluster {i+1}",
#             color=colors[i],
#             alpha=0.7,
#         )

#     ax6.set_xlabel("Season")
#     ax6.set_ylabel("Occurrence Probability [%]")
#     ax6.set_title("Seasonal Distribution")
#     ax6.set_xticks(np.arange(len(seasons)) + 0.25)
#     ax6.set_xticklabels(seasons)
#     ax6.legend()
#     ax6.grid(True, alpha=0.3, axis="y")

#     plt.tight_layout()

#     # Save the plot
#     save_name = f"temporal_patterns_{len(top_clusters)}clusters"
#     if alg_filter:
#         save_name += f"_{alg_filter}"
#     if fname:
#         save_name += f"_{fname.replace('.pklz','')}"
#     save_figure(fig, path_to_figures_sclopf, save_name)
#     print(
#         f"Temporal analysis plot saved to: {os.path.join(path_to_figures_sclopf, f'{save_name}.pdf')}"
#     )

#     return fig


if __name__ == "__main__":

    for n_subplots in [12]:
        # for alg_filter in ["_ACC", "_bACC"]:
        for metric in ["alpha"]:
            alg = "agg"
            alg_filter = [alg, metric]

            create_clustering_analysis_plot(
                n_subplots=n_subplots,
                alg_filter=alg_filter,
                # fname="agglomerative_clustering_n600_ACC_impurity_maxD1_decay1_linkageComplete_n_clusters1.3e+02_fitted.pklz",
                min_lost_load_share=0.05,
                average_over_classes=False,
                edge_log_scale=True,
            )

    # Run temporal analysis
    # analyse_clusters_temporal_occurence_patterns(
    #     fname="agglomerative_clustering_n600_bACC_katz_maxD4_decay1_linkageAverage_n_clusters1.3e+02_fitted.pklz",
    #     n_clusters_to_analyze=6,
    # )

    print("Clustering analysis plot generated successfully!")
