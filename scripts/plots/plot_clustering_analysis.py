#!/usr/bin/env python3
"""
Clustering Analysis Plot Generation Script

This file holds functions to generate clustering analysis visualizations:
- Network topology with centroids and failure probabilities
- Histograms of loss distribution by CO2 levels
- Colorbars for both node blackout probability and line failure probability
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
from matplotlib.patches import Rectangle

from sklearn.metrics import silhouette_score


from loguru import logger

# Add project root to path
root_path = "./"
sys.path.append(root_path)

# Import project utilities
from utils.clustering.data_handling import get_path_to_clustering_dir, load_clustering
from utils.data_handling import get_co2_levels, get_actual_co2_level
from utils.clustering_visualisation import *
from utils import data_handling
from utils.config import path_to_figures_sclopf, path_to_plot_data
from utils import cascade_simulation
from scripts.plots.plot_combined_generation_storage import LABEL_FONTSIZE
from utils.plot_style import *

# Import plot styling
setup_matplotlib_style()

# Plot data path
if not os.path.exists(path_to_plot_data):
    os.makedirs(path_to_plot_data, exist_ok=True)


def load_clustering_data(
    n_nodes=600,
):
    """Load clustering data and network setup."""
    # Load network graph and node positions
    snet_index = '0'
    network = data_handling.load_pypsa_network(0.6, n_nodes, True)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    pos = nx.get_node_attributes(nx_graph, "pos")

    # Calculate failure statistics
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    try:
        # TODO this loads it for one levels. Why even save it for every level!
        file_path = data_handling.path_to_grid_data + f"/n_2_failures_co2lvl{0.0}.pklz"
        with gzip.open(file_path, "rb") as fh:
            n_2_failures = pickle.load(fh)
    except FileNotFoundError:
        logger.warning("File not found. Fallback by creating possbile line failurres")
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

# TODO n_nodes is not loaded anymore
def load_processed_data(save_dir, n_nodes):
    """Load pre-processed clustering and split data."""
    # Load split properties
    split_properties_filtered = pd.read_hdf(save_dir + 
                                            f"data_filtered.h5", index_col=0)

    # Load masks and weights
    with gzip.open(save_dir + f"masks_dict.pklz", "rb") as f:
        masks_dict = pickle.load(f)
    # with gzip.open(save_dir + f"/weights_filtered_dict.pklz", "rb") as f:
    #     weights_dict = pickle.load(f)

    # Load blackout vectors
    with gzip.open(save_dir + f"/blackout_vectors_filtered_dict.pklz", "rb") as f:
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
    n_subplots: int = 12,
    alg_filter = None,
    average_over_classes: bool = False,
    colors_classes: list[str] = ["blue", "lightgray", "red"],
    edge_log_scale=True,
    plot_dir=path_to_figures_sclopf,
    n_nodes=600,
    co2l_list=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    sort_by="accumulative_lost_load",
    use_only_lines: bool = False,
    save_plot_data: bool = False
):
    """Create the main clustering analysis plot.
    Parameters
    sort_by: "accumulative_lost_load" or "frequency"
    """

    # Load data
    logger.info("Loading clustering data...")
    data = load_clustering_data(n_nodes=n_nodes)
    nx_graph = data["nx_graph"]
    pos = data["pos"]
    num_failures_weighted = data["num_failures_weighted"]
    cluster_res_dir = os.path.dirname(clustering_res_full_path) + "/"
    fname = os.path.basename(clustering_res_full_path)

    # Find and load clustering results
    logger.info("Loading clustering results...")
    if not fname:
        fname = find_best_clustering_file(cluster_res_dir, alg_filter)
    logger.info(f"Using clustering file: {fname}")
    clustering_res_path = os.path.join(cluster_res_dir, fname)

    logger.info("Loading processed data...")
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
        logger.info("Loaded pre-computed centroids from disk.")
    except FileNotFoundError:
        logger.info("Computing centroids from clusters...")
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
    logger.info("Creating visualization...")

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

    gs = GridSpec(2, 1, figure=fig, hspace=0.15, 
                  height_ratios=[0.05, n_rows][::-1])

    gs_clusters = GridSpecFromSubplotSpec(n_rows, 1, 
                                          subplot_spec=gs[0, 0], 
                                          hspace=0.2)

    plot_count = 0
    df_iter = centroids_df.iterrows()

    
    if save_plot_data:
        fpath_plot_cluster_fname = path_to_plot_data + f"/plot_data_{fname}"
            
        plot_data_dict = {
            "graph": nx_graph,
            "graph_pos": pos,
            "node_centroids": centroids,
            "edge_centroids": edge_centroids,
            "centroids_df": centroids_df,
            "group_masks": group_masks,
            "split_lost_load": split_lost_load,
            "co2l_masks": co2l_masks,
            "co2l_inds_hist": co2l_inds_hist,
            "co2_lvls_hist": co2_lvls_hist,
            "weights": weights,
            "num_failures_weighted": num_failures_weighted
            }
        
        with gzip.open(fpath_plot_cluster_fname, "wb") as fh_plot_data:
            pickle.dump(plot_data_dict, fh_plot_data)
            
        logger.warning(f"Saved plot data at '{os.path.basename(fpath_plot_cluster_fname)}'")
        
    
    logger.info(f"Plotting top {n_subplots} clusters...")
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
                use_only_lines=use_only_lines
            )

            
            ax.axis("off")
            ax.set_aspect("equal")

            # Add cluster information
            cluster_label = (
                f"{plot_count}"
                rf"\\"
                rf"$R={round(centroid_row.lost_load_share*100, ndigits=1)}\%$"
                rf"\\"
                rf"$\beta={round(centroid_row.n_samples/centroids_df.n_samples.sum()*100, ndigits=1)}\%$"
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
            ax_hist.set_xlabel("Share of load not served [\%]", rotation=0, labelpad=0)
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
    
    if use_only_lines:
        save_name += "_onlyLines"    
    
    save_figure(fig, save_name, plot_dir)
    logger.info(f"Plot saved to: {os.path.join(plot_dir, f'{save_name}.pdf')}")

    return fig


def compute_centroids_from_clusters(
    average_over_classes: bool,
    co2l_list: list[float],
    clustering_res_path,
    split_properties_filtered,
    weights_filtered,
    blackout_vectors_filtered_dict,
    failed_edges_indicator_vectors_filtered,
):
    weights = weights_filtered.values
    logger.info("total weighted number of events:", weights.sum())
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

    logger.info("Processing clustering results...")
    # Load clustering results
    with gzip.open(clustering_res_path, "rb") as f:
        clustering_res = pickle.load(f)["model"]

    labels_all = np.load(labels_all_path, allow_pickle=True)
    logger.info(f"{clustering_res.labels_.shape[0]} unique blackouts.")

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


def create_clustering_analysis_plot_from_data(fpath_plot_data: str, 
    n_subplots: int = 12,
    n_cols: int = 4,
    colors_classes: list[str] = ["blue", "lightgray", "red"],
    edge_log_scale: bool = True,
    plot_dir: str = path_to_figures_sclopf,
    sort_by: str="accumulative_lost_load",
    use_only_lines: bool = False,
    fig_scaling: float = 4.,
    save_prefix: str | None = None
    ):
    """Use the functions from 'create_clustering_analysis_plot'"""
    
    agg_hash_str = os.path.basename(fpath_plot_data).replace(".pklz", "")
    
    with gzip.open(fpath_plot_data) as fh_in:
        plot_data = pickle.load(fh_in)
    
    # TODO add agg_param and silhouette score
    nx_graph: nx.Graph = plot_data["graph"]
    pos_graph: dict = plot_data["graph_pos"]
    node_centroids: dict = plot_data["node_centroids"]
    edge_centroids: dict = plot_data["edge_centroids"]
    centroids_df: pd.DataFrame = plot_data["centroids_df"]
    group_masks = plot_data["group_masks"]
    split_lost_load = plot_data["split_lost_load"]
    co2l_masks = plot_data["co2l_masks"]
    co2l_inds_hist = plot_data["co2l_inds_hist"]
    co2_lvls_hist = plot_data["co2_lvls_hist"]
    weights = plot_data["weights"]
    num_failures_weighted = plot_data["num_failures_weighted"]
        
    n_rows = int(np.ceil(n_subplots/n_cols))
    
    # Plot it
    ## Setup colormaps
    # Color scale parameters
    node_cmap, edge_cmap = setup_colormaps()
    vmax = 1.0
    vmin = 0.001
    if edge_log_scale:
        vmin_edge = 1e-3
        vmax_edge = 1.0
    else:
        vmin_edge = 0.0
        vmax_edge = 1.0
    
    
    fig = plt.figure(figsize=(n_cols * fig_scaling, n_rows * fig_scaling * 1.3))
    
    gs = GridSpec(2, 1, figure=fig, hspace=.15,
                  height_ratios=[.05, n_rows][::-1])
    
    gs_clusters = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs[0, 0],
                                          hspace=.2)
    
    if sort_by == "frequency":
        centroids_df = centroids_df.sort_values(by="n_samples", ascending=False)
    elif sort_by == "accumulative_lost_load":
        centroids_df = centroids_df.sort_values(by="weighted_lost_load", ascending=False)
    else:
        raise ValueError("Sort method {sort_by} not implemented.")
    
    df_iter = centroids_df.iterrows()
    
    plot_count = 0
    for row in range(n_rows):
        if plot_count >= n_subplots:
            break
        
        
        gs_row = GridSpecFromSubplotSpec(2, 1,
                                         subplot_spec=gs_clusters[row, 0],
                                         hspace=-0.05,
                                         height_ratios=[3, 1],
                                         wspace=-0.01)
        
        gs_row_map = GridSpecFromSubplotSpec(1, n_cols, 
                                             subplot_spec=gs_row[0,0],
                                             wspace=0.0)
        gs_row_hist = GridSpecFromSubplotSpec(1, n_cols,
                                              subplot_spec=gs_row[1, 0],
                                              wspace=0.18, hspace=.5)
        
        for column in range(n_cols):
            if plot_count >= n_subplots:
                break
            
            label, centroid_row = next(df_iter, (None, None))
            if centroid_row is None or label is None:
                break
            
            plot_count += 1
            
            centroid_r = node_centroids[label]
            failed_edges_prob = edge_centroids[label]
            
            # Create map subplot
            ax_map = fig.add_subplot(gs_row_map[0, column])
            
            plot_centroid_with_failures(
                nx_graph,
                pos_graph,
                node_cmap,
                edge_cmap,
                vmax,
                vmin,
                vmin_edge,
                vmax_edge,
                centroid_r,
                failed_edges_prob,
                ax_map,
                colors_classes=colors_classes,
                radius=0.4,
                edge_log_scale=edge_log_scale,
                use_only_lines=use_only_lines
            )
            
            ax_map.axis('off')
            ax_map.set_aspect('equal')
            
            # Add cluster information
            cluster_label = (
                f"{plot_count}"
                rf"\\"
                rf"$R={round(centroid_row.lost_load_share*100, ndigits=1)}\%$"
                rf"\\"
                rf"$\beta={round(centroid_row.n_samples/centroids_df.n_samples.sum()*100, ndigits=1)}\%$"
            )
            
            ax_map.set_title(
                cluster_label,
                y=0.8,
                x=-0,
                fontsize=LABEL_FONTSIZE,
                loc="left",
                horizontalalignment="left",
            )
            
            
            # Histogram
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
            ax_hist.set_xlabel("Share of load not served [\%]", rotation=0, labelpad=0)
            if column != 0:
                ax_hist.set_ylabel("")
                ax_hist.set_yticklabels([])

            h, l = ax_hist.get_legend_handles_labels()
            ax_hist.legend().set_visible(False)
            
    
    # Add legend and colorbars       
    if centroid_r.ndim == 1:
        gs_legend_colorax = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 0.5, 0.5]
        )
    else:
        gs_legend_colorax = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 1, 0.5]
        )               

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
    
    ## Node colorbar
    if centroid_r.ndim == 1:
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

    ## Edge colorbar
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
    save_name = f"from_data_clustering_analysis_{n_cols}cols_{n_subplots}"
    if sort_by:
        save_name += f"_sortedBy{sort_by.replace('_', ' ').title().replace(' ', '')[0].lower() + sort_by.replace('_', ' ').title().replace(' ', '')[1:]}"
    
    save_name += f"_{agg_hash_str}"
    
    if use_only_lines:
        save_name += "_onlyLines"    
    
    if save_prefix is not None:
        save_name = save_prefix + "_" + save_name
    
    save_figure(fig, 
                save_name, 
                plot_dir)
    logger.info(f"Plot saved to: {os.path.join(plot_dir, f'{save_name}.pdf')}")

    return fig


def create_cluster_plot_only_lines_with_zoom(fpath_plot_data: str,
                                  cluster_rank: int,
                                  sort_by: str= "frequency",
                                  zoom_middle: tuple[float, float] | None = None,
                                  zoom_radius_x: float = 20.,
                                  edge_log_scale: bool = True,
                                  colors_classes: list[str] = ["blue", "lightgray", "red"],
                                  figsize=(10, 4),
                                  plot_dir: str = path_to_figures_sclopf):
    """Plot the cluster with rank 'cluster_rank' (as sortedy by 'sort_by') 
    with only lines and additionally zoom if 'zoom_window' is provided."""
    
    agg_hash_str = os.path.basename(fpath_plot_data).replace(".pklz", "")
    
    with gzip.open(fpath_plot_data) as fh_in:
        plot_data = pickle.load(fh_in)
    
    
    # TODO add agg_param and silhouette score
    nx_graph: nx.Graph = plot_data["graph"]
    pos_graph: dict = plot_data["graph_pos"]
    node_centroids: dict = plot_data["node_centroids"]
    edge_centroids: dict = plot_data["edge_centroids"]
    centroids_df: pd.DataFrame = plot_data["centroids_df"]
    group_masks = plot_data["group_masks"]
    split_lost_load = plot_data["split_lost_load"]
    co2l_masks = plot_data["co2l_masks"]
    co2l_inds_hist = plot_data["co2l_inds_hist"]
    co2_lvls_hist = plot_data["co2_lvls_hist"]
    weights = plot_data["weights"]
    num_failures_weighted = plot_data["num_failures_weighted"]
    
    
    if sort_by == "frequency":
        centroids_df = centroids_df.sort_values(by="n_samples", 
                                                ascending=False)
    elif sort_by == "accumulative_lost_load":
        centroids_df = centroids_df.sort_values(by="weighted_lost_load", 
                                                ascending=False)
    else:
        raise ValueError("Sort method {sort_by} not implemented.")
    
    centroid_row = centroids_df.iloc[cluster_rank]
    label = centroids_df.index[cluster_rank]
    
    # Plot it
    ## Setup colormaps
    # Color scale parameters
    node_cmap, edge_cmap = setup_colormaps()
    vmax = 1.0
    vmin = 0.001
    if edge_log_scale:
        vmin_edge = 1e-3
        vmax_edge = 1.0
    else:
        vmin_edge = 0.0
        vmax_edge = 1.0
    
    centroid_picked = node_centroids[label]
    edge_centroid_picked = edge_centroids[label]
    
    # Plot it
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1,1,.1])
    
    ax_map_large = fig.add_subplot(gs[0, 0])
    ax_map_zoom = fig.add_subplot(gs[0, 1])
    
    plot_centroid_with_failures(
                nx_graph,
                pos_graph,
                node_cmap,
                edge_cmap,
                vmax,
                vmin,
                vmin_edge,
                vmax_edge,
                centroid_picked,
                edge_centroid_picked,
                ax_map_large,
                colors_classes=colors_classes,
                radius=0.4,
                edge_log_scale=edge_log_scale,
                use_only_lines=True,
            )
    
    
    
    ax_map_large.axis('off')
    ax_map_large.set_aspect('equal')
    
    
    ## Zoom in map
    lw_fac=1.
    if zoom_middle is not None and zoom_radius_x is not None:
        xlim_large = ax_map_large.get_xlim()
        ylim_large = ax_map_large.get_ylim()
        
        
        ratio_large = (ylim_large[1] - ylim_large[0])/(xlim_large[1] - xlim_large[0])
        
        
        zoom_radius_y = ratio_large * zoom_radius_x
        
        zoom_xlims = (zoom_middle[0] - zoom_radius_x, zoom_middle[0] + zoom_radius_x)
        zoom_ylims = (zoom_middle[1] - zoom_radius_y, zoom_middle[1] + zoom_radius_y)
        
        ax_map_zoom.set_xlim(zoom_xlims)
        ax_map_zoom.set_ylim(zoom_ylims)
        
        lw_fac = (xlim_large[1] - xlim_large[0])/(2*zoom_radius_x)
        
        rect = Rectangle((zoom_xlims[0], zoom_ylims[0]),
                         2*zoom_radius_x, 2*zoom_radius_y,
                         linewidth=2., linestyle='--',
                         edgecolor='k',
                         facecolor='none', 
                         alpha=1.,
                         zorder=5)
        
        ax_map_large.add_patch(rect)
        
    plot_centroid_with_failures(
                nx_graph,
                pos_graph,
                node_cmap,
                edge_cmap,
                vmax,
                vmin,
                vmin_edge,
                vmax_edge,
                centroid_picked,
                edge_centroid_picked,
                ax_map_zoom,
                colors_classes=colors_classes,
                radius=0.4,
                edge_log_scale=edge_log_scale,
                use_only_lines=True,
                edge_width=lw_fac
            )
    
    ax_map_zoom.axis('off')
    ax_map_zoom.set_aspect('equal')
    
    if edge_log_scale:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, 
                                                   vmax=vmax_edge)
        )
    else:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.Normalize(vmin=vmin_edge, 
                                                     vmax=vmax_edge)
        )
    pos_ax_map_zoom = ax_map_zoom.get_position()
    cbar_ax = fig.add_axes([pos_ax_map_zoom.x1 + .01, 
                               pos_ax_map_zoom.y0,
                                0.025, pos_ax_map_zoom.height])
    cb_edge = fig.colorbar(sm_edge, cax=cbar_ax, orientation="vertical")
    cb_edge.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, 
                           width=1.0, which="both")
    cb_edge.set_label(
        "line failure probability", 
        size=COLORBAR_LABEL_FONTSIZE
    )
    #cb_edge.ax.xaxis.set_label_position("right")
    
    # Aesthetics
    add_panel_label(ax_map_large, 0)
    
    add_panel_label(ax_map_zoom, 1)
    
    cluster_label = (
                f"{cluster_rank+1}"
                rf"\\"
                rf"$R={round(centroid_row.lost_load_share*100, ndigits=1)}\%$"
                rf"\\"
                rf"$\beta={round(centroid_row.n_samples/centroids_df.n_samples.sum()*100, ndigits=1)}\%$"
            )
    
    fig.text(0. , 1., 
             cluster_label, transform=ax_map_large.transAxes,
             verticalalignment='top',
             fontsize=LABEL_FONTSIZE)
    
    # Save it
    save_name = "from_data_clustering_analysis"
    if sort_by:
        save_name += f"_sortedBy{sort_by.replace('_', ' ').title().replace(' ', '')[0].lower() + sort_by.replace('_', ' ').title().replace(' ', '')[1:]}"
    
    save_name += f"_cluster{cluster_rank}_{agg_hash_str}"
    
    save_figure(fig, save_name, plot_dir)
    
    logger.info(f"Plot saved to: {os.path.join(plot_dir, f'{save_name}.pdf')}")
    
    return
    