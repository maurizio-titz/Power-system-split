"""calculated the distance matrix for the indicator vectors and saves the results to disk"""

import argparse
from datetime import datetime
from functools import partial
import os
import random
import sys

from scipy import sparse


sys.path.append("./")

from utils.clustering import boundary_field, filter_splits_events
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from utils.clustering import (
    balanced_overlap_distance,
    balanced_overlap_distance_weighted,
    calc_distance_matrix,
    calc_distance_matrix_joblib,
    multiclass_accuracy_distance_weighted,
    multiclass_balanced_distance_weighted,
    neighborhood_homo_batch,
    prepare_clusters_for_analysis,
    typed_katz_centrality_batch,
    weighted_distance_wrapper,
)
from utils.data_handling import (
    get_co2_levels,
    load_networkx_graph,
    load_networkx_graph,
    load_pypsa_network,
)

from utils import data_handling
from utils.clustering import (
    get_path_to_clustering_dir,
)
from utils.clustering_visualisation import (
    plot_clusters_wrapper,
    plot_grid,
    plot_indicator_vectors,
)
from loguru import logger
from joblib import Parallel, delayed
from utils.config import (
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_lopf,
    path_to_vis_results_sclopf,
    path_to_indicator_vectors_sclopf,
)

decay_factor_clustering = 1
max_distance_clustering = 1
tau_clustering = 2.5
alpha_clustering = 0.5
# raise NotImplementedError(
#     "Remove this raise statement after setting decay_factor_clustering"
# )
use_sclopf = True

if use_sclopf:
    path_to_evaluation_results = path_to_evaluation_results_sclopf
    path_to_pypsa_network = path_to_pypsa_network_sclopf
    path_to_vis_results = path_to_vis_results_sclopf
else:
    path_to_evaluation_results = path_to_evaluation_results_lopf
    path_to_pypsa_network = path_to_pypsa_network_lopf
    path_to_vis_results = path_to_vis_results_lopf


def get_str_from_params(params):
    """returns a string representation of the parameters"""
    params_ = params.copy()
    params_.pop("metric", None)  # remove metric from params
    params_.pop("n_jobs", None)  # remove n_jobs from params
    sorted_keys = sorted(params_.keys())
    return "_".join(
        [
            f"{k}{params_[k].capitalize() if isinstance(params_[k], str) else format(params_[k], '.2g')}"
            for k in sorted_keys
        ]
    )


def calc_katz_centralities(
    use_sclopf,
    n_nodes,
    save_dir,
    unique_vecs,
    decay_factor,
    max_distance,
    network=None,
    nx_graph=None,
    adjacency_matrix=None,
):
    save_path_centralities = (
        save_dir + f"indicator_vector_katz_maxD{max_distance}_decay{decay_factor}"
    )

    try:
        typed_katz_centralities = np.load(save_path_centralities + ".npy")

        # If we have cached results, we still need to return graph data for plotting
        if nx_graph is None and network is not None:
            snet_index = 0
            nx_graph = data_handling.build_networkx_graph(
                network, snet_index=snet_index
            )
            pos = nx.get_node_attributes(nx_graph, "pos")
        elif nx_graph is not None:
            pos = nx.get_node_attributes(nx_graph, "pos")
        else:
            # If we don't have network or nx_graph, we need to load/create them
            if network is None:
                network = data_handling.load_pypsa_network(
                    co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
                )
            snet_index = 0
            nx_graph = data_handling.build_networkx_graph(
                network, snet_index=snet_index
            )
            pos = nx.get_node_attributes(nx_graph, "pos")

    except FileNotFoundError:
        logger.info(
            f"Computing Katz centralities (decay={decay_factor}, max_dist={max_distance})..."
        )

        snet_index = 0
        # Check if network is provided, otherwise load it
        if network is None:
            network = data_handling.load_pypsa_network(
                co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
            )
        else:
            pass

        if nx_graph is None:
            nx_graph = data_handling.build_networkx_graph(
                network, snet_index=snet_index
            )
        else:
            pass

        pos = nx.get_node_attributes(nx_graph, "pos")
        snapshot_weights = network.snapshot_weightings.objective

        if adjacency_matrix is None:
            adjacency_matrix = data_handling.get_adjacency_matrix_from_nx_graph(
                nx_graph
            )
        else:
            pass

        typed_katz_centralities = typed_katz_centrality_batch(
            adjacency_matrix=adjacency_matrix,
            node_class_vectors=unique_vecs,
            decay_factor=decay_factor,
            max_distance=max_distance,
        )

        np.save(save_path_centralities, typed_katz_centralities)

    return typed_katz_centralities, nx_graph, pos, adjacency_matrix


def calc_neighborhood_impurity(
    save_dir,
    unique_vecs,
    decay_factor=None,
    max_distance=None,
    method="impurity",
    tau=None,
    adjacency_matrix=None,
    err_on_missing=False,
):
    if save_dir is not None:
        if method == "impurity":
            if decay_factor is None or max_distance is None:
                raise ValueError(
                    "decay_factor and max_distance must be provided for method 'impurity'"
                )
            save_path_centralities = (
                save_dir
                + f"indicator_vector_impurity_maxD{max_distance}_decay{decay_factor}"
            )
        if method == "boundary":
            if tau is None:
                raise ValueError("tau must be provided for method 'boundary'")
            save_path_centralities = save_dir + f"indicator_vector_boundary_tau{tau}"
    try:
        if save_dir is not None:
            neighbourhood_impurities = np.load(save_path_centralities + ".npy")
        else:
            raise FileNotFoundError

    except FileNotFoundError:
        if err_on_missing:
            raise FileNotFoundError(
                f"neighborhood impurity file {save_path_centralities}.npy not found and err_on_missing is True."
            )
        logger.info(
            f"Computing neighbourhood impurities (decay={decay_factor}, max_dist={max_distance})..."
        )
        if adjacency_matrix is None:
            nx_graph = load_networkx_graph(
                snet_index=0, co2lvl=0.0
            )  # adjacency_matrix is independent of co2 level
            adjacency_matrix = data_handling.get_adjacency_matrix_from_nx_graph(
                nx_graph
            )
        if method == "impurity":
            neighbourhood_impurities = neighborhood_homo_batch(
                adjacency_matrix=adjacency_matrix,
                node_class_vectors=unique_vecs,
                decay_factor=decay_factor,
                max_distance=max_distance,
            )
        elif method == "boundary":
            I, B, _, _ = data_handling.load_grid_matrices(snet_index=0, co2lvl=0.0)
            L = I.dot(B).dot(I.T)
            neighbourhood_impurities = 1 - boundary_field(
                labels=unique_vecs,
                incidence_matrix=I,
                L=L,
                tau=tau,
            )
        if save_dir is not None:
            np.save(save_path_centralities, neighbourhood_impurities)

    return neighbourhood_impurities


def plot_blackout_and_nodeWeights(
    save_dir,
    unique_vecs,
    typed_katz_centralities_dict,
    node_weight_params,
    nx_graph,
    pos,
    n_outages_to_plots=8,
    plot_idxs=None,
    overWrite_plots=False,
):

    if isinstance(node_weight_params, dict):
        node_weight_params = ParameterGrid([node_weight_params])

    n_rows = len(node_weight_params.param_grid[0]["decay_factor"])
    n_cols = len(node_weight_params.param_grid[0]["max_distance"])

    plots_created = 0
    plots_skipped = 0

    if plot_idxs is None:
        plot_idxs = random.sample(range(unique_vecs.shape[0]), n_outages_to_plots)

    for plot_count, random_idx in tqdm(enumerate(plot_idxs), desc="Generating plots"):
        fig_path = save_dir + f"/blackout_vec_{random_idx}.png"

        if os.path.exists(fig_path) and overWrite_plots is False:
            print(f"Plot already exists, skipping plots altogether.")
            return

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_rows * 8, n_cols * 8),
        )
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])

        for i_decay_factor in range(n_rows):
            for i_max_distance in range(n_cols):
                decay_factor = node_weight_params.param_grid[0]["decay_factor"][
                    i_decay_factor
                ]
                max_distance = node_weight_params.param_grid[0]["max_distance"][
                    i_max_distance
                ]
                ax = axs[i_decay_factor, i_max_distance]

                # if len(n_rows.shape) == 1:
                katz_centrality = typed_katz_centralities_dict[
                    (decay_factor, max_distance)
                ][random_idx, :]

                vmin = katz_centrality.min()
                vmax = katz_centrality.max()
                plot_grid(
                    nx_graph,
                    pos,
                    katz_centrality,
                    ax=ax,
                    save_dir=None,
                    cmap="cividis",
                    vmax=vmax,
                    vmin=vmin,
                    node_size=50,
                )
                # axs[0].set_title(f"blackout vector")
                ax.set_title(f"decay {decay_factor}, max dist {max_distance}")

            katz_plot_path = save_dir + f"/node_weights_{random_idx}.png"
            if save_dir is not None:
                fig.savefig(katz_plot_path)
            plt.close(fig)

        blackout_vec = unique_vecs[random_idx, :]
        f = plot_grid(
            nx_graph,
            pos,
            blackout_vec,
            # ax=ax,
            save_dir=None,
            cmap="coolwarm",  # Use string colormap name instead
            vmax=blackout_vec.max(),
            vmin=blackout_vec.min(),
            node_size=50,
        )
        if save_dir is not None:
            f.savefig(fig_path)
        plt.close(f)
        plots_created += 1


if __name__ == "__main__":
    # add arg parser

    parser = argparse.ArgumentParser(
        description="Calculate distance matrix for blackout events."
    )
    parser.add_argument(
        "--min_loss",
        type=float,
        default=0.05,
        help="Minimum lost load share below which blackouts are filtered out. Lower values increase computation time.",
    )
    parser.add_argument(
        "--comp_mode",
        type=str,
        default="sequential",
        help="Computation mode for distance matrix calculation. Options are 'seq'(sequential), 'joblib' or 'vec'(vectorized).",
    )
    args = parser.parse_args()
    if args.comp_mode == "seq":
        comp_mode = "seq"
    elif args.comp_mode == "vec_joblib":
        comp_mode = "vec_joblib"
    elif args.comp_mode == "vec":
        comp_mode = "vec"
    else:
        raise ValueError("comp_mode must be one of 'seq', 'vec' or 'vec_joblib'")

    start_time = datetime.now()

    n_nodes = 600

    # ignore splits with less than 5% lost load share
    min_lost_load_share = args.min_loss
    print(f"Using min lost load share of {min_lost_load_share}")

    co2l_list = get_co2_levels(n_nodes)

    indicator_type, transformation = "rocof", "overUnder"
    # indicator_type, transformation = "rocof", "blackout"

    if transformation is None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""

    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l=co2l_list,
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=None,
        lost_load_share=min_lost_load_share,
        use_sclopf=use_sclopf,
    )

    logger.info("Starting data preprocessing...")
    unique_vecs, split_props = filter_splits_events(
        use_sclopf,
        path_to_indicator_vectors_sclopf,
        path_to_vis_results,
        n_nodes,
        min_lost_load_share,
        co2l_list,
        indicator_type,
        transformation,
        save_dir,
    )
    print(f"Found {unique_vecs.shape[0]} blackout unique vectors.")

    neighbourhood_impurities_dict = {}
    nx_graph = load_networkx_graph(snet_index=0, co2lvl=0.0)
    adjacency_matrix = data_handling.get_adjacency_matrix_from_nx_graph(nx_graph)
    pos = nx.get_node_attributes(nx_graph, "pos")

    logger.info("Computing neighbourhood impurities ")
    node_weights_params = ParameterGrid(
        {
            # "decay_factor": [1, 1.2, 1.5, 2],
            # "max_distance": [1, 2, 3],
            "tau": [1, 2, 2.5, 3, 4],
            "alpha": [alpha_clustering]
        }
    )

    for i, katz_params in enumerate(node_weights_params):
        tau = katz_params["tau"]

        neighborhood_impurities = calc_neighborhood_impurity(
            save_dir,
            unique_vecs,
            tau=tau,
            decay_factor=None,
            max_distance=None,
            adjacency_matrix=adjacency_matrix,
            method="boundary",
        )
        neighbourhood_impurities_dict[tau] = neighborhood_impurities

    # # calculate distance matrix
    decay_factor = decay_factor_clustering
    max_distance = max_distance_clustering
    tau = tau_clustering
    alpha = alpha_clustering

    neighborhood_impurities = neighbourhood_impurities_dict[tau]
    blackout_neighbourhood_impurities = np.concatenate(
        (unique_vecs, neighborhood_impurities), axis=1
    )
    print("plotting blackout and node weights...")
    # plot_blackout_and_nodeWeights(
    #     save_dir,
    #     unique_vecs,
    #     neighbourhood_impurities_dict,
    #     node_weights_params,
    #     nx_graph,
    #     pos,
    # )

    distance_metrics = ["boundary_field_ACC"]
    for dist_metric in distance_metrics:
        # weighting_str = f"_impurity_maxD{max_distance}_decay{decay_factor}"
        weighting_str = f"_boundary_tau{tau}_alpha{alpha}"

        save_path_distance_matrix = (
            save_dir + f"/distance_matrix_n{n_nodes}_{dist_metric}{weighting_str}"
        )

        if os.path.exists(save_path_distance_matrix + ".npy"):
            continue
            raise FileNotFoundError(
                "Distance matrix with neighbourhood impurities already exists."
            )
        logger.info(f"Computing weighted distance matrix for metric {dist_metric}...")
        # distance_matrix = calc_distance_matrix_joblib(
        distance_matrix = calc_distance_matrix(
            blackout_neighbourhood_impurities,
            mode=comp_mode,
            test_mode=False,
            metric=dist_metric,
            n_jobs=4,
        )
        # np.save(save_path_distance_matrix + "_" + comp_mode, distance_matrix)
        np.save(save_path_distance_matrix, distance_matrix)

        # elif node_weights == "":

        #     dist_metric_str = f"bACC"
        #     weighting_str = f""
        #     save_path_distance_matrix = (
        #         save_dir
        #         + f"/distance_matrix_n{n_nodes}_{dist_metric_str}{weighting_str}"
        #     )

        #     if os.path.exists(save_path_distance_matrix + ".npy"):
        #         distance_matrix = np.load(save_path_distance_matrix + ".npy")
        #     else:
        #         logger.info("Computing standard distance matrix...")
        #         distance_matrix = calc_distance_matrix(
        #             unique_vecs,
        #             metric=balanced_overlap_distance,
        #         )
        #         np.save(save_path_distance_matrix, distance_matrix)

    end_time = datetime.now()
    total_time = end_time - start_time
