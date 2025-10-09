"""clusters the filtered indicator vectors using kmeans and saves the results to disk"""

from datetime import datetime
from functools import partial
import gzip
import os
import pickle
import sys

from scipy import sparse

sys.path.append("./")

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from filter_splits import split_mask
from utils.clustering import (
    balanced_overlap_distance,
    balanced_overlap_distance_weighted,
    calc_distance_matrix,
    get_unique_vectors_with_weights,
    prepare_clusters_for_analysis,
    sparsify_distance_matrix,
    typed_katz_centrality_batch,
    weighted_bACC_dist,
    weighted_distance_wrapper,
)
from utils.indicator_utils import load_indicator_vectors
from utils.data_handling import get_co2_levels

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
)

decay_factor_clustering = 1
max_distance_clustering = 1
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


def filter_splits_events(
    use_sclopf,
    path_to_evaluation_results,
    path_to_vis_results,
    n_nodes,
    min_lost_load_share,
    co2l_list,
    indicator_type,
    transformation,
    save_dir,
    split_props=None,
    network=None,
):
    os.makedirs(save_dir, exist_ok=True)
    logger_num = logger.add(f"{save_dir}/log.txt")

    # get masks
    masks_dict_path = save_dir + "/masks_dict.pklz"
    if os.path.exists(masks_dict_path):
        with gzip.open(masks_dict_path, "rb") as fh_in:
            masks_dict = pickle.load(fh_in)
    else:
        logger.info("Computing masks for CO2 levels...")

        if split_props is None:
            split_props = pd.read_hdf(
                path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
            )
        split_props.lost_load_share_blackout = (
            split_props.lost_load_share_blackout.astype(float)
        )
        split_props = split_props[split_props.co2l.isin(co2l_list)]

        masks_dict = {
            co2l: split_mask(
                split_props[split_props.index.get_level_values("co2l") == co2l],
                min_lost_load_share,
                ignore_shedding=True,
            )
            for co2l in co2l_list
        }

        with gzip.open(masks_dict_path, "wb") as fh_out:
            pickle.dump(masks_dict, fh_out)

    save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_n{n_nodes}.pklz"
    if os.path.exists(save_path_unique_vecs):
        with gzip.open(save_path_unique_vecs, "rb") as fh_in:
            unique_vecs_dict = pickle.load(fh_in)
    else:
        logger.info("Loading indicator vectors and computing unique vectors...")

        if network is None:
            network = data_handling.load_pypsa_network(
                co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
            )
        snapshot_weights = network.snapshot_weightings.objective

        if split_props is None:
            split_props = pd.read_hdf(
                path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
            )

        (
            blackout_vectors_filtered_dict,
            weights_filtered_dict,
            split_props_filtered,
        ) = load_indicator_vectors(
            n_nodes,
            indicator_type,
            transformation,
            path_to_indicator_vectors=path_to_evaluation_results,
            mask=masks_dict,
            weights=snapshot_weights,
            split_props=split_props,
            co2_list=co2l_list,
        )

        # Save intermediate results
        with gzip.open(f"{save_dir}/weights_filtered_dict.pklz", "wb") as fh_out:
            pickle.dump(weights_filtered_dict, fh_out)
        split_props_filtered.to_hdf(
            f"{save_dir}/data_filtered_{n_nodes}.h5", key="split_props"
        )
        with gzip.open(
            f"{save_dir}/blackout_vectors_filtered_dict_{n_nodes}.pklz", "wb"
        ) as fh_out:
            pickle.dump(blackout_vectors_filtered_dict, fh_out)
        with gzip.open(
            f"{save_dir}/weights_filtered_dict_{n_nodes}.pklz", "wb"
        ) as fh_out:
            pickle.dump(weights_filtered_dict, fh_out)

        blackout_vectors_filtered = np.concatenate(
            list(blackout_vectors_filtered_dict.values())
        )
        weights_filtered = np.concatenate(list(weights_filtered_dict.values()))

        unique_vecs_dict = get_unique_vectors_with_weights(
            blackout_vectors_filtered, weights_filtered
        )

        with gzip.open(save_path_unique_vecs, "wb") as fh_out:
            pickle.dump(unique_vecs_dict, fh_out)

    unique_vecs = np.array(list(unique_vecs_dict.keys()))
    weights_filtered = [d["weight"] for d in unique_vecs_dict.values()]
    return unique_vecs, network, split_props


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


def plot_blackout_and_katz(
    save_dir,
    unique_vecs,
    typed_katz_centralities_dict,
    katz_param_grid,
    nx_graph,
    pos,
    blackout_katz_centralities,
):
    # set seed for reproducibility
    n_outages_to_plots = 8

    n_rows = len(katz_param_grid.param_grid[0]["decay_factor"])
    n_cols = len(katz_param_grid.param_grid[0]["max_distance"])

    plots_created = 0
    plots_skipped = 0

    for plot_count in tqdm(range(n_outages_to_plots), desc="Generating plots"):
        fig_path = save_dir + f"/blackout_vec_{plot_count}.png"
        if os.path.exists(fig_path):
            plots_skipped += 1
            continue

        random_idx = np.random.randint(0, blackout_katz_centralities.shape[0])

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_rows * 8, n_cols * 8),
        )

        for i_decay_factor in range(n_rows):
            for i_max_distance in range(n_cols):
                decay_factor = katz_param_grid.param_grid[0]["decay_factor"][
                    i_decay_factor
                ]
                max_distance = katz_param_grid.param_grid[0]["max_distance"][
                    i_max_distance
                ]
                ax = axs[i_decay_factor, i_max_distance]

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
                    node_size=100,
                )
                # axs[0].set_title(f"blackout vector")
                ax.set_title(f"decay {decay_factor}, max dist {max_distance}")

            katz_plot_path = save_dir + f"/katz_centrality_{plot_count}.png"
            fig.savefig(katz_plot_path)
            plt.close(fig)

        blackout_vec = unique_vecs[random_idx, :]
        f = plot_grid(
            nx_graph,
            pos,
            blackout_vec,
            # ax=ax,
            save_dir=None,
            cmap="YlOrBr",  # Use string colormap name instead
            # vmax=vmax,
            # vmin=vmin,
        )
        f.savefig(fig_path)
        plt.close(f)
        plots_created += 1


if __name__ == "__main__":
    start_time = datetime.now()

    n_nodes = 600

    # ignore splits with less than 5% lost load share
    min_n_nodes_split, min_lost_load_share = None, 0.05

    co2l_list = get_co2_levels(n_nodes)
    # co2l_list = [0.6]

    indicator_type, transformation = "rocof", "blackout"

    if transformation is None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""

    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l=co2l_list,
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=min_n_nodes_split,
        lost_load_share=min_lost_load_share,
        use_sclopf=use_sclopf,
    )
    # for params in clustering_params.values():
    #     rename_old_files(params, save_dir)

    logger.info("Starting data preprocessing...")
    unique_vecs, network, split_props = filter_splits_events(
        use_sclopf,
        path_to_evaluation_results,
        path_to_vis_results,
        n_nodes,
        min_lost_load_share,
        co2l_list,
        indicator_type,
        transformation,
        save_dir,
    )

    typed_katz_centralities_dict = {}
    nx_graph = None
    adjacency_matrix = None
    pos = None

    for sample_weights_type in ["katz", ""]:
        if sample_weights_type == "katz":
            logger.info("Computing Katz-weighted distance matrix...")
            katz_param_grid = ParameterGrid(
                {
                    "decay_factor": [1, 1.2, 1.5, 2],
                    "max_distance": [1, 2, 3],
                }
            )

            for i, katz_params in enumerate(katz_param_grid):
                decay_factor = katz_params["decay_factor"]
                max_distance = katz_params["max_distance"]

                typed_katz_centralities, nx_graph, pos, adjacency_matrix = (
                    calc_katz_centralities(
                        use_sclopf,
                        n_nodes,
                        save_dir,
                        unique_vecs,
                        decay_factor,
                        max_distance,
                        network,
                        nx_graph,
                        adjacency_matrix,
                    )
                )
                typed_katz_centralities_dict[(decay_factor, max_distance)] = (
                    typed_katz_centralities
                )

            # # calculate distance matrix
            decay_factor = decay_factor_clustering
            max_distance = max_distance_clustering

            typed_katz_centralities = typed_katz_centralities_dict[
                (decay_factor, max_distance)
            ]
            blackout_katz_centralities = np.concatenate(
                (unique_vecs, typed_katz_centralities), axis=1
            )

            plot_blackout_and_katz(
                save_dir,
                unique_vecs,
                typed_katz_centralities_dict,
                katz_param_grid,
                nx_graph,
                pos,
                blackout_katz_centralities,
            )

            dist_metric_str = f"bACC"
            weighting_str = f"_katz_maxD{max_distance}_decay{decay_factor}"

            save_path_distance_matrix = (
                save_dir
                + f"/distance_matrix_n{n_nodes}_{dist_metric_str}{weighting_str}"
            )

            if os.path.exists(save_path_distance_matrix + ".npy"):
                distance_matrix = np.load(save_path_distance_matrix + ".npy")
            else:
                logger.info("Computing Katz-weighted distance matrix...")
                distance_matrix = calc_distance_matrix(
                    blackout_katz_centralities,
                    metric=partial(
                        weighted_distance_wrapper,
                        weighted_distance_metric=balanced_overlap_distance_weighted,
                    ),
                )
                np.save(save_path_distance_matrix, distance_matrix)

        elif sample_weights_type == "":
            dist_metric_str = f"bACC"
            weighting_str = f""
            save_path_distance_matrix = (
                save_dir
                + f"/distance_matrix_n{n_nodes}_{dist_metric_str}{weighting_str}"
            )

            if os.path.exists(save_path_distance_matrix + ".npy"):
                distance_matrix = np.load(save_path_distance_matrix + ".npy")
            else:
                logger.info("Computing standard distance matrix...")
                distance_matrix = calc_distance_matrix(
                    unique_vecs,
                    metric=balanced_overlap_distance,
                )
                np.save(save_path_distance_matrix, distance_matrix)

    end_time = datetime.now()
    total_time = end_time - start_time
