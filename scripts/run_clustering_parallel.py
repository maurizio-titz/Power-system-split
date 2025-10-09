"""clusters the filtered indicator vectors using kmeans and saves the results to disk"""

from datetime import datetime
import gc
import gzip
import os
import pickle
from random import random
import sys
from typing import Callable

from scipy import sparse

sys.path.append("./")

from matplotlib import pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from filter_splits import split_mask
from utils.clustering_uncertainty import (
    get_unique_vectors_with_weights,
    prepare_clusters_for_analysis,
    typed_katz_centrality_batch,
    weighted_distance_wrapper,
)
from utils.indicator_utils import load_indicator_vectors
from utils.data_handling import get_co2_levels

from utils import data_handling
from utils.clustering import (
    cluster_kmeans,
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
from scipy.stats import uniform
from scipy.stats import loguniform
from scripts.calculate_distance_matrix import (
    decay_factor_clustering,
    max_distance_clustering,
)

use_sclopf = True

if use_sclopf:
    path_to_evaluation_results = path_to_evaluation_results_sclopf
    path_to_pypsa_network = path_to_pypsa_network_sclopf
    path_to_vis_results = path_to_vis_results_sclopf
else:
    path_to_evaluation_results = path_to_evaluation_results_lopf
    path_to_pypsa_network = path_to_pypsa_network_lopf
    path_to_vis_results = path_to_vis_results_lopf

n_clusters_list = [32, 64, 80, 128, 150, 256]
clustering_params = {
    # "optics": {
    #     "clustering_algorithm": OPTICS,
    #     "min_samples": loguniform(5, 100),  # sample min_samples on a log scale
    #     "cluster_method": ["xi"],
    #     "max_eps": loguniform(0.02, 0.2),
    #     "n_jobs": [1],
    #     "xi": loguniform(0.02, 0.2),  # sample xi on a log scale
    #     "metric": ["precomputed"],
    # },
    "agglomerative": {
        "clustering_algorithm": AgglomerativeClustering,
        "n_clusters": n_clusters_list,
        "linkage": [
            "average",
            "complete",
        ],  # "single" is not used as it leads to bad results
        "metric": ["precomputed"],
    },
    # "agglomerative": {
    #     "clustering_algorithm": AgglomerativeClustering,
    #     "linkage": ["average", "complete"],
    #     "n_clusters": [None],  # has to be set when using distance threshold
    #     "compute_full_tree": [True],  # has to be set when using distance threshold
    #     "distance_threshold": np.logspace(-1, np.log10(0.3), 16),
    #     "metric": ["precomputed"],
    # },
    "kmedoids": {
        "clustering_algorithm": KMedoids,
        "n_clusters": n_clusters_list,
        "metric": ["precomputed"],
        "method": ["pam"],
    },
}


def get_str_from_params(params):
    """returns a string representation of the parameters"""
    params_ = params.copy()
    params_.pop("metric", None)  # remove metric from params
    params_.pop("n_jobs", None)  # remove n_jobs from params
    sorted_keys = sorted(params_.keys())
    return "_".join(
        [
            f"{k}{params_[k].capitalize() if isinstance(params_[k], str) else ('None' if params_[k] is None else format(params_[k], '.2g'))}"
            for k in sorted_keys
        ]
    )


def get_str_from_params_old(params):
    """returns a string representation of the parameters"""
    params_ = params.copy()
    params_.pop("metric", None)  # remove metric from params
    params_.pop("n_jobs", None)  # remove n_jobs from params
    sorted_keys = sorted(params.keys())
    return "_".join(
        [
            f"{k}{params[k].upper() if isinstance(params[k], str) else str(params[k])}"
            for k in sorted_keys
        ]
    )


def rename_old_files(param_dict, directory):
    for params in ParameterGrid(param_dict):
        old_str = get_str_from_params_old(params)
        new_str = get_str_from_params(params)
        for file in os.listdir(directory):
            if old_str in file:
                new_file = file.replace(old_str, new_str)
                os.rename(directory + file, directory + new_file)
                logger.info(f"renamed {file} to {new_file}")
            if "groups_co2" in file:
                new_file = file.replace("groups_co2l", "_co2l")
                os.rename(directory + file, directory + new_file)
                logger.info(f"renamed {file} to {new_file}")


def run_clustering(
    algorithm_name: str,
    clustering_algorithm: Callable,
    params: dict,
    distance_matrix_path: str,
    save_dir: str,
    n_nodes: int,
    dist_metric_str: str,
    sparsified_distance_matrix: bool = False,
    skip_existing: bool = True,
):
    if "min_samples" in params:
        params["min_samples"] = int(params["min_samples"])
    params_str = get_str_from_params(params)

    save_path = (
        save_dir
        + f"/{algorithm_name}_clustering_n{n_nodes}_{dist_metric_str}_{params_str}_fitted.pklz"
    )
    if not os.path.exists(save_path):
        print(f"performing {algorithm_name} clustering with {params}")

        # Load distance matrix in worker process to avoid memory issues
        if sparsified_distance_matrix:
            distance_matrix = sparse.load_npz(distance_matrix_path + ".npz")
        else:
            distance_matrix = np.load(distance_matrix_path + ".npy")

        # Handle infinite values
        if np.isinf(distance_matrix).any():
            max_val = distance_matrix[~np.isinf(distance_matrix)].max()
            distance_matrix[np.isinf(distance_matrix)] = 2 * max_val

        opt = clustering_algorithm(**params)
        # Make distance matrix writable if it's sparse
        if sparse.issparse(distance_matrix):
            distance_matrix = distance_matrix.copy()
        opt.fit(distance_matrix)

        # Explicitly delete distance matrix to free memory
        del distance_matrix
        gc.collect()  # Force garbage collection

        print("opt.labels_shape: " + str(opt.labels_.shape))
        # save clustering results
        with gzip.open(
            save_path,
            "wb",
        ) as fh_out:
            pickle.dump(opt, fh_out)
    else:
        if skip_existing:
            print(f"already exists, skipping optics clustering for {save_path}")
        else:
            raise FileExistsError(
                f"File {save_path} already exists. Set skip_existing to True to skip this step."
            )


if __name__ == "__main__":

    indicator_type_transformation = [
        ("rocof", "blackout"),
    ]

    n_nodes = 600
    n_parallel_iter = 16  # Reduced from 25 to reduce memory pressure
    n_iter = 32

    # ignore splits with less than 5% lost load share
    min_n_nodes_split, min_lost_load_share = None, 0.05

    co2l_list = get_co2_levels(n_nodes)
    # co2l_list = 0.6
    print(f"co2l_list: {co2l_list}")

    indicator_type, transformation = indicator_type_transformation[0]

    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l=co2l_list,
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=min_n_nodes_split,
        lost_load_share=min_lost_load_share,
        use_sclopf=use_sclopf,
    )
    os.makedirs(save_dir, exist_ok=True)

    logger_num = logger.add(f"{save_dir}/log.txt")

    logger.info(f"save_dir: {save_dir}, n_nodes: {n_nodes}, co2l_list: {co2l_list}")
    logger.info(f"using {indicator_type} indicator type")
    logger.info(f"transforming with {transformation}")

    save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_n{n_nodes}.pklz"
    with gzip.open(save_path_unique_vecs, "rb") as fh_in:
        unique_vecs_dict = pickle.load(fh_in)

    # Load split_props_filtered at the top level to avoid repeated loading
    split_props_filtered_path = f"{save_dir}/data_filtered_{n_nodes}.h5"
    split_props_filtered = pd.read_hdf(split_props_filtered_path, key="split_props")

    logger.info("Starting distance matrix iteration for clustering")
    logger.info(f"Will process distance matrices from calculate_distance_matrix.py")

    for sparsified_distance_matrix in [False]:
        # Define the distance matrices that were calculated in calculate_distance_matrix.py
        # Use the same construction logic as in calculate_distance_matrix.py

        # Standard distance matrix (no Katz weighting)
        standard_dist_metric_str = "bACC"
        standard_weighting_str = ""
        standard_save_path = (
            save_dir
            + f"/distance_matrix_n{n_nodes}_{standard_dist_metric_str}{standard_weighting_str}"
        )
        logger.info(f"Standard distance matrix path: {standard_save_path}")

        # Katz-weighted distance matrix (using the same parameters as in calculate_distance_matrix.py)
        decay_factor = decay_factor_clustering
        max_distance = max_distance_clustering
        katz_dist_metric_str = "bACC"
        katz_weighting_str = f"_katz_maxD{max_distance}_decay{decay_factor}"
        katz_save_path = (
            save_dir
            + f"/distance_matrix_n{n_nodes}_{katz_dist_metric_str}{katz_weighting_str}"
        )
        logger.info(f"Katz-weighted distance matrix path: {katz_save_path}")

        distance_matrices = [
            {
                "name": "katz_weighted",
                "dist_metric_str": f"{katz_dist_metric_str}{katz_weighting_str}",
                "save_path": katz_save_path,
            },
            # {
            #     "name": "standard",
            #     "dist_metric_str": f"{standard_dist_metric_str}{standard_weighting_str}",
            #     "save_path": standard_save_path,
            # },
        ]

        for distance_config in distance_matrices:
            distance_name = distance_config["name"]
            dist_metric_str = distance_config["dist_metric_str"]
            save_path_distance_matrix = distance_config["save_path"]

            if sparsified_distance_matrix:
                dist_metric_str += "_sparse"
                save_path_distance_matrix += "_sparse"

            # Check if distance matrix exists
            matrix_file = save_path_distance_matrix + ".npy"
            if not os.path.exists(matrix_file):
                raise FileNotFoundError(f"Distance matrix not found: {matrix_file}")
                # logger.warning(
                #     f"Distance matrix not found: {matrix_file}. Skipping {distance_name} clustering."
                # )
                # continue

            logger.info(f"Processing distance matrix: {distance_name}")
            logger.info(
                f"Distance matrix will be loaded from: {save_path_distance_matrix}"
            )

            logger.info("performing clustering")
            np.random.seed(seed=42)
            for clust_alg_name in clustering_params.keys():
                logger.info(f"Using clustering algorithm: {clust_alg_name}")
                clustering_HPs = clustering_params[clust_alg_name]
                clust_alg = clustering_HPs.pop("clustering_algorithm")
                param_grid = list(
                    ParameterSampler(clustering_HPs, n_iter=n_iter, random_state=42)
                )

                logger.info(
                    f"Running {len(param_grid)} {clust_alg_name} clustering parameter combinations for {distance_name}"
                )

                Parallel(
                    n_jobs=n_parallel_iter,
                    backend="loky",
                    max_nbytes=None,
                    temp_folder=None,
                )(
                    delayed(run_clustering)(
                        clust_alg_name,
                        clust_alg,
                        params,
                        save_path_distance_matrix,
                        save_dir,
                        n_nodes,
                        dist_metric_str,
                        sparsified_distance_matrix,
                    )
                    for params in tqdm(
                        param_grid, desc=f"{clust_alg_name} {distance_name}"
                    )
                )
                logger.info(
                    f"{clust_alg_name} clustering completed for {distance_name}"
                )

    logger.info("Preparing clusters for analysis")
    processed_files = 0
    for file in os.listdir(save_dir):
        if file.endswith("_fitted.pklz"):
            save_path = os.path.join(save_dir, file)
            prepare_clusters_for_analysis(
                save_path, unique_vecs_dict, split_props_filtered
            )
            processed_files += 1
    logger.info(f"Clusters prepared for analysis - processed {processed_files} files")

    # logger.info("performing agglomerative clustering")
    # agglomerative_param_grid = ParameterGrid(clustering_params["agglomerative"])
    # for params in tqdm(agglomerative_param_grid):
    #     n_clusters = params["n_clusters"]
    #     linkage = params["linkage"]

    #     agglomerative_clustering_save_path = (
    #         save_dir
    #         + f"/agglo_clustering_n{n_nodes}_{dist_metric_str}_ncl{n_clusters}_{linkage}_fitted.pklz"
    #     )

    #     if not os.path.exists(agglomerative_clustering_save_path):
    #         logger.info(
    #             f"performing agglomerative clustering with {n_clusters} clusters and {linkage} linkage"
    #         )
    #         agg = AgglomerativeClustering(
    #             n_clusters=n_clusters,
    #             metric="precomputed",
    #             linkage=linkage,
    #         )
    #         agg.fit(distance_matrix)
    #         logger.info(
    #             f"agg.labels_shape: {agg.labels_.shape}, "
    #             f"agg.n_connected_components_: {agg.n_connected_components_}"
    #         )

    #         # save clustering results
    #         with gzip.open(
    #             agglomerative_clustering_save_path,
    #             "wb",
    #         ) as fh_out:
    #             pickle.dump(agg, fh_out)
    #     else:
    #         logger.info(
    #             f"already exists, skipping optics clustering for {save_path}"
    #         )
    #     prepare_clusters_for_analysis(
    #         agglomerative_clustering_save_path,
    #         unique_vecs_dict,
    #         split_props_filtered,
    #     )

    # logger.info("performing kmedoids clustering")
    # medoids_param_grid = ParameterGrid(clustering_params["kmedoids"])
    # for params in tqdm(medoids_param_grid):
    #     n_clusters = params["n_clusters"]
    #     method = params["method"]
    #     metric = params["metric"]

    #     kmedoids_clustering_save_path = (
    #         save_dir
    #         + f"/kmedoids_clustering_n{n_nodes}_{dist_metric_str}_ncl{n_clusters}_fitted.pklz"
    #     )

    #     if not os.path.exists(kmedoids_clustering_save_path):
    #         logger.info(f"performing medoids clustering with {n_clusters} clusters")
    #         k_med = KMedoids(
    #             n_clusters=n_clusters,
    #             metric=metric,
    #             method=method,
    #         )
    #         k_med.fit(distance_matrix)
    #         logger.info(f"k_med.labels_shape: {k_med.labels_.shape}, ")

    #         # save clustering results
    #         with gzip.open(
    #             kmedoids_clustering_save_path,
    #             "wb",
    #         ) as fh_out:
    #             pickle.dump(k_med, fh_out)
    #     else:
    #         logger.info(
    #             f"already exists, skipping optics clustering for {save_path}"
    #         )
    #     prepare_clusters_for_analysis(
    #         kmedoids_clustering_save_path,
    #         unique_vecs_dict,
    #         split_props_filtered,
    #     )

    # DBSCAN_param_grid = ParameterGrid(clustering_params["dbscan"])
    # for params in tqdm(DBSCAN_param_grid):
    #     params_str = get_str_from_params(params)

    #     eps = params["eps"]
    #     min_samples = params["min_samples"]
    #     metric = params["metric"]
    #     n_jobs = params["n_jobs"]

    #     dbscan_save_path = (
    #         save_dir
    #         + f"/dbscan_clustering_n{n_nodes}_{katz_params_str}_eps{eps}_{params_str}_fitted.pklz"
    #     )

    #     if not os.path.exists(dbscan_save_path):
    #         logger.info(
    #             f"performing DBSCAN clustering with eps {eps} and min_samples {min_samples}"
    #         )
    #         dbscan = DBSCAN(
    #             eps=eps,
    #             min_samples=min_samples,
    #             metric=metric,
    #             n_jobs=n_jobs,
    #         )
    #         dbscan.fit(distance_matrix, sample_weight=weights_filtered)

    #         # save clustering results
    #         with gzip.open(
    #             dbscan_save_path,
    #             "wb",
    #         ) as fh_out:
    #             pickle.dump(dbscan, fh_out)
    #     else:
    #         logger.info(
    #             f"already exists, skipping optics clustering for {save_path}"
    #         )
    #     prepare_clusters_for_analysis(
    #         dbscan_save_path, unique_vecs_dict, split_props_filtered
    #     )
