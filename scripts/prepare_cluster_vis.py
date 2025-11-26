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
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from filter_splits import split_mask
from utils.clustering import (
    get_unique_vectors_with_weights,
    prepare_clusters_for_analysis,
    typed_katz_centrality_batch,
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
from scipy.stats import uniform
from scipy.stats import loguniform
from scripts.calculate_distance_matrix import (
    decay_factor_clustering,
    max_distance_clustering,
)
from scripts.run_clustering_parallel import clustering_params

use_sclopf = True
test = False

if use_sclopf:
    path_to_evaluation_results = path_to_evaluation_results_sclopf
    path_to_pypsa_network = path_to_pypsa_network_sclopf
    path_to_vis_results = path_to_vis_results_sclopf
else:
    path_to_evaluation_results = path_to_evaluation_results_lopf
    path_to_pypsa_network = path_to_pypsa_network_lopf
    path_to_vis_results = path_to_vis_results_lopf

n_clusters_list = [32, 64, 80, 128, 150, 256]
if test:
    n_clusters_list = n_clusters_list[:2]


if __name__ == "__main__":

    indicator_type_transformation = [
        ("rocof", "overUnder"),
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

    logger.info("Preparing clusters for analysis and calculating silhouette scores")

    # Initialize list to store clustering results information
    clustering_results = []
    distance_matrix_full_path = None
    distance_matrix = None

    processed_files = 0
    files = os.listdir(save_dir)
    for file in files:
        if file.endswith("_fitted.pklz"):
            print(f"Processing file: {file}")
            save_path = os.path.join(save_dir, file)

            try:
                # Load the clustering result
                with gzip.open(save_path, "rb") as fh_in:
                    clustering_model = pickle.load(fh_in)

                # Extract algorithm name and parameters from filename
                filename_parts = file.replace("_fitted.pklz", "").split("_")
                algorithm_name = filename_parts[0]
                distance_matrix_path = (
                    "distance_matrix_" + "_".join(filename_parts[2:7]) + ".npy"
                )
                distance_matrix_full_path_new = os.path.join(
                    save_dir, distance_matrix_path
                )
                if distance_matrix_full_path_new != distance_matrix_full_path:
                    distance_matrix_full_path = distance_matrix_full_path_new
                    distance_matrix = np.load(distance_matrix_full_path_new)
                    distance_matrix = distance_matrix.astype(np.float32)

                # Get cluster labels
                if hasattr(clustering_model, "labels_"):
                    cluster_labels = clustering_model.labels_
                else:
                    logger.warning(f"No labels_ attribute found in {file}")
                    continue

                # Check if we have valid clusters (more than one cluster and not all noise)
                unique_labels = np.unique(cluster_labels)
                n_clusters = len(unique_labels)

                # Remove noise label (-1) if present for cluster count
                if -1 in unique_labels:
                    n_clusters -= 1

                silhouette_avg = None

                if n_clusters > 1 and len(cluster_labels) > 1:
                    silhouette_avg = silhouette_score(
                        distance_matrix, cluster_labels, metric="precomputed"
                    )
                else:
                    logger.warning(
                        f"Cannot calculate silhouette score for {file}: insufficient clusters (n_clusters={n_clusters})"
                    )

                # Store results
                result_info = {
                    "file_path": save_path,
                    "filename": file,
                    "algorithm_name": algorithm_name,
                    "n_clusters": n_clusters,
                    "n_samples": len(cluster_labels),
                    "silhouette_score": silhouette_avg,
                    "has_noise": -1 in unique_labels,
                    "n_noise_points": (
                        np.sum(cluster_labels == -1) if -1 in unique_labels else 0
                    ),
                }

                clustering_results.append(result_info)

                # Prepare clusters for analysis (original functionality)
                prepare_clusters_for_analysis(
                    save_path, unique_vecs_dict, split_props_filtered
                )
                processed_files += 1

            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                if e == FileNotFoundError:
                    raise e
                continue

    # Create DataFrame with clustering results and save it
    clustering_results_df = pd.DataFrame(clustering_results)

    # Save the results DataFrame
    results_save_path = os.path.join(save_dir, "clustering_res_info.csv")
    clustering_results_df.to_csv(results_save_path, index=False)

    logger.info(f"Clustering results summary saved to {results_save_path}")
    logger.info(f"Clusters prepared for analysis - processed {processed_files} files")

    # Display summary statistics
    if len(clustering_results_df) > 0:
        logger.info("Summary statistics:")
        logger.info(
            f"Total algorithms: {len(clustering_results_df['algorithm_name'].unique())}"
        )
        logger.info(
            f"Algorithm distribution:\n{clustering_results_df['algorithm_name'].value_counts()}"
        )
        logger.info(
            f"Silhouette score statistics:\n{clustering_results_df['silhouette_score'].describe()}"
        )
        logger.info(
            f"Best silhouette score: {clustering_results_df['silhouette_score'].max():.4f}"
        )
        best_result = clustering_results_df.loc[
            clustering_results_df["silhouette_score"].idxmax()
        ]
        logger.info(
            f"Best result: {best_result['filename']} (score: {best_result['silhouette_score']:.4f})"
        )
    else:
        logger.warning("No clustering results processed successfully")
