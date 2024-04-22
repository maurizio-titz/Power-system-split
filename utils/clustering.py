"""helper functions for clustering of splits, i.e. the corresponding indicator vectors"""

import gzip
import os
import pickle
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append("./")

from utils.config import path_to_clustering_results
from utils.indicator_utils import (
    calc_is_main_component_indicator_vectors,
    load_indicator_vectors,
)


def cluster_kmeans(
    n_nodes,
    co2l,
    n_clusters,
    indicator_type,
    path_to_indicator_vectors,
    safe_path,
    transformation=None,
    mask=None,
    weights=None,
):
    """generates clusters with k-means algorithm and saves the results

    Args:
        n_nodes (int): number of nodes in the network
        co2l (float): co2 level
        n_clusters (int): number of clusters
        indicator_type (string): which type of indicator vector to use
        path_to_indicator_vectors (string or ): ....
        path_to_clustering_results (string): ....
    """

    if weights is None:
        indicator_vectors = load_indicator_vectors(
            n_nodes,
            co2l,
            indicator_type,
            path_to_indicator_vectors=path_to_indicator_vectors,
            mask=mask,
            weights=None,
        )
    else:
        indicator_vectors, weights = load_indicator_vectors(
            n_nodes,
            co2l,
            indicator_type,
            path_to_indicator_vectors=path_to_indicator_vectors,
            mask=mask,
            weights=weights,
        )

    os.makedirs(safe_path, exist_ok=True)
    with gzip.open(f"{safe_path}/weights.pklz", "wb") as fh_out:
        pickle.dump(weights, fh_out)

    # transform indicator vector
    indicator_vectors = transform_indicator_vectors(
        indicator_vectors, transformation, indicator_type
    )

    # run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(indicator_vectors, sample_weight=weights)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(indicator_vectors, labels, sample_size=10000)
    samples_per_cluster = calculate_samples_per_cluster(n_clusters, weights, labels)

    # calculate the mean distance to the centroid for each cluster
    centroid_mean_distance = calculate_mean_distances_to_centroid(
        n_clusters, indicator_vectors, labels, centroids
    )

    with gzip.open(
        f"{safe_path}/kmeans{n_clusters}.pklz",
        "wb",
    ) as fh_out:
        pickle.dump(
            [
                labels,
                centroids,
                samples_per_cluster,
                centroid_mean_distance,
                inertia,
                silhouette_avg,
                #          weights,
            ],
            fh_out,
        )


def transform_indicator_vectors(indicator_vectors, transformation, indicator_type):
    """transforms indicator vectors to allow better clustering

    Args:
        indicator_vectors (np.ndarray): indicator vectors
        transformation (str): name of transformation

    Returns:
        np.ndarray: transformed indicator vectors
    """
    if transformation == "sign":
        transformed_indicator_vectors = np.sign(indicator_vectors)
    elif transformation == "tanh":
        transformed_indicator_vectors = np.tanh(indicator_vectors)
    elif transformation == "clipped_tanh":
        transformed_indicator_vectors = np.tanh(indicator_vectors)
        transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    elif transformation == "clipped":
        transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    elif transformation == "main_comp_max_lshare":
        if indicator_type != "lshare":
            raise ValueError(
                "is_main_component transformation only works for lshare indicator"
            )
        transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
            indicator_vectors
        )
    elif transformation == "main_comp_most_frequent_lshare":
        if indicator_type != "lshare":
            raise ValueError(
                "is_main_component transformation only works for lshare indicator"
            )
        transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
            indicator_vectors, main_component_by="main_comp_most_frequent_lshare"
        )
    elif transformation == "blackout":
        transformed_indicator_vectors = np.array(indicator_vectors < -1, dtype=int)
    elif transformation == "not_zero":
        transformed_indicator_vectors = np.array(indicator_vectors != 0, dtype=int)
    else:
        raise ValueError(f"transformation {transformation} not implemented")

    return transformed_indicator_vectors


def calculate_samples_per_cluster(
    n_clusters: int, weights: np.ndarray, labels: np.ndarray
) -> list:
    """calculate the weighted number of samples per cluster

    Args:
        n_clusters (int): number of clusters
        weights (list): weights given by snapshot lengths
        labels (list): label of each sample, i.e. which cluster it belongs to

    Returns:
        _type_: _description_
    """
    if weights is None:
        samples_per_centroid = np.unique(labels, return_counts=True)[1]
    else:
        # samples_per_centroid = [
        #     ((labels == i) * weights).sum() for i in range(n_clusters)
        # ]
        samples_per_centroid = np.bincount(
            labels, weights=weights, minlength=n_clusters
        ).astype(int)

    assert (
        isinstance(samples_per_centroid, np.ndarray)
        and len(samples_per_centroid.shape) == 1
    ), "samples_per_centroid is not a 1 dimensional numpy array"

    return samples_per_centroid


def calculate_mean_distances_to_centroid(
    n_clusters: int, indicator_vectors: np.ndarray, labels: list, centroids: list
):
    """calculates the mean distance of all samples in a cluster to the centroid

    Args:
        n_clusters (int): _description_
        indicator_vectors (np.ndarray): _description_
        labels (list): _description_
        centroids (list): _description_

    Returns:
        _type_: _description_
    """
    centroid_mean_distance = []
    for i_centroid in range(n_clusters):
        mean_distance = mean_distance_to_centroid(
            indicator_vectors, centroids, i_centroid, labels
        )
        centroid_mean_distance.append(mean_distance)
    return centroid_mean_distance


def mean_distance_to_centroid(indicator_vectors, centroids, i_centroid, cluster_labels):
    """Calculate Euclidean distance for each data point assigned to centroid

    Args:
        indicator_vectors (array): indicator
        i_centroid (_type_): the centroide of interest
        cluster_labels (array): cluster labels of indicator

    Returns:
        float: mean distance of all data points assigned to the cluster from the centroid
    """
    centroid = centroids[i_centroid]
    assert (
        indicator_vectors.shape[1] == centroid.shape[0]
    ), f"indicator_vectors.shape[1] != centroid.shape[0], {indicator_vectors.shape[1]} != {centroid.shape[0]}"

    distances = [
        np.linalg.norm(indicator_vector - centroid)
        for indicator_vector in indicator_vectors[cluster_labels == i_centroid]
    ]
    assert np.isnan(distances).any() is False, "NaN in distances"

    return np.mean(distances)


def load_clustering(
    n_nodes=None,
    co2l=None,
    n_clusters=None,
    indicator_type=None,
    load_dir=None,
    transformation=None,
    n_nodes_split=None,
    lost_load_share=None,
):
    """load clustering results from zip pickle file. Provide either load_dir or all clustering parameters

    Args:
        n_nodes (int): _description_
        co2l (listlike or float): _description_
        n_clusters (_type_): number of cluster in kmeans
        indicator_type (string): type of indicator vector
        load_dir (_type_): _description_

    Returns:
        (tuple of arrays): labels, centroids, inertia, silhouette_avg
    """

    if load_dir is None:
        load_dir = get_path_to_clustering_dir(n_nodes=n_nodes, co2l=co2l, indicator_type=indicator_type, transformation=transformation, n_nodes_split=n_nodes_split, lost_load_share=lost_load_share)
    
    with gzip.open(
        f"{load_dir}/kmeans{n_clusters}.pklz",
        "rb",
    ) as fh_out:
        (
            labels,
            centroids,
            samples_per_centroid,
            centroid_mean_distance,
            inertia,
            silhouette_avg,
        ) = pickle.load(fh_out)
    return (
        labels,
        centroids,
        samples_per_centroid,
        centroid_mean_distance,
        inertia,
        silhouette_avg,
    )

def get_path_to_clustering_dir(n_nodes, co2l, indicator_type, transformation, n_nodes_split, lost_load_share):
    assert n_nodes is not None, "n_nodes must not be None"
    assert co2l is not None, "co2l must not be None"
    assert indicator_type is not None, "indicator_type must not be None"
    assert transformation is not None, "transformation must not be None"
    assert n_nodes_split is not None, "n_nodes_split must not be None"
    assert lost_load_share is not None, "lost_load_share must not be None"

    if transformation is not None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""
    return f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2{str(co2l).replace(", ", "_")[1:-1]}_n{n_nodes}_ns{n_nodes_split}_lls{lost_load_share}/"
