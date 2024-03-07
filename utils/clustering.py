import numpy as np
from sklearn.cluster import KMeans
import pickle
import gzip
from sklearn.metrics import silhouette_score
import os
import sys
from utils.indicator_utils import *

sys.path.append("./")


def cluster_kmeans(
    n_nodes,
    co2l,
    n_clusters,
    indicator_type,
    path_to_indicator_vectors,
    path_to_clustering_results,
    transformation=None,
    mask=None,
    weights=None
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
        indicator_vectors = load_indicator_vectors(n_nodes, co2l, indicator_type, path_to_indicator_vectors=path_to_indicator_vectors, mask=mask, weights=None)
    else:
        indicator_vectors, weights = load_indicator_vectors(n_nodes, co2l, indicator_type, path_to_indicator_vectors=path_to_indicator_vectors, mask=mask, weights=weights)

    os.makedirs(path_to_clustering_results, exist_ok=True)
    with gzip.open(f"{path_to_clustering_results}/weights.pkl","wb") as fh_out:
        pickle.dump(weights, fh_out)
    
    # transform indicator vector
    indicator_vectors = transform_indicator_vectors(indicator_vectors, transformation)
    
    # run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(indicator_vectors, sample_weight=weights)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(indicator_vectors, labels, sample_size=10000)
    samples_per_cluster = calculate_samples_per_cluster(n_clusters, weights, labels)

    # calculate the mean distance to the centroid for each cluster
    centroid_mean_distance = calculate_mean_distances_to_centroid(n_clusters, indicator_vectors, labels, centroids, centroid_mean_distance)

    if transformation != None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""

    with gzip.open(
        f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2l{co2l}_n{n_nodes}_kmeans{n_clusters}.pkl",
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

    def transform_indicator_vectors(indicator_vectors, transformation):
        """transforms indicator vectors to allow better clustering

        Args:
            indicator_vectors (np.ndarray): indicator vectors
            transformation (str): name of transformation

        Returns:
            np.ndarray: transformed indicator vectors
        """
        if transformation == "sign":
            indicator_vectors = np.sign(indicator_vectors)
        elif transformation == "tanh":
            indicator_vectors = np.tanh(indicator_vectors)
        elif transformation == "clipped_tanh":
            indicator_vectors = np.tanh(indicator_vectors)
            indicator_vectors = np.clip(indicator_vectors, None, 0)
        elif transformation == "clipped":
            indicator_vectors = np.clip(indicator_vectors, None, 0)
        elif transformation == "main_comp_max_lshare":
            if indicator_type != "lshare":
                raise ValueError(
                    "is_main_component transformation only works for lshare indicator"
                )
            indicator_vectors = calc_is_main_component_indicator_vectors(indicator_vectors)
        elif transformation == "main_comp_most_frequent_lshare":
            if indicator_type != "lshare":
                raise ValueError(
                    "is_main_component transformation only works for lshare indicator"
                )
            indicator_vectors = calc_is_main_component_indicator_vectors(indicator_vectors, main_component_by="main_comp_most_frequent_lshare")
        elif transformation == "blackout":
            indicator_vectors = np.array(indicator_vectors < -1, dtype=int)
        elif transformation == "not_zero":
            indicator_vectors = np.array(indicator_vectors != 0, dtype=int)
        else:
            raise ValueError(f"transformation {transformation} not implemented")
        
        return indicator_vectors

def calculate_samples_per_cluster(n_clusters, weights, labels):
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
        samples_per_centroid = [(labels==i)*weights for i in n_clusters]
    return samples_per_centroid

def calculate_mean_distances_to_centroid(n_clusters, indicator_vectors, labels, centroids, centroid_mean_distance):
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
    assert np.isnan(distances).any() == False, "NaN in distances"

    return np.mean(distances)


def load_clustering(
    n_nodes, co2l, n_clusters, indicator_type, path_to_clustering_results, transformation=None
):
    """load clustering results from zip pickle file

    Args:
        n_nodes (int): _description_
        co2l (listlike or float): _description_
        n_clusters (_type_): number of cluster in kmeans
        indicator_type (string): type of indicator vector
        path_to_clustering_results (_type_): _description_

    Returns:
        (tuple of arrays): labels, centroids, inertia, silhouette_avg
    """

    if transformation != None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""

    with gzip.open(
        f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2l{co2l}_n{n_nodes}_kmeans{n_clusters}.pkl",
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