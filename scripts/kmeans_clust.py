import numpy as np
from sklearn.cluster import KMeans
import pickle
import gzip
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import sys

sys.path.append("./")
from utils.config import path_to_clustering_results, path_to_indicator_vectors


def cluster_kmeans(
    n_nodes,
    co2l,
    n_clusters,
    indicator_type,
    path_to_indicator_vectors,
    path_to_clustering_results,
    transformation=None,
    test=False,
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

    results = []
    if isinstance(co2l, float):
        file_name = f"indicator_vector_{indicator_type}_Co2l{co2l}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            results = pickle.load(out)[-1]
    elif isinstance(co2l, list):
        for co2 in co2l:
            file_name = f"indicator_vector_{indicator_type}_Co2l{co2}_n{n_nodes}.pklz"
            with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
                results.append(pickle.load(out)[-1])
        results = np.concatenate(results)
    else:
        Exception("co2l has to be float or list")

    indicator_vectors = results
    if test:
        indicator_vectors = indicator_vectors[:100]

    # transform indicator vector
    if transformation == "sign":
        indicator_vectors = np.sign(indicator_vectors)
    elif transformation == "tanh":
        indicator_vectors = np.tanh(indicator_vectors)
    elif transformation == "clipped_tanh":
        indicator_vectors = np.tanh(indicator_vectors)
        indicator_vectors = np.clip(indicator_vectors, None, 0)
    elif transformation == "clipped":
        indicator_vectors = np.clip(indicator_vectors, None, 0)

    # create KMeans object
    kmeans = KMeans(n_clusters=n_clusters)

    # fit KMeans object to dataset
    kmeans.fit(indicator_vectors)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(indicator_vectors, labels, sample_size=10000)
    samples_per_centroid = np.unique(labels, return_counts=True)[1]

    # calculate the mean distance to the centroid for each cluster
    centroid_mean_distance = []
    for i_centroid in range(n_clusters):
        mean_distance = mean_distance_to_centroid(
            indicator_vectors, centroids, i_centroid, labels
        )
        centroid_mean_distance.append(mean_distance)

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
                samples_per_centroid,
                centroid_mean_distance,
                inertia,
                silhouette_avg,
            ],
            fh_out,
        )


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


types = [
    "rocof",
    "failed_edges",
    "lshare",
]
n_clusters_list = [10, 20, 30, 40, 50, 100, 200]

co2l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for indicator_type in types:
    print(indicator_type)
    for n_nodes in [400]:
        for transformation in ["sign", "tanh", None]:
            print(transformation)
            if indicator_type != "rocof" and transformation != None:
                continue
            if not "clipped" in transformation:
                continue
            for n_clusters in tqdm(n_clusters_list):
                print(n_clusters)
                cluster_kmeans(
                    n_nodes,
                    co2l,
                    n_clusters,
                    indicator_type,
                    path_to_indicator_vectors,
                    path_to_clustering_results,
                    transformation=transformation,
                    test=False,
                )


co2l_list = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]

for indicator_type in types:
    print(indicator_type)
    for co2l in co2l_list:
        print(co2l)
        for n_nodes in [400]:
            for transformation in ["sign", "tanh", None]:
                print(transformation)
                if indicator_type != "rocof" and transformation != None:
                    continue
                if transformation != "clipped_tanh":
                    continue
                for n_clusters in tqdm(n_clusters_list):
                    print(n_clusters)
                    cluster_kmeans(
                        n_nodes,
                        co2l,
                        n_clusters,
                        indicator_type,
                        path_to_indicator_vectors,
                        path_to_clustering_results,
                        transformation=transformation,
                        test=False,
                    )
