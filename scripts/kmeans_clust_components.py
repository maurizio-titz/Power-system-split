import numpy as np
from sklearn.cluster import KMeans
import pickle
import gzip
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import sys
from filter_splits import split_mask, split_mask_component_vectors
from utils.config import path_to_evaluation_results
from utils.indicator_utils import *
import pandas as pd
from collections import Counter

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
    mask=None,
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

    results = load_indicator_vectors(n_nodes, co2l, indicator_type, path_to_indicator_vectors=path_to_indicator_vectors, mask=mask)

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
    elif transformation == "not_zero":
        indicator_vectors = np.array(indicator_vectors != 0, dtype=int)

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

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # create_component_indicator_vectors(400, 0.1, "lshare", path_to_indicator_vectors, mask)
    types = [
        "rocof_component",
        # "failed_edges",
        # "lshare",
    ]
    indicator_type = "rocof_components"
    n_clusters_list = [10, 15, 20, 35, 50]
    n_nodes = 400
    n_nodes_split, lost_load_share = 5, 0.005

    co2l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # get masks
    path = path_to_evaluation_results + f"/masks_all_n{n_nodes}.pklz"
    with gzip.open(path, 'rb') as out:
        masks = pickle.load(out)
    
    component_masks = []
    
    print("creating component masks")
    for co2,mask in zip(co2l,masks):
        #  get rocof indicator vectors
        # get rocof component indicator vectors
        file_name = f"indicator_vector_{indicator_type}_Co2l{co2}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            split_ind_to_component_ind, component_rocof_vectors = pickle.load(out)
        n_component_vectors = len(component_rocof_vectors)
        # all_inds = [
        #     x
        #     for xs in split_ind_to_component_ind
        #     for x in xs
        # ]
        # print(np.min(all_inds))
        # print(np.max(all_inds))
        # print(split_ind_to_component_ind[0], split_ind_to_component_ind[-1])
        # print(n_component_vectors)
        a = np.array(component_rocof_vectors != 0, dtype=int)
        a = a.sum(axis=1)
        plt.hist(a, bins=200)
        plt.savefig(f"hist_{co2}.png")
        plt.close()
        print("saved fig")
        
        print(len(split_ind_to_component_ind))
        # load_indicator_vectors(n_nodes, co2, indicator_type, path_to_indicator_vectors, mask=None)
        split_properties_df = pd.read_csv(path_to_evaluation_results + f'split_properties_Co2L{co2}_n{n_nodes}.csv', index_col=0)
        component_mask = split_mask_component_vectors(mask, split_ind_to_component_ind, n_component_vectors)
        component_masks.append(component_mask)

    path = path_to_evaluation_results + f"/components_masks_all_n{n_nodes}.pklz"
    with gzip.open(path, 'wb') as out:
        pickle.dump(component_masks, out)
        
    # for indicator_type in types:
    #     print(indicator_type)
    #     for n_nodes in [400]:
    #         for transformation in ["sign", "tanh", None, "clipped_tanh", "clipped"]:
    #             print(transformation)
    #             if indicator_type != "rocof" and transformation != None:
    #                 continue
    #             if indicator_type == "rocof" and transformation == None:
    #                 continue
    #             for n_clusters in tqdm(n_clusters_list):
    #                 print(n_clusters)
    #                 #split_properties_df = pd.read_csv(path_to_evaluation_results + f'split_properties_Co2L{co2l}_n{n_nodes}.csv', index_col=0)
    #                 #mask = split_mask(split_properties_df, n_nodes, lost_load_share)
    #                 cluster_kmeans(
    #                     n_nodes,
    #                     co2l,
    #                     n_clusters,
    #                     indicator_type,
    #                     path_to_indicator_vectors,
    #                     path_to_clustering_results,
    #                     transformation=transformation,
    #                     test=False,
    #                     mask = masks,
    #                 )

    n_nodes = 400
    # indicator_type = "lshare"
    transformations = ["tanh", "not_zero"]
    
    print("start clustering")
    for transformation in transformations:
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
                            mask = component_masks,
                        )
