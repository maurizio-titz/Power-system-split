import numpy as np
# from sklearn.cluster import KMeans
import pickle
import gzip
# from sklearn.metrics import silhouette_score
from tqdm import tqdm
import sys
# from filter_splits import split_mask, split_mask_component_vectors
from utils.config import path_to_evaluation_results
from utils.indicator_utils import *
import pandas as pd
# from collections import Counter
from utils.clustering import *

sys.path.append("./")
from utils.config import path_to_clustering_results, path_to_indicator_vectors


# def cluster_kmeans(
#     n_nodes,
#     co2l,
#     n_clusters,
#     indicator_type,
#     path_to_indicator_vectors,
#     path_to_clustering_results,
#     transformation=None,
#     mask=None,
#     weights=None
# ):
#     """generates clusters with k-means algorithm and saves the results

#     Args:
#         n_nodes (int): number of nodes in the network
#         co2l (float): co2 level
#         n_clusters (int): number of clusters
#         indicator_type (string): which type of indicator vector to use
#         path_to_indicator_vectors (string or ): ....
#         path_to_clustering_results (string): ....
#     """

#     if weights is None:
#         indicator_vectors = load_indicator_vectors(n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights=None)
#     else:
#         indicator_vectors, weights = load_indicator_vectors(n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights=weights)

#     with gzip.open(f"{path_to_clustering_results}/weights.pkl","wb") as fh_out:
#         pickle.dump(weights, fh_out)
    
#     # transform indicator vector
#     indicator_vectors = transform_indicator_vectors(indicator_vectors, transformation)
    
#     # run KMeans
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(indicator_vectors, sample_weight=weights)

#     labels = kmeans.labels_
#     centroids = kmeans.cluster_centers_
#     inertia = kmeans.inertia_
#     silhouette_avg = silhouette_score(indicator_vectors, labels, sample_size=10000)
#     samples_per_cluster = calculate_samples_per_cluster(n_clusters, weights, labels)

#     # calculate the mean distance to the centroid for each cluster
#     centroid_mean_distance = calculate_mean_distances_to_centroid(n_clusters, indicator_vectors, labels, centroids, centroid_mean_distance)

#     if transformation != None:
#         transformation_string = "_" + transformation
#     else:
#         transformation_string = ""

#     with gzip.open(
#         f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2l{co2l}_n{n_nodes}_kmeans{n_clusters}.pkl",
#         "wb",
#     ) as fh_out:
#         pickle.dump(
#             [
#                 labels,
#                 centroids,
#                 samples_per_cluster,
#                 centroid_mean_distance,
#                 inertia,
#                 silhouette_avg,
#       #          weights,
#             ],
#             fh_out,
#         )

#     def transform_indicator_vectors(indicator_vectors, transformation):
#         if transformation == "sign":
#             indicator_vectors = np.sign(indicator_vectors)
#         elif transformation == "tanh":
#             indicator_vectors = np.tanh(indicator_vectors)
#         elif transformation == "clipped_tanh":
#             indicator_vectors = np.tanh(indicator_vectors)
#             indicator_vectors = np.clip(indicator_vectors, None, 0)
#         elif transformation == "clipped":
#             indicator_vectors = np.clip(indicator_vectors, None, 0)
#         elif transformation == "main_comp_max_lshare":
#             if indicator_type != "lshare":
#                 raise ValueError(
#                     "is_main_component transformation only works for lshare indicator"
#                 )
#             indicator_vectors = calc_is_main_component_indicator_vectors(indicator_vectors)
#         elif transformation == "main_comp_most_frequent_lshare":
#             if indicator_type != "lshare":
#                 raise ValueError(
#                     "is_main_component transformation only works for lshare indicator"
#                 )
#             indicator_vectors = calc_is_main_component_indicator_vectors(indicator_vectors, main_component_by="main_comp_most_frequent_lshare")
#         elif transformation == "not_zero":
#             indicator_vectors = np.array(indicator_vectors != 0, dtype=int)
#         else:
#             raise ValueError(f"transformation {transformation} not implemented")
        
#         return indicator_vectors

# def calculate_samples_per_cluster(n_clusters, weights, labels):
#     if weights is None:
#         samples_per_centroid = np.unique(labels, return_counts=True)[1]
#     else:
#         samples_per_centroid = [(labels==i)*weights for i in n_clusters]
#     return samples_per_centroid

# def calculate_mean_distances_to_centroid(n_clusters, indicator_vectors, labels, centroids, centroid_mean_distance):
#     centroid_mean_distance = []
#     for i_centroid in range(n_clusters):
#         mean_distance = mean_distance_to_centroid(
#             indicator_vectors, centroids, i_centroid, labels
#         )
#         centroid_mean_distance.append(mean_distance)
#     return centroid_mean_distance

# def mean_distance_to_centroid(indicator_vectors, centroids, i_centroid, cluster_labels):
#     """Calculate Euclidean distance for each data point assigned to centroid

#     Args:
#         indicator_vectors (array): indicator
#         i_centroid (_type_): the centroide of interest
#         cluster_labels (array): cluster labels of indicator

#     Returns:
#         float: mean distance of all data points assigned to the cluster from the centroid
#     """
#     centroid = centroids[i_centroid]
#     assert (
#         indicator_vectors.shape[1] == centroid.shape[0]
#     ), f"indicator_vectors.shape[1] != centroid.shape[0], {indicator_vectors.shape[1]} != {centroid.shape[0]}"

#     distances = [
#         np.linalg.norm(indicator_vector - centroid)
#         for indicator_vector in indicator_vectors[cluster_labels == i_centroid]
#     ]
#     assert np.isnan(distances).any() == False, "NaN in distances"

#     return np.mean(distances)


# def load_clustering(
#     n_nodes, co2l, n_clusters, indicator_type, path_to_clustering_results, transformation=None
# ):
#     """load clustering results from zip pickle file

#     Args:
#         n_nodes (int): _description_
#         co2l (listlike or float): _description_
#         n_clusters (_type_): number of cluster in kmeans
#         indicator_type (string): type of indicator vector
#         path_to_clustering_results (_type_): _description_

#     Returns:
#         (tuple of arrays): labels, centroids, inertia, silhouette_avg
#     """

#     if transformation != None:
#         transformation_string = "_" + transformation
#     else:
#         transformation_string = ""

#     with gzip.open(
#         f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2l{co2l}_n{n_nodes}_kmeans{n_clusters}.pkl",
#         "rb",
#     ) as fh_out:
#         (
#             labels,
#             centroids,
#             samples_per_centroid,
#             centroid_mean_distance,
#             inertia,
#             silhouette_avg,
#         ) = pickle.load(fh_out)
#     return (
#         labels,
#         centroids,
#         samples_per_centroid,
#         centroid_mean_distance,
#         inertia,
#         silhouette_avg,
#     )

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import networkx as nx
    from utils import data_handling
    from utils.config import path_to_pypsa_network
    # create_component_indicator_vectors(400, 0.1, "lshare", path_to_indicator_vectors, mask)
    types = [
        # "rocof_component",
        # "failed_edges",
        "lshare",
    ]
    transformations = ["main_comp_most_frequent_lshare",
                    #    "sign", "tanh", None, "clipped_tanh", "clipped"
                       ]
    n_clusters_list = [8, 16, 32, 48, 64, 96, 128]
    n_clusters_list = [128]
    n_nodes = 400
    n_nodes_split, lost_load_share = 5, 0.005

    co2l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # get masks
    print("loading masks")
    split_properties_all = []
    for co2l in co2l_list:
        split_properties_df = pd.read_csv(path_to_evaluation_results + f'split_significance_Co2L{co2l}_n{n_nodes}.csv', index_col=0)
        split_properties_df['co2l'] = co2l
        split_properties_all.append(split_properties_df)
    split_properties_df = pd.concat(split_properties_all)
    masks = [split_mask(split_properties_df[split_properties_df.co2l==co2l], n_nodes, lost_load_share) for co2l in co2l_list]
    # path = path_to_evaluation_results + f"/masks_all_n{n_nodes}.pklz"
    path = path_to_evaluation_results + f"/masks_0.1-0.8_n{n_nodes}.pklz"
    with gzip.open(path, "wb") as out:
        pickle.dump(masks, out)
    # with gzip.open(path, 'rb') as out:
    #     masks = pickle.load(out)


    snet_index = 0
    network = data_handling.load_pypsa_network(0.5, n_nodes, path_to_pypsa_network)
    nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
    snapshot_weights = network.snapshot_weightings.objective
    
    print("performing clustering")
    for indicator_type in types:
        print(indicator_type)
        for transformation in transformations:
            print(transformation)
            for n_clusters in tqdm(n_clusters_list):
                print(n_clusters)
                #split_properties_df = pd.read_csv(path_to_evaluation_results + f'split_significance_Co2L{co2l}_n{n_nodes}.csv', index_col=0)
                #mask = split_mask(split_properties_df, n_nodes, lost_load_share)
                cluster_kmeans(
                    n_nodes,
                    co2l_list,
                    n_clusters,
                    indicator_type,
                    path_to_indicator_vectors,
                    path_to_clustering_results,
                    transformation=transformation,
                    test=False,
                    mask = masks,
                    weights=snapshot_weights
                )
