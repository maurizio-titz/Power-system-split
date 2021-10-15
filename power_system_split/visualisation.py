#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains useful methods to analyse and
visualise cascade results"""

import sys

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import RadiusNeighborsClassifier
from tqdm import tqdm

sys.path.append('../power_system_split/')
from utils import get_split_components


def get_split_adjacencies_from_rocof_solutions(solution_dict,
                                               splitting_cascades_in,
                                               nx_graph):
    """obtain the adjacency matrices for each split in the
    rocof solution dictionary (see evaluate_split_observables).

    NOTE: This only takes into account splits that have been
    prefiltered according to the 'nodes' criterion or the 'load' criterion
    passed to the evaluation function"""

    adjacencies = []

    for key in solution_dict.keys():
        for number in solution_dict[key][::2]:

            split = splitting_cascades_in[key][number]
            F = nx_graph.copy()
            F.remove_edges_from(split)
            A = nx.adjacency_matrix(F).A
            adjacencies.append(A.astype('bool'))

    return np.array(adjacencies)

def indicator_vectors_from_rocof_solution(solution_dict, splitting_cascades, nx_graph, criterion):
    """Construct boolean indicator vectors for split components and retrieve component properties.

    Args:
        solution_dict (dict): Dictionary with time stamps as keys. Each entry contains a list
        [split_number, result, split_number, result, ...] where split_number indicates the index of the split
        in the splitting_cascades dictionary and result summarizes component properties such as the inertia.
        splitting_cascades (dict):  Dictionary with time stamps as keys, which contains all splits occuring
        at this time stamp.
        nx_graph (networkx graph): Graph of PyPSA network.
        criterion (string): Criterion for selecting system splits.

    Returns:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes.
        split_component_props (DataFrame): Properties of components with shape n_vectors x 5.
    """


    list_of_nodes = list(nx_graph)
    n_nodes = len(list_of_nodes)

    indicator_vectors = np.empty((0, n_nodes), bool)
    split_component_props = pd.DataFrame(columns=['time_stamp',
                                                  'number_of_split',
                                                  'inertia_proxy',
                                                  'load_imbalance',
                                                  'causing_link'],
                                         index=[], dtype=float)


    for time_stamp in tqdm(solution_dict.keys()):

        if not solution_dict[time_stamp]:
            continue

        for i, split_number in enumerate(solution_dict[time_stamp][::2]):

            split = splitting_cascades[time_stamp][split_number]
            components = get_split_components(split, nx_graph, criterion = criterion)

            component_props = solution_dict[time_stamp][i*2+1]

            for j, component in enumerate(components):

                component_indicator_vec = np.isin(list_of_nodes, list(component))
                indicator_vectors = np.append(indicator_vectors, np.array([component_indicator_vec]),
                                              axis=0)
                causing_link = split[0]
                causing_link_index = float(nx_graph[causing_link[0]][causing_link[1]]['line_index'][0])

                props = {'time_stamp': time_stamp,
                         'number_of_split': split_number,
                         'inertia_proxy': component_props['inertia_proxy'][j],
                         'load_imbalance': component_props['load_imbalance'][j],
                         'causing_link' : causing_link_index
                         }

                # With `ignore_index=True`, the resulting axis will be labeled 0, 1, …, n - 1.
                # (such that split_component_props.loc[i] match indicator_vectors[i])
                split_component_props = split_component_props.append(props, ignore_index=True)


    return indicator_vectors, split_component_props


def calc_likelihood_failure(nx_graph,
                            splitting_cascades,
                            number_of_snapshots,
                            solution_dict={}):
    """Calculate the likelihood that a) a given edge causes a split
    if it fails (primary likelihood) and b) the likelihood that a
    given edge fails at some point during a cascade. The number of snapshots
    is the number of points in time (i.e. snapshots) used in the pypsa network
    simulation.

    If solution dict is given which is the output of evaluate_split_observables,
    only the cascades that were used in the solution dict are considered,
    otherwise all cascades are used to calculate the likelihood."""

    likelihood_primary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    likelihood_secondary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    number_of_edges = len(nx_graph.edges())

    if not len(solution_dict):
        print("""Did not pass solution_dict to determine which cascades
              to use. Evaluating all cascades.""")
        for key in splitting_cascades.keys():
            for current_cascade in splitting_cascades[key]:
                for failed_edge in current_cascade:
                    likelihood_secondary[failed_edge] += 1/(number_of_snapshots
                                                            * number_of_edges)

                likelihood_primary[current_cascade[0]] += 1/number_of_snapshots
    else:

        for key in solution_dict.keys():
            for split_number in solution_dict[key][::2]:
                current_cascade = splitting_cascades[key][split_number]
                for failed_edge in current_cascade:
                    likelihood_secondary[failed_edge] += 1/(number_of_snapshots
                                                            * number_of_edges)

                likelihood_primary[current_cascade[0]] += 1/number_of_snapshots

    return likelihood_primary, likelihood_secondary



def optimize_number_of_cluster(indicator_vectors, max_n_cluster=20,
                              cluster_distance_type='average'):

    """Estimate optimal number of cluster via the silhouette score.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        max_n_cluster (int, optional): Cluster sizes 2,3,...,max_n_cluster are tested. Defaults to 20.
        cluster_distance_type (str, optional): Defaults to 'average'.

    Returns:
        tuple: optimal number of cluster, array of tested numbers of clusters,
        silhouette scores for each tested number
    """


    silhouette_avg_scores = []
    n_cluster_test = np.arange(2, max_n_cluster+1, dtype=int)

    for i in n_cluster_test:
        print('\r {}'.format(i), end="\r", flush=True)
        agg_cluster = AgglomerativeClustering(affinity='hamming',
                                              n_clusters=i,
                                              linkage=cluster_distance_type)
        agg_cluster.fit(indicator_vectors)

        new_score = silhouette_score(indicator_vectors, agg_cluster.labels_,metric='hamming')
        silhouette_avg_scores.append(new_score)

    n_optimum = np.argmax(silhouette_avg_scores)

    return n_optimum, n_cluster_test, silhouette_avg_scores


def cluster_indicator_vectors_agglomerative(indicator_vectors, n_cluster=None, min_cluster_distance=0.1,
                                            cluster_distance_type='single'):
    """Cluster indicator vectors of graph components into similar groups with agglomerative clustering.

    The distance between vectors is quantified by the hamming distance,
    i.e. the relative number of nodes that are not in the same component.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        n_cluster (int, optional): Number of cluster. If this is not None, min_cluster_distance has to be None.
        Defaults to None.
        min_cluster_distance (float, optional): The minimum distance between two clusters. Below this distance, two
        clusters will be merged. Defaults to 0.1.
        cluster_distance_type (str, optional): Method to compute the distance between two cluster.
        Available: "ward","average","single" and "maximum". Defaults to 'average'.

    Returns:
        tuple: number of cluster and array of cluster labels for each indicator vector
    """

    if n_cluster != None and min_cluster_distance != None:
        raise ValueError(
            'If n_cluster is not None, min_cluster_distance has to be None')

    agg_cluster = AgglomerativeClustering(affinity='hamming',
                                          n_clusters=n_cluster,
                                          distance_threshold=min_cluster_distance,
                                          linkage=cluster_distance_type,
                                          compute_distances=False)
    agg_cluster.fit(indicator_vectors)

    return agg_cluster.n_clusters_, agg_cluster.labels_

def cluster_indicator_vectors_dbscan(indicator_vectors, neighbor_max_dist=0.1, neighbor_min_samples=5, n_jobs=10):
    """Cluster indicator vectors of graph components into similar groups with DBSCAN.

    The distance between vectors is quantified by the hamming distance,
    i.e. the relative number of nodes that are not in the same component.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        neighbor_max_sit (float): The maximum distance between two samples for one to be considered
        as in the neighborhood of the other (eps parameter in sklearn).
        neighbor_min_samples(float): The number of samples (or total weight) in a neighborhood
        for a point to be considered as a core point. This includes the point itself.
        n_jobs (int): Number of jobs.

    Returns:
        tuple: number of cluster, number of noise samples, and array of cluster labels for each indicator vector
    """


    dbscan_cluster = DBSCAN(eps=neighbor_max_dist, min_samples=neighbor_min_samples, metric='hamming', n_jobs=n_jobs)
    dbscan_cluster.fit(indicator_vectors)

    labels = dbscan_cluster.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return n_clusters, n_noise, labels


def cluster_indicator_vectors_combined(indicator_vectors, min_cluster_distance=0.1, subset_ratio=0.01,
                                       n_jobs=20):
    """Cluster indicator vectors of graph components into similar groups with Agglomerative clustering and 
    Nearest-Neighbor classification combined.

    The distance between vectors is quantified by the hamming distance,
    i.e. the relative number of nodes that are not in the same component (number between 0 and 1).
    
    First, a subset of indicator vectors is clustered with Agg. clustering. The rest is then assigned to the cluster
    with Nearest-Neighbor classification. We use a fixed radius to determine nearest neighbors to account
    for a minimum distance that should seperate the clusters. 

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        min_cluster_distance (float): The minimum hamming distance between two clusters (number between 0 and 1).
        Below this distance, two clusters will be merged during Agglomerative clustering. 
        subset_ratio (float):  Ratio of indicator vectors used in Agglomerative clustering. 
        n_jobs (int): Number of jobs.

    Returns:
        tuple: number of cluster and array of cluster labels for each indicator vector
    """

    # Select subset of indicator vectors
    indices_all_components = np.arange(indicator_vectors.shape[0])
    np.random.shuffle(indices_all_components)
    subset_indices = indices_all_components[:int(subset_ratio*indicator_vectors.shape[0])]
    remaining_indices = indices_all_components[int(subset_ratio*indicator_vectors.shape[0]):]
    
    print('\n',subset_indices.size, 'components will be used.')
    print(remaining_indices.size, 'components will be assigned via NN classification.')

    # Cluster subset of vectors
    n_cluster, subset_c_labels = cluster_indicator_vectors_agglomerative(indicator_vectors[subset_indices], 
                                                                         min_cluster_distance = min_cluster_distance,
                                                                         cluster_distance_type = 'average')
    


    # Assign cluster labels to remaining vectors
    rnc = RadiusNeighborsClassifier(radius= min_cluster_distance, metric='hamming', outlier_label=-1, n_jobs=n_jobs)
    rnc.fit(indicator_vectors[subset_indices], subset_c_labels)
    remaining_c_labels = rnc.predict(indicator_vectors[remaining_indices])
    
    all_c_labels = np.zeros(indicator_vectors.shape[0])
    all_c_labels[subset_indices] = subset_c_labels
    all_c_labels[remaining_indices] = remaining_c_labels

    print('There are ',n_cluster, 'unique split cluster.')

    return n_cluster, all_c_labels

def plot_component_cluster(indicator_vectors, plot_axs, cluster_labels, cluster_props, nx_graph, node_size=20):
    """Plot clusters of split components on a geographically embedded graph.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        plot_axs (ndarray): 1d array of axis to plot clusters on, with length n_cluster
        cluster_labels (ndarray): Cluster label for each vector in indicator_vectors.
        cluster_props (pd.DataFrame, string): Data frame containing "likelihood" (of cluster) and "counts" (of components)
        as columns. The index represents the cluster labels and each index should match one unique label in 'cluster_labels'.
        If None, plot titles are only cluster labels. A common title string can also be passed.
        nx_graph (graph): NetworkX graph with geographical location of nodes
    """

    node_positions = nx.get_node_attributes(nx_graph,'pos')
    if type(cluster_props) is pd.DataFrame:
        sorted_labels = cluster_props.likelihood.sort_values().index[::-1]
    else:
        sorted_labels = np.unique(cluster_labels)

    for i, label in enumerate(sorted_labels):

        mean_ind_vector = np.array([ indicator_vectors[cluster_labels==label].mean(0) ])
        ax = plot_axs[i]

        nx.draw(nx_graph,
                pos = node_positions,
                node_size = node_size,
                width = 1,
                alpha = 1,
                ax = ax,
                node_color = mean_ind_vector,
                cmap='cividis',
                vmin=0,
                vmax=1)

        if type(cluster_props)==str:
            ax.set_title(cluster_props)
        elif type(cluster_props) is pd.DataFrame:
            ax.set_title('{}: P = {:.2f} % ({})'.format(label,
                                                        cluster_props.loc[label].likelihood*100,
                                                        int(cluster_props.loc[label].counts)))
        else:
            ax.set_title('{}'.format(label))
