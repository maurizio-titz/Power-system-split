#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains useful methods to analyse and
visualise cascade results"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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


def indicator_vectors_from_adj(adjacency_matrices):
    """Construct boolean indicator vectors for connected components in a list of graphs.

    Args:
        adjacency_matrices (array_like): An array of shape n_graphs x n_nodes x n_nodes,
        which contains the adjacency matrices of multiple graphs.

    Returns:
        tuple: The array of indicator vectors and a list of the graph number i
        to which the indicated components belong
    """

    n_nodes = adjacency_matrices.shape[1]
    list_of_nodes = nx.convert_matrix.from_numpy_matrix(adjacency_matrices[0])
    list_of_nodes = list(list_of_nodes)

    indicator_vectors = np.empty((0, n_nodes, n_nodes), bool)
    # TODO: when we directly pass the split components to this function
    # we don't have to keep track of the graph number "index_of_adj_mat". Implement this in the future!
    index_of_adj_mat = []

    for i, adj_mat in enumerate(adjacency_matrices):
        print('\r {}'.format(i), end="\r", flush=True)

        adj_graph = nx.convert_matrix.from_numpy_matrix(adj_mat)
        connected_comps = nx.connected_components(adj_graph)

        for comp in connected_comps:

            new_indicator_vec = np.isin(list_of_nodes, list(comp))

            # TODO: Update constraint that selects "big enough" split components
            if new_indicator_vec.sum() > 19:
                indicator_vectors = np.append(indicator_vectors,
                                              np.array([new_indicator_vec]),
                                              axis=0)
                index_of_adj_mat.append(i)

    return indicator_vectors, index_of_adj_mat


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


def cluster_indicator_vectors(indicator_vectors, n_cluster=None, min_cluster_distance=0.1,
                              cluster_distance_type='average'):
    """Cluster indicator vectors of graph components into similar groups. 
    
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


def plot_component_cluster(indicator_vectors, plot_axs, cluster_labels, nx_graph):
    """Plot clusters of split components on a geographically embedded graph.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        plot_axs (ndarray): 1d array of axis to plot clusters on, with length n_cluster 
        cluster_labels (ndarray): Array of cluster labels for each vector in indicator_vectors
        nx_graph (graph): NetworkX graph with geographical location of nodes
    """
  
    
    n_samples = cluster_labels.shape[0]
    n_cluster = plot_axs.shape[0]
    
    #TODO: Implement a better estimate of the cluster likelihood!
    likelihood_of_cluster = [np.sum(cluster_labels==i) / n_samples for i in range(n_cluster)]
    sorted_labels = np.argsort(likelihood_of_cluster)[::-1]

    node_positions = nx.get_node_attributes(nx_graph,'pos')
    
    for i, label in enumerate(sorted_labels):
        
        mean_ind_vector = np.array([ indicator_vectors[cluster_labels==label].mean(0) ])
        ax = plot_axs[i]
        
        nx.draw(nx_graph,
                pos = node_positions,
                node_size = 20,
                width = 1,
                alpha = 1,
                ax = ax,
                node_color = mean_ind_vector,
                cmap='cividis',
                vmin=0,
                vmax=1)
    
        ax.set_title('Split {}: P = {:.2f} %'.format(i, likelihood_of_cluster))



#TODO: add requirements for new functions
