#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
Different functions to visualize system splits and their properties
"""

import gzip
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import RadiusNeighborsClassifier
from tqdm import tqdm

sys.path.append("./")
from utils import data_handling
from utils.clustering_visualisation import plot_clusters_wrapper
from utils.config import (
    path_to_clustering_results_sclopf,
    path_to_pypsa_network_sclopf,
    path_to_sclopf_data,
    path_to_sclopf_results,
)


def calc_likelihood_failure(
    nx_graph,
    splitting_cascades,
    number_of_simulations,
    generator_snapshot_weightings,
):
    """Calculate the likelihood that a) a given edge causes a split
    if it fails (primary likelihood) and b) the likelihood that a
    given edge fails at some point during a cascade (secondary likelihood).

    Args:
        nx_graph (networkx graph): Graph of power system
        splitting_cascades (dict): Nested dictionary with cascade results
        number_of_simulations (int): Number of initial failures times the number of time steps.
        generator_snapshot_weightings (pd.DataFrame): Number of hours for this snapshot.

    Returns:
        likelihood_primary, likelihood_secondary (dict, dict): Likelihood of primary and subsequent secondary
        failures induced by the link that is defined in the key.
    """

    likelihood_primary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    likelihood_secondary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    likelihood_total = {(u, v): 0.0 for u, v in nx_graph.edges()}

    # nr_edges = nx_graph.number_of_edges()
    # total_nr_splits = sum(len(vv) for vv in splitting_cascades.values())

    for timestamp, splits in tqdm(splitting_cascades.items()):
        snapshot_weight = generator_snapshot_weightings[timestamp]

        for init_failure, cascade_weight_tuple in splits.items():

            cascade = cascade_weight_tuple[0]
            trigger_weight = cascade_weight_tuple[1]

            cascade_edges = data_handling.matrix_indices_to_nx_edges(cascade, nx_graph)

            for failed_edge in cascade_edges:
                likelihood_secondary[failed_edge] += (
                    snapshot_weight * trigger_weight / number_of_simulations
                )
                likelihood_total[failed_edge] += (
                    snapshot_weight * trigger_weight / number_of_simulations
                )

            init_edges = data_handling.matrix_indices_to_nx_edges(
                (init_failure[0][0], init_failure[1][0]), nx_graph
            )

            for init_edge in init_edges:
                likelihood_primary[init_edge] += (
                    snapshot_weight * trigger_weight / number_of_simulations
                )
                if (
                    not init_edge in cascade_edges
                ):  # for total likelihood, avoid double counting
                    likelihood_total[init_edge] += (
                        snapshot_weight * trigger_weight / number_of_simulations
                    )

    return likelihood_primary, likelihood_secondary, likelihood_total


def cluster_indicator_vectors_agglomerative(
    indicator_vectors,
    n_cluster=None,
    min_cluster_distance=0.1,
    cluster_distance_type="single",
    affinity="hamming",
):
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
            "If n_cluster is not None, min_cluster_distance has to be None"
        )

    agg_cluster = AgglomerativeClustering(
        affinity=affinity,
        n_clusters=n_cluster,
        distance_threshold=min_cluster_distance,
        linkage=cluster_distance_type,
        compute_distances=False,
    )
    agg_cluster.fit(indicator_vectors)

    return agg_cluster.n_clusters_, agg_cluster.labels_


def calc_rocof(load_imbalance, inertia_proxy):
    return 50 * load_imbalance / (inertia_proxy * 2 * 6)
