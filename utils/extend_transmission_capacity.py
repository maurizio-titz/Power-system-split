#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Extend the transmission capacity of lines by increases num_parallel on 
lines that are likely to trigger a system split.
"""

import os

import networkx
import pypsa

import gzip
import pickle

from collections import defaultdict

from utils.data_handling import get_matrices_from_nx_graph, build_networkx_graph, nx_edges_to_matrix_indices
from utils.visualization import calc_likelihood_failure
from utils.cascade_simulation import calc_possible_double_line_failures
from utils.config import path_to_evaluation_results

if not os.path.exists("results/sclopf/line_extension_mitigation"):
    os.mkdir("results/sclopf/line_extension_mitigation")

def get_keys_of_largest_items_dict(dict_in: dict, nn: int) -> list:
    """Get a list with tuples (keys, values) with the largest values of the dictionary.

    Args:
        dict_in (dict): Input dictionary.
        nn (int): Number of keys of max items that is returned
    Returns:
        key_list (list)
        value_list (list)
    """
    
    res = sorted(dict_in.items(), key = lambda x: x[1],
                 reverse = True)[:nn]

    return res

    
def get_most_likely_primary_links(pypsa_net: pypsa.Network, nx_graph: networkx.graph, 
                                  casc_dict: dict, number_simulations: int, 
                                  nn_links: int):
    """Get a list with nn_links links that most likely trigger a cascade 
    leading to a system split, i.e., primary failures."""
        
    likelihood_primary_failures, likelihood_secondary_failures = calc_likelihood_failure(nx_graph, casc_dict,
                                                             number_simulations,
                                                             pypsa_net.snapshot_weightings.generators)
    
    
    tuple_vulnerable_links = get_keys_of_largest_items_dict(likelihood_primary_failures,
                                                            nn_links)
    
    list_names_vulnerable_links = [xx[0]for xx in tuple_vulnerable_links]
    
    return likelihood_primary_failures, likelihood_secondary_failures, list_names_vulnerable_links

def calc_impactful_primary_links(co2lvl=None, split_significance_df=None):
    """calculate lost load caused by each trigger/primary failure."""

    assert co2lvl is not None or split_significance_df is not None, "Either co2lvl or split_significance must be provided"
    
    if split_significance_df is None:
        split_significance_df = pd.read_csv(path_to_evaluation_results+f"split_significance_Co2L{co2lvl}_n400.csv", index_col=0)
    lost_load_total_shares = split_significance_df.lost_load_total_share.values
    
    cumulative_lost_load_share_by_initial_failure = defaultdict(float)
    init_failures_0 = split_significance_df.init_failures_0.values
    init_failures_1 = split_significance_df.init_failures_1.values
    
    for i in range(split_significance_df.shape[0]):
        lost_load_share_split = lost_load_total_shares[i]
        cumulative_lost_load_share_by_initial_failure[init_failures_0[i]] += lost_load_share_split
        cumulative_lost_load_share_by_initial_failure[init_failures_1[i]] += lost_load_share_split
    
    return cumulative_lost_load_share_by_initial_failure

def get_most_impactful_primary_links(pypsa_net: pypsa.Network, nx_graph: networkx.graph, 
                                  casc_dict: dict, number_simulations: int, 
                                  nn_links: int):
    """Get a list with nn_links links that most likely trigger a cascade 
    leading to a system split, i.e., primary failures."""
        
    cumulative_lost_load_share_by_initial_failure = calc_impactful_primary_links(nx_graph, casc_dict,
                                                             number_simulations,
                                                             pypsa_net.snapshot_weightings.generators)
    
    
    tuple_vulnerable_links = get_keys_of_largest_items_dict(likelihood_primary_failures,
                                                            nn_links)
    
    list_names_vulnerable_links = [xx[0]for xx in tuple_vulnerable_links]
    
    return cumulative_lost_load_share_by_initial_failure, list_names_vulnerable_links


def increase_capacity_most_likely_primary_links(pypsa_net: pypsa.Network,
                                                nx_graph_in: networkx.Graph,
                                                casc_dict: dict,
                                                nn_links: int,
                                                delta_num_parallel: float,
                                                rerun_likelihood_calc: bool = False):
    """Increase the transmission capacity of the nn_links links that are most likely trigger a cascade
    leading to a system split. 
    Args:
        pypsa_net (pypsa.Network): optimized PyPSA network.
        nx_graph (networkx.Graph): NetworksX graph extracted from pypsa_net
        casc_dict (dict): Dictionary with cascade results
        nn_links (int): number of links that will be extended.
        delta_num_parallel (float): Amount of line extension.

    Returns:
        nx_graph, vulnerable_edges (Graph, list): Modified networkx graph and list of links
            that were modified.
    """
    
    nx_graph = nx_graph_in.copy()
    
    _, _, num_parallels, _ = get_matrices_from_nx_graph(nx_graph)
    
    bridge_idxs = nx_edges_to_matrix_indices(networkx.bridges(nx_graph), nx_graph)
    nr_failures = calc_possible_double_line_failures(num_parallels,
                                                     ignored_idxs=bridge_idxs)
    number_simulations = len(nr_failures) * pypsa_net.snapshot_weightings.generators.sum()
    
    # List with edges to extend
    fpath_likelihood_res = 'results/sclopf/line_extension_mitigation/primary_n_secondary_link_failure_prob.pklz'
    
    if not os.path.exists(fpath_likelihood_res) or rerun_likelihood_calc:
        likeli_prim, likeli_sec, vulnerable_edges = get_most_likely_primary_links(pypsa_net, nx_graph, casc_dict, 
                                                                                  number_simulations, nn_links)

        with gzip.open(fpath_likelihood_res, 'wb') as fh_out:
            pickle.dump((likeli_prim, likeli_sec, vulnerable_edges), fh_out)
        
    else:
        with gzip.open(fpath_likelihood_res, 'rb') as fh_in:
            _, _, vulnerable_edges = pickle.load(fh_in)
    
    for edge_r in vulnerable_edges:
        nx_graph.edges[edge_r]['num_parallel'] += delta_num_parallel
    
    return nx_graph, vulnerable_edges
