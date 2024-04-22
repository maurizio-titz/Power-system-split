#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Extend the transmission capacity of lines by increases num_parallel on 
lines that are likely to trigger a system split.
"""

import os
from typing import Union

import networkx
import pandas as pd
import pypsa

from utils.cascade_simulation import calc_possible_double_line_failures
from utils.config import path_to_evaluation_results, path_to_line_extension_mitigation
from utils.data_handling import get_matrices_from_nx_graph, nx_edges_to_matrix_indices
from utils.visualization import calc_likelihood_failure

if not os.path.exists(path_to_line_extension_mitigation):
    os.mkdir(path_to_line_extension_mitigation)


def get_keys_of_largest_items_dict(dict_in: dict, nn: int) -> list:
    """Get a list with tuples (keys, values) with the largest values of the dictionary.

    Args:
        dict_in (dict): Input dictionary.
        nn (int): Number of keys of max items that is returned
    Returns:
        key_list (list)
        value_list (list)
    """

    res = sorted(dict_in.items(), key=lambda x: x[1], reverse=True)[:nn]

    return res


def get_most_likely_primary_links(
    pypsa_net: pypsa.Network,
    nx_graph: networkx.Graph,
    casc_dict: dict,
    number_simulations: int,
    nn_links: int,
):
    """Get a list with nn_links links that most likely trigger a cascade
    leading to a system split, i.e., primary failures."""

    likelihood_primary_failures, likelihood_secondary_failures = (
        calc_likelihood_failure(
            nx_graph,
            casc_dict,
            number_simulations,
            pypsa_net.snapshot_weightings.generators,
        )
    )

    tuple_vulnerable_links = get_keys_of_largest_items_dict(
        likelihood_primary_failures, nn_links
    )

    list_names_vulnerable_links = [xx[0] for xx in tuple_vulnerable_links]

    return (
        likelihood_primary_failures,
        likelihood_secondary_failures,
        list_names_vulnerable_links,
    )


def calc_impact_primary_links(
    nx_graph, co2lvl=None, n_nodes=400, split_properties_df=None
) -> dict:
    """Calculate the impact of primary failures on the network.

    Args:
        nx_graph (_type_): _description_
        co2lvl (_type_, optional): _description_. Defaults to None.
        n_nodes (int, optional): _description_. Defaults to 400.
        split_properties_df (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: keys are nx.edges and values are the weighted cumulative lost load share caused by the failure of the edge.
    """

    assert (
        co2lvl is not None or split_properties_df is not None
    ), "Either co2lvl or split_properties must be provided"

    if split_properties_df is None:
        split_properties_df = pd.read_csv(
            path_to_evaluation_results
            + f"split_properties_Co2L{co2lvl}_n{n_nodes}.csv",
            index_col=0,
        )
    split_properties_df.lost_load_rocof_share_weighted = (
        split_properties_df.lost_load_rocof_share
        * split_properties_df.snapshot_weighting
    )

    cumulative_lost_load_share_by_initial_failure = {
        edge: 0 for edge in nx_graph.edges()
    }
    init_failures_0 = set(list(split_properties_df.init_failures_0.unique()))
    init_failures_1 = set(list(split_properties_df.init_failures_1.unique()))
    init_failures = init_failures_0.union(init_failures_1)

    edge_number_to_nx_edge = {i: (u, v) for i, (u, v) in enumerate(nx_graph.edges())}

    for idx_line in init_failures:
        cumulative_lost_load_share_by_initial_failure[
            edge_number_to_nx_edge[idx_line]
        ] = split_properties_df.loc[
            (split_properties_df.init_failures_0 == idx_line)
            | (split_properties_df.init_failures_1 == idx_line),
            "lost_load_rocof_share",
        ].sum()

    return cumulative_lost_load_share_by_initial_failure


def get_most_impactful_primary_links(
    nx_graph: networkx.Graph,
    co2lvl: float,
    nn_links: int,
):
    """Get a list with nn_links links that trigger the most lost load share."""

    cumulative_lost_load_share_by_initial_failure = calc_impact_primary_links(
        nx_graph, co2lvl=co2lvl
    )

    tuple_vulnerable_links = get_keys_of_largest_items_dict(
        cumulative_lost_load_share_by_initial_failure, nn_links
    )

    list_names_vulnerable_links = [xx[0] for xx in tuple_vulnerable_links]

    return cumulative_lost_load_share_by_initial_failure, list_names_vulnerable_links


def increase_capacity_most_important_lines(
    pypsa_net: pypsa.Network,
    nx_graph_in: networkx.Graph,
    nn_links: int,
    delta_num_parallel: float,
    co2lvl: float,
    extension_type: str,
    casc_dict: Union[dict, None] = None,
):
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
    nr_failures = calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )
    number_simulations = (
        len(nr_failures) * pypsa_net.snapshot_weightings.generators.sum()
    )

    if extension_type == "most_likely_primary":
        # List with edges to extend
        importance_all_lines, _, selected_edges = get_most_likely_primary_links(
            pypsa_net, nx_graph, casc_dict, number_simulations, nn_links
        )

    elif extension_type == "most_impactful_primary":
        # List with edges to extend
        importance_all_lines, selected_edges = get_most_impactful_primary_links(
            nx_graph, co2lvl=co2lvl, nn_links=nn_links
        )
    else:
        raise ValueError(
            "extension_type must be either most_likely_primary or most_impactful_primary"
        )

    # extend lines
    for edge_r in selected_edges:
        num_parallel = nx_graph.edges[edge_r]["num_parallel"]
        num_parallel_new = nx_graph.edges[edge_r]["num_parallel"] + delta_num_parallel
        num_par_factor = num_parallel_new / num_parallel
        nx_graph.edges[edge_r]["num_parallel"] = num_parallel_new
        nx_graph.edges[edge_r]["s_nom"] *= num_par_factor
        nx_graph.edges[edge_r]["weight"] *= num_par_factor

    assert nn_links == len(
        selected_edges
    ), "Number of links extended must be equal to nn_links"

    return nx_graph, importance_all_lines, selected_edges
