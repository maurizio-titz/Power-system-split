#!/usr/bin/python3
# -*- coding: utf-8 -*

import networkx as nx
import numpy as np

from power_system_split import utils


def test_cascade_no_overloads():
    # Build a small triangular network with two alternative pathways that
    # connect nodes 1 and 2
    G = nx.Graph()
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [0],
               s_nom = 1)
    G.add_edge(1,3,
               weight = 1,
               orientation = (1,3),
               line_index = [1],
               s_nom = 1)
    G.add_edge(3,2,
               weight = 1,
               orientation = (3,2),
               line_index = [2],
               s_nom = 1)

    ## first scenario: Failure does not overload another edge and thus not disconnect the grid
    initial_loading = {(1,2):0.3,(1,3):0.5,(3,2):0.5}

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    line_limits = nx.get_edge_attributes(G,'s_nom')
    failing_edge = (1,3)
    failing_links,new_loading_dict,system_split = utils.simulate_cascade_PTDF_based(G,
                                                                    trigger_links = [failing_edge],
                                                                    line_limits = line_limits,
                                                                    initial_flows = initial_loading)
    assert(np.isclose(new_loading_dict[(1,2)],0.8))
    assert(np.isclose(new_loading_dict[(3,2)],0.0))
    assert(len(failing_links) == 1)
    assert(system_split == False)

    failing_links,new_loading_dict,system_split = utils.simulate_cascade_PTDF_based_edge_based(G,
                                                                trigger_links = [failing_edge],
                                                                line_limits = line_limits,
                                                                initial_flows = initial_loading,
                                                                return_loading_dict = True)

    assert(len(failing_links) == 1)
    assert(system_split == False)
    assert(np.isclose(new_loading_dict[(1,2)],0.8))
    assert(np.isclose(new_loading_dict[(3,2)],0.0))
    
    return


def test_cascade_with_overloads():
    # Build a small triangular network with two alternative pathways that
    # connect nodes 1 and 2
    G = nx.Graph()
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [0],
               s_nom = 1)
    G.add_edge(1,3,
               weight = 1,
               orientation = (1,3),
               line_index = [1],
               s_nom = 1)
    G.add_edge(3,2,
               weight = 1,
               orientation = (3,2),
               line_index = [2],
               s_nom = 1)

    ## first scenario: Failure does not overload another edge and thus not disconnect the grid
    initial_loading = {(1,2):0.6,(1,3):0.5,(3,2):0.5}

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    line_limits = nx.get_edge_attributes(G,'s_nom')
    failing_edge = (1,3)
    failing_links,new_loading_dict,system_split = utils.simulate_cascade_PTDF_based(G,
                                                                    trigger_links = [failing_edge],
                                                                    line_limits = line_limits,
                                                                    initial_flows = initial_loading)
    failing_edges = [list(G.edges())[i] for i in failing_links]
    assert(system_split == True)
    assert(sorted(failing_edges) == sorted([(1,3),(1,2)]))
    return

def test_cascade_no_overloads_multiedges():
    # Build a small triangular network with two alternative pathways that
    # connect nodes 1 and 2
    G = nx.MultiGraph()
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [0],
               s_nom = 1)
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [1],
               s_nom = 1)

    ## first scenario: Failure does not overload another edge and thus not disconnect the grid
    initial_loading = {(1,2,0):0.4,(1,2,1):0.4}

    line_limits = nx.get_edge_attributes(G,'s_nom')
    failing_edge = (1,2,0)
    failing_links,new_loading_dict,system_split = utils.simulate_cascade_PTDF_based(G,
                                                                    trigger_links = [failing_edge],
                                                                    line_limits = line_limits,
                                                                    initial_flows = initial_loading)
    failing_edges = [list(G.edges(keys = True))[i] for i in failing_links]
    assert(np.isclose(new_loading_dict[(1,2,1)],0.8))
    return

def test_cascade_w_overloads_multiedges():
    # Build a small triangular network with two alternative pathways that
    # connect nodes 1 and 2
    G = nx.MultiGraph()
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [0],
               s_nom = 1)
    G.add_edge(1,2,
               weight = 1,
               orientation = (1,2),
               line_index = [1],
               s_nom = 1)

   ## first scenario: Failure does not overload another edge and thus not disconnect the grid
    initial_loading = {(1,2,0):0.6,(1,2,1):0.6}#

    line_limits = nx.get_edge_attributes(G,'s_nom')
    failing_edge = (1,2,0)
    failing_links,new_loading_dict,system_split = utils.simulate_cascade_PTDF_based(G,
                                                                    trigger_links = [failing_edge],
                                                                    line_limits = line_limits,
                                                                    initial_flows = initial_loading)
    failing_edges = [list(G.edges(keys = True))[i] for i in failing_links]
    assert(system_split == True)
    assert(sorted(failing_edges) == sorted([(1,2,0),(1,2,1)]))
    return
