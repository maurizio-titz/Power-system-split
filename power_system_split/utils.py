#!usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains useful methods to calculate and evaluate system
splits in solved PyPSA networks"""

import numpy as np
import networkx as nx
import pandas as pd
import itertools

def construct_incidencematrix_from_orientation(Graph):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order"""
    B = np.zeros((len(Graph.nodes()),len(Graph.edges())))
    orientations = nx.get_edge_attributes(Graph,'orientation')
    node_list = list(Graph.nodes())
    if isinstance(Graph, nx.MultiGraph):
        for i,edge in enumerate(Graph.edges(keys = True)):
            orientation = orientations[edge]
            n1 = node_list.index(orientation[0])
            B[n1,i] = 1.
            n2 = node_list.index(orientation[1])
            B[n2,i] = -1.
    elif isinstance(Graph, nx.Graph):
        for i,edge in enumerate(Graph.edges()):
            orientation = orientations[edge]
            n1 = node_list.index(orientation[0])
            B[n1,i] = 1.
            n2 = node_list.index(orientation[1])
            B[n2,i] = -1.
    return B

def redefined_index(Graph,element):
    """Get index of element in edge list for graph Graph"""
    if isinstance(Graph,nx.MultiGraph):
        edge_list = list(Graph.edges(keys = True))
        try:
            index = edge_list.index(element)
        except ValueError:
            index = edge_list.index(element[1::-1] + (element[2],))
    elif isinstance(Graph, nx.Graph):
        edge_list = list(Graph.edges())
        try:
            index = edge_list.index(element)
        except ValueError:
            index = edge_list.index(element[::-1])

    return index

def simulate_cascade_PTDF_based(Graph,trigger_links,initial_flows,line_limits):
    """Simulate a cascade of failures using the network topology G, which is assumed to have a property 'orientation'
    for each edge, the initially failing links, the initial flows (as dictionary) and the line limits (as dictionary).
    The approach used to simulate the cascade is based on the Power Transfer Distribution Factors assuming fixed power
    injections and subsequently calculating the flows using the PTDFs after the removal of all failing links from
    the network"""
    print("""NOTE: Using simulate_cascade_PTDF_based_edge_based for cascade
             simulations is preferred due to better data storing, since the
             return values there are edges not edge indices.""")
    # store indices of failing links
    failure_cascade = []

    multi_graph = False
    if isinstance(Graph,nx.MultiGraph):
        multi_graph = True

    if not multi_graph:
        flows_G = np.array([initial_flows[(u,v)] for u,v in Graph.edges()])
        smax_G = np.array([line_limits[(u,v)] for u,v in Graph.edges()])
    else:
        flows_G = np.array([initial_flows[(u,v,key)] for u,v,key in Graph.edges(keys = True)])
        smax_G = np.array([line_limits[(u,v,key)] for u,v,key in Graph.edges(keys = True)])

    if not multi_graph:
        Graph_edges = list(Graph.edges())
    else:
        Graph_edges = list(Graph.edges(keys = True))

    failure_cascade = [redefined_index(Graph,trigger_link) for trigger_link in trigger_links]

    if len(np.where(np.abs(flows_G)>smax_G)[0]):
        print("Setup has initial overloads!")

    stop = 0
    system_split = False
    while not stop:
        H = Graph.copy()
        I0 = construct_incidencematrix_from_orientation(H)

        if not multi_graph:
            flows0 = np.array([initial_flows[(u,v)] for u,v in H.edges()])
        else:
            flows0 = np.array([initial_flows[(u,v,key)] for u,v,key in H.edges(keys = True)])

        P0 = np.dot(I0,flows0)

        H.remove_edges_from([Graph_edges[i] for i in failure_cascade])

        if not nx.is_connected(H):
            system_split = True
            break

        I = construct_incidencematrix_from_orientation(H)

        line_weights = nx.get_edge_attributes(H,'weight')
        if not multi_graph:
            line_susceptances = np.array([line_weights[(u,v)] for u,v in H.edges()])
        else:
            line_susceptances = np.array([line_weights[(u,v,key)] for u,v,key in H.edges(keys = True)])

        B_d = np.diag(line_susceptances)
        L = np.linalg.multi_dot([I,B_d,I.T])

        #L = nx.laplacian_matrix(H)
        try:
            theta = np.linalg.solve(L,P0)
        except np.linalg.LinAlgError:
            # pseudoinverse
            L_inv = np.linalg.pinv(L)
            theta = np.dot(L_inv,P0)

        flows_H = np.linalg.multi_dot([B_d,I.T,theta])

        if not multi_graph:
            smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        else:
            smax = np.array([line_limits[(u,v,key)] for u,v,key in H.edges(keys = True)])

        next_indices = np.where(np.abs(flows_H)>smax)[0]

        ### map next failing indices to original graph
        if not multi_graph:
            H_edges = list(H.edges())
        else:
            H_edges = list(H.edges(keys = True))
        for ind in next_indices:
            index = redefined_index(Graph,element = H_edges[ind])
            if not index in failure_cascade:
                failure_cascade.append(index)

        if len(next_indices)==0:
            stop = 1

        loading_dict = {}
        if not multi_graph:

            for i,edge in enumerate(H.edges()):
                loading_dict[edge] = flows_H[i]
                loading_dict[edge[::-1]] = flows_H[i]
        else:
            for i,edge in enumerate(H.edges(keys = True)):
                loading_dict[edge] = flows_H[i]

    return failure_cascade,loading_dict,system_split


def get_effective_injections(Graph,initial_flows):
    """get effective injections from the initial flows"""
    multi_graph = False
    if isinstance(Graph,nx.MultiGraph):
        multi_graph = True

    if not multi_graph:
        flows_G = np.array([initial_flows[(u,v)] for u,v in Graph.edges()])
    else:
        flows_G = np.array([initial_flows[(u,v,key)] for u,v,key in Graph.edges(keys = True)])

    I = construct_incidencematrix_from_orientation(Graph)
    P0 = np.dot(I,flows_G)
    return P0

def simulate_cascade_PTDF_based_edge_based_reduced(Graph,
                                                   P0,
                                                   trigger_links,
                                                   line_limits,
                                                   return_loading_dict = False):
    """Simulate a cascade of failures using the network topology G, which is assumed to have a property 'orientation'
    for each edge, the initially failing links, the initial flows (as dictionary) and the line limits (as dictionary).
    The approach used to simulate the cascade is based on the Power Transfer Distribution Factors assuming fixed power
    injections and subsequently calculating the flows using the PTDFs after the removal of all failing links from
    the network"""

    multi_graph = False
    if isinstance(Graph,nx.MultiGraph):
        multi_graph = True

    failure_cascade = trigger_links

    stop = 0
    system_split = False

    while not stop:
        H = Graph.copy()

        H.remove_edges_from(failure_cascade)

        if not nx.is_connected(H):
            system_split = True
            break

        I = construct_incidencematrix_from_orientation(H)

        line_weights = nx.get_edge_attributes(H,'weight')

        if not multi_graph:
            line_susceptances = np.array([line_weights[(u,v)] for u,v in H.edges()])
        else:
            line_susceptances = np.array([line_weights[(u,v,key)] for u,v,key in H.edges(keys = True)])

        B_d = np.diag(line_susceptances)

        L = np.linalg.multi_dot([I,B_d,I.T])
        #L = nx.laplacian_matrix(H)
        try:
            theta = np.linalg.solve(L,P0)
        except np.linalg.LinAlgError:
            # pseudoinverse
            L_inv = np.linalg.pinv(L)
            theta = np.dot(L_inv,P0)

        flows_H = np.linalg.multi_dot([B_d,I.T,theta])

        if not multi_graph:
            smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        else:
            smax = np.array([line_limits[(u,v,key)] for u,v,key in H.edges(keys = True)])

        next_indices = np.where(np.abs(flows_H)>smax)[0]

        ### map next failing indices to original graph
        if not multi_graph:
            H_edges = list(H.edges())
            for ind in next_indices:
                edge = H_edges[ind]
                if not ((edge in failure_cascade) or (edge[::-1] in failure_cascade)):
                    failure_cascade.append(edge)
        else:
            H_edges = list(H.edges(keys = True))
            for ind in next_indices:
                edge = H_edges[ind]
                if not ((edge in failure_cascade) or ((*edge[:2][::-1],edge[2]) in failure_cascade)):
                    failure_cascade.append(edge)

        if len(next_indices)==0:
            stop = 1

        if return_loading_dict:
            loading_dict = {}
            if not multi_graph:

                for i,edge in enumerate(H.edges()):
                    loading_dict[edge] = flows_H[i]
                    loading_dict[edge[::-1]] = flows_H[i]
            else:
                for i,edge in enumerate(H.edges(keys = True)):
                    loading_dict[edge] = flows_H[i]

    if not return_loading_dict:
        return_vals = [failure_cascade,system_split]
    else:
        return_vals = [failure_cascade,loading_dict,system_split]
    return return_vals




def simulate_cascade_PTDF_based_edge_based(Graph,trigger_links,initial_flows,line_limits,return_loading_dict = False):
    """Simulate a cascade of failures using the network topology G, which is assumed to have a property 'orientation'
    for each edge, the initially failing links, the initial flows (as dictionary) and the line limits (as dictionary).
    The approach used to simulate the cascade is based on the Power Transfer Distribution Factors assuming fixed power
    injections and subsequently calculating the flows using the PTDFs after the removal of all failing links from
    the network"""

    multi_graph = False
    if isinstance(Graph,nx.MultiGraph):
        multi_graph = True

    if not multi_graph:
        flows_G = np.array([initial_flows[(u,v)] for u,v in Graph.edges()])
        smax_G = np.array([line_limits[(u,v)] for u,v in Graph.edges()])
    else:
        flows_G = np.array([initial_flows[(u,v,key)] for u,v,key in Graph.edges(keys = True)])
        smax_G = np.array([line_limits[(u,v,key)] for u,v,key in Graph.edges(keys = True)])

    failure_cascade = trigger_links

    if len(np.where(np.abs(flows_G)>smax_G)[0]):
        print("Setup has initial overloads!")

    stop = 0
    system_split = False
    while not stop:
        H = Graph.copy()
        I0 = construct_incidencematrix_from_orientation(H)

        if not multi_graph:
            flows0 = np.array([initial_flows[(u,v)] for u,v in H.edges()])
        else:
            flows0 = np.array([initial_flows[(u,v,key)] for u,v,key in H.edges(keys = True)])

        P0 = np.dot(I0,flows0)

        H.remove_edges_from(failure_cascade)

        if not nx.is_connected(H):
            system_split = True
            break

        I = construct_incidencematrix_from_orientation(H)

        line_weights = nx.get_edge_attributes(H,'weight')
        if not multi_graph:
            line_susceptances = np.array([line_weights[(u,v)] for u,v in H.edges()])
        else:
            line_susceptances = np.array([line_weights[(u,v,key)] for u,v,key in H.edges(keys = True)])

        B_d = np.diag(line_susceptances)

        L = np.linalg.multi_dot([I,B_d,I.T])
        #L = nx.laplacian_matrix(H)
        try:
            theta = np.linalg.solve(L,P0)
        except np.linalg.LinAlgError:
            # pseudoinverse
            L_inv = np.linalg.pinv(L)
            theta = np.dot(L_inv,P0)

        flows_H = np.linalg.multi_dot([B_d,I.T,theta])

        if not multi_graph:
            smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        else:
            smax = np.array([line_limits[(u,v,key)] for u,v,key in H.edges(keys = True)])

        next_indices = np.where(np.abs(flows_H)>smax)[0]

        ### map next failing indices to original graph
        if not multi_graph:
            H_edges = list(H.edges())
            for ind in next_indices:
                edge = H_edges[ind]
                if not ((edge in failure_cascade) or (edge[::-1] in failure_cascade)):
                    failure_cascade.append(edge)
        else:
            H_edges = list(H.edges(keys = True))
            for ind in next_indices:
                edge = H_edges[ind]
                if not ((edge in failure_cascade) or ((*edge[:2][::-1],edge[2]) in failure_cascade)):
                    failure_cascade.append(edge)

        if len(next_indices)==0:
            stop = 1

        if return_loading_dict:
            loading_dict = {}
            if not multi_graph:

                for i,edge in enumerate(H.edges()):
                    loading_dict[edge] = flows_H[i]
                    loading_dict[edge[::-1]] = flows_H[i]
            else:
                for i,edge in enumerate(H.edges(keys = True)):
                    loading_dict[edge] = flows_H[i]

    if not return_loading_dict:
        return_vals = [failure_cascade,system_split]
    else:
        return_vals = [failure_cascade,loading_dict,system_split]
    return return_vals


def build_networkx_graph(snet_branches,multi_graph = False):
    """Build a networkx graph from the pypsa networks"""
    if not multi_graph:
        F = nx.Graph()
    else:
        F = nx.MultiGraph()

    for line_index,line in snet_branches.iterrows():

        if not F.has_edge(line['bus0'],line['bus1']):
            F.add_edge(line['bus0'],line['bus1'],
                       weight = 1/line['x_pu_eff'],
                       orientation = (line['bus0'],line['bus1']),
                       line_index = [line_index[1]],
                       s_nom = line['s_nom'])
        else:
            if not multi_graph:
                ### add up line susceptance and line limit to add the lines to a bulk and store the line indices of all lines
                ### that were grouped together
                F[line['bus0']][line['bus1']]['weight'] += 1/line['x_pu_eff']
                F[line['bus0']][line['bus1']]['s_nom'] += line['s_nom']
                F[line['bus0']][line['bus1']]['line_index'].append(line_index[1])
            else:

                F.add_edge(line['bus0'],line['bus1'],

                           weight = 1/line['x_pu_eff'],
                           orientation = (line['bus0'],line['bus1']),
                           line_index = [line_index[1]],
                           s_nom = line['s_nom'])
    return F


def get_split_components(split,Graph,criterion):
    """Return the split resulting from the edge list in split if the resulting subgraphs are larger than 10 nodes"""
    assert isinstance(split[0],tuple),"""Format of split data has been changed.
                   Please use transform_cascade_results to adjust to new format"""
    F = Graph.copy()
    #cascade_edges = [list(F.edges())[index] for index in split]
    F.remove_edges_from(split)
    subgraphs =  list((F.subgraph(c).copy() for c in nx.connected_components(F)))
    if criterion == 'nodes':
        ## old criterion of considering only cases where at least two subgraphs
        ## with at least two nodes each exist
        relevant_subgraphs = [i for i in range(len(subgraphs)) if len(subgraphs[i].nodes())>10]
        if len(relevant_subgraphs)<2:
            return_val = []
        else:
            return_val = [subgraphs[relevant_subgraphs[i]] for i in range(len(relevant_subgraphs))]
    elif criterion == 'load':
        ## new criterion based on load where all subgraphs are considered
        ## independent of their number of nodes
        relevant_subgraphs = [i for i in range(len(subgraphs))]
        return_val = [subgraphs[relevant_subgraphs[i]] for i in range(len(relevant_subgraphs))]

    return return_val

def evaluate_system_split_inertia(split,Graph,generators,current_generation, use_pnom = True):
    """For a given list of edge indices split, a networkx graph Graph,

    generators: pandas dataframe
        a pypsa solved network generators (passed as network.generators)
    current_generation: pandas Series
        time dependent generation of pypsa network for a given timestamp
        (passed via network.generators_t.p.loc[timestamp])

    in total, for a given solved network you can call this function as

    evaluate_system_split_inertia(cascade,G,network.generators,network.generators_t.p.loc[timestamp])

    here, cascade is an output of simulate_cascade_PTDF_based corresponding to a given timestamp
    and system split
    """
    inertiaplants = ['CCGT','OCGT','coal','nuclear','oil','ror']
    # threshold below which a generator is not counted as being participating
    participation_threshold = 1e0
    subgraphs = get_split_components(split,Graph)
    if len(subgraphs) < 2 :
        return 0

    inertia_generations = []
    for subgraph in subgraphs:
        gens = generators[generators["bus"].isin(list(subgraph.nodes()))]
        existing_inertia_sources = list(set(inertiaplants)-(set(inertiaplants)-set(list(gens.carrier))))
        current_generators = current_generation.loc[list(gens.index)]
        if use_pnom:
            participating = current_generation.loc[list(gens.index)]>participation_threshold
            reduced_gens = gens[participating]
            inertia_generation = (reduced_gens["p_nom"]*reduced_gens["p_max_pu"]).groupby(reduced_gens.carrier).sum().loc[existing_inertia_sources].sum()
        else:
            inertia_generation = current_generators.groupby(gens.carrier).sum().loc[existing_inertia_sources].sum()
        inertia_generations.append(inertia_generation)
    return inertia_generations


def likelihood_systemsplit_edge_based(split_dict,Graph,only_large_splits = True):
    """ Calculate the empirical likelihood that a given edge is involved
    in a system split based on the split_dict

    If only_large_splits is True, only splits which yield components
    with more than ten nodes each are evaluated"""

    likelihood_dict = {(u,v):0.0 for u,v in Graph.edges()}

    if only_large_splits:
        # counts the number of relevant splits
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                relevant_subgraphs = get_split_components(cascade,Graph)
                if len(relevant_subgraphs) < 2:
                    continue
                for edge in cascade:
                    likelihood_dict[edge] += 1
                split_counter += 1
        try:
            # normalisation only possible if splits occured at all
            for edge in likelihood_dict.keys():
                likelihood_dict[edge] /= split_counter
        except ZeroDivisionError:
            pass
    else:
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                assert isinstance(cascade[0],tuple),"""Format of split data has been changed.
                                   Please use transform_cascade_results to adjust to new format"""

                for edge in cascade:
                    likelihood_dict[edge] += 1
                    split_counter += 1
        try:
            for edge in likelihood_dict.keys():
                likelihood_dict[edge] /= split_counter
        except ZeroDivisionError:
            pass
        #for count, edge  in enumerate(list(Graph.edges())):
        #    likelihood_dict[edge] += likelihoods[count]

    return likelihood_dict



def likelihood_systemsplit_node_based(split_dict,Graph,only_large_splits = True):
    """ Calculate the empirical likelihood that a given edge is involved
    in a system split based on the split_dict

    If only_large_splits is True, only splits which yield components
    with more than ten nodes each are evaluated"""
    complete_graph_edges = list(itertools.permutations(list(Graph.nodes()),2))
    likelihood_dict = {(u,v) : 0.0 for u,v in complete_graph_edges}

    if only_large_splits:
        # counts the number of relevant splits
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                relevant_subgraphs = get_split_components(cascade,Graph)
                if len(relevant_subgraphs) < 2:
                    continue
                for subgraph in relevant_subgraphs:
                    for u,v in list(itertools.permutations(list(subgraph.nodes()),2)):
                        likelihood_dict[(u,v)] += 1
                split_counter += 1
        try:
            # normalisation only possible if splits occured at all
            for edge in likelihood_dict.keys():
                likelihood_dict[edge] /= split_counter
        except ZeroDivisionError:
            pass
    else:
        # counts the number of relevant splits
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                relevant_subgraphs = get_split_components(cascade,Graph)
                for subgraph in relevant_subgraphs:
                    for u,v in list(itertools.permutations(list(subgraph.nodes()),2)):
                        likelihood_dict[(u,v)] += 1
                split_counter += 1
            print(timestamp)
        try:
            # normalisation only possible if splits occured at all
            for edge in likelihood_dict.keys():
                likelihood_dict[edge] /= split_counter
        except ZeroDivisionError:
            pass
    return likelihood_dict

def solution_key_to_pandas_timestamp(key):
    """Convert from dictionary key used in solution dictionaries
    to pandas datetime index used in pypsa networks"""
    day,month,year,hour = key.split('_')
    snapshot = pd.Timestamp(year = int(year),month = int(month),hour = int(hour),day = int(day))
    return snapshot

def pandas_timestamp_to_solution_key(timestamp):
    """Convert from pandas datetime index to
    to dictionary key used in solution dictionaries used in pypsa networks"""
    return timestamp.strftime('%d_%m_%Y_%H')

def verify_cascade_results(G,test_cascades,initial_loading):
    """Verify if the cascades contained in the list test_cascades
    (list of indices of the edges in list(G.edges())[index])
    are triggered by the first link in each list given
    the initial loading and the Graph G"""

    line_limits = nx.get_edge_attributes(G,'s_nom')

    ### iterate only over non-bridge edges for now

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    for i in range(len(test_cascades)):
        trigger_link = list(G.edges())[test_cascades[i][0]]
        failing_links,new_loading_dict,system_split = simulate_cascade_PTDF_based(G,
                                                                 trigger_links = [trigger_link],
                                                                 line_limits = line_limits,
                                                                 initial_flows = initial_loading)
        assert failing_links == test_cascades[i]
    print("All results correct!")
    return

def get_inertia_gen_subgraph(subgraph,generators,current_generation,storages,current_storage,use_pnom,inertiaplants = None, inertia_storages = None):
    """get inertia generation for a subgraph"""
    if not inertiaplants:
        inertiaplants = ['CCGT','OCGT','coal','nuclear','oil','ror','lignite','biomass']
    if not inertia_storages:
        inertia_storages = ['PHS']

    # threshold below which a generator or  is not counted as being participating
    participation_threshold = 0.05

    gens                     = generators[generators["bus"].isin(list(subgraph.nodes()))]
    existing_inertia_sources = list(set(inertiaplants)-(set(inertiaplants)-set(list(gens.carrier))))
    current_generators       = current_generation.loc[list(gens.index)]

    stores                    = storages[storages["bus"].isin(list(subgraph.nodes()))]
    existing_inertia_storages = list(set(inertia_storages)-(set(inertia_storages)-set(list(stores.carrier))))
    current_storages          = current_storage.loc[list(stores.index)]

    if use_pnom:
        participating_gens = current_generation.loc[list(gens.index)]>participation_threshold*gens["p_nom"]
        reduced_gens       = gens[participating_gens]
        existing_inertia_sources = list(set(existing_inertia_sources)-(set(existing_inertia_sources)-set(reduced_gens.carrier)))
        nominal_power            = reduced_gens["p_nom"]*reduced_gens["p_max_pu"]
        inertia_generation       = (nominal_power).groupby(reduced_gens.carrier).sum().loc[existing_inertia_sources].sum()

        participating_storages    = current_storages.loc[list(stores.index)]>participation_threshold*stores["p_nom"]
        reduced_stores            = stores[participating_storages]
        existing_inertia_storages = list(set(existing_inertia_storages)-(set(existing_inertia_storages)-set(reduced_stores.carrier)))
        inertia_generation += (reduced_stores["p_nom"]).groupby(reduced_stores.carrier).sum().loc[existing_inertia_storages].sum()

    else:
        inertia_generation = current_generators.groupby(gens.carrier).sum().loc[existing_inertia_sources].sum()
        inertia_generation += current_storages.groupby(stores.carrier).sum().loc[existing_inertia_storages].sum()
    return inertia_generation


def get_load_imbalance_subgraph(subgraph,generators,current_generation,storages,current_storage,loads,current_load):
    """Calculate load imbalance due to subgraph"""
    gens   = generators[generators["bus"].isin(list(subgraph.nodes()))]
    loads  = loads[loads["bus"].isin(list(subgraph.nodes()))]
    stores = storages[storages["bus"].isin(list(subgraph.nodes()))]


    load_imbalance = current_generation.loc[list(gens.index)].sum() + current_storage.loc[list(stores.index)].sum()-\
                        current_load.loc[list(loads.index)].sum()
    return load_imbalance


def get_available_flexible_generation(subgraph,
                                      generators,
                                      current_generation,
                                      storages,
                                      current_storage,
                                      flexible_plants = None,
                                      flexible_storages = None):
    """Calculate avaible generation in flexible technologies"""
    if not flexible_plants:
        flexible_plants = ['OCGT','ror']
    if not flexible_storages:
        flexible_storages = ['PHS']

    gens                     = generators[generators["bus"].isin(list(subgraph.nodes()))]
    existing_flexible_sources = list(set(flexible_plants)-(set(flexible_plants)-set(list(gens.carrier))))
    current_generators       = current_generation.loc[list(gens.index)]

    stores                    = storages[storages["bus"].isin(list(subgraph.nodes()))]
    existing_flexible_storages = list(set(flexible_storages)-(set(flexible_storages)-set(list(stores.carrier))))
    current_storages          = current_storage.loc[list(stores.index)]

    maximal_generation = (gens["p_nom"]*gens["p_max_pu"]).groupby(gens.carrier).sum().loc[existing_flexible_sources].sum()
    flexible_generation = maximal_generation - current_generators.groupby(gens.carrier).sum().loc[existing_flexible_sources].sum()

    #maximal_storage = (stores["p_nom"]).groupby(stores.carrier).sum().loc[existing_flexible_storages].sum()
    #flexible_generation += maximal_storage - current_storages.groupby(stores.carrier).sum().loc[existing_flexible_storages].sum()
    return flexible_generation

def check_load_criterion(subgraphs,generators,current_generation,storages,current_storage,loads,current_load)):
    """ Check load/generation criterion which means
    that none of the subgraph accounts for 90 % of the load
    or generation at the current timestamp"""

    load_criterion = True

    threshold = 0.9

    overall_generation = current_generation.sum() + current_storage.sum()
    overall_load       = current_load.sum()

    subgraph_contributions = np.zeros((len(subgraphs),2))

    for count,subgraph in enumerate(subgraphs):
        gens                = generators[generators["bus"].isin(list(subgraph.nodes()))]
        loads               = loads[loads["bus"].isin(list(subgraph.nodes()))]
        stores              = storages[storages["bus"].isin(list(subgraph.nodes()))]
        subgraph_generation = current_generation.loc[list(gens.index)].sum() + current_storage.loc[list(stores.index)].sum()
        subgraph_load       = current_load.loc[list(loads.index)].sum()

        subgraph_contributions[count] = np.array([subgraph_generation/overall_generation,subgraph_load/overall_load])

    if np.any(subgraph_contributions>threshold):
        load_criterion = False
    return load_criterion


def evaluate_split_observables(split,
                               pypsa_network,
                               timestamp,
                               use_pnom = True,
                               inertiaplants = None,
                               inertia_storages = None,
                               flexible_plants = None,
                               flexible_storage = None,
                               snet_index = None,
                               criterion = 'nodes'):
    """
    split: list of edges split to be used with networkx graph created from pypsa network

    pypsa_network: pypsa network object containing solution for timestamps
    """

    assert isinstance(timestamp,pd.Timestamp)

    try:
        snet = pypsa_network.sub_networks['obj'][snet_index]
    except KeyError:
        snet = pypsa_network

    branches           = snet.branches()[["bus0","bus1","x_pu_eff","s_nom"]]
    Graph              = build_networkx_graph(branches)

    # Rescale load shedding since units for load shedding are different
    # than for generation, storage and load
    load_shedding_indices = pypsa_network.generators[pypsa_network.generators.carrier.isin(['load'])].index

    current_generation = pypsa_network.generators_t.p.loc[timestamp].copy()
    current_generation[load_shedding_indices] /= 1e3
    current_storage    = pypsa_network.storage_units_t.p.loc[timestamp]
    current_load       = pypsa_network.loads_t.p.loc[timestamp]

    #try:
    assert np.abs(current_generation.sum()+current_storage.sum()-current_load.sum())<1e-3
    #except AssertionError:
    #    print(np.abs(current_generation.sum()+current_storage.sum()-current_load.sum()))
    results_dict = {'inertia_proxy': [],'load_imbalance': [], 'available_flexible_generation': [] }

    evaluate_results = False

    subgraphs = get_split_components(split,Graph, criterion = criterion)

    if criterion == 'nodes':
        if len(subgraphs) >= 2:
            evaluate_results = True
    elif criterion == 'load':
        evaluate_results = check_load_criterion(subgraphs,
                                                pypsa_network.generators,
                                                current_generation,
                                                pypsa_network.storage_units,
                                                current_storage,
                                                pypsa_network.loads,
                                                current_load)

    if evaluate_results:
        for subgraph in subgraphs:
            inertia_generation = get_inertia_gen_subgraph(subgraph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                          use_pnom,
                                                          inertiaplants = inertiaplants,
                                                          inertia_storages = inertia_storages)

            load_imbalance = get_load_imbalance_subgraph(subgraph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                          pypsa_network.loads,
                                                          current_load)

            available_flexible_generation = get_available_flexible_generation(subgraph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                          flexible_plants = flexible_plants,
                                                          flexible_storages = flexible_storage)

            results_dict['inertia_proxy'].append(inertia_generation)
            results_dict['load_imbalance'].append(load_imbalance)
            results_dict['available_flexible_generation'].append(available_flexible_generation)

    return results_dict


def transform_cascade_results(Graph,cascade):
    """Cascade model used to save indices of edges
    but the new format should be the edges itself"""
    if isinstance(Graph, nx.MultiGraph):
        Graph_edges = list(Graph.edges(keys = True))
    elif isinstance(Graph, nx.Graph):
        Graph_edges = list(Graph.edges())
    return [Graph_edges[index] for index in cascade]
