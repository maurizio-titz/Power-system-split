#!usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains useful methods to calculate and evaluate system
splits in solved PyPSA networks"""

import itertools

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import lil_matrix,spdiags
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy import linalg as sc_linalg
from scipy.sparse.csgraph import connected_components

import power_system_split.visualisation as vis

def construct_incidencematrix_from_orientation(Graph,return_np_array = True):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order
    NOTE: This function was tweaked based on networkx incidence matrix function
    https://networkx.org/documentation/stable/_modules/networkx/linalg/graphmatrix.html#incidence_matrix"""
    if isinstance(Graph, nx.MultiGraph):
        edgelist = list(Graph.edges(keys=True))
    else:
        edgelist = list(Graph.edges())
    nodelist = list(Graph.nodes())

    node_index = {node: i for i, node in enumerate(nodelist)}
    B = lil_matrix((len(nodelist),len(edgelist)))
    orientations = nx.get_edge_attributes(Graph,'orientation')

    for i,edge in enumerate(edgelist):
        (u,v) = orientations[edge]
        n1 = node_index[u]
        B[n1,i] = 1.
        n2 = node_index[v]
        B[n2,i] = -1.
    if return_np_array:
        return_val = B.toarray()
    else:
        return_val = B.asformat("csr")
    return return_val

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
                                                   return_loading_dict = False):
    """Simulate a cascade of failures using the network topology G, which is assumed to have a property 'orientation'
    for each edge, the initially failing links, the initial flows (as dictionary) and the line limits (as dictionary).
    The approach used to simulate the cascade is based on the Power Transfer Distribution Factors assuming fixed power
    injections and subsequently calculating the flows using the PTDFs after the removal of all failing links from
    the network"""

    multi_graph = False
    if isinstance(Graph,nx.MultiGraph):
        multi_graph = True

    failure_cascade = trigger_links.copy()

    stop = 0
    system_split = False

    H = Graph.copy()

    while not stop:
        

        for e in failure_cascade:
            if H.has_edge(*e):
                apply_line_failure(H, e)

        if not nx.is_connected(H):
            system_split = True
            break

        I = construct_incidencematrix_from_orientation(H,return_np_array = False)

        line_weights = nx.get_edge_attributes(H,'weight')

        if not multi_graph:
            line_susceptances = np.array([line_weights[(u,v)] for u,v in H.edges()])
        else:
            line_susceptances = np.array([line_weights[(u,v,key)] for u,v,key in H.edges(keys = True)])

        #B_d = np.diag(line_susceptances)
        #L = np.dot(np.dot(I,B_d),I.T)
        B_d = spdiags(line_susceptances,0,len(line_susceptances),len(line_susceptances))
        L = I.dot(B_d).dot(I.T)
        #L = nx.laplacian_matrix(H)
        try:
            theta = spsolve(L,P0)#solve(L,P0)
        except np.linalg.LinAlgError:
            # pseudoinverse
            L_inv = np.linalg.pinv(L.toarray())
            theta = np.dot(L_inv,P0)

        flows_H = B_d.dot((I.T).dot(theta))
        
        # Get a list of (updated) line limits
        line_limits = nx.get_edge_attributes(H, 's_nom')
        smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        
        #TODO: clean following code up if not needed anymore
        # if not multi_graph:
        #     smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        # else:
        #     smax = np.array([line_limits[(u,v,key)] for u,v,key in H.edges(keys = True)])

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


def build_networkx_graph(pypsa_network,multi_graph = False, snet_index = None):
    """Build a networkx graph from the pypsa networks"""
    pypsa_network.determine_network_topology()

    try:
        snet = pypsa_network.sub_networks['obj'][snet_index]
    except KeyError:
        snet = pypsa_network

    branches = snet.branches()
    positions = pypsa_network.buses[["x","y"]]
    pos = dict(zip(positions.index,list(zip(positions.x,positions.y))))

    branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom", "num_parallel"]]

    if not multi_graph:
        F = nx.Graph()
    else:
        F = nx.MultiGraph()
    
    for line_index,line in branches.iterrows():

        if not F.has_edge(line['bus0'],line['bus1']):
            F.add_edge(line['bus0'],line['bus1'],
                       weight = 1/line['x_pu_eff'],
                       orientation = (line['bus0'],line['bus1']),
                       line_index = [line_index[1]],
                       s_nom = line['s_nom'],
                       num_parallel= line['num_parallel'])
        else:
            #TODO check wether we really have to add up lines here. If not: remove this option and state clearly that we
            # assume single line graphs!
            print('WARNING: Multiple lines between two buses are not supported! The results might be incorrect!')
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
    nx.set_node_attributes(F,pos,'pos')
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
    elif (criterion == 'load') or (criterion=='all') :
        ## all subgraphs are considered
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

    for cascade in test_cascades:
        trigger_link = list(G.edges())[cascade[0]]
        failing_links,new_loading_dict,system_split = simulate_cascade_PTDF_based(G,
                                                                 trigger_links = [trigger_link],
                                                                 line_limits = line_limits,
                                                                 initial_flows = initial_loading)
        assert failing_links == cascade
    print("All results correct!")
    return

def calc_system_inertia_over_time(pypsa_network,
                                  snet_index = None,
                                  use_pnom = True,
                                  inertiaplants = None,
                                  inertia_storages = None):
    """get inertia generation for a subgraph"""

    Graph              = build_networkx_graph(pypsa_network,snet_index = snet_index)

    inertia_over_time = np.zeros(len(pypsa_network.snapshots))

    for count,timestamp in enumerate(pypsa_network.snapshots):

        # Rescale load shedding since units for load shedding are different
        # than for generation, storage and load
        # (Note: In this project load shedding is not implement)
        load_shedding_indices = pypsa_network.generators[pypsa_network.generators.carrier.isin(['load'])].index

        current_generation = pypsa_network.generators_t.p.loc[timestamp].copy()
        current_generation[load_shedding_indices] /= 1e3
        current_storage    = pypsa_network.storage_units_t.p.loc[timestamp]


        inertia_over_time[count] = get_inertia_gen_subgraph(Graph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                           use_pnom,
                                                          inertiaplants = inertiaplants,
                                                          inertia_storages = inertia_storages)

    return inertia_over_time


def get_inertia_gen_subgraph(subgraph,generators,current_generation,storages,current_storage,use_pnom,inertiaplants = None, inertia_storages = None):
    """get inertia generation for a subgraph"""
    if not inertiaplants:
        inertiaplants = ['CCGT','OCGT','coal','nuclear','oil','ror','lignite','biomass']
    if not inertia_storages:
        inertia_storages = ['PHS']

    # threshold below which a generator or storage is not counted as being participating
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


def get_load_imbalance_subgraph(subgraph,
                                generators,
                                current_generation,
                                storages,
                                current_storage,
                                loads,
                                current_load,
                                HVDC_transport):
    """Calculate load imbalance due to subgraph"""
    gens   = generators[generators["bus"].isin(list(subgraph.nodes()))]
    loads  = loads[loads["bus"].isin(list(subgraph.nodes()))]
    stores = storages[storages["bus"].isin(list(subgraph.nodes()))]


    load_imbalance = current_generation.loc[list(gens.index)].sum() + current_storage.loc[list(stores.index)].sum()-\
                        current_load.loc[list(loads.index)].sum()
    load_imbalance -= HVDC_transport[HVDC_transport.index.isin(list(subgraph.nodes()))].sum()
    return load_imbalance

def get_load_subgraph(subgraph,
                    generators,
                    current_generation,
                    storages,
                    current_storage,
                    loads,
                    current_load,
                    HVDC_transport):
    """Calculate load in subgraph (ingnoring HVDC and storage consumption)"""
    loads  = loads[loads["bus"].isin(list(subgraph.nodes()))]
    #stores = storages[storages["bus"].isin(list(subgraph.nodes()))]
    #HVDC =...
    
    subgraph_load = current_load.loc[list(loads.index)].sum()
    
    return subgraph_load

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

def check_load_criterion(subgraphs,
                         generators,
                         current_generation,
                         storages,
                         current_storage,
                         loads,
                         current_load,
                         HVDC_transport):
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
        subgraph_generation -= HVDC_transport[HVDC_transport.index.isin(list(subgraph.nodes()))].sum()
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
                               snet_index = None,
                               criterion = 'nodes'):
    """
    ARGUMENTS:

    split: list of edges split to be used with networkx graph created from pypsa network

    pypsa_network: pypsa network object containing solution for timestamps

    timestamp: a pandas timestamp that is contained in the snapshots of the
               pypsa network

    use_pnom (optional): boolean, whether or not to use the nominal power of a generator to
              to estimate its inertia
              defaults to True

    inertiaplants (optional): list of strings that describes the carriers that are assumed
                   to contribute to the system inertia

                   defaults to None which uses the list of inertia plants provided
                   in the function "get_inertia_gen_subgraph"

    inertia_storages (optional): list of strings that describes the carriers that are assumed
                   to contribute to the system inertia

                   defaults to None which uses the list of inertia storages provided
                   in the function "get_inertia_gen_subgraph"

    snet_index (optional): integer, gives the index of the subnet of the pypsa network to use
                 defaults to None

    criterion (optional): string, that describes which criterion to use to determine which
               subgraphs are evaluated.
               defaults to "nodes"

               If "nodes" is chosen, only subgraphs with
               at least 10 nodes are evaluated.

               If "load" is chosen, the split is only evaluated if none of the split
               components contains 90 % of the total load or generation. In this case,
               every split component is evaluated independent of the component size
               
               If "all" is chosen, all subgraphs are evaluated.

    RETURNS:

    results_dict: dictionary of results with keywords 'inertia_proxy' and
                  'load_imbalance'.
                  NOTE: TO GET THE ROCOF FROM LOAD IMBALANCE AND INERTIA PROXY
                  YOU NEED TO MULTIPLY INERTIA PROXY BY THE INERTIA CONSTANT
                  (SEE BELOW) AND TAKE INTO ACCOUNT THE SYSTEM FREQUENCY

                  Each keyword contains a list with length corresponding to the number
                  of split components as evaluated based on the criterion given.
                  To reproduce these split components call
                  "get_split_components(split,Graph, criterion = criterion)" with
                  the split and the networkx Graph representing the subnetwork under
                  consideration.

                  'inertia_proxy' is the inertia estimate based on the sum of nominal powers
                  of all plants and storages that contribute with at least 5 % of their
                  nominal power at the given point in time and are listed in the inertia
                  plants list.
                  NOTE: THIS ESTIMATE DOES NOT INCORPORATE THE INERTIA CONSTANT.
                  TO GET THE ACTUAL INERTIA, MULTIPLY EITHER BY A GLOBAL CONSTANT
                  (e.g. H = 6s^{-1}) OR DEFINE INDDIVIDUAL INERTIA CONSTANTS IN
                  THE FUNCTION "get_inertia_gen_subgraph"

                  'load_imbalance' is the load imbalance for each subgraph in units
                  of MW, i.e. its generation surplus or the missing generation.
                  It assumes that HVDC transport stays the same as before the split,
                  since the split is assumed to occur instantenously and thus leaves
                  HVDC transport unchanged.

    """

    assert isinstance(timestamp,pd.Timestamp)

    Graph              = build_networkx_graph(pypsa_network,snet_index = snet_index)


    # Rescale load shedding since units for load shedding are different
    # than for generation, storage and load
    # (Note: In this project load shedding is not implement)
    load_shedding_indices = pypsa_network.generators[pypsa_network.generators.carrier.isin(['load'])].index

    current_generation = pypsa_network.generators_t.p.loc[timestamp].copy()
    current_generation[load_shedding_indices] /= 1e3
    current_storage    = pypsa_network.storage_units_t.p.loc[timestamp]
    current_load       = pypsa_network.loads_t.p.loc[timestamp]
    HVDC_transport     = pypsa_network.links_t.p0.loc[timestamp].groupby(pypsa_network.links["bus0"]).sum()
    HVDC_transport     = HVDC_transport.add(pypsa_network.links_t.p1.loc[timestamp].groupby(pypsa_network.links["bus1"]).sum(),
                                            fill_value = 0)

    assert np.abs(current_generation.sum()+current_storage.sum()-current_load.sum())<5e-2

    results_dict = {'inertia_proxy': [],'load_imbalance': [], 'load':[]} #, 'available_flexible_generation': [] }

    evaluate_results = False

    subgraphs = get_split_components(split,Graph, criterion = criterion)

    if (criterion == 'nodes') or (criterion =='all'):
        if len(subgraphs) >= 2:
            evaluate_results = True
    elif criterion == 'load':
        evaluate_results = check_load_criterion(subgraphs,
                                                pypsa_network.generators,
                                                current_generation,
                                                pypsa_network.storage_units,
                                                current_storage,
                                                pypsa_network.loads,
                                                current_load,
                                                HVDC_transport)

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
                                                          current_load,
                                                          HVDC_transport)

            subgraph_load = get_load_subgraph(subgraph,
                                            pypsa_network.generators,
                                            current_generation,
                                            pypsa_network.storage_units,
                                            current_storage,
                                            pypsa_network.loads,
                                            current_load,
                                            HVDC_transport)

            #available_flexible_generation = get_available_flexible_generation(subgraph,
            #                                              pypsa_network.generators,
            #                                              current_generation,
            #                                              pypsa_network.storage_units,
            #                                              current_storage,
            #                                              flexible_plants = flexible_plants,
            #                                              flexible_storages = flexible_storage)

            results_dict['inertia_proxy'].append(inertia_generation)
            results_dict['load_imbalance'].append(load_imbalance)
            results_dict['load'].append(subgraph_load) 
            #results_dict['available_flexible_generation'].append(available_flexible_generation)

    return results_dict


def transform_cascade_results(Graph,cascade):
    """Cascade model used to save indices of edges
    but the new format should be the edges itself"""
    if isinstance(Graph, nx.MultiGraph):
        Graph_edges = list(Graph.edges(keys = True))
    elif isinstance(Graph, nx.Graph):
        Graph_edges = list(Graph.edges())
    return [Graph_edges[index] for index in cascade]

def add_syn_inertia(indicator_vectors, node_count, m):
    """ Add inertia to split components that contain the node "node_count".

    Args:
        indicator_vectors (ndarray): Indicator vectors of split components
        node_count (int): Index of node to place inertia on
        m (float): Amount of inertia generation to place (in MW)

    Returns:
        ndarray: Inertia per split component
    """    
    # identify splits where this node is involved
    indices = np.where(indicator_vectors[:, node_count])[0]

    # assign additional_synthetic_inertia to all split components
    synthetic_inertia = np.zeros(indicator_vectors.shape[0])
    synthetic_inertia[indices] += m
    
    return  synthetic_inertia


def inertia_placement(comp_props, indicator_vectors, var_ref, m, q, number_of_simulations, node_list, epsilon):
    """Optimal placement of synthetic inertia through greedy search. 

    Args:
        comp_props (pandas.DataFrame): Split component properties for one Co2 level, with 
        'time_stamp', 'number_of_split', 'inertia_proxy' and 'load_imbalance' as columns. 
        indicator_vectors (ndarray): Indicator vectors of split components
        var_ref (float): Reference Value at risk that should be reached through inertia placement
        m (float): Amount of incremental inertia to place in each step
        q (float): Quantile for VaR calculation
        number_of_simulations (int): Number of cascade simulations for one CO2 level
        node_list (list): List of nodes in graph
        epsilon (float): Relative deviation from reference value that is allowed after optimization.

    Returns:
        tuple: component properties with synthetic inertia and information on greedy search steps.
    """    
    
    # Initialize values for inertia placement
    step=1
    comp_props_new = comp_props.copy()
    var_targ = vis.val_at_risk(comp_props_new, number_of_simulations, q, method='abs')
    inertia_placement_info = pd.DataFrame(columns=['new_var', 'added_node_index'], dtype=np.float)
    inertia_placement_info.loc[0, 'added_node_index'] = np.nan
    inertia_placement_info.loc[0, 'new_var'] = var_targ
    
    
    # Continue placing inertia until the reference value is reached
    while var_targ > var_ref*(1+epsilon) :

        var_proposals = np.zeros(len(node_list))
        comp_props_proposal = comp_props_new.copy()
        
        
        # Try all potential nodes 
        for count, node in enumerate(node_list):


            # add new synthetic inertia
            new_inertia = add_syn_inertia(indicator_vectors, count, m)
            comp_props_proposal.loc[:,'inertia_proxy'] = comp_props_new.loc[:,'inertia_proxy'] + new_inertia
            comp_props_proposal.loc[:, 'rocof'] = vis.calc_rocof(comp_props_proposal.load_imbalance, 
                                                                 comp_props_proposal.inertia_proxy)
            
            var_proposals[count] = vis.val_at_risk(comp_props_proposal, number_of_simulations, q,
                                                   method='abs')

        
        # Select proposal and place inertia at best node
        count_opt = np.argmin(var_proposals)
        comp_props_new.loc[:,'inertia_proxy'] += add_syn_inertia(indicator_vectors,
                                                                 count_opt, m)
        var_targ = var_proposals[count_opt]
        
        print(step, ' New var:', var_targ, 'Target Var:', var_ref)
        
        # Save placement information    
        inertia_placement_info.loc[step, 'added_node_index'] = count_opt
        inertia_placement_info.loc[step, 'new_var'] = var_targ

        
        # Go to next placement step
        step+=1
    
    return comp_props_new, inertia_placement_info



# def apply_line_failure(graph, edge):

#     num_parallel = graph.edges[edge]['num_parallel']
#     s_nom = graph.edges[edge]['s_nom']
#     weight = graph.edges[edge]['weight']  

#     new_num_parallel =  calc_num_parallel_after_failure(num_parallel) 
    
#     if new_num_parallel==0:
#         graph.remove_edge(*edge)
#     else:
#         graph.edges[edge]['num_parallel'] = new_num_parallel
#         graph.edges[edge]['weight'] = weight * new_num_parallel / num_parallel
#         graph.edges[edge]['s_nom'] =  s_nom * new_num_parallel / num_parallel







