#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

def construct_incidencematrix_from_orientation(G):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order"""
    B = np.zeros((len(G.nodes()),len(G.edges())))
    orientations = nx.get_edge_attributes(G,'orientation')
    for i in range(len(G.edges())):
        if type(G) == type(nx.Graph()):
            edge = list(G.edges())[i]
            orientation = orientations[edge]#G[edge[0]][edge[1]]['orientation']
        elif type(G) == type(nx.MultiGraph()):
            edge = list(G.edges(keys = True))[i]
            orientation = orientations[edge]#G[edge[0]][edge[1]]['orientation']
        n1 = list(G.nodes()).index(orientation[0])
        B[n1,i] = 1.
        n2 = list(G.nodes()).index(orientation[1])
        B[n2,i] = -1.
    return B

def redefined_index(G,element):
    """Get index of element in edge list for graph G"""
    if type(G) == type(nx.Graph()):
        edge_list = list(G.edges())
        try:
            index = edge_list.index(element)
        except ValueError:
            index = edge_list.index(element[::-1])
    elif type(G) == type(nx.MultiGraph()):
        edge_list = list(G.edges(keys = True))
        try:
            index = edge_list.index(element)
        except ValueError:
            index = edge_list.index(element[1::-1] + (element[2],))
    return index

def simulate_cascade_PTDF_based(G,trigger_links,initial_flows,line_limits):
    """Simulate a cascade of failures using the network topology G, which is assumed to have a property 'orientation'
    for each edge, the initially failing links, the initial flows (as dictionary) and the line limits (as dictionary).
    The approach used to simulate the cascade is based on the Power Transfer Distribution Factors assuming fixed power
    injections and subsequently calculating the flows using the PTDFs after the removal of all failing links from
    the network"""

    # store indices of failing links
    failure_cascade = []

    multi_graph = False
    if type(G) == type(nx.MultiGraph()):
        multi_graph = True

    if not multi_graph:
        flows_G = np.array([initial_flows[(u,v)] for u,v in G.edges()])
        smax_G = np.array([line_limits[(u,v)] for u,v in G.edges()])
    else:
        flows_G = np.array([initial_flows[(u,v,key)] for u,v,key in G.edges(keys = True)])
        smax_G = np.array([line_limits[(u,v,key)] for u,v,key in G.edges(keys = True)])

    failure_cascade = [redefined_index(G,trigger_link) for trigger_link in trigger_links]

    I = construct_incidencematrix_from_orientation(G)

    if len(np.where(np.abs(flows_G)>smax_G)[0]):
        print("Setup has initial overloads!")
    stop = 0
    system_split = False
    while not stop:
        H = G.copy()
        I0 = construct_incidencematrix_from_orientation(H)
        if not multi_graph:
            flows0 = np.array([initial_flows[(u,v)] for u,v in H.edges()])
        else:
            flows0 = np.array([initial_flows[(u,v,key)] for u,v,key in H.edges(keys = True)])

        P0 = np.dot(I0,flows0)

        if not multi_graph:
            H.remove_edges_from([list(G.edges())[i] for i in failure_cascade])
        else:
            H.remove_edges_from([list(G.edges(keys = True))[i] for i in failure_cascade])


        if not nx.is_connected(H):
            system_split = True
            break

        I = construct_incidencematrix_from_orientation(H)
        L = nx.laplacian_matrix(H)
        try:
            theta = np.linalg.solve(L.A,P0)
        except np.linalg.LinAlgError:
            # pseudoinverse
            L_inv = np.linalg.pinv(L.A)
            theta = np.dot(L_inv,P0)

        line_weights = nx.get_edge_attributes(H,'weight')
        if not multi_graph:
            line_susceptances = np.array([line_weights[(u,v)] for u,v in H.edges()])
        else:
            line_susceptances = np.array([line_weights[(u,v,key)] for u,v,key in H.edges(keys = True)])

        B_d = np.diag(line_susceptances)

        flows_H = np.linalg.multi_dot([B_d,I.T,theta])

        if not multi_graph:
            smax = np.array([line_limits[(u,v)] for u,v in H.edges()])
        else:
            smax = np.array([line_limits[(u,v,key)] for u,v,key in H.edges(keys = True)])

        next_indices = np.where(np.abs(flows_H)>smax)[0]

        ### map next failing indices to original graph
        for n in next_indices:
            if not multi_graph:
                index = redefined_index(G,element = list(H.edges())[n])
            else:
                index = redefined_index(G,element = list(H.edges(keys = True))[n])

            if not index in failure_cascade:
                failure_cascade.append(index)

        if len(next_indices)==0:
            stop = 1

        loading_dict = {}
        if not multi_graph:
            for i in range(len(H.edges())):
                loading_dict[list(H.edges())[i]] = flows_H[i]
                loading_dict[list(H.edges())[i][::-1]] = flows_H[i]
        else:
            for i in range(len(H.edges())):
                loading_dict[list(H.edges(keys = True))[i]] = flows_H[i]

    return failure_cascade,loading_dict,system_split


def build_networkx_graph(snet_branches,multi_graph = False):
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


def get_split_components(split,G):
    """Return the split resulting from the edge list in split if the resulting subgraphs are larger than 10 nodes"""
    F = G.copy()
    cascade_edges = [list(F.edges())[index] for index in split]
    F.remove_edges_from(cascade_edges)
    subgraphs =  list((F.subgraph(c).copy() for c in nx.connected_components(F)))
    relevant_subgraphs = [i for i in range(len(subgraphs)) if len(subgraphs[i].nodes())>10]
    if len(relevant_subgraphs)<2:
        return_val = []
    else:
        return_val = [subgraphs[relevant_subgraphs[i]] for i in range(len(relevant_subgraphs))]
    return return_val

def evaluate_system_split_inertia(split,G,generators,current_generation, use_pnom = True):
    """For a given list of edge indices split, a networkx graph G,

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
    participation_threshold = 1e-6
    subgraphs = get_split_components(split,G)
    if len(subgraphs) < 2 :
        return 0
    else:
        inertia_generations = []
        for subgraph in subgraphs:
            gens = network.generators[network.generators["bus"].isin(list(subgraph.nodes()))]
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


def likelihood_systemsplit_edge_based(split_dict,G,only_large_splits = True):
    """ Calculate the empirical likelihood that a given edge is involved
    in a system split based on the split_dict

    If only_large_splits is True, only splits which yield components
    with more than ten nodes each are evaluated"""

    likelihood_dict = {(u,v):0.0 for u,v in G.edges()}

    if only_large_splits:
        # counts the number of relevant splits
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                F = G.copy()
                cascade_edges = [list(F.edges())[index] for index in cascade]
                F.remove_edges_from(cascade_edges)
                subgraphs =  list((F.subgraph(c).copy() for c in nx.connected_components(F)))
                relevant_subgraphs = [i for i in range(len(subgraphs)) if len(subgraphs[i].nodes())>10]

                if len(relevant_subgraphs) < 2:
                    continue
                for edge in cascade_edges:
                    likelihood_dict[edge] += 1
                split_counter += 1
        for edge in likelihood_dict.keys():
            likelihood_dict[edge] /= split_counter

    else:
        likelihoods = np.zeros(len(G.edges()))
        split_counter = 0
        for timestamp in split_dict.keys():
            for cascade in split_dict[timestamp]:
                likelihoods[cascade] += 1
                split_counter += 1
        likelihoods /= split_counter
        for count, edge  in enumerate(list(G.edges())):
            likelihood_dict[edge] += likelihoods[count]

    return likelihood_dict
