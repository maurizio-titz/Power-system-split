""" 
Preparation and conversion of data for system split simulation and evaluation  
"""

import pypsa
import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm

def build_networkx_graph(pypsa_network, snet_index = None):
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


    F = nx.Graph()

    for line_index,line in branches.iterrows():

        if not F.has_edge(line['bus0'],line['bus1']):
            F.add_edge(line['bus0'],line['bus1'],
                       weight = 1/line['x_pu_eff'],
                       orientation = (line['bus0'],line['bus1']),
                       line_index = [line_index[1]],
                       s_nom = line['s_nom'],
                       num_parallel= line['num_parallel'])
        else:
            raise(RuntimeError('There duplicated edges in the PyPSA network'))

    nx.set_node_attributes(F,pos,'pos')
    return F

def construct_incidencematrix_from_orientation(Graph,return_np_array = True):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order
    NOTE: This function was tweaked based on networkx incidence matrix function
    https://networkx.org/documentation/stable/_modules/networkx/linalg/graphmatrix.html#incidence_matrix"""


    edgelist = list(Graph.edges())
    nodelist = list(Graph.nodes())

    node_index = {node: i for i, node in enumerate(nodelist)}
    B = sparse.lil_matrix((len(nodelist),len(edgelist)))
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

def get_effective_injections(Graph,initial_flows):
    """get effective injections from the initial flows"""

    flows_G = np.array([initial_flows[(u,v)] for u,v in Graph.edges()])

    I = construct_incidencematrix_from_orientation(Graph)
    P0 = np.dot(I,flows_G)
    return P0


def load_pypsa_network(path_to_pypsa_network='./data/sclopf_test_data/'):


    # Load PyPSA network
    network = pypsa.Network()
    network.import_from_netcdf(path_to_pypsa_network+ 'sclopf-elec_s_200_ec_lv1.0_Co2L0.0-2920SEG-0.nc') #f'sclopf-elec_s_800_ec_lv1.0_Co2L{co2l}-3H.nc')

    # The following line is needed to remove the outage lines used for SCLOPF. For SCLOPF,
    # the Lines are split into 2, one that fails during N-1 stability test, except from
    # those lines that only have one circuit. We have to add these line up again to obtain a simple graph 
    duplicated_lines = network.lines[network.lines.index.str[-6:]!='outage'].index
    network.lines.loc[duplicated_lines + '_outage', 'num_parallel'] +=  network.lines.loc[duplicated_lines, 'num_parallel'].values
    network.lines.loc[duplicated_lines + '_outage', 's_nom'] +=  network.lines.loc[duplicated_lines, 's_nom'].values
    network.lines_t.p0.loc[:, duplicated_lines + '_outage'] += network.lines_t.p0.loc[:, duplicated_lines].values
    network.lines_t.p1.loc[:, duplicated_lines + '_outage'] += network.lines_t.p1.loc[:, duplicated_lines].values

    network.mremove('Line', duplicated_lines)
    network.determine_network_topology()
    network.calculate_dependent_values()


    ######
    #TODO: remove this line if the networks contain timestamp snapshots
    network.snapshots = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))

    network.snapshot_weightings.index = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))

    for component in network.all_components:
        pnl = network.pnl(component)
        attrs = network.components[component]["attrs"]

        for k,default in attrs.default[attrs.varying].iteritems():
            pnl[k].index = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))
    ######

    # Select a particular subnetwork for calculations (if the pypsa network has different ones).
    # For our data set, "0" indicates the Continental European AC grid. 
    snet_index = 0

    # Build graph for subnetwork
    G = build_networkx_graph(network, snet_index= snet_index)


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

def transform_cascade_results(Graph,cascade):
    """Cascade model used to save indices of edges
    but the new format should be the edges itself"""

    Graph_edges = list(Graph.edges())
    return [Graph_edges[index] for index in cascade]

def edge_list_to_subgraph(split,Graph,criterion):
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


def split_evaluation_to_indicator_vectors(solution_dict, splitting_cascades, nx_graph, criterion):
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
            components = edge_list_to_subgraph(split, nx_graph, criterion = criterion)

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