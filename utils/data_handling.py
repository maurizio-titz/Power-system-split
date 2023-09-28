""" 
Preparation and conversion of data for system split simulation and evaluation  
"""

import pypsa
import networkx as nx
import numpy as np
from scipy import sparse
import os

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

    for ii, edge in enumerate(edgelist):
        (uu, vv) = orientations[edge]
        n1 = node_index[uu]
        B[n1, ii] = 1.
        n2 = node_index[vv]
        B[n2, ii] = -1.
    if return_np_array:
        return_val = B.toarray()
    else:
        return_val = B.asformat("csr")
        
    return return_val


def get_effective_injections(network, snapshot, nx_graph):
    """Get effective nodal injections on nx_graph for certain snapshot from PyPSA network. 

    Args:
        network (pypsa.network): PyPSA network
        snapshot (string): Format '%Y-%m-%d %H:00'
        nx_graph (networkx graph): Graph pf power system under investigation.

    Returns:
        numpy array: n_nodes x 1 array with nodal injections
    """

    I_m = construct_incidencematrix_from_orientation(nx_graph)

    flows_network = network.lines_t.p0.loc[snapshot]        
    flows_matrix = np.array([flows_network[attribs['line_index']] for u,v, attribs in nx_graph.edges(data=True)])

    P0 = np.dot(I_m,flows_matrix)

    return P0


def load_pypsa_network(co2l, n_nodes, path_to_pypsa_network):
    """
    Load PyPSA network from path with certain Co2 constraint and aggregation level of n_nodes.
    """

    # Select a particular subnetwork for calculations (if the pypsa network has different ones).
    # For our data set, "0" indicates the Continental European AC grid. -> snet doc
    
    # Load PyPSA network
    file_name = 'sclopf-elec_s_{0}_ec_lv1.0_Co2L{1:.1f}-2920SEG.nc'.format(n_nodes, co2l)
    assert os.path.isfile(path_to_pypsa_network + file_name) == True, f'File "{path_to_pypsa_network + file_name}" does not exist'
    network = pypsa.Network(path_to_pypsa_network + file_name)

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

    return network


def get_subgraphs_from_edges(edge_indices, nx_graph):
    """Generate subgraphs that result from removing the edges. 

    Args:
        edge_indices (iterable): indices of edges to remove (indices from matrix format)
        nx_graph (networkx graph): Graph from which subgraphs are generated

    Returns:
        list: List of networkx graphs
    """    

    nx_edges = matrix_indices_to_nx_edges(edge_indices, nx_graph)
    
    F = nx_graph.copy()
    F.remove_edges_from(nx_edges)
    subgraphs =  list((F.subgraph(c).copy() for c in nx.connected_components(F)))

    return subgraphs


def nx_edges_to_matrix_indices(nx_edges, nx_graph):
    """
    Transform edge names from networkx graph format to matrix format.
    """
    lookup_dict = dict(zip(nx_graph.edges(), range(nx_graph.number_of_edges())))
    matrix_indices = [lookup_dict[link_name] for link_name in nx_edges]

    return matrix_indices


def matrix_indices_to_nx_edges(indices, nx_graph):
    """
    Transform edge indices from matrix format to edge names in networkx graph.
    """
    
    lookup_dict = dict(zip(range(nx_graph.number_of_edges()), nx_graph.edges()))
    edge_names = [lookup_dict[ind] for ind in indices]

    return edge_names


def get_matrices_from_nx_graph(nx_graph):
    """
    Extract incidence matrix, susceptance matrix, effective number of parallel lines per edge 
    and line limits per edge from networkx graph. 
    """
    
    # Build incidence matrix and susceptance matrix
    I_m = construct_incidencematrix_from_orientation(nx_graph,return_np_array = False) 
    B_d = sparse.spdiags(np.array([attribs['weight'] for u,v, attribs in nx_graph.edges(data=True)]),
                         0,nx_graph.number_of_edges(),nx_graph.number_of_edges()).asformat('csr')

    # Get array of num_parallels and of line limits
    num_parallels = np.array([attribs['num_parallel'] for u,v, attribs in nx_graph.edges(data=True)])
    line_limits = np.array([attribs['s_nom'] for u,v, attribs in nx_graph.edges(data=True)])


    return I_m, B_d, num_parallels, line_limits

