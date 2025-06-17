#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preparation and conversion of data for system split simulation and evaluation
"""

import gzip
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pypsa
from scipy import sparse

from utils.config import (
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_lopf,
    path_to_vis_results_sclopf,
    path_to_cascade_results_sclopf,
    path_to_cascade_results_lopf,
)


def build_networkx_graph(pypsa_network, snet_index=None):
    """Build a networkx graph from the pypsa networks"""
    pypsa_network.determine_network_topology()

    try:
        snet = pypsa_network.sub_networks["obj"][snet_index]
    except KeyError:
        snet = pypsa_network

    branches = snet.branches()
    positions = pypsa_network.buses[["x", "y"]]
    pos = dict(zip(positions.index, list(zip(positions.x, positions.y))))

    branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom", "num_parallel"]]

    F = nx.Graph()

    for line_index, line in branches.iterrows():

        if not F.has_edge(line["bus0"], line["bus1"]):
            F.add_edge(
                line["bus0"],
                line["bus1"],
                weight=1 / line["x_pu_eff"],
                orientation=(line["bus0"], line["bus1"]),
                line_index=[line_index[1]],
                s_nom=line["s_nom"],
                num_parallel=line["num_parallel"],
            )
            assert line["bus0"] < line["bus1"], f"Line {line_index} is not ordered"
        else:
            raise (RuntimeError("There duplicated edges in the PyPSA network"))

    nx.set_node_attributes(F, pos, "pos")
    return F


def check_order_networkx_line_order_equals_pypsa(
    nx_graph: nx.Graph, pypsa_network: pypsa.Network, snet=None
):
    """
    Check if the edges in the networkx graph are sorted in ascending order
    """
    if snet:
        lines_subnet = pypsa_network.lines[
            pypsa_network.lines.sub_network.astype(int) == snet
        ]
    else:
        lines_subnet = pypsa_network.lines

    # return np.array(list(nx.get_edge_attributes(nx_graph, "orientation").values())) == np.array(list(list(zip(lines_subnet.bus0.astype(str), lines_subnet.bus1.astype(str)))))

    for i, (u, v) in enumerate(
        nx.get_edge_attributes(nx_graph, "orientation").values()
    ):
        if (u, v) != (lines_subnet.bus0[i], lines_subnet.bus1[i]):
            raise (
                RuntimeError(
                    f"Edge {i} in networkx ({u}, {v}) does not match pypsa ({lines_subnet.bus0[i]}, {lines_subnet.bus1[i]})"
                )
            )
    return True

    # zip(nx.get_edge_attributes(nx_graph, "orientation").values(), list(zip(pypsa_network.lines.bus0, pypsa_network.lines.bus1)))


def check_if_edges_sorted(nx_graph: nx.Graph):
    """
    Check if the edges in the networkx graph are sorted in ascending order
    """

    for i, (u, v) in enumerate(
        nx.get_edge_attributes(nx_graph, "orientation").values()
    ):
        if u > v:
            raise (RuntimeError(f"Edge {i} ({u}, {v}) is not sorted"))
    return True


def construct_incidencematrix_from_orientation(Graph, return_np_array=True):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order
    NOTE: This function was tweaked based on networkx incidence matrix function
    https://networkx.org/documentation/stable/_modules/networkx/linalg/graphmatrix.html#incidence_matrix
    """

    edgelist = list(Graph.edges())
    nodelist = list(Graph.nodes())

    node_index = {node: i for i, node in enumerate(nodelist)}
    B = sparse.lil_matrix((len(nodelist), len(edgelist)))
    orientations = nx.get_edge_attributes(Graph, "orientation")

    for ii, edge in enumerate(edgelist):
        (uu, vv) = orientations[edge]
        n1 = node_index[uu]
        B[n1, ii] = 1.0
        n2 = node_index[vv]
        B[n2, ii] = -1.0
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
    flows_matrix = np.array(
        [
            flows_network[attribs["line_index"]]
            for u, v, attribs in nx_graph.edges(data=True)
        ]
    )

    P0 = np.dot(I_m, flows_matrix)

    return P0


def load_pypsa_network(co2lvl, n_nodes, use_sclopf: bool = True):
    if not use_sclopf:
        return load_pypsa_network_from_path(
            path_to_pypsa_network_lopf
            + f"elec_s_{n_nodes}_ec_lv1.0_Co2L{co2lvl}-3H.nc",
            use_sclopf,
        )

    else:
        return load_pypsa_network_from_path(
            path_to_pypsa_network_sclopf
            + f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2lvl}-2920SEG.nc",
            use_sclopf,
        )


def load_pypsa_network_from_path(path_to_pypsa_network: str, use_sclopf: bool):
    """Load PyPSA network from path with certain Co2 constraint and aggregation level of n_nodes.

    Args:
        path_to_pypsa_network (str): path to PyPSA network as ".nc" file.
        use_sclopf (bool): If 'True' the network is used was evaluated using security constrained lopf and
            the outage lines are removed beforehand.

    Returns:
        network (PyPSA network): _description_
    """

    # Select a particular subnetwork for calculations (if the pypsa network has different ones).
    # For our data set, "0" indicates the Continental European AC grid. -> snet doc

    # Load PyPSA network
    print(path_to_pypsa_network)

    assert (
        os.path.isfile(path_to_pypsa_network) == True
    ), f'File "{path_to_pypsa_network}" does not exist'
    network = pypsa.Network(path_to_pypsa_network)

    # The following line is needed to remove the outage lines used for SCLOPF. For SCLOPF,
    # the Lines are split into 2, one that fails during N-1 stability test, except from
    # those lines that only have one circuit. We have to add these line up again to obtain a simple graph
    if use_sclopf:
        duplicated_lines = network.lines[network.lines.index.str[-6:] != "outage"].index
        network.lines.loc[
            duplicated_lines + "_outage", "num_parallel"
        ] += network.lines.loc[duplicated_lines, "num_parallel"].values
        network.lines.loc[duplicated_lines + "_outage", "s_nom"] += network.lines.loc[
            duplicated_lines, "s_nom"
        ].values
        network.lines_t.p0.loc[
            :, duplicated_lines + "_outage"
        ] += network.lines_t.p0.loc[:, duplicated_lines].values
        network.lines_t.p1.loc[
            :, duplicated_lines + "_outage"
        ] += network.lines_t.p1.loc[:, duplicated_lines].values

        network.mremove("Line", duplicated_lines)

    network.determine_network_topology()
    network.calculate_dependent_values()

    return network


def load_cascades(co2lvl, n_nodes, use_sclopf: bool = True):
    if use_sclopf:
        path_to_cascade_results = path_to_cascade_results_sclopf
    else:
        path_to_cascade_results = path_to_cascade_results_lopf

    full_path_to_cascades = (
        path_to_cascade_results + f"system_splits_Co2L{co2lvl}_n{n_nodes}.pklz"
    )
    with gzip.open(full_path_to_cascades, "rb") as fh_casc:
        splitting_cascades = pickle.load(fh_casc)

    return splitting_cascades


def get_subgraphs_from_edges(edge_indices, nx_graph):
    """Generate subgraphs that result from removing the edges.

    Args:
        edge_indices (iterable): indices of edges to remove (indices from matrix format)
        nx_graph (networkx graph): Graph from which subgraphs are generated

    Returns:
        list: List of networkx graphs
    """

    check_if_edges_sorted(nx_graph)

    nx_edges = matrix_indices_to_nx_edges(edge_indices, nx_graph)

    F = nx_graph.copy()
    F.remove_edges_from(nx_edges)
    subgraphs = list((F.subgraph(c).copy() for c in nx.connected_components(F)))

    for subgraph in subgraphs:
        check_if_edges_sorted(subgraph)

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
    I_m = construct_incidencematrix_from_orientation(nx_graph, return_np_array=False)
    B_d = sparse.spdiags(
        np.array([attribs["weight"] for u, v, attribs in nx_graph.edges(data=True)]),
        0,
        nx_graph.number_of_edges(),
        nx_graph.number_of_edges(),
    ).asformat("csr")

    # Get array of num_parallels and of line limits
    num_parallels = np.array(
        [attribs["num_parallel"] for u, v, attribs in nx_graph.edges(data=True)]
    )
    line_limits = np.array(
        [attribs["s_nom"] for u, v, attribs in nx_graph.edges(data=True)]
    )

    return I_m, B_d, num_parallels, line_limits


def get_adjacency_matrix_from_nx_graph(nx_graph):
    """returns the adjacency matrix of the nx graph"""
    I_m, B_d, num_parallels, line_limits = get_matrices_from_nx_graph(nx_graph)
    I_m = np.absolute(I_m)
    adjacency_matrix = I_m @ I_m.T
    # Remove self-loops for sparse matrices
    if hasattr(adjacency_matrix, "setdiag"):
        adjacency_matrix.setdiag(0)
    else:
        np.fill_diagonal(adjacency_matrix, 0)  # remove self-loops
    return adjacency_matrix


def load_split_props(n_nodes: int, co2l=None, use_sclopf: bool = True):
    """Load component properties for a certain Co2 constraint and aggregation level of n_nodes. If co2l is None, return all lvls.

    Args:
        co2l (float): Co2 constraint
        n_nodes (int): Aggregation level of nodes
        use_sclopf (bool): If 'True' the network is used was evaluated using security constrained lopf and
            the outage lines are removed beforehand.

    Returns:
        split_properties (pandas.DataFrame): dataframe containing the properties of each split, e.g. loss, number of components etc.
    """

    if use_sclopf:
        load_dir = path_to_vis_results_sclopf
    else:
        load_dir = path_to_vis_results_lopf

    # load hdf pandas
    split_properties = pd.read_hdf(load_dir + f"split_properties_all_n{n_nodes}.h5")

    if co2l is not None:
        return split_properties[split_properties.index.get_level_values("co2l") == co2l]

    return split_properties
