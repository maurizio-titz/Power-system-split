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
import re

from utils.cascade_simulation import (
    remove_highest_volt_lvl_circuit,
    line_type_num_parallels,
)
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_pypsa_network_lopf,
    path_to_vis_results_sclopf,
    path_to_cascade_results_sclopf,
    path_to_cascade_results_lopf,
    path_to_grid_data,
)

if not os.path.exists(path_to_grid_data):
    os.mkdir(path_to_grid_data)

def update_line_params(n: pypsa.Network):
    """Return updated line parameters in pypsa network if lines extension used in optimization.
    Args:
        n (pypsa.Network or pypsa.SubNetwork): PyPSA network or subnetwork
    """

    if isinstance(n, pypsa.SubNetwork):
        n = n.network

    base_s_nom = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
    )
    # s_nom_prev = n.lines.num_parallel * base_s_nom
    factor = n.lines.s_nom_opt / n.lines.s_nom
    print("mean extension level= ", factor.mean())

    n.lines.loc[:, "x_pu_eff"] /= factor
    n.lines.loc[:, "num_parallel"] = n.lines.s_nom_opt / base_s_nom

    # return n.lines


def get_networkx_graph_path(snet_index: str | None = '0', co2lvl=None):
    """Get the path to a networkx graph from the pypsa networks"""
    graph_path = path_to_grid_data + f"/nx_graph"
    if co2lvl is not None:
        graph_path = f"{graph_path}_Co2{co2lvl}"
    if snet_index is not None:
        graph_path = f"{graph_path}_snet{snet_index}"
    graph_path = f"{graph_path}.gml"

    return graph_path


def load_networkx_graph(snet_index: str | None=None, co2lvl: None = None):
    """Get a networkx graph from the pypsa networks"""
    graph_path = get_networkx_graph_path(snet_index, co2lvl)
    graph = nx.read_gml(graph_path)  # [, stringizer])
    return graph


def save_networkx_graph(graph, snet_index: str | None =None, co2lvl=None, overwrite=False):
    """Save a networkx graph from the pypsa networks"""
    graph_path = get_networkx_graph_path(snet_index, co2lvl)
    if not overwrite and os.path.exists(graph_path):
        raise FileExistsError(f"File {graph_path} already exists. Set overwrite=True.")

    if not os.path.exists(os.path.dirname(graph_path)):
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    nx.write_gml(graph, graph_path)  # [, stringizer])


def build_networkx_graph(
    pypsa_network,
    snet_index: str  | None = None,
    assert_order: bool = True,
    inplace: bool = False,
    update_lines: bool = False,
):
    """Build a networkx graph from the pypsa networks"""
    if not inplace:
        pypsa_network = pypsa_network.copy()
    pypsa_network.determine_network_topology()

    if snet_index is not None:
        snet = pypsa_network.sub_networks["obj"][snet_index]
    else:
        snet = pypsa_network

    if update_lines:
        update_line_params(snet)
        print("Updating line parameters based on optimized values.")

    branches = pypsa_network.lines.loc[snet.branches().index.get_level_values(1)]
    branches.index = pd.MultiIndex.from_tuples(
        [("Line", idx) for idx in branches.index]
    )

    positions = pypsa_network.buses[["x", "y"]]
    pos = dict(zip(positions.index, list(zip(positions.x, positions.y))))

    branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom_opt", "num_parallel"]]

    G = nx.Graph()

    for line_index, line in branches.iterrows():
        # circuit_counts = get_circuit_counts(line["num_parallel"], use_sclopf=True)

        if not G.has_edge(line["bus0"], line["bus1"]):
            G.add_edge(
                line["bus0"],
                line["bus1"],
                weight=1 / line["x_pu_eff"],
                orientation=(line["bus0"], line["bus1"]),
                line_index=[line_index[1]],
                s_nom=line["s_nom_opt"],
                num_parallel=line["num_parallel"],
                # circuit_counts=circuit_counts,
            )
            if assert_order:
                assert line["bus0"] < line["bus1"], f"Line {line_index} is not ordered"
        else:
            raise (RuntimeError("There duplicated edges in the PyPSA network"))

    nx.set_node_attributes(G, pos, "pos")

    # if save:
    #     graph_path = path_to_grid_data + f"nx_graph_snet{snet_index}.pklz"
    #     nx.write_gml(G, graph_path)  # [, stringizer])

    return G


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


def get_circuit_counts(num_parallel, use_sclopf: bool = True):
    """Get the number of each sub line type in a given line based on num_parallel values in PyPSA network.

    Args:
        num_parallel (int): Effective number of parallel lines for a given line.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network.
    Returns:
        dict: Dictionary of number of each sub line type in a given line.
    """
    line_types = np.zeros_like(line_type_num_parallels)
    while num_parallel > 1e-6:
        num_parallel_new = remove_highest_volt_lvl_circuit(
            num_parallel, use_sclopf=use_sclopf
        )
        num_par_removed = num_parallel - num_parallel_new
        line_type_idx = np.argwhere(
            np.isclose(line_type_num_parallels, num_par_removed)
        )[0, 0]
        line_types[line_type_idx] += 1
        num_parallel = num_parallel_new
    return dict(zip(line_type_num_parallels, line_types))


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


def save_effective_injections(co2lvl, effective_injections, overwrite=False):
    """Save effective injections to disk.

    Args:
        co2lvl (float): CO2 level used in filename.
        effective_injections (dict): Dictionary of effective injections per snapshot.
    Returns:
        str: Path to the written file.
    """

    file_path = path_to_grid_data + f"/effective_injections_co2lvl{co2lvl}.pklz"

    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists. Set overwrite=True.")

    with gzip.open(file_path, "wb") as fh:
        pickle.dump(effective_injections, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return file_path


def load_effective_injections(co2lvl):
    """Load effective injections from disk for certain snet and co2lvl.

    Args:
        co2lvl (float): CO2 constraint
    Returns:
        effective_injections (dict): Dictionary of effective injections per snapshot.
    """
    with gzip.open(
        path_to_grid_data + f"/effective_injections_co2lvl{co2lvl}.pklz",
        "rb",
    ) as f:
        effective_injections = pickle.load(f)

    return effective_injections


def load_snapshot_list(co2lvl):
    """Load snapshot list from disk for certain co2lvl.

    Args:
        co2lvl (float): CO2 constraint

    Returns:
        snapshot_list (list): List of snapshots.
    """
    with gzip.open(
        path_to_grid_data + f"/snapshot_list_co2lvl{co2lvl}.pklz",
        "rb",
    ) as f:
        snapshot_list = pickle.load(f)

    return snapshot_list


def save_snapshot_list(snapshots, co2lvl, overwrite=False):
    """Save snapshot list to disk.

    Args:
        snapshots (list): List of snapshots.
        co2lvl (float): CO2 level used in filename.
    Returns:
        str: Path to the written file.
    """

    file_path = path_to_grid_data + f"/snapshot_list_co2lvl{co2lvl}.pklz"

    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists. Set overwrite=True.")

    with gzip.open(file_path, "wb") as fh:
        pickle.dump(snapshots, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return file_path


def load_pypsa_network(
    co2lvl: float,
    n_nodes: int,
    use_sclopf: bool = True,
    lopt: bool = False,
):
    """Load PyPSA network with certain Co2 constraint and aggregation level of n_nodes.
    Args:
        co2lvl (float): Co2 constraint
        n_nodes (int): Aggregation level of nodes
        use_sclopf (bool): If 'True' the network is used was evaluated using security constrained lopf
        lopt (bool): If 'True' line extensions were used in the optimization
    Returns:
        network (PyPSA network)"""

    data_path = (
        path_to_pypsa_network_sclopf if use_sclopf else path_to_pypsa_network_lopf
    )
    files = os.listdir(data_path)

    files = [
        f
        for f in files
        if re.search(rf"Co2L{re.escape(str(co2lvl))}(?![0-9])", f)
        and f"_{n_nodes}_" in f
    ]  # get files for co2lvl and n_nodes
    # print("Filtered files:", files)
    files = [f for f in files if ("lcopt" in f) == lopt]  # filter for lopt or not
    # print("Filtered for lopt files:", files)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No file found for n_nodes={n_nodes} and co2lvl={co2lvl}"
        )
    elif len(files) > 1:
        raise RuntimeError(
            f"Multiple files found for n_nodes={n_nodes} and co2lvl={co2lvl}: {files}"
        )
    else:
        return load_pypsa_network_from_path(data_path + "/" +files[0], use_sclopf)


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


def load_cascades(co2lvl: float, n_nodes: int, use_sclopf: bool = True):
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


def load_grid_matrices(snet_index, co2lvl):
    """Load grid matrices from file for certain snet and co2lvl.

    Args:
        snet_index (int): Subnetwork index
        co2lvl (float): CO2 constraint
    Returns:
        I_m (scipy.sparse matrix): Incidence matrix
        B_d (scipy.sparse matrix): Susceptance matrix
        num_parallels (numpy array): Effective number of parallel lines per edge
        line_limits (numpy array): Line limits per edge
    """
    with gzip.open(
        path_to_grid_data + f"/grid_matrices_snet{snet_index}_co2lvl{co2lvl}.pklz", "rb"
    ) as f:
        grid_matrices = pickle.load(f)

    I_m = grid_matrices["I_m"]
    B_d = grid_matrices["B_d"]
    num_parallels = grid_matrices["num_parallels"]
    line_limits = grid_matrices["line_limits"]

    return I_m, B_d, num_parallels, line_limits


def save_grid_matrices(
    snet_index,
    co2lvl,
    I_m=None,
    B_d=None,
    num_parallels=None,
    line_limits=None,
    nx_graph=None,
    overwrite=False,
):
    """Save grid matrices to disk. Provide either nx_graph or all matrix arguments.

    Args:
        snet_index (int): Subnetwork index used in filename.
        co2lvl (float): CO2 level used in filename.
        I_m (scipy.sparse matrix or numpy array): Incidence matrix.
        B_d (scipy.sparse matrix): Susceptance diagonal matrix.
        num_parallels (np.ndarray): Effective parallel line counts.
        line_limits (np.ndarray): Line limits per edge.
        nx_graph (networkx.Graph, optional): If provided, matrices are extracted from this graph.
    Returns:
        str: Path to the written file.
    """

    file_path = (
        path_to_grid_data + f"/grid_matrices_snet{snet_index}_co2lvl{co2lvl}.pklz"
    )

    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists. Set overwrite=True.")

    if nx_graph is not None:
        I_m, B_d, num_parallels, line_limits = get_matrices_from_nx_graph(nx_graph)

    if any(x is None for x in (I_m, B_d, num_parallels, line_limits)):
        raise ValueError(
            "Either provide nx_graph or all of I_m, B_d, num_parallels and line_limits."
        )

    out = {
        "I_m": I_m,
        "B_d": B_d,
        "num_parallels": num_parallels,
        "line_limits": line_limits,
    }

    with gzip.open(file_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return file_path


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

    assert isinstance(n_nodes, int), "n_nodes must be an integer"
    # load hdf pandas
    split_properties = pd.read_hdf(load_dir + f"split_properties_all_n{n_nodes}.h5")
    split_properties["lost_load_share_blackout"] = split_properties[
        "lost_load_share_blackout"
    ].astype(float)

    if co2l is not None:
        return split_properties[split_properties.index.get_level_values("co2l") == co2l]

    return split_properties


def get_co2_levels(n_nodes, ignore_lvls=()):
    """Get available CO2 levels from SCLOPF data files.

    Args:
        n_nodes (int): Number of nodes in the network
        ignore_lvls (tuple): CO2 levels to ignore

    Returns:
        np.array: Sorted array of CO2 levels (descending)
    """
    from utils.config import path_to_sclopf_data

    sclopf_files = os.listdir(path_to_sclopf_data)
    sclopf_files = [file for file in sclopf_files if f"_{n_nodes}_" in file]
    co2l_list = [float(file.split("Co2L")[-1].split("-")[0]) for file in sclopf_files]
    co2l_list = sorted(list(set(co2l_list)), reverse=True)

    if isinstance(ignore_lvls, (float, int)):
        ignore_lvls = [ignore_lvls]
    for lvl in ignore_lvls:
        if lvl in co2l_list:
            co2l_list.remove(lvl)

    return np.array(sorted(co2l_list, reverse=True))

# FIXME n_nodes goes not into this. Maybe other names filenames.
def get_actual_co2_level(lvls, n_nodes=600, percent=False):
    """Get actual CO2 levels from results file.

    Args:
        lvls: CO2 level(s) to look up
        n_nodes (int): Number of nodes (not used currently)
        percent (bool): Return as percentage if True

    Returns:
        Actual CO2 level(s)
    """
    from utils.config import path_to_sclopf_results

    lvls_actual = (
        pd.read_csv(path_to_sclopf_results + "actual_co2_levels.csv", index_col=0)
        .loc[lvls]
        .values.squeeze()
    )

    if percent:
        lvls_actual = (lvls_actual * 100).round().astype(int)

    return lvls_actual
