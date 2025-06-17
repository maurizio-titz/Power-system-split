#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Find indicator vector for a split that either has the number
of functioning links or the inertia in the split."""

import gzip
import multiprocessing as mp
import os
import pickle
import sys
from functools import partial
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import (  # path_to_cascade_results_lopf,
    path_to_cascade_results_lopf,
    path_to_cascade_results_sclopf,
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_indicator_vectors_lopf,
    path_to_indicator_vectors_sclopf,
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
)
from utils.data_handling import (  # nx_edges_to_matrix_indices,
    build_networkx_graph,
    load_pypsa_network_from_path,
    load_pypsa_network,
    load_split_props,
)

fpath_out_root = path_to_indicator_vectors_sclopf
if not os.path.exists(fpath_out_root):
    os.mkdir(fpath_out_root)


def failing_edges_in_split(
    idx_edges_array: np.ndarray, list_failed_edges: list
) -> np.ndarray:
    """Return a binary vector for each split that collects
    if the links are active (0) or if they have failed during the cascade (1)

    Args:
        list_idx_edges (numpy array): List of with integers giving
        the indices of the link
        list_failed_edges (list): list containing the links that
        have failed during a cascade

    Returns:
        functioning_edge_indicator_vector: Vector that collects if an edge is working
    """

    functioning_edge_indicator_vector = np.zeros(len(idx_edges_array), dtype=int)

    idx_failed_edges = np.where(np.isin(list_failed_edges, idx_edges_array))[0]

    functioning_edge_indicator_vector[idx_failed_edges] = 1

    return functioning_edge_indicator_vector


def nodal_rocof_in_split(co2_lvl: float, n_nodes: int, save_res: bool = True):
    """Return the vector for each split that has an
    entry for each node, which quantifies the rocof in its component."""

    # TODO write version to create version from scratch

    return


def extract_nodal_rocof_and_load_share_in_split_from_old_results(
    co2_lvl: float,
    n_nodes: int,
    save_res: bool = True,
    verbose: bool = True,
    overwrite: bool = False,
    use_sclopf: bool = True,
):
    """Use the results from the old indicator vectors to arrive at the
    vectors that give the RoCoF for every node's component"""

    # Check before if the results already exists
    if use_sclopf:
        path_to_cascade_results = path_to_cascade_results_sclopf
        path_to_evaluation_results = path_to_evaluation_results_sclopf
    else:
        path_to_evaluation_results = path_to_evaluation_results_lopf
        path_to_cascade_results = path_to_cascade_results_lopf

    if save_res:
        fpath_indi_vec_rocof_out = (
            path_to_evaluation_results
            + "/indicator_vector_rocof_Co2L"
            + f"{co2_lvl}_n{n_nodes}.pklz"
        )
        fpath_indi_vec_lshare_out = (
            path_to_evaluation_results
            + "/indicator_vector_lshare_Co2L"
            + f"{co2_lvl}_n{n_nodes}.pklz"
        )
        print(fpath_indi_vec_rocof_out)

        if (
            os.path.exists(fpath_indi_vec_rocof_out)
            or os.path.exists(fpath_indi_vec_lshare_out)
        ) and not overwrite:
            raise IOError(
                "Output was written previously. "
                + "Please move or delete the previous results, or chose 'overwrite=True'."
            )

    if verbose:
        print(
            "Loading cascade results, indicator vectors and split component properties:",
            flush=True,
        )

    # Load component properties and indicator vector
    if use_sclopf:
        path_to_cascade_results_file = (
            path_to_cascade_results
            + "/system_splits_Co2L"
            + f"{co2_lvl}_n{n_nodes}.pklz"
        )
    else:
        path_to_cascade_results_file = (
            path_to_cascade_results
            + "/system_splits_singlelinefailures_Co2L"
            + f"{co2_lvl}_n{n_nodes}_lopf.pklz"
        )

    with gzip.open(path_to_cascade_results_file, "rb") as fh_in_casc:
        cascade_dict = pickle.load(fh_in_casc)

    indicator_vectors_file_path = (
        path_to_evaluation_results
        + f"component_indicator_vectors_Co2L{co2_lvl}_n{n_nodes}.pklz".format(
            co2_lvl, n_nodes
        )
    )
    with gzip.open(indicator_vectors_file_path, "rb") as fh_in_indi:
        indicator_vector_arr = pickle.load(fh_in_indi)

    df_comp_props = pd.read_hdf(
        path_to_evaluation_results
        + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5".format(co2_lvl, n_nodes),
        key="df",
    )
    if verbose:
        print("Finished loading data.\n")

    # Convert index to datetime
    df_comp_props.time_stamp = pd.to_datetime(df_comp_props.time_stamp)

    # Add list of tuples of initial failures
    if use_sclopf:
        df_comp_props["init_failure_tuples"] = list(
            zip(
                df_comp_props.init_failure_0.astype(int),
                df_comp_props.init_failure_1.astype(int),
            )
        )

    # iterate through each time
    unique_times = pd.unique(df_comp_props.time_stamp)
    total_nr_splits = sum(len(vv) for vv in cascade_dict.values())

    indicator_vector_rocof = np.full(
        (total_nr_splits, indicator_vector_arr.shape[1]), np.nan, dtype=float
    )
    indicator_vector_load_share = np.full(
        (total_nr_splits, indicator_vector_arr.shape[1]), np.nan, dtype=float
    )

    if verbose:
        print("Starting to extract rocof and load share indicator vectors:")

    index_tuple_time_split = list()
    out_idx = 0
    for idx_time, time_stamp_r in enumerate(tqdm(unique_times, disable=not verbose)):
        df_time_r = df_comp_props.loc[df_comp_props.time_stamp == time_stamp_r]

        # Find the number of unique tuples aka number of splits
        if use_sclopf:
            unique_init_tuples = pd.unique(df_time_r["init_failure_tuples"])
        else:
            unique_init_tuples = pd.unique(df_time_r["init_failure"])

        for idx_split, init_tuple_r in enumerate(unique_init_tuples):
            if use_sclopf:
                df_event_r = df_time_r.loc[
                    df_time_r.init_failure_tuples == init_tuple_r
                ]
            else:
                df_event_r = df_time_r.loc[df_time_r.init_failure == init_tuple_r]

            indicator_vector_view = indicator_vector_arr[df_event_r.index]
            assert (np.sum(indicator_vector_view, axis=0) == 1).all()

            index_tuple_time_split.append((time_stamp_r, init_tuple_r, idx_split))
            for idx_in_split_r, row_r in df_event_r.iterrows():

                idx_vec_in_split = np.argwhere(
                    indicator_vector_arr[idx_in_split_r] == 1
                )
                indicator_vector_rocof[out_idx, idx_vec_in_split] = row_r.rocof
                indicator_vector_load_share[out_idx, idx_vec_in_split] = (
                    row_r.load_share
                )

            out_idx += 1

    if save_res:

        with gzip.open(fpath_indi_vec_rocof_out, "wb") as fh_rocof_out:
            pickle.dump((index_tuple_time_split, indicator_vector_rocof), fh_rocof_out)

        with gzip.open(fpath_indi_vec_lshare_out, "wb") as fh_out_lshare:
            pickle.dump(
                (index_tuple_time_split, indicator_vector_load_share), fh_out_lshare
            )

    return index_tuple_time_split, indicator_vector_rocof, indicator_vector_load_share


def find_failed_edge_indicator_vector_for_cascade_results(
    co2_lvl: float,
    n_nodes: int,
    snet_idx: int = 0,
    save_res: bool = True,
    verbose: bool = True,
    overwrite: bool = False,
    use_sclopf: bool = True,
) -> tuple:
    """Find the indicator vectors that has an entry for every split that has a 'True'
    if a link failed during the split.

    Args:
        co2_lvl (float): CO2 level of the considered PyPSA scenario
        n_nodes (int): Number of nodes of the PyPSA network
        snet_idx (int, optional): Subnet index of PyPSA with 0 indicating CE. Defaults to 0.
        save_res (bool, optional): If 'True', save the results to a file. Defaults to True.
        verbose (bool, optional): If 'True', print additional details. Defaults to True.
        overwrite (bool, optional): If 'True', overwrite previous run. Defaults to False.

    Raises:
        IOError: _description_

    Returns:
        edge_names_ls, edge_pypsa_index_ls,
        index_tuple_splits, indicator_failed_edges_arr: _description_
    """

    # Check if files already exists
    if save_res:
        if use_sclopf:
            fpath_out_edge_base = (
                path_to_evaluation_results_sclopf
                + "/failed_edges_indicator_vector_Co2l"
                + f"{co2_lvl}_n{n_nodes}.pklz"
            )
        else:
            fpath_out_edge_base = (
                path_to_evaluation_results_lopf
                + "/failed_edges_indicator_vector_Co2l"
                + f"{co2_lvl}_n{n_nodes}.pklz"
            )
        if os.path.exists(fpath_out_edge_base) and not overwrite:
            raise IOError(
                "File already exists! Please remove or choose 'overwrite=True'."
            )
        os.makedirs(os.path.dirname(fpath_out_edge_base), exist_ok=True)

    # Load file
    if verbose:
        print("Loading cascade results and PyPSA output to generate graph:", flush=True)

    ## Results path of cascade simulations
    if use_sclopf:
        path_to_cascade_results_file = (
            path_to_cascade_results_sclopf
            + f"system_splits_Co2L{co2_lvl}_n{n_nodes}.pklz"
        )
    else:
        path_to_cascade_results_file = (
            path_to_cascade_results_lopf
            + f"system_splits_singlelinefailures_Co2L{co2_lvl}_n{n_nodes}_lopf.pklz"
        )

    with gzip.open(path_to_cascade_results_file, "rb") as fh:
        cascade_dict = pickle.load(fh)

    ## PyPSA network

    pypsa_net = load_pypsa_network(co2_lvl, n_nodes, use_sclopf=use_sclopf)
    graph_nx = build_networkx_graph(pypsa_net, snet_index=snet_idx)

    edge_names_ls = list(graph_nx.edges())
    edge_pypsa_index_ls = np.array(
        [
            int(data["line_index"][0].split("_out")[0])
            for uu, vv, data in graph_nx.edges(data=True)
        ]
    )

    assert len(np.unique(edge_pypsa_index_ls)) == len(edge_pypsa_index_ls)

    if verbose:
        print("\nFinished loading data and building vector now:", flush=True)

    # node_names_ls = list(graph_nx.nodes())

    nr_edges = len(edge_names_ls)
    total_nr_splits = sum(len(vv) for vv in cascade_dict.values())

    indicator_failed_edges_arr = np.full((total_nr_splits, nr_edges), False, dtype=bool)

    # iterate through time and split
    index_tuple_splits = list()
    out_idx = 0
    for idx_time, [time_str, trigger_split_dict] in enumerate(
        tqdm(cascade_dict.items(), disable=not verbose)
    ):
        for idx_split, [trigger_tuple, failing_links] in enumerate(
            trigger_split_dict.items()
        ):
            index_tuple_splits.append((time_str, trigger_tuple, idx_split))

            indicator_failed_edges_arr[out_idx, failing_links] = True
            out_idx += 1

    if save_res:
        with gzip.open(fpath_out_edge_base, "wb") as fh_out_edges:
            pickle.dump(
                (
                    edge_names_ls,
                    edge_pypsa_index_ls,
                    index_tuple_splits,
                    indicator_failed_edges_arr,
                ),
                fh_out_edges,
            )

    return (
        edge_names_ls,
        edge_pypsa_index_ls,
        index_tuple_splits,
        indicator_failed_edges_arr,
    )


def run_all_co2_lvl_edge_based(
    n_nodes: int, overwrite: bool = False, use_sclopf: bool = True
) -> None:
    """Calculate and save the edge based indicator vectors, i.e. the failed edge vectors, for all CO2 levels.

    Args:
        n_nodes (int): _description_
        overwrite (bool, optional): _description_. Defaults to False.
        use_sclopf (bool, optional): _description_. Defaults to True.
    """

    # Identify all input files
    ## Cascade and PyPSA files need to exist
    if use_sclopf:
        path_to_pypsa_network = path_to_pypsa_network_sclopf
    else:
        path_to_pypsa_network = path_to_pypsa_network_lopf

    glob_pypsa_search_str = path_to_pypsa_network + "*elec_s_{0}*.nc".format(n_nodes)
    pypsa_file_ls = glob(glob_pypsa_search_str)
    co2_lvl_ls = [float(xx.split("Co2L")[-1].split("-")[0]) for xx in pypsa_file_ls]
    print(f"running edge based alternative indicator vectors: {co2_lvl_ls}")
    for co2_r in co2_lvl_ls:
        print(co2_r)
        find_failed_edge_indicator_vector_for_cascade_results(
            co2_r,
            n_nodes,
            save_res=True,
            verbose=True,
            overwrite=overwrite,
            use_sclopf=use_sclopf,
        )

    return


def run_all_co2_lvl_node_based(n_nodes: int, use_sclopf=True) -> None:

    # Identify all input files
    ## Cascade and PyPSA files need to exist
    if use_sclopf:
        path_to_pypsa_network = path_to_pypsa_network_sclopf
    else:
        path_to_pypsa_network = path_to_pypsa_network_lopf

    glob_pypsa_search_str = path_to_pypsa_network + "*elec_s_{0}*.nc".format(n_nodes)
    pypsa_file_ls = glob(glob_pypsa_search_str)
    co2_lvl_ls = [float(xx.split("Co2L")[-1].split("-")[0]) for xx in pypsa_file_ls]
    print(f"running node based alternative indicator vectors: {co2_lvl_ls}")
    for co2_r in co2_lvl_ls:
        print(co2_r)
        extract_nodal_rocof_and_load_share_in_split_from_old_results(
            co2_r, n_nodes, save_res=True, verbose=True
        )

    return


def check_rocof_lshare_indicator_vectors(co2_lvl, nn_nodes=400, show_progress=False):
    """Check if the indicator that gives the rocof and load share gives
    consistent results with the previous results."""

    path_in = path_to_evaluation_results_sclopf

    ## Load Data
    # Load indicator vector rocof
    fpath_rocof_in = path_in + f"/indicator_vector_rocof_Co2L{co2_lvl}_n{nn_nodes}.pklz"
    with gzip.open(fpath_rocof_in, "rb") as fh_rocof_in:
        indi_vec_rocof = pickle.load(fh_rocof_in)[-1]

    # Load indicator vector lshare
    fpath_lshare_in = (
        path_in + f"/indicator_vector_lshare_Co2L{co2_lvl}_n{nn_nodes}.pklz"
    )
    with gzip.open(fpath_lshare_in, "rb") as fh_lshare_in:
        indi_vec_lshare = pickle.load(fh_lshare_in)[-1]

    # Load indicator vector failed edges
    fpath_failed_edges = (
        path_in + f"/indicator_vector_failed_edges_Co2L{co2_lvl}_n{nn_nodes}.pklz"
    )
    with gzip.open(fpath_failed_edges) as fh_edges_in:
        indi_vec_fedges = pickle.load(fh_edges_in)[-1]

    # PyPSA network + graph_nx
    pypsa_net = load_pypsa_network_from_path(
        co2_lvl, nn_nodes, "data/European_networks_sclopf/"
    )
    # translate to network for CE
    graph_nx = build_networkx_graph(pypsa_net, snet_index=0)

    ## Separate graph according to failed edges
    # and check if same nodes in node based indicator vectors
    list_nodes = list(graph_nx.nodes())
    node_lookup_dict = {xx: idx for idx, xx in enumerate(list_nodes)}

    list_edges = list(graph_nx.edges())

    for idx, row in enumerate(tqdm(indi_vec_fedges, disable=not show_progress)):

        row_rocof = indi_vec_rocof[idx]
        graph_r = graph_nx.copy()
        assert nx.is_connected(graph_r), "Graph should be connected but is not!"

        idx_failed_edegs = np.argwhere(row)
        list_tuple_fedges = [list_edges[int(xx)] for xx in idx_failed_edegs]

        graph_r.remove_edges_from(list_tuple_fedges)

        assert not nx.is_connected(
            graph_r
        ), "Graph should be disconnected, since the failed edges, which lead to a system split, have been removed!"

        # Check if other indicator vectors have same nodes
        ## Start with only two components
        counter_larger_components = 0
        components = [
            set([node_lookup_dict[yy] for yy in xx])
            for xx in nx.connected_components(graph_r)
        ]

        unique_rocof_row = np.unique(row_rocof)
        for uni_rocof_r in unique_rocof_row:
            if uni_rocof_r != 0:
                nodes_idx_comp = set(np.argwhere(row_rocof == uni_rocof_r).flatten())

                # Is this a component of the disconnected graph
                is_in_connected_comp = nodes_idx_comp in components

                if not is_in_connected_comp:
                    return co2_lvl, False

    return co2_lvl, True


def single_call(nn_nodes, co2_lvl_rr):
    return check_rocof_lshare_indicator_vectors(co2_lvl_rr, nn_nodes=nn_nodes)


def check_component_properties_indicatorvectors_all_co2lvl(
    nn_nodes=400, nn_procs=4, save_it: bool = True
):
    """Check if the component properties of all co2 vectors agree
    with the failed edges vectors

    Args:
        nn_scan (int, optional): Number of nodes in PyPSA network. Defaults to 400.
    """

    glob_search_str = f"results/sclopf/indicator_vectors_rocof_lshare_edges/indicator_vector_rocof_*n{nn_nodes}.pklz"
    co2_levels = [
        float(xx.split("_n")[0].split("Co2l")[-1]) for xx in glob(glob_search_str)
    ]

    funci = partial(single_call, nn_nodes)

    def dummy_callback(_):
        pbar.update()

    with mp.Pool(processes=nn_procs) as pool:
        with tqdm(total=len(co2_levels)) as pbar:
            async_results = [
                pool.apply_async(funci, args=(xx,), callback=dummy_callback)
                for xx in co2_levels
            ]

            results = [async_r.get() for async_r in async_results]

    if save_it:
        fpath_out = (
            path_to_indicator_vectors_sclopf
            + f"/check_rocof_allCo2lvls_nn{nn_nodes}.pklz"
        )
        with gzip.open(fpath_out, "wb") as fh_out:
            pickle.dump(results, fh_out)

    return results


def get_indicator_vector_snapshot_weightings(
    co2lvl, n_nodes=400, use_sclopf: bool = True
):
    split_properties = load_split_props(
        n_nodes=n_nodes, co2l=co2lvl, use_sclopf=use_sclopf
    )
    return split_properties.snapshot_weighting.values
