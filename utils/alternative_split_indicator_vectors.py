#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Find indicator vector for a split that either has the number
of functioning links or the inertia in the split."""

import os
import sys

import numpy as np
import pandas as pd

import gzip
import pickle

from tqdm import tqdm
from glob import glob

from utils.data_handling import load_pypsa_network, build_networkx_graph

fpath_out_root = "results/sclopf/indicator_vectors_rocof_lshare_edges"
if not os.path.exists(fpath_out_root):
    os.mkdir(fpath_out_root)


def failing_edges_in_split(idx_edges_array: np.ndarray, list_failed_edges: list):
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
    
    idx_failed_edges = np.where(np.isin(idx_edges_array, list_failed_edges))[0]
    
    functioning_edge_indicator_vector[idx_failed_edges] = 1
    
    return functioning_edge_indicator_vector
 

def nodal_rocof_in_split(co2_lvl: float, n_nodes: int, save_res: bool = True):
    """Return the vector for each split that has an 
    entry for each node, which quantifies the rocof in its component."""
    
    # TODO write version to create version from scratch
    
    
    return


def extract_nodal_rocof_and_load_share_in_split_from_old_results(co2_lvl: float, n_nodes: int,
                                                                 save_res: bool = True,
                                                                 verbose: bool = True,
                                                                 overwrite: bool = False):
    """Use the results from the old indicator vectors to arrive at the 
    vectors that give the rocof for every node's component"""
    
    # Check before if the results already exists
    if save_res:
        fpath_indi_vec_rocof_out = (fpath_out_root + 
                                    "/indicator_vector_rocof_Co2l" + 
                                    "{0:.1f}_n{1}.pklz".format(co2_lvl, n_nodes))
        fpath_indi_vec_lshare_out = (fpath_out_root + 
                                     "/indicator_vector_lshare_Co2l" + 
                                     "{0:.1f}_n{1}.pklz".format(co2_lvl, n_nodes))
        
        if ((os.path.exists(fpath_indi_vec_rocof_out) or os.path.exists(fpath_indi_vec_lshare_out)) 
            and not overwrite):
            raise IOError("Output was written previously. " +
                          "Please move or delete the previous results, or chose 'overwrite=True'.")    
    
    if verbose:
        print("Loading cascade results, indicator vectors and split component properties:",
              flush=True)
        
    # Load component properties and indicator vector
    path_to_cascade_results = ("results/sclopf/cascade_results/system_splits_Co2L" +
                               "{0:.1f}_n{1}.pickle".format(co2_lvl, n_nodes))
    with open(path_to_cascade_results, 'rb') as fh_in_casc:
        cascade_dict = pickle.load(fh_in_casc)
        
    path_to_indicator_vectors = ("results/sclopf/evaluation_results/" + 
                                 "indicator_vectors_Co2L{0:.1f}_n{1}.npy".format(co2_lvl, n_nodes))
    indicator_vector_arr = np.load(path_to_indicator_vectors)  
        
    df_comp_props = pd.read_hdf("results/sclopf/evaluation_results/" + 
                                "component_properties_Co2L{0:.1f}_n{1}.h5".format(co2_lvl, n_nodes),
                                key='df')
    if verbose:
        print("Finished loading data.\n")
        
    # Convert index to datetime
    df_comp_props.time_stamp = pd.to_datetime(df_comp_props.time_stamp)
    
    # Add list of tuples of initial failures
    df_comp_props['init_failure_tuples'] = list(zip(df_comp_props.init_failure_0.astype(int),
                                                    df_comp_props.init_failure_1.astype(int)))
    
    # iterate through each time
    unique_times = pd.unique(df_comp_props.time_stamp)
    total_nr_splits = sum(len(vv) for vv in cascade_dict.values())
        
    indicator_vector_rocof = np.full((total_nr_splits, indicator_vector_arr.shape[1]), 
                                     np.nan, dtype=float)
    indicator_vector_load_share = np.full((total_nr_splits, indicator_vector_arr.shape[1]),
                                          np.nan, dtype=float)
    
    if verbose:
        print("Starting to extract rocof and load share indicator vectors:")
    
    index_tuple_time_split = list()
    out_idx = 0
    for idx_time, time_stamp_r in enumerate(tqdm(unique_times, disable=not verbose)):
        df_time_r = df_comp_props.loc[df_comp_props.time_stamp == time_stamp_r]
        
        # Find the number of unique tuples aka number of splits
        unique_init_tuples = pd.unique(df_time_r['init_failure_tuples'])
        for idx_split, init_tuple_r in enumerate(unique_init_tuples):
            df_event_r = df_time_r.loc[df_time_r.init_failure_tuples == init_tuple_r]
            
            indicator_vector_view = indicator_vector_arr[df_event_r.index]
            assert (np.sum(indicator_vector_view, axis=0) == 1).all()
            
            index_tuple_time_split.append((time_stamp_r, init_tuple_r, idx_split))
            for idx_in_split_r, row_r in df_event_r.iterrows():
                            
                idx_vec_in_split = np.where(indicator_vector_arr[idx_in_split_r] == 1)
                indicator_vector_rocof[out_idx, idx_vec_in_split] = row_r.rocof
                indicator_vector_load_share[out_idx, idx_vec_in_split] = row_r.load_share

            out_idx += 1
            
    if save_res:
        
        with gzip.open(fpath_indi_vec_rocof_out, 'wb') as fh_rocof_out:
            pickle.dump((index_tuple_time_split, indicator_vector_rocof), fh_rocof_out)
        
        
        with gzip.open(fpath_indi_vec_lshare_out, 'wb') as fh_out_lshare:
            pickle.dump((index_tuple_time_split, indicator_vector_load_share), fh_out_lshare)
                    
    
    return index_tuple_time_split, indicator_vector_rocof, indicator_vector_load_share
            

def find_failed_edge_indicator_vector_for_cascade_results(co2_lvl: float, n_nodes: int,
                                                          snet_idx: int = 0,
                                                          save_res: bool = True,
                                                          verbose: bool = True,
                                                          overwrite: bool = False):
    """Find the indicator vectors that give the working edges and
    """
    
    # Check if files already exisits
    if save_res:
        fpath_out_edge_base = (fpath_out_root + "/indicator_vector_failed_edges_Co2l" +
                               "{0:.1f}_n{1}.pklz".format(co2_lvl, n_nodes))
        if os.path.exists(fpath_out_edge_base) and not overwrite:
            raise IOError("File already exists! Please remove or choose 'overwrite=True'.")
    
    
    # Load file
    if verbose:
        print("Loading cascade results and PyPSA output to generate graph:",
              flush=True)
    ## Results path of cascade simulations
    path_to_cascade_results = ("results/sclopf/cascade_results/" + 
                               "system_splits_Co2L{0:.1f}_n{1}.pickle".format(co2_lvl, n_nodes))
    
    with open(path_to_cascade_results, 'rb') as fh:
        cascade_dict = pickle.load(fh)
        
    ## PyPSA network
    pypsa_net = load_pypsa_network(co2_lvl, n_nodes,
                                   'data/European_networks_sclopf/')
    graph_nx = build_networkx_graph(pypsa_net, snet_index=snet_idx)
    
    edge_names_ls = list(graph_nx.edges())
    edge_index_ls = np.array([int(data['line_index'][0].split('_out')[0]) 
                              for uu, vv, data in graph_nx.edges(data=True)])
    
    assert len(np.unique(edge_index_ls)) == len(edge_index_ls)
    
    if verbose:
        print("\nFinished loading data and building vector now:", flush=True)
    
    #node_names_ls = list(graph_nx.nodes())
    
    nr_edges = len(edge_names_ls)    
    total_nr_splits = sum(len(vv) for vv in cascade_dict.values())
    
    indicator_failed_edges_arr = np.full((total_nr_splits, nr_edges), np.nan, dtype=float)
    
    # iterate through time and split
    index_tuple_splits = list()
    out_idx = 0
    for idx_time, [time_str, trigger_split_dict] in enumerate(tqdm(cascade_dict.items(), disable=not verbose)):
        for idx_split, [trigger_tuple, failing_links] in enumerate(trigger_split_dict.items()):
            index_tuple_splits.append((time_str, trigger_tuple, idx_split))
            
            indicator_failed_edges_arr[out_idx, :] = failing_edges_in_split(edge_index_ls, failing_links)
            out_idx += 1
            
    if save_res:
        with gzip.open(fpath_out_edge_base, 'wb') as fh_out_edges:
            pickle.dump((edge_names_ls, edge_index_ls,
                         index_tuple_splits, indicator_failed_edges_arr), fh_out_edges)
        
    return [edge_names_ls, edge_index_ls,
            index_tuple_splits, indicator_failed_edges_arr]


def run_all_co2_lvl_edge_based(n_nodes: int):
    
    # Identify all input files
    ## Cascade and PyPSA files need to exist
    glob_pypsa_search_str = ("data/European_networks_sclopf/" +
                             "sclopf-elec_s_{0}*.nc".format(n_nodes))    
    pypsa_file_ls = glob(glob_pypsa_search_str)
    co2_lvl_ls = [float(xx.split("Co2L")[-1].split("-")[0]) for xx in pypsa_file_ls]
    
    for co2_r in co2_lvl_ls:
        if co2_r > 0.:
            find_failed_edge_indicator_vector_for_cascade_results(co2_r, n_nodes,
                                                                  save_res=True,
                                                                  verbose=True)
    
    return


def run_all_co2_lvl_node_based(n_nodes: int):
    
    
    # Identify all input files
    ## Cascade and PyPSA files need to exist
    glob_pypsa_search_str = ("data/European_networks_sclopf/" +
                             "sclopf-elec_s_{0}*.nc".format(n_nodes))    
    pypsa_file_ls = glob(glob_pypsa_search_str)
    co2_lvl_ls = [float(xx.split("Co2L")[-1].split("-")[0]) for xx in pypsa_file_ls]
    
    for co2_r in co2_lvl_ls:
        if co2_r > 0.:
            extract_nodal_rocof_and_load_share_in_split_from_old_results(co2_r, n_nodes,
                                                                         save_res=True,
                                                                         verbose=True)
    
    return
