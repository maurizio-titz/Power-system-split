#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Place synthetic inertia to mitigate dangerous system splits 
and lost load. This is done via adding rot energy to individual nodes
iteratively (i.e., greedily to the one mitigating the most lost load)."""

import os

import numpy as np
import pandas as pd

import gzip
import pickle

from tqdm import tqdm

from utils.data_handling import load_pypsa_network

results_path_mitigation = "results/sclopf/syn_inertia_mitigation"
if not os.path.exists(results_path_mitigation):
    os.mkdir(results_path_mitigation)


def build_idx_to_node_idx_list_n_reverse(indicator_vec_arr: np.ndarray):
    """Build a dictionary that connects individual nodes to the components
    of each split component.

    Args:
        indicator_vec_arr (np.ndarray): _description_
    """
    
    split_node_idx_ls = list()
    node_split_idx_ls = list()
    
    for idx_r, row_r in enumerate(indicator_vec_arr):
        split_node_idx_ls.append(np.argwhere(row_r).flatten())
        
    for column_r in indicator_vec_arr.T:
        node_split_idx_ls.append(np.argwhere(column_r).flatten())
    
    return split_node_idx_ls, node_split_idx_ls


def greedy_inertia_placement_step(component_props_arr: np.ndarray,
                                  indicator_vectors: np.ndarray,
                                  snapshot_weightings_arr: np.ndarray,
                                  split_to_node_list: list, delta_rot_energy: float,
                                  rocof_neg_threshold: float,
                                  load_share_threshold: float,
                                  freq_ref: float = 50.,
                                  verbose: bool = False):
    """Place inertia according to the highest amount of mitigated load."""
    
    
    # Check if component properties has all the entries needed (e.g., last day and first day)
    nn_nodes = indicator_vectors.shape[1]
    mitigate_load_loss_node_arr = np.zeros(nn_nodes)
    
    # Try which lost load can be mitigated when placing delta_rot_energy at each node
    # to pick the node with the maximum change (i.e., greedy optimization).
    for idx_split, split_r in enumerate(component_props_arr):  
        
        modified_rocof = freq_ref * (split_r[1]/(2 * (split_r[0] + delta_rot_energy)))
        if split_r[2] < rocof_neg_threshold and modified_rocof > rocof_neg_threshold and split_r[4] > load_share_threshold:
            idx_node_for_split = split_to_node_list[idx_split]
            mitigated_load_loss_r = split_r[3]
            mitigate_load_loss_node_arr[idx_node_for_split] += mitigated_load_loss_r * snapshot_weightings_arr[idx_split]
            
    return mitigate_load_loss_node_arr


def run_greedy_inertia_placement(component_df: pd.DataFrame,
                                 indicator_vec_arr: np.ndarray,
                                 snapshot_weightings_generators: pd.DataFrame,
                                 delta_rot_energy: float,
                                 rocof_threshold_Hz_s: float = -1.,
                                 max_iter: int = 10000,
                                 freq_ref: float = 50,
                                 load_share_threshold: float = 0,
                                 show_progress: bool = True):
    """Run inertia placement to reduce the amount of lost load, which is defined as the 
    load in a component that suffers a rocof small er as 'rocof_threshold_Hz_s'."""
    
    assert rocof_threshold_Hz_s < 0
    
    # Find out significant splits and reduce comp and ind
    cut_idxs = (component_df['rocof'] < rocof_threshold_Hz_s).values
    
    component_df_cut = component_df[cut_idxs]
    len_cut_df = len(component_df_cut.index)
    indicator_vec_arr_cut = indicator_vec_arr[cut_idxs]
    
    # Deal with snapshot weightings
    cut_time_stamps = component_df.time_stamp[cut_idxs].values
    snapshot_weightings_arr_cut = snapshot_weightings_generators[cut_time_stamps].values
    
    # convert component_df to array 
    modified_component_df = component_df_cut.loc[:, ['rot_energy',
                                                     'power_imbalance',
                                                     'rocof', 'load', 'load_share']].copy()
    modified_comp_arr = np.array(modified_component_df.values)
    #return modified_comp_arr
    modified_comp_index = modified_component_df.index.values
    
    inertia_placed_loss_mitigated_ls = list()
    
    split_to_node_idx_ls, node_to_split_idx_ls = build_idx_to_node_idx_list_n_reverse(indicator_vec_arr_cut)
    
    # step, place_added, multi_added
    delta_rot_energy_factor = 1.
    
    pbar = tqdm(range(max_iter), disable=not show_progress)
    for idx_step in pbar:
        count_beyond_threshold = np.count_nonzero(modified_comp_arr[:, 2] <  rocof_threshold_Hz_s)
        pbar.set_description(f"Beyond threshold {count_beyond_threshold/ len_cut_df*100:.1f}%")
        ch_rot_energy_r = delta_rot_energy_factor * delta_rot_energy
        
        # Find candidate for largest mitigated lost load
        proposed_load_loss_change = greedy_inertia_placement_step(modified_comp_arr, 
                                                                  indicator_vec_arr_cut,
                                                                  snapshot_weightings_arr_cut,
                                                                  split_to_node_idx_ls,
                                                                  delta_rot_energy,
                                                                  rocof_threshold_Hz_s,
                                                                  load_share_threshold,
                                                                  freq_ref=freq_ref)
        
        # If no change was detected, more synthetic rot. energy is added 
        if (proposed_load_loss_change == 0).all():
            delta_rot_energy_factor += 1.
            
        else:
            max_change = np.max(proposed_load_loss_change)
            idx_node = np.where(proposed_load_loss_change == max_change)[0]
            if len(idx_node) > 1:
                idx_node = np.random.choice(idx_node)
            else:
                idx_node = idx_node[0]
            
            # Collect change details
            results_out_r = [idx_step, idx_node, delta_rot_energy_factor, max_change]
            inertia_placed_loss_mitigated_ls.append(results_out_r)
            
            # Modify rest of the components due to the placement of inertia
            try:
                idx_node_in_split = node_to_split_idx_ls[idx_node]
            except TypeError:
                print(idx_node)
                raise TypeError
            
            modified_comp_arr[idx_node_in_split, 0] += ch_rot_energy_r
            new_rocof = (freq_ref *
                          (modified_comp_arr[idx_node_in_split, 1]/
                           (2*modified_comp_arr[idx_node_in_split, 0])))
            modified_comp_arr[idx_node_in_split, 2] = new_rocof
            
            delta_rot_energy_factor = 1.
        
        # Break when all splits are pushed below rocof threshold
        if count_beyond_threshold == 0:
            break
            
    return (modified_comp_index, modified_comp_arr,
            inertia_placed_loss_mitigated_ls)


def run_specific_level_n_size(co2_lvl: float, nn_nodes: int = 400, 
                              delta_rot_energy: float = 100, max_iter: int=10000,
                              rocof_threshold_Hz_s: float = -1., lshare_threshold: float = 0., 
                              save_it: bool = True):
    """Run the inertia placement for an optimized power system that was analyzed by 
    running cascade experiments. 

    Args:
        co2_lvl (float): CO2 level of PyPSA network.
        nn_nodes (int, optional): Number of nodes of PyPSA network. Defaults to 800.
        delta_rot_energy (float, optional): Change in rotational energy per step. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10000.
        rocof_threshold_Hz_s (float, optional): _description_. Defaults to -1..
        lshare_threshold (float, optional): _description_. Defaults to 0..
        save_it (bool, optional): _description_. Defaults to True.

    Returns:
        res_tuple: _description_
    """
    
    assert rocof_threshold_Hz_s < 0
    
    # Load files for DataFrame collecting component properties, indicator vectors and PyPSA network.
    fpath_component_in = f"results/sclopf/evaluation_results/component_properties_Co2L{co2_lvl:.1f}_n{nn_nodes}.h5"
    component_df = pd.read_hdf(fpath_component_in, key='df')
        
    fpath_indicator_vec_in = f"results/sclopf/evaluation_results/indicator_vectors_Co2L{co2_lvl:.1f}_n{nn_nodes}.npy"
    indicator_vec_arr = np.load(fpath_indicator_vec_in)
    
    fpath_pypsa_network = ("data/European_networks_sclopf/" +
                           f"sclopf-elec_s_{nn_nodes}_ec_lv1.0_Co2L{co2_lvl}-2920SEG.nc")
    pypsa_net = load_pypsa_network(fpath_pypsa_network, True)
    snapshot_weightings_generators = pypsa_net.snapshot_weightings.generators
    
    res_tuple = run_greedy_inertia_placement(component_df, indicator_vec_arr, snapshot_weightings_generators,
                                             delta_rot_energy, rocof_threshold_Hz_s=rocof_threshold_Hz_s,
                                             max_iter=max_iter, load_share_threshold=lshare_threshold)
    
    if save_it:
        fpath_out = (results_path_mitigation + f"/synthetic_inertia_placement_Co2{co2_lvl:.2f}_N{nn_nodes}" + 
                     f"_deltarotE{delta_rot_energy:.2f}_rocofthres{rocof_threshold_Hz_s:.2f}" + 
                     f"_lshare{lshare_threshold:.2f}_maxiter{max_iter}.pklz")
        with gzip.open(fpath_out, 'wb') as fh_out:
            pickle.dump(res_tuple, fh_out)
    
    return res_tuple
