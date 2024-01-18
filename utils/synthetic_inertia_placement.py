#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Mitigation strategies for system splits. Either place virtual
inertia or add lines."""

import numpy as np
import pandas as pd

import gzip
import pickle

from tqdm import tqdm


def build_idx_to_node_idx_list_n_reverse(indicator_vec_arr):
    """Build a dictionary that connects individual nodes to the components
    of each split component.

    Args:
        indicator_vec_arr (_type_): _description_
        split_numbers (_type_): _description_
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
                                  split_to_node_list: list, delta_rot_energy: float,
                                  rocof_neg_threshold:float,
                                  freq_ref: float = 50.):
    """Place inertia according to the highest amount of mitigated load."""
    
    
    # Check if component properties has all the entries needed (e.g., last day and first day)
    nn_nodes = indicator_vectors.shape[1]
    mitigate_load_loss_node_arr = np.zeros(nn_nodes)
    
    # For each of these significant cascades...check what happens if
    # inertia is placed
    for idx_split, split_r in enumerate(component_props_arr):  
        
        modified_rocof = freq_ref * (split_r[1]/(2 * (split_r[0] + delta_rot_energy)))
        if split_r[2] < rocof_neg_threshold and modified_rocof > rocof_neg_threshold:
            idx_node_for_split = split_to_node_list[idx_split]
            mitigated_load_loss_r = split_r[3]
            mitigate_load_loss_node_arr[idx_node_for_split] += mitigated_load_loss_r
            
    return mitigate_load_loss_node_arr


def run_greedy_inertia_placement(component_df: pd.DataFrame,
                                 indicator_vec_arr: np.ndarray,
                                 delta_rot_energy: float,
                                 rocof_threshold_Hz_s: float = -1.,
                                 max_iter: int = 2000,
                                 freq_ref: float = 50):
    """Run inertia placement to reduce the ammount of lost load, which is defined as the 
    load in a component that suffers a rocof small er as 'rocof_threshold_Hz_s'."""
    
    # Find out significant splits and reduce comp and ind
    cut_idxs = (component_df['rocof'] < rocof_threshold_Hz_s).values
    component_df_cut = component_df[cut_idxs]
    indicator_vec_arr_cut = indicator_vec_arr[cut_idxs]
    
    # convert component_df to array 
    modified_component_df = component_df_cut.loc[:, ['rot_energy',
                                                     'power_imbalance',
                                                     'rocof', 'load']].copy()
    modified_comp_arr = np.array(modified_component_df.values)
    modified_comp_index = modified_component_df.index.values
    
    inertia_placed_loss_mitigated_ls = list()
    
    split_to_node_idx_ls, node_to_split_idx_ls = build_idx_to_node_idx_list_n_reverse(indicator_vec_arr_cut)
    
    # step, place_added, multi_added
    delta_rot_energy_factor = 1.
    for idx_step in tqdm(range(max_iter)):
        ch_rot_energy_r = delta_rot_energy_factor * delta_rot_energy
        
        # Find candidate for largest mitigated lost load
        proposed_load_loss_change = greedy_inertia_placement_step(modified_comp_arr, 
                                                                  indicator_vec_arr_cut,
                                                                  split_to_node_idx_ls,
                                                                  delta_rot_energy,
                                                                  rocof_threshold_Hz_s,
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
            
            
    output_tuple = (modified_comp_index, modified_comp_arr,
                    inertia_placed_loss_mitigated_ls)
    
    return output_tuple


def run_specific_level_n_size(save_it: bool = True):
    
    
    # Load files for DataFrame collecting component d 
    fpath_component_in = "results/"
    with gzip.open(fpath_component_in) as fh_comp_in:
        component_df = pickle.load(fh_comp_in)
        
    fpath_indicator_vec_in = "results/"
    indicator_vec_arr = np.load(fpath_indicator_vec_in)
    
    result, run_greedy_inertia_placement(component_df, indicator_vec_arr)
    
    
    return
