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

import multiprocessing
from functools import partial

from utils.data_handling import load_pypsa_network

# Send messages to mattermost
import utils.config as cfg
from utils import send_mattermost_messages

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


def get_lost_load_in_member_compontents(node_arr: np.ndarray, 
                                        component_props_arr: np.ndarray,
                                        snapshot_weightings_arr: np.ndarray,
                                        node_to_split_idx_ls: list, change_rot_energy: float,
                                        rocof_neg_threshold: float, freq_ref: float = 50.) -> np.ndarray:
    """Get how much load there is in all components that have the nodes in 'node_list'
    as members."""
    
    lost_load_member_comps = np.zeros(len(node_arr))
    
    for idx_ele, node_idx in enumerate(node_arr):
        # Find idxs of splits where node_r is a member
        idx_node_in_split = node_to_split_idx_ls[node_idx]
        
        cut_snapshow_weighting = snapshot_weightings_arr[idx_node_in_split]
        
        node_cut_comp_arr = component_props_arr[idx_node_in_split, :].copy()
        
        # Place inertia and evaluate rocof new
        node_cut_comp_arr[:, 0] += change_rot_energy
        new_rocof = (freq_ref * 
                     (node_cut_comp_arr[:, 1]/(2*node_cut_comp_arr[:, 0])))
        node_cut_comp_arr[:, 2] = new_rocof
        
        # See where rocof exceeds threshold and how much load is therefore lost
        lost_idx = node_cut_comp_arr[:, 2] < rocof_neg_threshold
        lost_load_member_comps[idx_ele] = (node_cut_comp_arr[lost_idx, 3] *
                                           cut_snapshow_weighting[lost_idx]).sum()
    
    return lost_load_member_comps


def greedy_inertia_placement_step(component_props_arr: np.ndarray,
                                  indicator_vectors: np.ndarray,
                                  snapshot_weightings_arr: np.ndarray,
                                  split_to_node_list: list, delta_rot_energy: float,
                                  rocof_neg_threshold: float,
                                  load_share_threshold: float,
                                  freq_ref: float = 50.):
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
                                 show_progress: bool = True,
                                 resolve_equal_randomly: bool = False, 
                                 atol: float = 1e-8):
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
    
    modified_comp_index = modified_component_df.index.values
    
    inertia_placed_loss_mitigated_ls = list()
    
    split_to_node_idx_ls, node_to_split_idx_ls = build_idx_to_node_idx_list_n_reverse(indicator_vec_arr_cut)
    
    # step, place_added, multi_added
    resolve_equality_counter = 0
    still_used_random_node_choice = 0
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
            max_change = proposed_load_loss_change.max()
            idx_node = np.where(abs(proposed_load_loss_change - max_change) < atol)[0]
            
            # Check if a decision has to be made due to two nodes being equal
            resolve_equality_counter += 1
            if len(idx_node) > 1:
                
                if resolve_equal_randomly:
                    idx_node = np.random.choice(idx_node)
                else:
                    # Check for which node placed, the lost load in its components is higher
                    # resolve conflict again randomly
                    list_lost_load_of_member_components = get_lost_load_in_member_compontents(idx_node, modified_comp_arr,
                                                                                              snapshot_weightings_arr_cut, node_to_split_idx_ls,
                                                                                              ch_rot_energy_r, rocof_threshold_Hz_s,
                                                                                              freq_ref=freq_ref)
                    
                    max_val_lost = list_lost_load_of_member_components.max()
                    idxs_max_lost_load = np.where(abs(list_lost_load_of_member_components - max_val_lost) < 1e-8)[0]
                    if len(idxs_max_lost_load) > 1:
                        idx_node = np.random.choice(idxs_max_lost_load)
                        still_used_random_node_choice += 1
                    else:
                        idx_node = idxs_max_lost_load[0]
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
            inertia_placed_loss_mitigated_ls, 
            resolve_equality_counter, still_used_random_node_choice)


def run_specific_co2lvl_n_size(co2_lvl: float, nn_nodes: int = 400, 
                              delta_rot_energy: float = 100, max_iter: int=10000,
                              rocof_threshold_Hz_s: float = -1., lshare_threshold: float = 0., 
                              use_random_resolve: bool = False,
                              save_it: bool = True, show_progress: bool = True):
    """Run the inertia placement for an optimized power system that was analyzed by 
    running cascade experiments. 

    Args:
        co2_lvl (float): CO2 level of PyPSA network.
        nn_nodes (int, optional): Number of nodes of PyPSA network. Defaults to 800.
        delta_rot_energy (float, optional): Change in rotational energy per step. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10000.
        rocof_threshold_Hz_s (float, optional): Threshold after which a component is counted as 
            experiencing a black out aka the load is counted as lost. Defaults to -1..
        lshare_threshold (float, optional): Amount of load share of components that are being considered
            in the greedy mitigation procedure. Defaults to 0, which corresponds to all components
            being considered.
        use_random_resolve (bool, optional): If 'True', nodes that would lead to the same
            lost load mitigation would lead to a random node being picked. Otherwise the node
            that remains in the components with the highest lost load is picked in a subsequent step.
            If this does not resolve the choice, a random choice is taken again.
        save_it (bool, optional): If 'True', the results are being pickled and saved. Defaults to True.

    Returns:
        res_tuple: see output of run_greedy_inertia_placement
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
                                             max_iter=max_iter, load_share_threshold=lshare_threshold,
                                             resolve_equal_randomly=use_random_resolve,
                                             show_progress=show_progress)
    
    if save_it:
        fpath_out = (results_path_mitigation + f"/synthetic_inertia_placement_Co2{co2_lvl:.2f}_N{nn_nodes}" + 
                     f"_deltarotE{delta_rot_energy:.2f}_rocofthres{rocof_threshold_Hz_s:.2f}" + 
                     f"_lshare{lshare_threshold:.2f}_maxiter{max_iter}")
        if use_random_resolve:
            fpath_out += "_randomresolve"
            
        with gzip.open(fpath_out + ".pklz", 'wb') as fh_out:
            pickle.dump(res_tuple, fh_out)
        
        if cfg.mattermost_url is not None:
            message_text = (f"Finished synthetic inertia placement for  " + 
                        f"N={nn_nodes}, C02_lvl={co2_lvl} and saved results in '" + fpath_out + "'.")
            send_mattermost_messages.post_message(message_text, cfg.mattermost_url)
    
    return res_tuple

def single_call(nn_nodes, co2_lvl, max_iter, delta_rot_e):
    
    run_specific_co2lvl_n_size(co2_lvl, nn_nodes=nn_nodes, delta_rot_energy=delta_rot_e,
                               show_progress=True, save_it=True, max_iter=max_iter)
    
    return


def run_different_parameters_for_co2lvl(co2_lvl: float, nn_nodes: int, delta_rot_ls: list,
                                        max_iter=10000, nr_processes: int = 5):
    """Run the function 'run_specific_co2lvl_n_size' for the parameters
    giving delta_rot_energy."""
    
    # Parallelize it please
    
    with multiprocessing.Pool(processes=nr_processes) as pool:
        partial_func = partial(single_call, nn_nodes, co2_lvl, max_iter)
        
        [xx for xx in tqdm(pool.imap(partial_func, delta_rot_ls), total=len(delta_rot_ls))]
        
    
    return
