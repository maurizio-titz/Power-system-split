#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the results of the cascade experiments to determine
the properties of the system splits."""

import sys

from tqdm import tqdm
import pandas as pd
import numpy as np

import gzip
import pickle

from datetime import datetime as dt
import re

sys.path.append('./') 
from utils import data_handling, subgraph_evaluation

# Post messages to mattermost
import utils.config as cfg
from utils import send_mattermost_messages

# Setup paths to solved PyPSA networks and results own scripts
#path_to_pypsa_network = './data/European_networks_sclopf/'
#path_to_cascade_results  = './results/sclopf/cascade_results/'
#save_path =  './results/sclopf/evaluation_results/'

path_to_pypsa_network_lopf = './data/European_networks_lopf/'
path_to_pypsa_network_sclopf = './data/European_networks_sclopf/'

path_to_cascades_sclopf =  './results/sclopf/cascade_results/'
path_to_cascades_lopf =  './results/lopf/cascade_results/'

save_path_sclopf =  './results/sclopf/cascade_results/'
save_path_lopf =  './results/lopf/cascade_results/'

# Dummy decorator to not get stuck on @profile
if 'profile' not in globals():
    def profile(func):
        return func

@profile
def evaluate_cascade(co2l: float, n_nodes: int, snet_index: int = 0,
                     start_time_str=None, end_time_str=None, 
                     eval_indicator_vectors: bool = True, 
                     verbose: bool = False, show_progress: bool = True,
                     use_sclopf=True):
    """Find the properties of the splits (i.e., RoCoF or lost load) and 
    indicator vectors describing the network.

    Args:
        co2l (float): Target CO2 level in PyPSA Optimization.
        n_nodes (int): Number of nodes in PyPSA network.
        snet_index (int, optional): Select a particular subnetwork for calculations (if the pypsa network has different ones).
                    For our data set, "0" indicates the Continental European AC grid. Defaults to 0.
        start_time_str, end_time_str (str Format "YYYY-mm-dd HH:MM"): If either or both are not None, the evaluation will only be 
                    done between start_time and end_time.
    """
    
    def check_format_time_str(time_str):
        
        correct_len_str = len("xxxx-xx-xx xx:xx")
        has_correct_len = len(time_str) == correct_len_str
        
        rr = re.compile('\d{4}/\d{2}/\d{2} \d{2}:\d{2}')
        does_match = rr.match(time_str) is not None
        
        return has_correct_len, does_match

    # Check if correct format of start or end time has been given
    if start_time_str is not None:
        check_format_r = check_format_time_str(start_time_str)
        assert check_format_r, "Provided start time does not have correct format. "
        start_time_dt = dt.strptime(start_time_str, "%Y-%m-%d %H:%M")
        
    if end_time_str is not None:
        check_format_r = check_format_time_str(end_time_str)
        assert check_format_r, "Provided end time does not have correct format."
        end_time_dt = dt.strptime(end_time_str, "%Y-%m-%d %H:%M")
        
    # Load PyPSA network, the graph of the subnetwork and its matrices
    if verbose:
        print("Loading PyPSA Network and converting it to NetworkX Graph.\n")
        
    if use_sclopf:
        path_to_pypsa_network = path_to_pypsa_network_sclopf
        full_path_to_file = path_to_pypsa_network + f'sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l:.1f}-2920SEG.nc'
        
        full_path_to_cascades = path_to_cascades_sclopf + f'system_splits_Co2L{co2l}_n{n_nodes}.pklz'
    else:
        path_to_pypsa_network = path_to_pypsa_network_lopf
        full_path_to_file = path_to_pypsa_network + f'elec_s_{n_nodes}_ec_lv1.0_Co2L{co2l}-3H.nc'
        
        full_path_to_cascades = path_to_cascades_lopf + + f'system_splits_Co2L{co2l}_n{n_nodes}.pklz'
    network = data_handling.load_pypsa_network(full_path_to_file, use_sclopf)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)

    # Load cascade results
    with gzip.open(full_path_to_cascades, 'rb') as fh_casc:
        splitting_cascades = pickle.load(fh_casc)

    # Initialize results
    comp_cols = ['time_stamp', 'init_failure_0','init_failure_1',
                 'split_number','rot_energy', 'power_imbalance',
                 'load', 'rocof', 'load_share']
    '''component_props = pd.DataFrame(columns=['time_stamp', 'init_failure_0','init_failure_1',
                                            'split_number','rot_energy', 'power_imbalance',
                                            'load', 'rocof', 'load_share'],
                                index=[], dtype=float)'''

    splitting_cascades_dtkeys = {dt.strptime(key, "%Y-%m-%d %H:%M"): value 
                                 for key, value in splitting_cascades.items()}
    
    if verbose:
        print("Starting evaluation of System Splits.\n")
    did_cut_dict = False
    if start_time_str is not None or end_time_str is not None:
        if start_time_str is not None and end_time_str is not None:
            splitting_cascades_cut = {key: value for key, value in splitting_cascades_dtkeys.items() 
                                      if key >= start_time_dt and key <= end_time_dt}
            
        elif start_time_str is not None:
            splitting_cascades_cut = {key: value for key, value in splitting_cascades_dtkeys.items() 
                                      if key >= start_time_dt}
            
        elif end_time_str is not None:
            splitting_cascades_cut = {key: value for key, value in splitting_cascades_dtkeys.items() 
                                      if  key <= end_time_dt}

        splitting_cascades_dtkeys = splitting_cascades_cut
        did_cut_dict = True
    
    total_nr_splits = sum(len(vv) for vv in splitting_cascades_dtkeys.values())
    if eval_indicator_vectors:
        indicator_vectors = np.empty((total_nr_splits, nx_graph.number_of_nodes()), int)
    
    component_props_dict = dict()
    out_dict_key = 0   
    for timestamp, splits in tqdm(splitting_cascades_dtkeys.items(), disable=not show_progress):
        
        for ii, (init_failure, cascade) in enumerate(tqdm(splits.items(), leave=False, disable=not show_progress)):

            subgraphs = data_handling.get_subgraphs_from_edges(cascade,
                                                               nx_graph)
            
            observables_arr = subgraph_evaluation.evaluate_observables_for_subgraphs(subgraphs, network,
                                                                                 timestamp)
            #observables['init_failure_0'] = init_failure[0]
            #observables['init_failure_1'] = init_failure[1]
            #observables['split_number'] = ii
            
            for row in observables_arr:
                dict_out_ele = [timestamp, init_failure[0], init_failure[1],
                                ii, *row]
                component_props_dict[out_dict_key] = dict_out_ele
                out_dict_key += 1
                
            #component_props = component_props.append(observables, ignore_index=True)
            
            # Append properties and vectors such that component_props.iloc[ii] refers to 
            # indicator_vectors[ii]
            if eval_indicator_vectors:
                indicator_vectors[ii, :] = subgraph_evaluation.get_indicator_vectors_of_subgraphs(subgraphs, nx_graph)

    
        #component_props.to_hdf(save_path + f'component_properties_Co2L{co2l}_n{n_nodes}.h5',
        #                    key='df', mode= 'w')

    # Convert dictionary to component props pdDataFrame and save
    cut_path_str = ""
    if did_cut_dict:
        start_time_str_df = dt.strftime(min(component_props.time_stamp), "%Y-%m-%d_%H:%M")
        end_time_str_df = dt.strftime(max(component_props.time_stamp), "%Y-%m-%d_%H:%M")
        
        cut_path_str = (f"_from{start_time_str_df}" + 
                        f"to{end_time_str_df}")
        
    
    if eval_indicator_vectors:
            indi_vec_save_path = save_path + f'indicator_vectors_Co2L{co2l}_n{n_nodes}{cut_path_str}.pklz'
            with gzip.open(indi_vec_save_path) as fh_vec_out:
                pickle.dump(indicator_vectors, fh_vec_out)
    
    component_props = pd.DataFrame.from_dict(component_props_dict, orient="index",
                                             columns=comp_cols)
    save_df_path = save_path + f"component_properties_Co2L{co2l}_n{n_nodes}_dict"
    save_df_path += cut_path_str
        
    component_props.to_hdf(save_df_path + ".h5",
                           key='df', mode= 'w')
    
    if cfg.mattermost_url is not None:
        message_text = f"Evaluation of N={n_nodes}, C02_lvl={co2l} finished and results saved"
        send_mattermost_messages.post_message(message_text, cfg.mattermost_url)
    
    return component_props

    
if __name__ == "__main__":
    
    # Load arguments
    co2l_in = float(sys.argv[1]) 
    n_nodes_in = int(sys.argv[2])
    
    evaluate_cascade(co2l_in, n_nodes_in)
