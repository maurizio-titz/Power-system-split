#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module takes the PyPSA networks and
evaluates the line loadings in the undisturbed system."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_handling import load_pypsa_network, build_networkx_graph, \
    load_grid_matrices, load_effective_injections, load_snapshot_list

from utils.cascade_simulation import solve_lpf

from loguru import logger

from utils.config import path_to_vis_results_sclopf


def calculate_line_loadings_for_all_snapshots(co2lvl: float, n_nodes: int=600,
                                              condition_on_split: bool = True,
                                              blackout_threshold: float | None = None,
                                              verbose: bool = False) -> tuple[dict[str, np.ndarray], 
                                                                                                dict[str, np.ndarray]]:
    """Get the line loadings for each snapshot by solving
    the linear power flows. This return"""
    
    snapshot_list = load_snapshot_list(co2lvl=co2lvl)
    if condition_on_split:
        split_property_df = pd.read_hdf(path_to_vis_results_sclopf + 
                                        f"/split_properties_all_n{n_nodes}.h5", 
                                        index_col=0)
        sprops_co2l = split_property_df[split_property_df.index.get_level_values("co2l") == co2lvl]
        
        if blackout_threshold is None:
            split_snapshots = sprops_co2l.index.get_level_values("time_stamp").unique()
            
            snapshot_list_cut = [pd.Timestamp(xx) for xx in split_snapshots.values]
            
        else:
            mask_above_threshold = sprops_co2l['lost_load_share_blackout'] > blackout_threshold
            split_above_thres_snapshots = sprops_co2l[mask_above_threshold].index.get_level_values("time_stamp").unique()
            snapshot_list_cut = [pd.Timestamp(xx) for xx in split_above_thres_snapshots.values]
    
    
        is_cut_subset = set(snapshot_list_cut).issubset(set(snapshot_list))
        if not is_cut_subset:
            raise ValueError("The conditioned snapshots should be a subset of all snapshots!")
    
    injections_all_snapshot = load_effective_injections(co2lvl=co2lvl)
    I_m, B_d, num_parallels, line_limits = load_grid_matrices('0', co2lvl)
    
    LL = I_m.dot(B_d).dot(I_m.T)
    
    flow_snapshot_dict: dict[str, np.ndarray] = dict()
    line_loading_snapshot_dict: dict[str, np.ndarray] = dict()
    for snapshot_r in tqdm(snapshot_list, desc="Line Loading", 
                           disable=not verbose):
        
        key_now = snapshot_r.strftime("%Y-%m-%d %H:%M")
        
        P0_now = injections_all_snapshot[snapshot_r]
        
        flows_now = solve_lpf(P0_now, B_d, I_m, LL)
        line_loading_now = np.divide(abs(flows_now), line_limits)
        
        flow_snapshot_dict[key_now] = flows_now
        line_loading_snapshot_dict[key_now] = line_loading_now
    
    if verbose:
        logger_cond_str = "conditioned on split" if condition_on_split else ""
        if blackout_threshold is not None:
            logger_cond_str += f"(BlackOutThres {blackout_threshold:.3f})"
        logger_str = f"Found Line Loading for Co2L={co2lvl}" + logger_cond_str
        
        logger.info(logger_str)
        
    return flow_snapshot_dict, line_loading_snapshot_dict
