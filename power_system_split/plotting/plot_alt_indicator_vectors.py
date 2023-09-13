#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot details on the indicators values"""

import pandas as pd
import numpy as np

import gzip
import pickle

import cartopy.crs as cartopy_crs
import networkx as nx

from utils.data_handling import (load_pypsa_network, build_networkx_graph,
                                 matrix_indices_to_nx_edges, nx_edges_to_matrix_indices,
                                 get_matrices_from_nx_graph)
from utils.cascade_simulation import calc_possible_double_line_failures


import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 24,
    'axes.labelsize': 24,'axes.titlesize': 28, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

def draw_map(graph: nx.Graph, xlims: tuple = (-11, 29), ylims: tuple = (36, 59),
             save_prefix=None):
    
    ax = plt.axes(projection=cartopy_crs.PlateCarree())
    ax.coastlines()
    fig = plt.gcf()
    
    pos_nx = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx_nodes(graph, pos_nx, node_size=20, ax=ax)
    nx.draw_networkx_edges(graph, pos_nx, width=2., ax=ax)
    
    # Aesthetics 
    ax.set_xlim(left=min(xlims), right=max(xlims))
    ax.set_ylim(bottom=min(ylims), top=max(ylims))
    
    ## Remove background color and frame
    ax.axis('off')
    
    if save_prefix is None:
        plt.show()
    else:
        fig_path = "plots/" + save_prefix
        fig.savefig(fig_path, bbox_inches='tight', transparent=True)
        
        fig.clear()
        plt.close(fig)
        
    return
    

def calc_likelihood_failure_alt_indicator(co2_lvl, n_nodes):
    """"""
    
    # Load PyPSA and generate networkx graph
    pypsa_net = load_pypsa_network(co2_lvl, n_nodes,
                                   'data/European_networks_sclopf/')
    nx_graph = build_networkx_graph(pypsa_net, snet_index=0)# snet_idx is Europe
    
    # Find number of total experiments
    _, _, num_paralles, _ = get_matrices_from_nx_graph(nx_graph)
    bridge_idxs = nx_edges_to_matrix_indices(nx.bridges(nx_graph),nx_graph)
    n_2_failures = calc_possible_double_line_failures(num_paralles,
                                                      ignored_idxs=bridge_idxs)
    snapshot_weights = pypsa_net.snapshot_weightings.generators
    number_total_experiments = int(len(n_2_failures) * 
                                   snapshot_weights.sum())
    
    # Load edge split indicator vector
    fpath_indicator_vector = ("results/sclopf/indicator_vectors_rocof_lshare_edges/" + 
                              "indicator_vector_active_edges_Co2l" + 
                              "{0:.1f}_n{1}.pklz".format(co2_lvl, n_nodes))
    with gzip.open(fpath_indicator_vector) as fh_indi_in:
        [edge_names_ls, edge_index_ls, index_tuple_splits,
         active_edges_arr] = pickle.load(fh_indi_in)
    
    active_edges_arr /= number_total_experiments
    time_stamps = pd.to_datetime([xx[0] for xx in index_tuple_splits])
    unique_times = np.unique(time_stamps)
    
    for uni_time_r in unique_times:
        snapshot_weight_r = snapshot_weights[uni_time_r]
        idxs_time = np.where(time_stamps == uni_time_r)[0]
        active_edges_arr[idxs_time] *= snapshot_weight_r
        
    return active_edges_arr
    
    # Output dictionaries with likelihood as values and links as keys
    likelihood_primary = {(uu, vv): 0.0 for uu, vv in nx_graph.edges()}
    likelihood_secondary = {(uu, vv): 0.0 for uu, vv in nx_graph.edges()}
      
    

def plot_initial_trigger_likelihood():
    """Plot the likelihood of link triggering an initial failure."""
    return


def plot_secondary_failure_likelihood():
    """Plot the likelihood of a linked being a secondary failure."""
    
    return


def plot_histogram_number_failed_links():
    """Plot the histogram of number failed links."""

    return


def plot_rocof_failure_map():
        
    return
