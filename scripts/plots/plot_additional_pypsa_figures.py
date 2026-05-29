#!/usr/bin/env python

"""Plot figure that give more insights into the scenarios
generated with PyPSA."""

import os 

import gzip
import pickle

from utils.data_handling import \
    (load_pypsa_network, build_networkx_graph, 
     get_co2_levels)

from utils.plot_style import *
setup_matplotlib_style()

from utils.config import path_to_figures_sclopf, path_to_plot_data

from utils import calculate_line_loadings

import networkx as nx

from tqdm import tqdm
from loguru import logger

from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors



def histogram_lineloading_accross_co2lvls(n_nodes: int = 600, 
                                          calc_again: bool = False,
                                          condition_on_split: bool = False,
                                          n_bins: int = 100, 
                                          xscale_log: bool = False,
                                          bin_range: tuple[float, float] = (0., 1.),
                                          x_lim: tuple[float, float] = (.5, 1.),
                                          save_prefix: str | None = None):
    """Draw histograms of line loadings for different co2 levels
    """
    
    condition_str = "_givenSplit" if condition_on_split else ""
    
    path_to_results = os.path.join(path_to_plot_data , 
                                   f"data_line_loading_n{n_nodes}{condition_str}.pklz")
    
    available_co2_lvls = get_co2_levels(n_nodes=n_nodes)
    
    if not os.path.exists(path_to_results) or calc_again:
        # Find line loading for all co2_lvls
        
        flow_n_lineloading_dict: dict[float, tuple[dict, dict]] = dict()
        
        for co2l_r in tqdm(available_co2_lvls, desc="Co2lvls"):
            flow_snapshot_dict, line_loading_dict = calculate_line_loadings.calculate_line_loadings_for_all_snapshots(co2l_r, n_nodes=n_nodes,
                                                                              condition_on_split=False)
        
            flow_n_lineloading_dict[co2l_r] = flow_snapshot_dict, line_loading_dict
        
        with gzip.open(path_to_results, "wb") as fh_out:
            pickle.dump(flow_n_lineloading_dict, fh_out)
        
    else:
        with gzip.open(path_to_results, "rb") as fh_in:
            flow_n_lineloading_dict = pickle.load(fh_in)
        
        if not(set(flow_n_lineloading_dict) == set(available_co2_lvls)):
            raise ValueError("Dictionary should contain all available co2 levels. Please check if all PyPSA Sclopf Scenarios" + 
                             " are in the right place and calcualte line loading again!")
    
    # Plot it
    fig, ax = plt.subplots()
    
    cmap_co2 = plt.get_cmap("cividis").copy()
    
    if xscale_log:
        line_loading_bins = np.logspace(np.log10(min(bin_range)), np.log10(bin_range), n_bins, endpoint=True)
    else:
        line_loading_bins = np.linspace(min(bin_range), max(bin_range), n_bins, endpoint=True)
    
    for key_r, [flow_dict_r, ll_dict_r] in tqdm(flow_n_lineloading_dict.items(), desc="Plotting hists..."):
        
        color_r = cmap_co2(np.where(available_co2_lvls == key_r)[0][0] / (len(co2ls) - 1))
        
        all_line_loadings = [item for sublist in ll_dict_r.values() for item in sublist]
        ax.hist(all_line_loadings, bins=line_loading_bins,
                histtype='step', density=True,
                color=color_r, 
                linewidth=LINE_WIDTH, label=f"{key_r * 100}\\%")
    
    # Aesthetics
    if xscale_log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Line Loading $\\ell_{\\text{load}}$")
    ax.set_ylabel("$P(\\ell_{\\text{load}})$")
    
    ax.set_xlim(min(x_lim), max(x_lim))
    
    ax.legend(loc="lower left",
              title="CO$_2$ level [\\% of 1990]",
              fontsize=14, title_fontsize=14)
    
    fname_fig = f"line_loading_histogram_n{n_nodes}"
    if xscale_log:
        fname_fig += "_xlog"
        
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return


def graph_with_num_parallel(co2lvl: float = .6, n_nodes: int = 600, 
                            snet_index: str = '0',
                            edge_width: float = 2.,
                            log_scale: bool = False,
                            vlims_edges: tuple[float, float] | None = (1e-1, 1e1),
                            save_prefix: str | None = None):
    """Plot the graph of the CE network with num_parallel on the acis"""
    
    net = load_pypsa_network(co2lvl=co2lvl, n_nodes=n_nodes)
    
    nx_graph = build_networkx_graph(net, snet_index=snet_index)
    pos_graph = nx.get_node_attributes(nx_graph, "pos")
    num_parallel_vals = np.array([xx for _, _, xx in nx_graph.edges(data="num_parallel")])
    
    if any(num_parallel_vals < -1e-8):
        raise ValueError("Num Parallel values should be >= 0")
    if log_scale:
        cval_edges: list[float] = [np.log10(xx) if xx > 1e-12 else -np.inf 
                      for xx in num_parallel_vals ]
    else:
        cval_edges: list[float] = list(num_parallel_vals)
    
    # Plot it
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    """nodes = nx.draw_networkx_nodes(nx_graph, pos=pos_graph, 
                                   ax=ax, node_color="k",
                                   node_size=0)
    nodes.set_zorder(-1)"""
    
    
    cmap_edges = plt.get_cmap("viridis").copy()
    
    edges = nx.draw_networkx_edges(nx_graph, pos=pos_graph, 
                                   edge_color=cval_edges,
                                   edge_cmap=cmap_edges,
                                   width=edge_width)
    
    edges.set_zorder(0)
    
    if vlims_edges is None:
        vlims_edges = (min(num_parallel_vals), max(num_parallel_vals))
        
    
    ## Colorbar
    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes([ax_pos.x1-.075, ax_pos.y0, 
                            .025, ax_pos.height])
    if log_scale:
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, 
                                        norm=mplcolors.LogNorm(vmin=min(vlims_edges), 
                                                               vmax=max(vlims_edges)))
    else:
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, 
                                        norm=mplcolors.Normalize(vmin=min(vlims_edges), 
                                                                 vmax=max(vlims_edges)))
    
    cb_edges = fig.colorbar(sm_edge, cax=cbar_ax)
    cb_edges.set_label("$\\ell_{\\text{parallel}}$", size=AXIS_LABEL_FONTSIZE*1.5)
    
    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    
    fname_fig = f"graph_num_parallel_co2l{co2lvl}_n{n_nodes}"
    
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return



