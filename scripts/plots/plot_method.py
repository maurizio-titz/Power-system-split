#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Script that generate the parts of the figure describing the approach.
The individual parts were used to generate a figure using inkscape."""

import os
import numpy as np
import networkx as nx

import gzip
import pickle

from tqdm import tqdm

from matplotlib import pyplot as plt
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.Dark2.colors)

from networkx.drawing.nx_agraph import to_agraph 

from utils import data_handling
from utils.config import path_to_figures_sclopf

from loguru import logger


def setup_network_with_layout(case_str='norm', 
                              save_fig: bool = True):
    """Setup the network"""

    gra = nx.DiGraph()

    link_weight_ls = [0, .5, .7, 1., .6, .6, .6]
    vmin = min(link_weight_ls)
    vmax = max(link_weight_ls)
    
    if case_str == 'norm':
        edge_weight = plt.cm.viridis(link_weight_ls)

    elif case_str == 'sec':
        edge_weight = plt.cm.viridis(link_weight_ls)
        edge_weight[3] = [0.66,0.66,0.66,1.]
        edge_weight[4] = [0.66,0.66,0.66,1.]
        edge_weight[6] = [1,0.2,0.2,1]
        #edge_weight[2] = [1,0.2,0.2,1]
        
    elif case_str == 'split':
        edge_weight = plt.cm.viridis(link_weight_ls)
        edge_weight[4] = [0.66,0.66,0.66,1.]
        edge_weight[6] = [0.66,0.66,0.66,1.]
        edge_weight[3] = [0.66,0.66,0.66,1.]

    else:
        raise IOError

    
    gra.add_edge(1, 2, weight=edge_weight[0])
    gra.add_edge(1, 3, weight=edge_weight[1])
    gra.add_edge(2, 3, weight=edge_weight[2])
    gra.add_edge(3, 4, weight=edge_weight[3])
    gra.add_edge(2, 5, weight=edge_weight[4])
    gra.add_edge(4, 5, weight=edge_weight[5])
    
    gra.add_edge(5, 2, weight=edge_weight[6])

    pos = {1: (1, 5), 2: (2, 3.75), 3:(0.25, 3.5), 4:(0, 0), 5:(1, 1.)}

    idx_curved_edges = [3, 6]
    idx_norm_edges = [0, 1, 2, 4, 5]
    
    
    #edges, weights_norm = zip(*[list(nx.get_edge_attributes(gra, 'weight').items())[xx] for xx in idx_norm_edges])
    edge_exist_norm, weights_exist_norm = zip(*[(xx, weights) for xx, weights
                                 in [list(nx.get_edge_attributes(gra, 'weight').items())[kk] for kk in idx_norm_edges]
                                 if not np.equal(weights, np.asarray([.66,.66,.66,1.])).all()])
    

    if len(edge_exist_norm) < len(idx_norm_edges):
        edge_nexist_norm, weights_nexist_norm = zip(*[(xx, weights) for xx, weights
                                 in [list(nx.get_edge_attributes(gra, 'weight').items())[kk] for kk in idx_norm_edges]
                                 if np.equal(weights, np.asarray([.66,.66,.66,1.])).all()])
        
    try:
        edge_exist_curved, weights_exist_curved = zip(*[(xx, weights) for xx, weights
                                 in [list(nx.get_edge_attributes(gra, 'weight').items())[kk] for kk in idx_curved_edges]
                                 if not np.equal(weights, np.asarray([.66,.66,.66,1.])).all()])
    except ValueError:
        edge_exist_curved, weights_exist_curved = [], []
    

    if len(edge_exist_curved) < len(idx_curved_edges):
        edge_nexist_curved, weights_nexist_curved = zip(*[(xx, weights) for xx, weights
                                 in [list(nx.get_edge_attributes(gra, 'weight').items())[kk] for kk in idx_curved_edges]
                                 if np.equal(weights, np.asarray([.66,.66,.66,1.])).all()])

    node_opts = {
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 7,
    }

    edge_opts = {
        "width": 10,
    }

    dashed_linestyle = (0, (1, 2))
    nx.draw_networkx_nodes(gra, pos, **node_opts)
    
    # Normal Edges
    nx.draw_networkx_edges(gra, pos, edgelist=edge_exist_norm, edge_color=list(weights_exist_norm), **edge_opts)
    if len(edge_exist_norm) < len(idx_norm_edges):
        collection = nx.draw_networkx_edges(gra, pos, edgelist=edge_nexist_norm, edge_color=list(weights_nexist_norm), style='dashed',**edge_opts)
        for patch in collection:
            patch.set_linestyle(dashed_linestyle)
    
    # Double Edge
    nx.draw_networkx_edges(gra, pos, edgelist=edge_exist_curved,
                           edge_color=list(weights_exist_curved), connectionstyle='arc3, rad = 0.25',
                           **edge_opts)
    if len(edge_exist_curved) < len(idx_curved_edges):
        collection = nx.draw_networkx_edges(gra, pos, edgelist=edge_nexist_curved, edge_color=list(weights_nexist_curved),
                                            connectionstyle='arc3, rad = 0.25' ,style='dashed',**edge_opts)
        for patch in collection:
            patch.set_linestyle(dashed_linestyle)


    fig = plt.gcf()
    ax = plt.gca()
    
    ax.margins(0.2)
    plt.axis('off')

    if case_str == 'norm':
        fig2 = plt.figure(figsize=(.5, 8))
        ax = plt.gca()
        img = plt.imshow(np.array([[vmin, vmax]]))
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, .2, .8, .6])
        cbar = plt.colorbar(orientation='vertical', cax=cax, ticks=[0, 1])
        #cbar.set_label("$F_{ij}$", fontsize=30)
        cbar.ax.set_yticklabels(['', ''], size=30)

    if save_fig:
        fig_path = case_str + "_network.svg"
        fig.savefig(os.path.join(path_to_figures_sclopf, fig_path), bbox_inches='tight', transparent=True)

        fig.clear()
        plt.close(fig)

        if case_str == 'norm':
            fig_path_cbar = 'colorbar.svg'
            fig2.savefig(os.path.join(path_to_figures_sclopf, fig_path_cbar), transparent=True)
            fig2.clear()
            plt.close(fig2)

    else:
        plt.show()
        
    return gra, edge_exist_curved



def ax_plot_split_network(nx_graph: nx.Graph, ax: plt.Axes | None, 
                          failed_links: list[int],
                          node_size: int = 100,
                          node_edge_width: float = 2,
                          edge_width: float = 3.):
    """Plot the split network components on given axis from networkx graph and failed links.
    Args:
        nx_graph (nx.Graph): NetworkX graph of the network.
        ax (plt.Axes | None): Matplotlib axis to plot on. If None, skip plotting.
        failed_links (list[int]): List of failed link indices.
        node_size (int): Size of the nodes.
        node_edge_width (float): Width of the node edges.
        edge_width (float): Width of the edges.
    Returns:
        failed_edges_names (list): List of failed edge names.
        size_split_components (list): List of sizes of the split components."""
    
    nx_graph_int = nx_graph.copy()
    
    # Split nx graph in subgraphs
    edge_list = list(nx_graph_int.edges)
    pos = nx.get_node_attributes(nx_graph_int, 'pos')

    failed_edges_names = [edge_list[xx] for xx in failed_links]

    nx_graph_int.remove_edges_from(failed_edges_names)

    assert nx.is_connected(nx_graph_int) is False, "Network is still connected!"

    subgraphs = [nx_graph_int.subgraph(c).copy() 
                 for c in nx.connected_components(nx_graph_int)]

    # Plot subgraphs with different colors
    node_kwargs = {
        'node_size': node_size,
        'edgecolors': 'black',
        'linewidths': node_edge_width,
    }
    
    edges_kwargs = {
        'width': edge_width,
    }

    # Plot failed edges in gray dashed lines in background
    if ax is not None:
        nx.draw_networkx_edges(nx_graph, pos, edgelist=failed_edges_names,
                               edge_color='gray', style='dashed',
                               **edges_kwargs)
    
    size_split_components = [len(subgraph.nodes) for subgraph in subgraphs]
    for idx_sg, subgraph in enumerate(subgraphs):
        if ax is not None:
            nx.draw_networkx_nodes(subgraph, pos, ax=ax, 
                node_color=f"C{idx_sg}", **node_kwargs)
            
            nx.draw_networkx_edges(subgraph, pos, ax=ax, 
                edge_color=f"k", **edges_kwargs)

    return failed_edges_names, size_split_components


def ax_plot_full_network(nx_graph: nx.Graph, ax: plt.Axes,
                         color_nodes: str = 'gray',
                         node_edge_width: float = 2,
                         node_size: int = 100,
                         edge_width: float = 3.,
                         fail_color: str | tuple = 'r',
                         failed_link_names: list | None = None):
    """Plot the full network on given axis from networkx graph.
    If failed_link_names is given, plot them in fail_color.
    Args:
        nx_graph (nx.Graph): NetworkX graph of the network.
        ax (plt.Axes): Matplotlib axis to plot on.
        color_nodes (str): Color of the nodes.
        node_edge_width (float): Width of the node edges.
        node_size (int): Size of the nodes.
        edge_width (float): Width of the edges.
        fail_color (str | tuple): Color of the failed edges.
        failed_link_names (list | None): List of failed link names to plot in fail_color.
    Returns:
        None
    """
    
    pos = nx.get_node_attributes(nx_graph, 'pos')
    
    
    nx.draw_networkx_nodes(nx_graph, pos, node_color=color_nodes,
                           edgecolors='black', linewidths=node_edge_width,
                           node_size=node_size, ax=ax)
    if failed_link_names is not None:
        # Plot failed edges in gray dashed lines in background
        nx.draw_networkx_edges(nx_graph, pos, edgelist=failed_link_names,
                               edge_color=fail_color,
                               width=edge_width, ax=ax)
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, width=edge_width,
                               edgelist=[edge for edge in nx_graph.edges if edge not in failed_link_names])
        
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, width=edge_width)

    return


def plot_entire_network_n_split_components(trigger_tuple: tuple[int, int] = (387, 453),
                                           path_to_pypsa_network = "sclopf-elec_s_600_ec_lv1.0_Co2L0.5-2920SEG.nc",
                                           path_to_cascade_results = "multi2_system_splits_Co2L0.5_n600_2013-12-13 18:00:00_.pklz",
                                           snapshot_time = '2013-01-13 12:00',
                                           save_fig: bool = True, 
                                           rasterize: bool = False):
    """Plot entire network and split components for a given trigger line.
    Args:
        trigger_tuple (tuple[int, int]): Tuple of line indices that trigger the cascade.
        path_to_pypsa_network (str): Path to the PyPSA network file.
        path_to_cascade_results (str): Path to the cascade results file.
        snapshot_time (str): Snapshot time to consider for the cascade results.
        save_fig (bool): Whether to save the figures or show them.
    Returns:
        None
    """

    # Load network
    pypsa_nw = data_handling.load_pypsa_network_from_path(path_to_pypsa_network, 
                                                          use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_nw, snet_index=0)

    

    # Load cascade results
    with gzip.open(path_to_cascade_results, 'rb') as f:
        cascade_results = pickle.load(f)
    
    snapshot_cascade_results = cascade_results[snapshot_time]
    
    failed_links = snapshot_cascade_results[trigger_tuple]
    
    print(f"Trigger line: {trigger_tuple}, Failed links: {failed_links}")
    
    # Plot full network
    fig_full, ax_full = plt.subplots(figsize=(20, 15))
    ax_plot_full_network(nx_graph, ax_full, 
                         failed_link_names=[list(nx_graph.edges)[xx] for xx in failed_links])
    
    fig_split, ax_split = plt.subplots(figsize=(20, 15))
    ax_plot_split_network(nx_graph, ax_split, failed_links)
    
    # Aesthetics
    ax_full.axis('off')
    ax_split.axis('off')
    
    if rasterize:
        ax_full.set_rasterized(True)
        ax_split.set_rasterized(True)
    
    dpi_fig = 150 if rasterize else None
    
    if save_fig:
        
        fig_path_full = f"entire_network_trigger_{trigger_tuple[0]}_{trigger_tuple[1]}"
        if rasterize:
            fig_path_full += "_rasterized.png"
        else:
            fig_path_full += ".svg"            
        fig_full.savefig(os.path.join(path_to_figures_sclopf, fig_path_full), bbox_inches='tight',
                         transparent=True, dpi=dpi_fig)
        fig_full.clear()
        plt.close(fig_full)
        
        fig_path_split = f"split_network_trigger_{trigger_tuple[0]}_{trigger_tuple[1]}"
        if rasterize:
            fig_path_split += "_rasterized.png"
        else:
            fig_path_split += ".svg"
        fig_split.savefig(os.path.join(path_to_figures_sclopf, fig_path_split), bbox_inches='tight',
                          transparent=True, dpi=dpi_fig)
        fig_split.clear()
        plt.close(fig_split)
    else:
        plt.show()
    
    return


def iterate_over_all_system_split_trigger(path_to_pypsa_network: str = "sclopf-elec_s_600_ec_lv1.0_Co2L0.5-2920SEG.nc",
                                          path_to_cascade_results: str = "multi2_system_splits_Co2L0.5_n600_2013-12-13 18:00:00_.pklz",
                                          snapshot_time: str = '2013-01-13 12:00'):
    """Iterate over all system split triggers and 
    give out the splits if they involve a central link between Spain and France.
    This is done to identify a split of the Iberian Peninsula from the rest of Europe for
    illustration purposes."""
    
    # Load network
    pypsa_nw = data_handling.load_pypsa_network_from_path(path_to_pypsa_network, 
                                                          use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_nw, snet_index='0')
    
    edge_list = list(nx_graph.edges)
    # Load cascade results
    with gzip.open(path_to_cascade_results, 'rb') as f:
        cascade_results = pickle.load(f)
    
    snapshot_cascade_results = cascade_results[snapshot_time]
    
    results_ls = list()
    
    for trigger_line, failed_links in tqdm(snapshot_cascade_results.items()):
        # Search for iberian peninsula split

        line_es_fr = ("ES1 4", "FR1 4")
        line_names = [edge_list[xx] for xx in failed_links]
        if line_es_fr not in line_names and line_es_fr[::-1] not in line_names:
            continue

        fedges_names, size_subgraphs = ax_plot_split_network(nx_graph, None,
                                                             failed_links)
        results_ls.append((trigger_line, fedges_names, size_subgraphs))

    return results_ls


if __name__ == "__main__":

    # Principle netwok which splits into two compoments
    logger.info("Starting to generate parts of method figure. Need to be put togehter, e.g., with inkscape!")
    setup_network_with_layout(case_str='norm', save_fig=True)
    setup_network_with_layout(case_str='sec', save_fig=True)
    setup_network_with_layout(case_str='split', save_fig=True)
    
    logger.info("Finished parts of small abstract network")
    
    #plot_entire_network_n_split_components()