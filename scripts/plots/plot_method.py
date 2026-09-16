#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Script that generate the parts of the figure describing the approach.
The individual parts were used to generate a figure using inkscape.
The same holds for the parts of the graphical abstract."""

import os

os.environ["PLOT_STYLE"] = "joule"

import glob
import warnings
import numpy as np
import pandas as pd
import networkx as nx

import gzip
import pickle

from tqdm import tqdm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Not via ``pypsa.plot``: that alias is deprecated and goes away in pypsa 1.0.
from pypsa.plot.maps.static import add_legend_circles

from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib import ticker as mticker
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.Dark2.colors)

from networkx.drawing.nx_agraph import to_agraph

from utils import data_handling
from utils.config import (path_to_cascade_results_sclopf,
                          path_to_figures_sclopf,
                          path_to_inertia_mitigation_results_sclopf,
                          path_to_plot_data,
                          path_to_pypsa_network_sclopf,
                          path_to_vis_results_sclopf)
from utils.plot_style import (add_panel_label, map_width_for_height,
                              set_axes_size_in_inches, setup_matplotlib_style)

from scripts.plots.plot_clustering_analysis import create_single_cluster_plot_from_data
# Same carrier aggregation the Fig. 1 maps use, so the wedges of the scenario
# map below are built from exactly the same groups.
from scripts.plots.plot_combined_generation_storage import aggregate_carriers

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
                          node_size: float = 100,
                          node_edge_width: float = 0.,
                          edge_width: float = 3.,
                          failed_edge_width_scale: float = 1.5,
                          component_cmap: str = "Dark2"):
    """Plot the split network components on given axis from networkx graph and failed links.
    Args:
        nx_graph (nx.Graph): NetworkX graph of the network.
        ax (plt.Axes | None): Matplotlib axis to plot on. If None, skip plotting.
        failed_links (list[int]): List of failed link indices.
        node_size (int): Size of the nodes.
        node_edge_width (float): Width of the outline around each node. 0 draws
            no outline, which is the default.
        edge_width (float): Width of the edges.
        failed_edge_width_scale (float): How much wider the dashed lines of the
            failed corridors are drawn than the intact ones. They are the point
            of the panel and are dashed and grey, which already makes them
            lighter than the solid black lines around them.
        component_cmap (str): Qualitative colormap the components are colored
            from. Taken explicitly instead of through the "C0", "C1", ... cycle,
            because importing ``utils.plot_style`` resets the cycle to the
            matplotlib default.
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
        'edgecolors': 'black' if node_edge_width else 'none',
        'linewidths': node_edge_width,
    }
    
    edges_kwargs = {
        'width': edge_width,
    }

    # Plot failed edges in gray dashed lines in background
    if ax is not None:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, edgelist=failed_edges_names,
                               edge_color='gray', style='dashed',
                               width=edge_width * failed_edge_width_scale)
    
    cmap_components = plt.get_cmap(component_cmap)

    size_split_components = [len(subgraph.nodes) for subgraph in subgraphs]
    for idx_sg, subgraph in enumerate(subgraphs):
        if ax is not None:
            nx.draw_networkx_nodes(subgraph, pos, ax=ax,
                node_color=[cmap_components(idx_sg % cmap_components.N)],
                **node_kwargs)
            
            nx.draw_networkx_edges(subgraph, pos, ax=ax, 
                edge_color=f"k", **edges_kwargs)

    return failed_edges_names, size_split_components


def ax_plot_full_network(nx_graph: nx.Graph, ax: plt.Axes,
                         color_nodes: str = 'gray',
                         node_edge_width: float = 0.,
                         node_size: float = 100,
                         edge_width: float = 3.,
                         failed_edge_width_scale: float = 1.5,
                         fail_color: str | tuple = 'r',
                         failed_link_names: list | None = None):
    """Plot the full network on given axis from networkx graph.
    If failed_link_names is given, plot them in fail_color.
    Args:
        nx_graph (nx.Graph): NetworkX graph of the network.
        ax (plt.Axes): Matplotlib axis to plot on.
        color_nodes (str): Color of the nodes.
        node_edge_width (float): Width of the outline around each node. 0 draws
            no outline, which is the default, as in 'ax_plot_split_network'.
        node_size (int): Size of the nodes.
        edge_width (float): Width of the edges.
        failed_edge_width_scale (float): How much wider the failed corridors are
            drawn than the intact ones, as in 'ax_plot_split_network'.
        fail_color (str | tuple): Color of the failed edges.
        failed_link_names (list | None): List of failed link names to plot in fail_color.
    Returns:
        None
    """
    
    pos = nx.get_node_attributes(nx_graph, 'pos')
    
    
    nx.draw_networkx_nodes(nx_graph, pos, node_color=color_nodes,
                           edgecolors='black' if node_edge_width else 'none',
                           linewidths=node_edge_width,
                           node_size=node_size, ax=ax)
    if failed_link_names is not None:
        # Failed corridors wider than the rest of the grid.
        nx.draw_networkx_edges(nx_graph, pos, edgelist=failed_link_names,
                               edge_color=fail_color,
                               width=edge_width * failed_edge_width_scale, ax=ax)
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, width=edge_width,
                               edgelist=[edge for edge in nx_graph.edges if edge not in failed_link_names])
        
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, width=edge_width)

    return


def get_failed_links_of_trigger(snapshot_cascade_results: dict,
                                trigger_tuple: tuple[int, int]) -> list[int]:
    """Look up the failed links of one trigger in the splits of a single snapshot.

    The keys of the cascade results carry the outage weighting next to the line
    index (``((line, weight), (line, weight))``) and the values the failed links
    next to the number of resulting components, hence the unpacking here.

    Args:
        snapshot_cascade_results (dict): Splits of one snapshot, keyed by trigger.
        trigger_tuple (tuple[int, int]): The two triggering line indices.

    Returns:
        list[int]: Indices of the links that failed for this trigger.
    """

    for trigger_key, (failed_links, _) in snapshot_cascade_results.items():
        if set(line for line, _ in trigger_key) == set(trigger_tuple):
            return list(failed_links)

    raise KeyError(f"Trigger {trigger_tuple} does not cause a split in this snapshot.")


def add_geomap_to_ax(ax: plt.Axes,
                     geomap_color: str = "0.65",
                     geomap_linewidth: float = 4.5,
                     resolution: str = "50m"):
    """Draw country borders and coastlines behind the network on ``ax``.

    The same two Natural Earth features, at the same 50 m resolution, that
    pypsa puts under the scenario map of 'plot_generation_map_scenario' -- see
    ``NetworkMapPlotter.add_geomap_features`` in ``pypsa/plot/maps/static.py``,
    which is what ``geomap_colors`` there ends up in. Land and ocean are left
    out: pypsa only fills them when asked, and the map panels are saved
    transparent.

    ``ax`` has to be a ``cartopy`` geographic axes, and its limits have to be
    final -- the features are clipped to them.

    Args:
        ax (plt.Axes): Geographic axes (``ccrs.PlateCarree()``) to draw on.
        geomap_color (str): Colour of both features. The "0.65" of the scenario
            map, light enough that the network stays the darkest thing on the
            panel.
        geomap_linewidth (float): Width of both features in points. pypsa fixes
            this at 0.3 pt, but it fixes it at the size the panel is *drawn*,
            and these two panels are drawn at ~15.6 x 15 in (the 20 x 15 in
            canvas cropped to the equal-aspect map) to be placed at about an
            inch in inkscape. 4.5 pt is that 0.3 pt divided by the ~1/14 the
            panel is scaled by. Rescale it with whatever size the panel
            actually gets, otherwise the borders come out hairline or heavy.
        resolution (str): Natural Earth resolution, "10m", "50m" or "110m".
            pypsa's default is "50m" and the scenario map takes it as is.

    Returns:
        None
    """
    for feature in (cfeature.BORDERS, cfeature.COASTLINE):
        ax.add_feature(feature.with_scale(resolution),
                       linewidth=geomap_linewidth,
                       edgecolor=geomap_color,
                       # Behind the network: networkx draws its edges at
                       # zorder 1 and its nodes above them.
                       zorder=0)

    return


def node_size_for_data_diameter(ax: plt.Axes, diameter_deg: float) -> float:
    """``node_size`` in points squared for a node ``diameter_deg`` degrees wide.

    ``draw_networkx_nodes`` sizes its markers by AREA in points squared, which
    pins the node to the figure rather than to the map: the same value gives a
    different-looking node on a 20 in panel than on a 5 in one. The map panels
    of the graphical abstract are placed at whatever scale the layout wants, so
    what has to agree between them is how much of the map a node covers -- a
    diameter in degrees. This converts one into the other through the axes' own
    transform, so the limits AND the aspect have to be final before it is
    called.

    Args:
        ax (plt.Axes): Axes the nodes go on, limits and aspect already set.
        diameter_deg (float): Node diameter in degrees of longitude.

    Returns:
        float: Value to pass as ``node_size``.
    """
    fig = ax.get_figure()
    # The aspect only takes effect on the axes box at draw time, and the
    # transform is read straight after, so the draw has to happen first.
    fig.canvas.draw()

    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    pixels_per_deg = (ax.transData.transform((x0 + 1., y0))[0]
                      - ax.transData.transform((x0, y0))[0])
    points_per_deg = pixels_per_deg * 72. / fig.dpi

    return (diameter_deg * points_per_deg) ** 2


def plot_entire_network_n_split_components(trigger_tuple: tuple[int, int] = (387, 453),
                                           path_to_pypsa_network = os.path.join(
                                               path_to_pypsa_network_sclopf,
                                               "sclopf-elec_s_600_ec_lv1.0_Co2L0.5-2920SEG.nc"),
                                           path_to_cascade_results = os.path.join(
                                               path_to_cascade_results_sclopf,
                                               "system_splits_Co2L0.5_n600.pklz"),
                                           snapshot_time = '2013-01-13 12:00',
                                           node_diameter_deg: float = 0.8,
                                           map_pad_deg: float = 1.,
                                           geomap: bool = True,
                                           geomap_color: str = "0.65",
                                           geomap_linewidth: float = 0.3,
                                           panel_box_in: tuple = (1.125, 0.879),
                                           fit_box: bool = True,
                                           box_fill: float = 0.98,
                                           plot_dir: str | None = None,
                                           edge_width: float = 0.2,
                                           save_fig: bool = True,
                                           rasterize: bool = False):
    """Plot entire network and split components for a given trigger line.

    Both panels are put on one scale and on equal aspect before anything is
    drawn. Left to autoscale they end up neither: each panel fits itself to its
    own content, which made the split panel come out 4 % wider than the full
    one, and latitude gets stretched against longitude by about a third, so a
    round marker covers an elliptical patch of map. Neither survives being
    placed next to the cluster map of the abstract, which is equal-aspect.

    Both panels stand on the coastlines and country borders of the scenario map
    ('plot_generation_map_scenario'), so the three panels of the abstract share
    a geography. That is what makes the axes geographic ones: the positions were
    already longitude and latitude, but only a ``cartopy`` axes knows it and can
    put the Natural Earth features under them.

    Args:
        trigger_tuple (tuple[int, int]): Tuple of line indices that trigger the cascade.
        path_to_pypsa_network (str): Path to the PyPSA network file.
        path_to_cascade_results (str): Path to the cascade results file.
        snapshot_time (str): Snapshot time to consider for the cascade results.
        node_diameter_deg (float): Node diameter in degrees, the unit the map
            panels of the abstract share. The default matches the pie nodes of
            'create_single_cluster_plot_from_data'.
        map_pad_deg (float): Margin in degrees around the node positions, again
            as in the cluster map.
        geomap (bool): Whether to draw the borders and coastlines at all. False
            gives back the bare graph panels these figures used to be.
        geomap_color (str): Colour of borders and coastlines, see
            'add_geomap_to_ax'.
        geomap_linewidth (float): Their width in points at the size the panels
            are drawn here, see 'add_geomap_to_ax'. 0.3 pt is what pypsa draws
            them at on the scenario map, so the three panels of the top row
            carry the same coastline.
        panel_box_in (tuple): Box in inches the panel has to fit into. The map
            is drawn as large as fits, and the canvas is made exactly that size,
            so the svg carries no margin and goes into inkscape at 100 %.
        fit_box (bool): Whether the map window is opened up to the proportions
            of 'panel_box_in' so the map fills the box. The grid is 1.7 times as
            wide as it is tall and the box is 1.3, so without this the panel
            comes out a wide strip that uses three quarters of the height it is
            given. False frames the grid alone.
        box_fill (float): Fraction of 'panel_box_in' the panel is drawn at, so
            it sits inside the box with a margin instead of exactly on its
            edges. Applied to both dimensions, so the proportions are kept.
        edge_width (float): Width of the transmission lines in points, at the
            size the panel is actually drawn. Far below the value that suited
            the old 20 in figures: line widths do not scale with the panel, so
            drawing at final size means setting them for that size.
        save_fig (bool): Whether to save the figures or show them.
        plot_dir (str, optional): Directory the svg is written to. None is
            the project's figure directory, which already carries the
            'PLOT_STYLE' subfolder.
    Returns:
        None
    """
    if plot_dir is None:
        plot_dir = path_to_figures_sclopf

    # Load network
    pypsa_nw = data_handling.load_pypsa_network_from_path(path_to_pypsa_network, 
                                                          use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_nw, snet_index='0')

    # Load cascade results
    with gzip.open(path_to_cascade_results, 'rb') as f:
        cascade_results = pickle.load(f)

    snapshot_cascade_results = cascade_results[snapshot_time]

    failed_links = get_failed_links_of_trigger(snapshot_cascade_results, trigger_tuple)

    print(f"Trigger line: {trigger_tuple}, Failed links: {failed_links}")
    
    # PlateCarree, so that the data coordinates stay plain degrees and the
    # networkx draw calls below need no transform of their own.
    subplot_kw = {"projection": ccrs.PlateCarree()} if geomap else {}
    fig_full, ax_full = plt.subplots(figsize=(20, 15), subplot_kw=subplot_kw)
    fig_split, ax_split = plt.subplots(figsize=(20, 15), subplot_kw=subplot_kw)

    # Fixed before drawing; setting the limits also switches autoscaling off.
    pos_arr = np.array(list(nx.get_node_attributes(nx_graph, 'pos').values()))
    map_xlim = (pos_arr[:, 0].min() - map_pad_deg,
                pos_arr[:, 0].max() + map_pad_deg)
    map_ylim = (pos_arr[:, 1].min() - map_pad_deg,
                pos_arr[:, 1].max() + map_pad_deg)

    if fit_box:
        # Open the window up to the box's proportions; never crop.
        box_ratio = panel_box_in[0] / panel_box_in[1]
        lon_span = map_xlim[1] - map_xlim[0]
        lat_span = map_ylim[1] - map_ylim[0]
        if lon_span / lat_span > box_ratio:
            grown = lon_span / box_ratio
            centre = 0.5 * (map_ylim[0] + map_ylim[1])
            map_ylim = (centre - 0.5 * grown, centre + 0.5 * grown)
        else:
            grown = lat_span * box_ratio
            centre = 0.5 * (map_xlim[0] + map_xlim[1])
            map_xlim = (centre - 0.5 * grown, centre + 0.5 * grown)

    for axis in (ax_full, ax_split):
        axis.set_xlim(*map_xlim)
        axis.set_ylim(*map_ylim)
        axis.set_aspect('equal')
        axis.axis('off')
        # After the limits: the features are clipped to the window in place.
        if geomap:
            add_geomap_to_ax(axis, geomap_color=geomap_color,
                             geomap_linewidth=geomap_linewidth)

    # Drawn at final size, so the panel goes into the assembly at 100 %.
    map_ratio = ((map_xlim[1] - map_xlim[0]) / (map_ylim[1] - map_ylim[0]))
    map_width_in = min(panel_box_in[0] * box_fill,
                       panel_box_in[1] * box_fill * map_ratio)
    map_height_in = map_width_in / map_ratio
    for figure, axis in ((fig_full, ax_full), (fig_split, ax_split)):
        figure.set_size_inches(map_width_in, map_height_in)
        axis.set_position([0., 0., 1., 1.])

    # After the resize: the conversion reads the axes' scale.
    node_size = node_size_for_data_diameter(ax_full, node_diameter_deg)

    ax_plot_full_network(nx_graph, ax_full, node_size=node_size,
                         edge_width=edge_width,
                         failed_link_names=[list(nx_graph.edges)[xx] for xx in failed_links])

    ax_plot_split_network(nx_graph, ax_split, failed_links, node_size=node_size,
                          edge_width=edge_width)

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
        # No tight bbox: the canvas is the map.
        fig_full.savefig(os.path.join(plot_dir, fig_path_full),
                         transparent=True, dpi=dpi_fig)
        fig_full.clear()
        plt.close(fig_full)
        
        fig_path_split = f"split_network_trigger_{trigger_tuple[0]}_{trigger_tuple[1]}"
        if rasterize:
            fig_path_split += "_rasterized.png"
        else:
            fig_path_split += ".svg"
        fig_split.savefig(os.path.join(plot_dir, fig_path_split),
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


def setup_arimo_style(fontsize: float = 8):
    """Matplotlib rc settings that render everything in Arimo at ``fontsize``.

    Returns an rc dict for use with ``plt.rc_context`` rather than setting the
    style globally, so it does not leak into the network sketches above.

    Arimo (the metric-compatible open counterpart of Arial) ships with the
    croscore fonts but is not in matplotlib's font list, so its files are
    registered here first. LaTeX text rendering is switched OFF deliberately:
    ``usetex`` routes text through LaTeX, which would ignore the system TTF and
    typeset Computer Modern instead. Consequence for the caller: label strings
    must be plain text plus mathtext (``CO$_2$``, a literal ``%``), not LaTeX
    escapes such as ``\\%``.

    Args:
        fontsize (float): Size in points for every text element.

    Returns:
        dict: rc parameters.
    """
    for font_file in glob.glob("/usr/share/fonts/truetype/croscore/Arimo-*.ttf"):
        font_manager.fontManager.addfont(font_file)

    # Concrete family names rather than the generic "sans-serif" alias: text
    # artists store the family they were given and resolve it lazily at draw
    # time. With the alias, a figure saved outside this rc context would
    # re-resolve "sans-serif" against the ambient rcParams and silently come out
    # in the default font.
    font_families = ["Arimo", "Liberation Sans", "DejaVu Sans"]

    return {"text.usetex": False,
            "font.family": font_families,
            "font.sans-serif": font_families,
            # Keep the mathtext subscript of CO2 in Arimo as well.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arimo",
            "mathtext.it": "Arimo:italic",
            "mathtext.bf": "Arimo:bold",
            "mathtext.cal": "Arimo",
            "mathtext.default": "regular",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            # Text stays text in the svg, so it remains editable in inkscape.
            "svg.fonttype": "none"}


def plot_lost_load_share_distribution(n_nodes: int = 600,
                                      fontsize: float = 6,
                                      label_fontsize: float = 8,
                                      ticklabel_fontsize: float = 5.25,
                                      figsize: tuple = (1.118, 1.55),
                                      legend_ncols: int = 2,
                                      panel_label: int | None = None,
                                      plot_dir: str | None = None,
                                      save_fig: bool = True):
    """Number of system splits per not-served-load-share bin over the CO2 levels.

    This is panel c of ``plot_combined_flow_split_statistics.py`` as a figure of
    its own, so it can go into the graphical abstract without its neighbouring
    panels. The numbers are identical -- same bins and the same
    ``trigger_weighting * snapshot_weighting`` weighting. The layout is reduced
    for the abstract: steps instead of lines, only the split-count axis (no
    secondary contingency-fraction axis) and 8 pt type throughout.

    Typeset with the project style, like the other figures, so the abstract and
    the paper share one typeface. That style renders text through LaTeX, which
    has two consequences here: a literal percent sign has to be escaped as
    ``\\%``, and the svg carries the glyphs as paths rather than as text, so the
    labels cannot be edited in inkscape afterwards. ``setup_arimo_style`` is the
    alternative -- Arial-metric Arimo, no LaTeX, editable svg text -- and is
    left in place above for the network sketches and in case that is wanted back.

    The panel is drawn at its FINAL physical size and saved WITHOUT a tight
    bounding box, so the svg really is ``figsize`` -- a tight box would grow the
    file past it. Place it at 100 % in inkscape; scaling it afterwards scales
    the text along with it and breaks the 8 pt minimum.

    At this width the axis labels have to be short: the full
    "Number of system splits" is ~1.3 in of text at 8 pt and would not fit along
    the count axis, so the labels are abbreviated rather than the font shrunk.

    The axis labels run at ``label_fontsize`` and everything else at
    ``fontsize``, because the two are limited by different things. The legend is
    what fixes the panel width -- it already spans 1.036 of the 1.118 in at 6 pt
    -- so it cannot grow without widening the panel. The labels are bounded by
    the axes instead: the abbreviated "System Splits" is only about half the
    height of the count axis, and the row's extra 0.05 in (1.55 rather than 1.5)
    pays for the taller x-label with room left over, so at 8 pt the data area
    ends up slightly LARGER than it was with 6 pt labels at 1.5 in.

    The legend sits in the figure, below the axes, so the panel is a single
    self-contained svg. It is what sets the figure width: "Lost load:" and the
    5 bins fill a 3-row x 2-column grid, reading left to right, and are wider
    than the axes themselves. The layout is constrained rather than tight: an
    outside legend is space the tight layout does not reserve, so the axis
    labels would end up underneath it.

    "Lost load:" is the first legend entry, on an invisible handle, rather than
    a legend title -- as a title it would cost a text row of its own, and here
    the axes are the scarcer resource. That it is a share is left to the
    percentages of the bins rather than spelled out, which keeps the entry from
    padding the first column out much beyond the bin labels.

    Args:
        n_nodes (int): Network resolution the split properties were computed for.
        fontsize (float): Point size of every text element other than the axis
            and tick labels, i.e. in practice the legend -- which is what the
            panel width is set by, so raising this widens the panel.
        label_fontsize (float): Point size of the two axis labels alone. Free to
            run above ``fontsize`` because it is paid for out of the axes rather
            than out of the panel width; see the note above.
        ticklabel_fontsize (float): Point size of the tick labels only. A little
            below ``fontsize``, so the numbers step back behind the axis labels
            and leave more of the canvas to the axes.
        figsize (tuple): Final size of the panel in inches, legend included.
            1.55 in of height is the bottom row of the graphical abstract, which
            the two map panels are built to as well.
        legend_ncols (int): Columns of the legend grid. The 5 bins are laid out
            row by row, so 2 gives 3 rows with the last cell empty.
        panel_label (int, optional): If given, stamp a panel label (0 = A, ...).
            The graphical abstract wants none, hence the None default.
        save_fig (bool): Whether to save the figure or show it.
        plot_dir (str, optional): Directory the svg is written to. None is
            the project's figure directory, which already carries the
            'PLOT_STYLE' subfolder.

    Returns:
        tuple: The figure and its axes.
    """
    if plot_dir is None:
        plot_dir = path_to_figures_sclopf

    co2ls = data_handling.get_co2_levels(n_nodes)

    split_props = pd.read_hdf(path_to_vis_results_sclopf
                              + f"/split_properties_all_n{n_nodes}.h5",
                              index_col=0)
    split_props.lost_load_share_blackout = \
        split_props.lost_load_share_blackout.astype(float)
    data_handling.check_columns_exist(split_props, ["total_weighting"],
                                      name="Split properties")

    cmap = plt.get_cmap("inferno_r")

    bins = np.linspace(0, 100, 6).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data = []
    for co2l in co2ls[::-1]:
        split_props_co2 = split_props[split_props.co2l == co2l]
        # Clip into the bin range so total-blackout splits land in the last bin.
        vals = data_handling.clip_to_bin_range(
            split_props_co2.lost_load_share_blackout.values * 100,
            bins,
        )
        # Weighted like the denominator: trigger_weighting * snapshot_weighting.
        counts, _ = np.histogram(vals, bins=bins,
                                 weights=split_props_co2.total_weighting)
        data.append(counts)

    data = pd.DataFrame(np.stack(data), index=co2ls[::-1], columns=bin_centers)

    markers = ["o", "s", "D", "^", "v", "."]
    co2_axis = data_handling.get_actual_co2_level(co2ls[::-1], percent=True)

    # Project style; the rc_context below only overrides the sizes.
    setup_matplotlib_style()
    # with plt.rc_context(setup_arimo_style(fontsize)):  # Arimo, no LaTeX
    with plt.rc_context({"font.size": fontsize,
                         "axes.labelsize": label_fontsize,
                         "xtick.labelsize": ticklabel_fontsize,
                         "ytick.labelsize": ticklabel_fontsize,
                         "legend.fontsize": fontsize,
                         "legend.title_fontsize": fontsize}):
        # Constrained: the legend's strip is subtracted from the axes, not drawn over it.
        fig = plt.figure(figsize=figsize, layout="constrained")
        # w_pad=0: the axes and the outside-right legend then both end at the
        # figure edge, which is what lines them up.
        fig.get_layout_engine().set(w_pad=0., h_pad=0.01,
                                    wspace=0., hspace=0.)
        ax = fig.add_subplot()

        colors = [cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
                  for i in range(len(bin_centers))]
        # Escaped percent: the project style typesets through LaTeX, where a
        # bare '%' would comment out the rest of the label.
        labels = [rf"{bins[i]}-{bins[i+1]}\%" for i in range(len(bin_centers))]

        for i, bin_center in enumerate(bin_centers):
            ax.step(co2_axis,
                    data.loc[:, bin_center],
                    where="mid",
                    label=labels[i],
                    alpha=0.8,
                    color=colors[i],
                    linewidth=0.8,
                    marker=markers[i % len(markers)],
                    markersize=1.8)

        # Abbreviated -- see the docstring: the full wording does not fit at 8 pt.
        ax.set_xlabel(r"CO$_2$ level [\%]", labelpad=1.5)
        ax.set_ylabel("System Splits", labelpad=1.5)
        ax.set_xticks([0, 30, 60])
        ax.set_xticks([10, 20, 40, 50], minor=True)
        # Only the gridlines of the count axis -- horizontal lines at its major
        # ticks. Thinned down so they do not compete with the curves at this size.
        # The explicit grid(False) is needed: importing the project modules turns
        # rcParams["axes.grid"] on globally, so enabling axis="y" alone would
        # leave the inherited vertical gridlines in place.
        ax.grid(False)
        ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.3)
        ax.set_yscale("log")
        # The counts span ~7 decades; a tick per decade is unreadable this small.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=100, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="both", which="major", length=2., width=0.5, pad=1.,
                       labelsize=ticklabel_fontsize)
        ax.tick_params(axis="both", which="minor", length=1.2, width=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.invert_xaxis()

        if panel_label is not None:
            add_panel_label(ax, panel_label, y_offset=0.1, x_offset=-.125,
                            fontsize=fontsize)

        # Proxy artists, so legend marker sizes are independent of the curves.
        # "Lost load:" is the first entry on an invisible handle rather than a
        # legend title, which saves the row a title would take.
        handles = [plt.Line2D([], [], color="none")] + \
            [plt.Line2D([], [], color=colors[i],
                        marker=markers[i % len(markers)],
                        markersize=2.5, linewidth=1.0, alpha=0.8)
             for i in range(len(bin_centers))]
        entry_labels = ["Lost load:"] + labels

        # ``legend`` fills columns top to bottom; transpose to read row by row.
        n_legend_rows = int(np.ceil(len(handles) / legend_ncols))
        row_major_order = [row * legend_ncols + col
                           for col in range(legend_ncols)
                           for row in range(n_legend_rows)
                           if row * legend_ncols + col < len(handles)]
        handles = [handles[i] for i in row_major_order]
        legend_labels = [entry_labels[i] for i in row_major_order]

        # "outside" is what makes the layout engine shrink the axes by the
        # height of the legend. Right rather than centre, and then nudged into
        # place below: centred it sat 0.03 in inside the axes, because the axes
        # is pushed right by the y-label while the legend is centred on the
        # whole figure.
        legend = fig.legend(handles, legend_labels,
                            loc="outside lower right",
                            ncols=legend_ncols,
                            frameon=True,
                            fancybox=True,
                            handlelength=1.2,
                            handletextpad=0.5,
                            columnspacing=1.0,
                            labelspacing=0.3,
                            borderpad=0.4,
                            borderaxespad=0.)

        # Same frame as the legend of the single-cluster map, so the two parts
        # of the abstract carry the same box: rounded, a 1 pt light grey edge
        # over a white fill, both at 80 % alpha. Spelled out rather than left to
        # the rcParams the project style happens to set, so the panel does not
        # change with whatever else has been imported. Note the fill is not
        # transparent, unlike the rest of this svg.
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_edgecolor((0.8, 0.8, 0.8, 0.8))
        legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))

        # Collapse the empty marker slot in front of "Lost load:".
        # ``handlelength`` is legend-wide, so this reaches into the offsetbox
        # tree: HPacker of columns -> VPacker of entries -> HPacker of
        # handle + text. Private attribute, checked against matplotlib 3.10.
        first_entry = \
            legend._legend_handle_box.get_children()[0].get_children()[0]
        first_entry.get_children()[0].set_width(0.)
        first_entry.sep = 0.

        if save_fig:
            # svg only, like the other parts here -- assembled in inkscape.
            # No tight bbox: the panel has to keep exactly the requested figsize.
            fig.savefig(os.path.join(plot_dir,
                                     "graphical_abstract_risk.svg"),
                        transparent=True)

            fig.clear()
            plt.close(fig)
        else:
            plt.show()

    return fig, ax


def plot_generation_map_scenario(n_nodes: int = 600,
                                 co2_level: float = 0.6,
                                 month: int = 12,
                                 fontsize: float = 8,
                                 figsize: tuple = (0.962, 0.861),
                                 bus_size_scale: float = 7e6,
                                 line_width: float = 0.1,
                                 geomap_color: str = "0.65",
                                 carrier_share_threshold: float = 0.01,
                                 legend_circles_twh: tuple | None = None,
                                 n_legend_carriers: int = 3,
                                 legend_fontsize: float | None = 4.5,
                                 map_xlim: tuple = (-10.6, 28.5),
                                 map_ylim: tuple = (35, 70),
                                 panel_box_in: tuple = (1.125, 0.879),
                                 fit_box: bool = True,
                                 box_fill: float = 0.98,
                                 panel_label: int | None = None,
                                 plot_dir: str | None = None,
                                 save_fig: bool = True):
    """Map of the generation mix in one month of one CO2 scenario.

    This is the top-left map of Fig. 1 (``create_combined_generation_storage_plot``)
    as a figure of its own, so the scenario can go into the graphical abstract
    without its neighbouring panels. The pie per bus is the generation summed
    over the snapshots of ``month``, weighted with the snapshot weightings and
    signed like the generators -- identical to Fig. 1 -- and the carriers are
    aggregated with the same keyword rules (all ``offwind*`` into one offshore
    wind wedge, ``solar-hsat`` into solar).

    One thing differs from Fig. 1 on purpose: carriers below
    ``carrier_share_threshold`` of the month's generation are dropped, which
    removes the artificial ``load``/``art_load`` generators along with the
    slivers that are invisible at this size; the threshold is the one Fig. 1
    applies to its generation-vs-CO2 panel.

    The map is the whole output: one svg, and what legend there is sits on it.
    Nine carrier colours at a readable 8 pt need more room than the map itself,
    so the key names only the ``n_legend_carriers`` biggest and ends in an
    ellipsis; the full list is Fig. 1's job.

    The panel is drawn at its FINAL physical size, so place the svg at 100 % in
    inkscape -- scaling it afterwards scales the text along with it and breaks
    the 8 pt minimum.

    At roughly an inch across, the panel is a thumbnail and the defaults are set
    for that: no size key (see ``legend_circles_twh``) and lines at a third of
    the width Fig. 1 gives them, so that the pies rather than the 600-node mesh
    carry the panel. Bus sizes, on the other hand, are areas in map coordinates,
    so ``bus_size_scale`` is independent of ``figsize`` and Fig. 1's 7e6 (MWh per
    unit area) carries over unchanged: bigger pies start merging into one blob
    over Germany and the Benelux at this size.

    Args:
        n_nodes (int): Network resolution.
        co2_level (float): CO2 level of the scenario to show.
        month (int): Month whose generation is summed, 1-12.
        fontsize (float): Point size of every text element. 8 pt is the floor
            for readability, hence the default.
        figsize (tuple): Final size of the panel in inches, filling the
            1.125 x 0.879 in box the panel has in the abstract as far as the
            aspect of ``map_xlim`` x ``map_ylim`` allows. It has to follow that
            aspect (39.1 deg x 35 deg = 1.12), otherwise the map is letterboxed
            inside the canvas. Since 1.12 is narrower than the box (1.28), it is
            the height that binds here and the width follows: 0.879 x 1.12.
            Recompute both numbers when changing the map window.
        bus_size_scale (float): MWh per unit of bus area.
        line_width (float): Width of the transmission lines and links in points.
            Unlike the bus sizes this is a physical width, so it does not shrink
            with the panel and has to be set for the panel's size.
        geomap_color (str): Colour of the coastlines and country borders. Light
            by default so the network stays the darkest thing on the panel;
            their width is fixed at 0.3 pt inside PyPSA.
        carrier_share_threshold (float): Carriers below this share of the
            month's total generation are dropped.
        legend_circles_twh (tuple, optional): Reference circle sizes of the size
            key, in TWh. Off by default: the key is set in the same 8 pt as
            everything else, which at this panel size puts a block of text over
            central Europe half as wide as the whole map. Pass e.g. (5, 1) to get
            it back -- worth it only if the panel is ever drawn much larger.
        n_legend_carriers (int): How many carriers the key in the upper right
            names, largest contributor of the month first, before the ellipsis
            that stands for the rest. 0 leaves the key off entirely.
        legend_fontsize (float, optional): Point size of the key alone; None
            ties it to ``fontsize``. CAUTION, this is the one place where the
            8 pt floor is broken on purpose, and it is a decision to revisit
            rather than inherit. A key this small cannot be set at 8 pt: at that
            size it covers two thirds of the panel and leaves no map to speak
            of. Share of the panel area by size: 8 pt 67 %, 6 pt 44 %, 5 pt
            26 %, 4.5 pt 18 %, 4 pt 15 %. The way back to 8 pt is a bigger box,
            not a smaller key -- at 8 pt the key alone is as wide as the whole
            panel is now.
        map_xlim (tuple): Longitude window of the map, cropped much closer to the
            network than the ``MAP_XLIM`` of the full-page figures, which leaves
            5 deg resp. 7 deg of empty sea beside it. What sets the edges is not
            the outermost bus but the outermost *pie*: the westernmost bus sits
            in Ireland at -8.97 deg and carries a radius of 0.37 deg, so its
            wedge reaches -9.34 deg, and the easternmost one in Romania reaches
            27.97 deg. The west edge then goes a further degree out to -10.6, to
            clear the west coast of Ireland, which the 50m coastline puts at
            -10.39 deg -- otherwise the island is sliced. Widening costs nothing
            here as long as the range stays below ~44.8 deg: the height is what
            binds against the box, so the map is drawn at 0.0251 in/deg either
            way and only the panel gets wider.
        map_ylim (tuple): Latitude window of the map. Untouched, so the map
            still stands on the same latitudes as Fig. 1.
        panel_box_in (tuple): Box in inches the panel has to fit into, the same
            one the two split panels are built to.
        fit_box (bool): Whether the window is opened up to the proportions of
            'panel_box_in' so the map fills the box, as in
            'plot_entire_network_n_split_components'. Here it is longitude that
            grows: this window is nearly square against a box half again as wide
            as it is tall, so left alone the panel stands well short of the
            box's width. Only ever opened up, never cropped.
        box_fill (float): Fraction of 'panel_box_in' the panel is drawn at, so
            it sits inside the box with a margin rather than on its edges.
        panel_label (int, optional): If given, stamp a panel label (0 = A, ...).
            The graphical abstract wants none, hence the None default.
        save_fig (bool): Whether to save the figure or show it.
        plot_dir (str, optional): Directory the svg is written to. None is
            the project's figure directory, which already carries the
            'PLOT_STYLE' subfolder.

    Returns:
        tuple: The panel figure and its axes.
    """
    if plot_dir is None:
        plot_dir = path_to_figures_sclopf

    network = data_handling.load_pypsa_network(co2lvl=co2_level, n_nodes=n_nodes,
                                               use_sclopf=True, lopt=False)

    snapshots_month = network.snapshots[network.snapshots.month == month]
    if len(snapshots_month) == 0:
        raise ValueError(f"Network has no snapshots in month {month}.")

    generation = (network.generators_t.p
                  .mul(network.snapshot_weightings.generators, axis="index")
                  .loc[snapshots_month]
                  .sum()
                  .mul(network.generators.sign))
    generation = generation.groupby([network.generators["bus"],
                                     network.generators["carrier"]]).sum()

    generation = aggregate_carriers(generation,
                                    {"offwind": "offwind-dc", "solar": "solar"},
                                    inplace=False)

    # Drop the carriers that cannot be seen anyway. Shares of the month's total,
    # not of the individual bus, so a carrier is either in the figure everywhere
    # or nowhere and the set of colours is the same for every pie.
    share_by_carrier = (generation.groupby(level="carrier").sum()
                        / generation.sum())
    kept_carriers = share_by_carrier[share_by_carrier
                                     > carrier_share_threshold].index
    dropped = share_by_carrier.drop(kept_carriers).sort_values(ascending=False)
    logger.info(f"Dropping {len(dropped)} carriers below "
                f"{carrier_share_threshold:.0%} of the month's generation: "
                f"{dict(dropped.round(5))}")
    generation = generation[generation.index
                            .get_level_values("carrier").isin(kept_carriers)]

    if fit_box:
        # As in 'plot_entire_network_n_split_components'; here longitude grows.
        box_ratio = panel_box_in[0] / panel_box_in[1]
        lon_span = map_xlim[1] - map_xlim[0]
        lat_span = map_ylim[1] - map_ylim[0]
        if lon_span / lat_span > box_ratio:
            grown = lon_span / box_ratio
            centre = 0.5 * (map_ylim[0] + map_ylim[1])
            map_ylim = (centre - 0.5 * grown, centre + 0.5 * grown)
        else:
            grown = lat_span * box_ratio
            centre = 0.5 * (map_xlim[0] + map_xlim[1])
            map_xlim = (centre - 0.5 * grown, centre + 0.5 * grown)
        figsize = (panel_box_in[0] * box_fill, panel_box_in[1] * box_fill)

    with plt.rc_context(setup_arimo_style(fontsize)):
        fig = plt.figure(figsize=figsize)
        # The axes covers the whole canvas and the figure is saved without a
        # tight bbox, so the svg is exactly ``figsize`` and the map fills every
        # bit of it. Going through add_subplot + tight_layout + bbox_inches
        # ='tight' instead would leave the final size to matplotlib: the content
        # comes out a little under ``figsize`` and savefig then pads it back out
        # by 0.1 in per side, which at this size overshoots the box by ~15 %.
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

        # The window goes in as ``boundaries`` rather than through set_xlim /
        # set_ylim afterwards: the size key below converts its circles with the
        # axis extent that is in place when it is built, so the extent has to be
        # final before that call -- and n.plot sets it, equal aspect and all.
        # ``n.plot.map`` rather than the ``n.plot(...)`` of the other plot
        # modules: the callable form is deprecated since pypsa 0.34 and the two
        # draw the same thing.
        network.plot.map(bus_sizes=generation / bus_size_scale,
                         boundaries=(*map_xlim, *map_ylim),
                         geomap_colors={"border": geomap_color,
                                        "coastline": geomap_color},
                         line_colors="black",
                         link_colors="black",
                         line_widths=line_width,
                         link_widths=line_width,
                         ax=ax)

        # Size key in the lower right, over the empty sea east of the network.
        if legend_circles_twh is not None:
            circle_sizes = [twh * 1e6 / bus_size_scale
                            for twh in legend_circles_twh]
            with warnings.catch_warnings():
                # add_legend_circles warns on every geographic axis that the
                # extent must be final before it converts its circles. It is:
                # n.plot set it from ``boundaries`` just above, and nothing
                # changes it afterwards.
                warnings.filterwarnings("ignore",
                                        message="When combining n.plot()")
                add_legend_circles(
                    ax,
                    sizes=circle_sizes,
                    labels=[f"{twh:g} TWh" for twh in legend_circles_twh],
                    patch_kw={"facecolor": "none", "edgecolor": "black",
                              "linewidth": 0.5},
                    legend_kw={"loc": "lower right", "frameon": False,
                               "labelspacing": 0.8, "borderpad": 0.2,
                               "handletextpad": 0.4, "fontsize": fontsize},
                )

        # Carrier key in the upper right. Only the biggest contributors of the
        # month, with an ellipsis for the rest: all nine carriers at a readable
        # 8 pt would need more room than the map itself, and the point here is
        # to say what the colours mean, not to be exhaustive -- Fig. 1 is.
        if n_legend_carriers:
            legend_fontsize = (fontsize if legend_fontsize is None
                               else legend_fontsize)
            carriers_ranked = (generation.groupby(level="carrier").sum()
                               .sort_values(ascending=False).index)
            shown = list(carriers_ranked[:n_legend_carriers])
            handles = [plt.Line2D([], [], linestyle="none", marker="o",
                                  markersize=legend_fontsize * 0.45,
                                  markerfacecolor=network.carriers.color[carrier],
                                  markeredgecolor="none")
                       for carrier in shown]
            # "Onshore Wind" is broken over two lines, the qualifier below the
            # carrier: set on one line it is the widest thing in the key by a
            # third, and the panel has no width to spare. And after the
            # aggregation above the DC entry carries every offshore wind
            # technology, so its nice name would be wrong.
            label_overrides = {"onwind": "Wind\n(onshore)",
                               "offwind-dc": "Offshore wind"}
            labels = [label_overrides.get(
                          carrier,
                          network.carriers.nice_name[carrier].capitalize())
                      for carrier in shown]
            # An invisible handle, so the ellipsis sits in the label column
            # under the carrier names rather than under the colour dots.
            handles.append(plt.Line2D([], [], linestyle="none", marker="none"))
            labels.append("...")

            # The key sits on top of the network, so it gets a white backing
            # rather than floating over the lines; without it the labels are
            # unreadable wherever a line crosses them.
            legend = ax.legend(handles, labels,
                               title="Carriers:",
                               loc="upper right",
                               alignment="left",
                               fontsize=legend_fontsize,
                               title_fontsize=legend_fontsize,
                               frameon=True,
                               fancybox=True,
                               handlelength=1.0,
                               handletextpad=0.4,
                               borderpad=0.2,
                               borderaxespad=0.4,
                               labelspacing=0.25)
            # Same frame as the risk and cluster legends. The fill also keeps
            # the key readable where the network runs under it.
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_edgecolor((0.8, 0.8, 0.8, 0.8))
            legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.8))
            # Attached to the axes explicitly: a later ax.legend call -- the size
            # key, if it is switched on -- would otherwise drop this one.
            ax.add_artist(legend)

        if panel_label is not None:
            add_panel_label(ax, panel_label, y_offset=0.1, x_offset=-.125,
                            fontsize=fontsize)

        if save_fig:
            fname = (f"graphical_abstract_generation_map_co2l{co2_level}"
                     f"_month{month:02d}.svg")
            # svg only, like the other parts here -- assembled in inkscape.
            # No bbox_inches and no pad: the canvas already is the panel, and
            # the default pad of 0.1 in per side would add 0.2 in to both
            # dimensions of the saved file.
            fig.savefig(os.path.join(plot_dir, fname),
                        pad_inches=0, transparent=True)

            fig.clear()
            plt.close(fig)
        else:
            plt.show()

    return fig, ax


# Same node colour as the map in Fig. 5 (``utils.mitigation_visualization.color1``,
# the orange of Dark2). Hard-coded rather than imported so this panel does not
# change colour if the mitigation module's default does.
INERTIA_NODE_COLOR = "#d95f02"


def load_inertia_placement_by_node(n_graph_nodes: int,
                                   n_nodes: int = 600,
                                   co2_level: float = 0.2,
                                   target: str = "num_GSS",
                                   blackoutthreshold: float = 0.8,
                                   ref_loss_factor: int = 1):
    """Inertia placed per node, accumulated up to the step that reaches the target.

    Reads the very pickle Fig. 5 reads, so the map cannot drift away from it.
    The file is written by ``create_combined_mitigation_plot`` (or by
    ``calc_inertia_placement_ref_loss`` directly) and is expected to exist --
    recomputing it here would take hours and would silently use this script's
    defaults instead of the ones the figure was made with.

    ``res_tuple[2]`` is the greedy placement log, one row per step, of which
    columns 1 and 2 are the node index and how many ``delta_Erot`` quanta went
    there. Rows beyond ``idx_reached_target`` are steps the algorithm took past
    the reference level and are dropped.

    Args:
        n_graph_nodes (int): Number of nodes in the networkx graph the map is
            drawn on. Taken from the graph rather than from ``n_nodes``, which
            is the clustering target and not necessarily the bus count.
        n_nodes (int): Network resolution the placement was computed for.
        co2_level (float): CO2 level of the scenario to show.
        target (str): Mitigation target the placement optimized ("num_GSS",
            "total_loss" or "lost_load_GSS").
        blackoutthreshold (float): Lost-load share above which a split counts
            as globally severe.
        ref_loss_factor (int): Which reference-loss run to read; 1 is the only
            one Fig. 5 uses.

    Returns:
        tuple: (inertia per node in MWs as an array of length n_graph_nodes,
            total placed inertia in MWs).
    """
    fname = (f"/inertia_placement_results_refLoss_allCo2lvls_n{n_nodes}"
             f"_{target}_blackoutthres{blackoutthreshold}.pkl")
    fpath = path_to_inertia_mitigation_results_sclopf + fname

    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Inertia placement results not found at {fpath}. Generate them by "
            "running create_combined_mitigation_plot with the same target and "
            "blackout threshold.")

    with gzip.open(fpath, "rb") as f:
        (res_tuples_by_lvl,
         _,
         _,
         steps_to_reach_target_by_ref_loss,
         delta_Erot_by_lvl) = pickle.load(f)

    res_tuple = res_tuples_by_lvl[co2_level]
    idx_reached_target = steps_to_reach_target_by_ref_loss[ref_loss_factor][co2_level]
    delta_Erot = delta_Erot_by_lvl[co2_level]

    # Columns 1 and 2 of the placement log: node index, number of quanta placed.
    placement_log = np.array(res_tuple[2])[:idx_reached_target, 1:3]

    inertia_by_node = np.zeros(n_graph_nodes)
    for node_idx, n_quanta in placement_log:
        inertia_by_node[int(node_idx)] += n_quanta
    inertia_by_node *= delta_Erot

    return inertia_by_node, inertia_by_node.sum()


def plot_inertia_placement_map_method(n_nodes: int = 600,
                                      co2_level: float = 0.2,
                                      target: str = "num_GSS",
                                      blackoutthreshold: float = 0.8,
                                      ref_loss_factor: int = 1,
                                      fontsize: float = 6.75,
                                      figsize: tuple = (2.6, 2.2),
                                      max_node_diameter_deg: float = 1.9,
                                      edge_width: float = 0.25,
                                      edge_color: str = "black",
                                      legend_fractions: tuple = (1., .4),
                                      legend_values_gws: tuple | None = None,
                                      legend_loc: str = "upper center",
                                      legend_bbox_to_anchor: tuple | None = (0.5, 0.0),
                                      legend_ncols: int = 2,
                                      legend_title: str | None = "Additional inertia:",
                                      legend_title_fontsize: float | None = 8,
                                      annotate_total: bool = False,
                                      total_xy: tuple = (0.0, 1.0),
                                      map_xlim: tuple | None = None,
                                      map_ylim: tuple | None = None,
                                      map_pad_deg: float = 1.0,
                                      map_height_in: float = 0.814,
                                      panel_height_in: float = 1.55,
                                      aspect: float | str = "equal",
                                      panel_label: int | None = None,
                                      plot_dir: str | None = None,
                                      save_fig: bool = True):
    """Map of where the greedy placement puts synthetic inertia.

    This is panel b of ``create_combined_mitigation_plot`` (the ``ax_inertia_map``
    axes of Fig. 5) as a figure of its own, so it can go into the method figure
    without its neighbouring panels: the European grid with node areas
    proportional to the synthetic inertia the greedy placement puts at each node
    until the 60 % CO2 reference level of globally severe splits is reached. The
    numbers are identical -- the same pickled placement results Fig. 5 reads,
    accumulated up to the same ``steps_to_reach_target``.

    Differences to Fig. 5, all because this is a method panel and not a results
    panel: no title, the size legend sits below the map, no numbers are printed
    on the panel, and everything is 8 pt Arimo.

    Like ``plot_lost_load_share_distribution``, the panel is drawn at its FINAL
    physical size, so place the svg at 100 % in inkscape -- scaling it afterwards
    scales the text along with it and breaks the 8 pt minimum.

    Node area is proportional to the inertia placed at that node, so the eye
    compares areas and not radii -- ``nx.draw_networkx_nodes`` takes the marker
    AREA in points squared, hence the linear scaling of ``node_size`` below.
    Nodes that receive nothing get size 0 and disappear; the grid stays visible
    through the edges, which is the intent -- the panel is about where inertia
    concentrates, not about the topology.

    Args:
        n_nodes (int): Network resolution.
        co2_level (float): CO2 level of the scenario to show. 0.2 is the level
            Fig. 5 maps.
        target (str): Mitigation target the placement optimized.
        blackoutthreshold (float): Lost-load share above which a split counts
            as globally severe.
        ref_loss_factor (int): Which reference-loss run to read.
        fontsize (float): Point size of every text element. 8 pt is the floor
            for readability, hence the default.
        figsize (tuple): Final size of the panel in inches. The saved svg comes
            out slightly smaller in width and taller in height, because
            ``bbox_inches='tight'`` crops the side margins and takes in the
            legend, which sits outside the axes.
        max_node_diameter_deg (float): Diameter in degrees of the node with
            the most inertia; everything else scales linearly in AREA below it.
            In degrees, like 'node_diameter_deg' of the split panels, so the
            bubbles keep their share of the map at any panel size -- as a fixed
            area in points they grew relative to the map every time the panel
            was made smaller.
        edge_width (float): Line width of the transmission lines.
        edge_color (str): Colour of the transmission lines.
        legend_fractions (tuple): Fractions of the largest node to show as size
            references in the legend. Ignored when ``legend_values_gws`` is given.
        legend_values_gws (tuple | None): Size references as absolute values in
            GWs, for round legend numbers instead of whatever fraction of the
            largest node happens to come out. Values above the largest node are
            allowed but then show a circle no node reaches.
        legend_loc (str): Anchor point of the size legend. Together with the
            default ``legend_bbox_to_anchor`` this puts it centred below the map.
        legend_bbox_to_anchor (tuple | None): Where the legend is anchored, in
            axes coordinates. ``(0.5, 0.0)`` is the bottom centre of the axes;
            because the aspect ratio shrinks the axes box onto the map itself,
            this sits directly under the drawn network rather than under an
            empty margin. None places the legend inside the axes at ``legend_loc``.
        legend_ncols (int): Columns of the size legend.
        legend_title (str | None): Title above the size legend; None for no title.
        legend_title_fontsize (float | None): Point size of that title alone;
            None ties it to ``fontsize``. Separated out because the title is the
            one line of the panel that is read rather than looked up, while
            ``fontsize`` also sets the GWs labels, which have to stay small
            enough for two of them to fit side by side under the map. What caps
            it is the width: the svg is cropped to its content, so a title wider
            than the map widens the file and the panel stops matching the
            cluster map it is placed next to.
        annotate_total (bool): Whether to write the total placed inertia onto
            the map. Off by default; the caption carries the number instead.
        total_xy (tuple): Position of that annotation in axes coordinates; it
            is anchored at its top left corner.
        map_xlim (tuple | None): Longitude range. None frames the nodes that
            are actually there plus ``map_pad_deg``. Fixed continental limits
            would leave a wide empty band, because the synchronous area drawn
            here reaches only to about 57 deg N.
        map_ylim (tuple | None): Latitude range; None as for ``map_xlim``.
        map_pad_deg (float): Padding around the nodes in degrees when the
            limits are derived from the data.
        map_height_in (float): Height of the map itself in inches, not of the
            panel. 'create_single_cluster_plot_from_data' is given the same
            value, which is what makes the two networks come out identical --
            change the one and the other has to follow.
        panel_height_in (float): Height the whole panel has to fit into, size
            legend included. The bottom row of the abstract is 1.55 in high;
            ``map_height_in`` was scaled by the same 1.55/1.5 when the row grew
            from 1.5, so the map keeps its share of the panel.
        aspect (float | str): Aspect ratio of the axes. "equal", so this map is
            drawn on the same geometry as the other maps of the abstract -- the
            two split panels and the cluster map are all equal-aspect, and a
            different aspect here would show the same grid as a differently
            shaped Europe. "latitude" is the alternative, 1/cos(mean latitude),
            which is the honest projection for distances but stretches this map
            about 1.5 times in latitude against the others.
        panel_label (int, optional): If given, stamp a panel label (0 = A, ...).
        save_fig (bool): Whether to save the figure or show it.
        plot_dir (str, optional): Directory the svg is written to. None is
            the project's figure directory, which already carries the
            'PLOT_STYLE' subfolder.

    Returns:
        tuple: The figure and its axes.
    """
    if plot_dir is None:
        plot_dir = path_to_figures_sclopf

    pypsa_nw = data_handling.load_pypsa_network(co2_level, n_nodes, True)
    nx_graph = data_handling.build_networkx_graph(pypsa_nw, snet_index='0')
    pos_nodes = nx.get_node_attributes(nx_graph, "pos")

    inertia_by_node, total_inertia = load_inertia_placement_by_node(
        n_graph_nodes=len(nx_graph),
        n_nodes=n_nodes,
        co2_level=co2_level,
        target=target,
        blackoutthreshold=blackoutthreshold,
        ref_loss_factor=ref_loss_factor,
    )

    max_inertia_node = inertia_by_node.max()

    # MWs -> GWs, the unit Fig. 5 reports.
    unit_factor = 1e-3

    with plt.rc_context(setup_arimo_style(fontsize)):
        fig, ax = plt.subplots(figsize=figsize)

        # === LAYOUT ===
        # Ahead of the drawing: the bubble areas below come from the map scale.
        pos_arr = np.array(list(pos_nodes.values()))
        if map_xlim is None:
            map_xlim = (pos_arr[:, 0].min() - map_pad_deg,
                        pos_arr[:, 0].max() + map_pad_deg)
        if map_ylim is None:
            map_ylim = (pos_arr[:, 1].min() - map_pad_deg,
                        pos_arr[:, 1].max() + map_pad_deg)
        ax.set_xlim(*map_xlim)
        ax.set_ylim(*map_ylim)

        if aspect == "latitude":
            aspect = 1 / np.cos(np.deg2rad(pos_arr[:, 1].mean()))
        ax.set_aspect(aspect)
        ax.axis("off")

        # Pinned to an exact height, so this and the cluster panel draw the
        # grid at the same size. No ``tight_layout``: it would undo the pinning.
        map_width_in = map_width_for_height(ax, map_height_in)
        side_pad_in = 0.02
        fig.set_size_inches(map_width_in + 2 * side_pad_in, panel_height_in)
        set_axes_size_in_inches(ax, map_width_in, map_height_in,
                                side_pad_in, side_pad_in)

        # In degrees, so the bubbles keep their share of the map at any panel
        # size. Area stays proportional to the inertia placed.
        max_node_size = node_size_for_data_diameter(ax, max_node_diameter_deg)
        node_sizes = inertia_by_node / max_inertia_node * max_node_size

        nx.draw_networkx_edges(nx_graph, pos_nodes, ax=ax,
                               width=edge_width, edge_color=edge_color)
        nx.draw_networkx_nodes(nx_graph, pos_nodes, ax=ax,
                               node_size=node_sizes,
                               node_color=INERTIA_NODE_COLOR,
                               linewidths=0)

        if legend_values_gws is not None:
            fractions = [value / (max_inertia_node * unit_factor)
                         for value in legend_values_gws]
        else:
            fractions = list(legend_fractions)

        # Size reference. ``plot`` sizes markers by DIAMETER in points while
        # ``draw_networkx_nodes`` sizes them by area, hence the square root.
        handles, labels = [], []
        for fraction in fractions:
            handles.append(plt.Line2D([], [], marker="o", lw=0,
                                      color=INERTIA_NODE_COLOR,
                                      markersize=np.sqrt(max_node_size * fraction)))
            labels.append(f"{max_inertia_node * unit_factor * fraction:.0f} GWs")
        ax.legend(handles, labels,
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncols=legend_ncols,
                  title=legend_title,
                  title_fontsize=(fontsize if legend_title_fontsize is None
                                  else legend_title_fontsize),
                  frameon=False,
                  labelspacing=0.4,
                  handletextpad=0.2,
                  columnspacing=1.2,
                  borderpad=0.1,
                  borderaxespad=0.2)

        if annotate_total:
            # Top left, the corner the legend vacated when it moved below the
            # map. Keeping it at the bottom would crowd the legend title.
            ax.text(*total_xy,
                    f"{total_inertia * unit_factor:.0f} GWs placed in total",
                    transform=ax.transAxes,
                    ha="left", va="top")

        if panel_label is not None:
            add_panel_label(ax, panel_label, y_offset=0.02, x_offset=-0.02,
                            fontsize=fontsize)

        fig.canvas.draw()
        content_height_in = fig.get_tightbbox().height
        if content_height_in > panel_height_in:
            logger.warning(
                f"Inertia panel comes out {content_height_in:.3f} in high but "
                f"was given {panel_height_in:.3f} in; reduce 'map_height_in' by "
                f"{content_height_in - panel_height_in:.3f} in."
            )

        if save_fig:
            fname = (f"method_inertia_placement_map_co2l{co2_level}"
                     f"_{target}_blackoutthres{blackoutthreshold}.svg")
            # svg only, like the other parts of the method figure -- assembled
            # in inkscape. pad_inches=0: 'tight' otherwise adds the rcParam
            # default of 0.1 in on every side, which both pushes the file over
            # the height budget and hangs invisible margins on a panel that is
            # placed by hand.
            fig.savefig(os.path.join(plot_dir, fname),
                        bbox_inches='tight', pad_inches=0., transparent=True)
            logger.info(f"Saved {fname} to {plot_dir}")

            fig.clear()
            plt.close(fig)
        else:
            plt.show()

    return fig, ax


def generate_graphical_abstract_parts(
        plot_dir: str | None = None,
        fpath_cluster_plot_data: str | None = None) -> str:
    """Generate the six parts of the graphical abstract into one folder.

    The parts are assembled by hand in inkscape, so what this buys is that they
    are all current and all in one place, rather than scattered among the
    manuscript figures with whatever was last regenerated.

    Every part is drawn at the size it is placed at, so they go into the
    assembly at 100 %: scaling one afterwards scales its text with it and takes
    the type off the point size it was set in. The row budgets they are built
    to are a 1.125 x 0.879 in box for the top row and 1.55 in of height for the
    bottom row; the two panels that overrun warn rather than silently rescale.
    All three bottom panels carry that height, so the row is set by changing the
    three defaults together -- the risk panel's ``figsize`` and the two maps'
    ``panel_height_in``, with ``map_height_in`` scaled alongside.

    The six, in the order they are drawn:

    ==========================  ==========================================
    top    generation map       the Fig. 1 scenario, one December
    top    entire network       the grid with the triggering corridors
    top    split network        the two components the trigger leaves
    bottom risk                 splits per lost-load bin over the CO2 levels
    bottom inertia map          where the Fig. 5 placement puts inertia
    bottom cluster 1 map        the most frequent split pattern of Fig. 3
    ==========================  ==========================================

    Args:
        plot_dir (str, optional): Directory the parts are written to. None puts
            them in a 'graphical_abstract' folder inside the project's figure
            directory, which already carries the 'PLOT_STYLE' subfolder -- so
            with PLOT_STYLE=joule they land next to the manuscript figures
            rather than in a tree of their own.
        fpath_cluster_plot_data (str, optional): Aggregated cluster plot data
            ('.pklz') the cluster panel is built from. None takes the same
            aggregation the manuscript figures use. This is the one part that
            cannot be computed from the network alone; 'plot_clusts.py' writes
            it.

    Returns:
        str: The directory the parts were written to.
    """
    if plot_dir is None:
        plot_dir = os.path.join(path_to_figures_sclopf, "graphical_abstract")
    if fpath_cluster_plot_data is None:
        fpath_cluster_plot_data = os.path.join(
            path_to_plot_data, "plot_data_agg_params_01bf408e.pklz")

    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Generating the graphical abstract parts in {plot_dir}")

    logger.info("[1/6] top row: scenario generation map")
    plot_generation_map_scenario(plot_dir=plot_dir, save_fig=True)

    # Two panels from one call: both read the same network and the same cascade
    # results, and they have to end up on one scale, which is what that function
    # guarantees by drawing them together.
    logger.info("[2/6, 3/6] top row: entire network and split components")
    plot_entire_network_n_split_components(plot_dir=plot_dir, save_fig=True)

    logger.info("[4/6] bottom row: lost load share distribution")
    plot_lost_load_share_distribution(plot_dir=plot_dir, save_fig=True)

    logger.info("[5/6] bottom row: inertia placement map")
    plot_inertia_placement_map_method(plot_dir=plot_dir, save_fig=True)

    # Last, and from the clustering module rather than this one: it is the only
    # part that needs the aggregated plot data instead of the network.
    logger.info("[6/6] bottom row: single cluster map")
    figure = create_single_cluster_plot_from_data(fpath_cluster_plot_data,
                                                  plot_dir=plot_dir)
    plt.close(figure)

    logger.info(f"Graphical abstract parts written to {plot_dir}")

    return plot_dir


if __name__ == "__main__":

    # Principle netwok which splits into two compoments
    logger.info("Starting to generate parts of method figure. Need to be put togehter, e.g., with inkscape!")
    setup_network_with_layout(case_str='norm', save_fig=True)
    setup_network_with_layout(case_str='sec', save_fig=True)
    setup_network_with_layout(case_str='split', save_fig=True)

    logger.info("Finished parts of small abstract network")

    generate_graphical_abstract_parts()