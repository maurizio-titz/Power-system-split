#!/usr/bin/env python

"""Plot the spatial power inhomogenetiy, i.e., SPI, vectors and their magnitude 
for different co2 scenarios."""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec

import networkx as nx

from utils.config import path_to_pre_outage_sclopf, path_to_figures_sclopf
from utils.data_handling import get_co2_levels, load_pypsa_network, build_networkx_graph, get_actual_co2_level

from tqdm import tqdm

from utils.plot_style import save_figure, setup_matplotlib_style, \
    LINE_WIDTH, AXIS_LABEL_FONTSIZE, add_panel_label, add_panel_label_fig_position

import cartopy.crs as ccrs

from pypsa import Network

from loguru import logger


def histograms_of_spi_norm(fig: Figure, ax: Axes, n_nodes:int = 600, bin_width=1, bin_lims: tuple[float, float] | None = None,
                           selected_co2_lvls: tuple[float, ...] | None = (0.6, 0.2, 0.0),
                           cmap: str = 'cividis',
                           verbose: bool = True, show_median: bool = True):
    """Draw the histogram of the spi norm on the axis 'ax'"""
    
    co2_lvls_all = get_co2_levels(n_nodes=n_nodes)
    actual_co2_lvls = get_actual_co2_level(co2_lvls_all, n_nodes=n_nodes)
    
    if selected_co2_lvls is None:
        co2_lvl_selected_internal = sorted(co2_lvls_all)[::-1]
    else:
        co2_lvl_selected_internal = sorted(list(selected_co2_lvls).copy())[::-1]
    
    if not set(co2_lvl_selected_internal).issubset(set(co2_lvls_all)):
        raise ValueError("Selected co2 levels should be in available ones!")
    
    # Load data
    path_to_dipole_vec = path_to_pre_outage_sclopf + f"/dipole_vector_time_series_all_co2ls_{n_nodes}.npy"
    path_to_vec_norm = path_to_pre_outage_sclopf + f"/spi_time_series_all_co2ls_{n_nodes}.npy"
    
    dipole_vecs = np.load(path_to_dipole_vec)
    vec_norm = np.load(path_to_vec_norm)
    
    same_first_dims = dipole_vecs.shape[0] == len(co2_lvls_all) and \
        vec_norm.shape[0] == len(co2_lvls_all)
        
    if not same_first_dims:
        raise ValueError("Numpy Arrays' first dimension should be the same as the number of Co2 levels.")
    
    if bin_lims is None:
        bin_lims = (0, np.quantile(vec_norm, 0.9999))
    
    bin_arr = np.arange(min(bin_lims), max(bin_lims), bin_width)
    cmap_co2 = plt.get_cmap(cmap).copy()
    
    for _, co2_l_r in tqdm(enumerate(co2_lvl_selected_internal), 
                             total=len(co2_lvl_selected_internal),
                             desc="Hist.", disable=not verbose):
        
        # Choose co2 level from list
        diff_list = abs(np.array(co2_lvls_all) - co2_l_r)
        idx_list = np.argmin(diff_list)
        co2_lvl_list_r = co2_lvls_all[idx_list]
        
        if diff_list[idx_list] > 1e-6 or np.count_nonzero(diff_list < 1e-6) > 1:
            raise ValueError("co2 value either not in list or duplicate in list.")
        
        network = load_pypsa_network(co2lvl=co2_lvl_list_r, 
                                     n_nodes=n_nodes, use_sclopf=True)
        
        vec_norm_r = vec_norm[idx_list]
        
        color_r = cmap_co2(idx_list / (len(co2_lvls_all) - 1))
        
        label_str_r = f"{round(actual_co2_lvls[idx_list] * 100)}\\%"
        
        median_val = np.median(vec_norm_r)
        std_val = np.std(vec_norm_r)
        if verbose:
            info_str = f"{co2_l_r}: {median_val:.4f} med, {std_val} std"
            print(info_str)
        
        if show_median:
            
            ax.axvline(x=median_val, color=color_r, ls="--", lw=1.5)
        
        ax.hist(vec_norm_r, 
            weights=network.snapshot_weightings.generators.values.astype(float),
            bins=bin_arr, histtype='step',
            color=color_r, label=label_str_r,
            linewidth=LINE_WIDTH)
        
    # Aesthetics
    ax.set_xlabel("$\\lvert SPI(t) \\rvert \\; [\\mathrm{TW} \\cdot \\mathrm{km}]$", size=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Hours", size=AXIS_LABEL_FONTSIZE)
    ax.legend(title="CO$_2$ level [\\% of 1990]")
    
    return
    
        
def draw_monthly_average_spi_vectors(fig: Figure, 
                                     grid_spec_ele, 
                                     dipole_vector_arr: np.ndarray,
                                     selected_co2_lvls: tuple[float, ...],
                                     n_nodes: int = 600,
                                     cmap: str = 'cividis',
                                     scale_factor: float = 1.2*1e4,
                                     head_width=3.,
                                     width=2., 
                                     use_latlon: bool = True,
                                     verbose: bool = True):
    """Draw the arrow for the spatial power inhomogeneity averaged for each month."""
    
    ax_ylims = 31.707498915424157, 60.29928495567561
    
    co2_lvls_all = get_co2_levels(n_nodes=n_nodes)
    actual_co2_levels = get_actual_co2_level(co2_lvls_all, n_nodes)

    cmap_co2 = plt.get_cmap(cmap).copy()
    
    ## Add axes to grid spec
    gs_all_vectors = GridSpecFromSubplotSpec(len(selected_co2_lvls), 13, 
                                             subplot_spec=grid_spec_ele)
    
    ax_all = list()
    for idx, co2_lvl_r in tqdm(enumerate(selected_co2_lvls), total=len(selected_co2_lvls),
                               desc="SPI-Vecs.", disable=not verbose):
        
        
        ax_ls = [fig.add_subplot(gs_all_vectors[idx, ii+1], 
                                 projection=ccrs.PlateCarree()) 
             for ii in range(12)]    
        # Choose co2 level from list
        diff_list = abs(np.array(co2_lvls_all) - co2_lvl_r)
        idx_co2_list = np.argmin(diff_list)
        co2_lvl_list_r = co2_lvls_all[idx_co2_list]
        pypsa_net = load_pypsa_network(co2lvl=co2_lvl_list_r, n_nodes=n_nodes)
        
        if use_latlon:
            nx_graph = build_networkx_graph(pypsa_network=pypsa_net, 
                                            snet_index='0')
            position_vector = np.array([pos for _, pos in nx_graph.nodes(data='pos')])
            center_pos = position_vector.mean(axis=0)
        else:
            center_pos = (0, 0)
        if diff_list[idx_co2_list] > 1e-6 or np.count_nonzero(diff_list < 1e-6) != 1:
            raise ValueError("co2 value either not in list or duplicate in list.")
        
        for month_nr in range(12):
            
            mask_month = pypsa_net.snapshots.month == month_nr + 1
            color_r = cmap_co2(idx_co2_list / (len(co2_lvls_all) -1))
            
            dipole_vec_month = dipole_vector_arr[idx_co2_list, :, mask_month]
            snapshot_weights_month = pypsa_net.snapshot_weightings.generators[mask_month].values.astype(int)
            
            mean_dipole_vec_0 = np.mean(np.repeat(dipole_vec_month[:, 0], snapshot_weights_month))
            mean_dipole_vec_1 = np.mean(np.repeat(dipole_vec_month[:, 1], snapshot_weights_month))
            
            ax_r = ax_ls[month_nr] 
            ax_r.arrow(center_pos[0], center_pos[1],
                       mean_dipole_vec_0 / scale_factor,
                       mean_dipole_vec_1 / scale_factor,
                       head_width=head_width,
                       width=width,
                       linewidth=.5,
                       facecolor=color_r)
            
            if idx == 0:
                title_month_str = f"{pypsa_net.snapshots[mask_month][0].month_name()[:3]}."
                ax_r.set_title(title_month_str)
            
            if use_latlon:
                ax_r.set_ylim(bottom=min(ax_ylims), top=max(ax_ylims))
            ax_r.axis('off')
            
            ax_all.append(ax_r)
    
        ax_grid_r = fig.add_subplot(gs_all_vectors[idx, 1:], zorder=-1)
        for _, spine in ax_grid_r.spines.items():
            spine.set_visible(False)
            
        ax_grid_r.tick_params(which='both', left=False, labelleft=False,
               bottom=False, labelbottom=False)
        ax_grid_r.axis('off')
        ax_grid_r.grid(False)
        
        ax_grid_r.set_ylim(ax_r.get_ylim())
        hline_r = ax_grid_r.axhline(y=center_pos[1], linestyle="-", zorder=-1,
                          color="darkgrey", linewidth=.5*LINE_WIDTH)
        
        pos_ax_grid_r = ax_grid_r.get_position()
        x_pos_co2_label = pos_ax_grid_r.x0 - 0.005
        
        to_figure = ax_grid_r.transData + fig.transFigure.inverted()
        y_pos_co2_label = to_figure.transform([0, center_pos[1]])[1]
        
        label_str_r = f"CO$_2$ level $={round(actual_co2_levels[idx_co2_list]*100)} \\%$"
        
        fig.text(x_pos_co2_label, y_pos_co2_label, label_str_r,
                 horizontalalignment='right', va='center',
                 fontsize=AXIS_LABEL_FONTSIZE)
    
    [xx.set_aspect('equal') for xx in ax_all]
    
    return ax_all  

    
def plot_spi_histograms_n_mean_spi_per_month(selected_co2_lvls: tuple[float, ...] = (.6, .2, 0.),
                              n_nodes: int=600, verbose: bool = True, 
                              save_prefix: str | None = None):
    """Plot the complete picture with histograms of SPI vectors and the vectors for each month
    for a selected subset of CO2 levels."""
    
    setup_matplotlib_style()
    
    fig = plt.figure(figsize=(10, 4 + 4 * (len(selected_co2_lvls)/3.)))
    
    gs0 = GridSpec(2, 1, figure=fig, hspace=0.5, height_ratios=[1, 2])
    ax_hist = fig.add_subplot(gs0[0])
    
    histograms_of_spi_norm(fig, ax_hist, 
                           selected_co2_lvls=selected_co2_lvls,
                           verbose=verbose)
    
    # Load dipole vectors for all co2 levels and snapshots
    dipole_vector_arr = np.load(path_to_pre_outage_sclopf + f"/dipole_vector_time_series_all_co2ls_{n_nodes}.npy")
    
    ax_vec_all = draw_monthly_average_spi_vectors(fig, gs0[1], dipole_vector_arr, 
                                     selected_co2_lvls, verbose=verbose)
    
    fig_fname = f"spi_vectors_co2_levels_" + "_".join([f"{xx:.2f}" for xx in selected_co2_lvls])
    
    pos_ax_hist = ax_hist.get_position()
    
    label_offset_x = -0.1
    label_offset_y = 0.025
    
    label_pos_x = pos_ax_hist.x0 + label_offset_x
    
    add_panel_label_fig_position(fig, 0, label_pos_x, 
                                 pos_ax_hist.y1 + label_offset_y)
    add_panel_label_fig_position(fig, 1, label_pos_x, 
                                 ax_vec_all[0].get_position().y1 + label_offset_y)
    
    if save_prefix is not None:
        fig_fname = save_prefix + "_" + fig_fname
    
    save_figure(fig, fig_fname, path_to_figures_sclopf)
    
    return gs0

if __name__ == "__main__":
    
    logger.info("Starting to plot 'Spatial Power Inhomogeneity'")
    plot_spi_histograms_n_mean_spi_per_month(verbose=False)
    logger.info("Finished")
    