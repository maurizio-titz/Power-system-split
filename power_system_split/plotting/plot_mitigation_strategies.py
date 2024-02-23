#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the results that came out of the mitigation strategies.
The two mitigation strategies are line extension and inertia placement."""

import numpy as np

import networkx
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 16,
    'axes.labelsize': 16,'axes.titlesize': 16, 'figure.titlesize' : 16})
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

from utils import data_handling

import gzip
import pickle

from glob import glob

color1 = "#d95f02"
color2 = "#7570b3"
color3 = "#1b9e77"
color4 = "#e7298a"

color_ls = [color1, color2,
            color3, color4]

def plot_map_inertia_placement(co2_lvl, nn=400, max_iter=10000,
                               max_node_size=800, 
                               edge_width=.2, delta_Erot=10, resolve_strategy="random",
                               save_fig=False):
    """Plot the results of the inertia placement"""
    
    
    # Load graph
    fpath_pypsa = ("data/European_networks_sclopf/sclopf-elec_s_" +
                   f"{nn}_ec_lv1.0_Co2L{co2_lvl}-2920SEG.nc")
    pypsa_net = data_handling.load_pypsa_network(fpath_pypsa, use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)
    pos_nodes = networkx.get_node_attributes(nx_graph, 'pos')
    
    # Load synthetic inertia placement
    fpath_in = ("results/sclopf/syn_inertia_mitigation/"+ 
                f"synthetic_inertia_placement_Co2{co2_lvl:.2f}" + 
                f"_N{nn}_deltarotE{delta_Erot:.2f}_rocofthres-1.00" + 
                f"_lshare0.00_maxiter{max_iter}_{resolve_strategy}.pklz")
    with gzip.open(fpath_in) as fh_in:
        modified_comp_idx, _, inertia_placed_ls, resolve_counter, still_random_counter = pickle.load(fh_in)
        
    inertia_placed_res_arr = np.array(inertia_placed_ls)
    
    inertia_node_idx = inertia_placed_res_arr[:, 1:3]
    
    node_count_ls = [0] * len(nx_graph)
    for indi_idx_r, indi_count_r in inertia_node_idx:
        node_count_ls[int(indi_idx_r)] += indi_count_r
    
    # Plot graph
    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(18,10))   
    ax2_twin = ax2.twinx()
    ## Normalize node_size
    max_node_count = max(node_count_ls)
    node_width_arr = (np.array(node_count_ls) / max_node_count) * max_node_size
    
    networkx.draw_networkx_edges(nx_graph, pos_nodes, width=edge_width, ax=ax)
    networkx.draw_networkx_nodes(nx_graph, pos_nodes, node_size=node_width_arr,
                                 node_color=color1, ax=ax)
    
    ## Plot mitigated lost load and remaining lost splits over time
    total_dangerous_splits = len(modified_comp_idx)
    inertia_placed_arr = np.array(inertia_placed_ls)
    ax2_twin.plot(inertia_placed_arr[:, 0], np.cumsum(inertia_placed_arr[:, 3])/total_dangerous_splits, color=color2,
                  linestyle='--')
    ax2.plot(inertia_placed_arr[:, 0], (1-inertia_placed_arr[:, 4]/total_dangerous_splits),
             color=color2, 
             linestyle="-")
    
    # Asthetics
    legend_small_node_size = max_node_size * 0.25
    legend_large_node_size = max_node_size * 0.75
    ls_legend = [legend_small_node_size, legend_large_node_size]
    for size_r in ls_legend:
        ax.plot([], [], marker='o', color=color1,
                 markersize=np.sqrt(size_r), label=f"{int(size_r*delta_Erot)} MWs", lw=0,
                 )
    ax.legend(labelspacing=1, loc='center left', bbox_to_anchor=(.7, 0.975),
               frameon=False, fontsize=22)
    
    ax.axis('off')
    
    para_text = (f"$N={nn}$, CO$_2$-Level $={co2_lvl}$, \n$\\Delta E_0 = {delta_Erot:.2f}$MWs")
    fig.text(0.0, 0.975, para_text, horizontalalignment='left', 
             verticalalignment='top',fontsize=22, transform=ax.transAxes)
    if resolve_strategy == "random":
        resolve_strategy_str = "Place inertia randomly"
        
    elif resolve_strategy == "concentrate":
        resolve_strategy_str = "Concentrate inertia where inertia was placed previously"
        
    elif resolve_strategy == "hindsight":
        resolve_strategy_str = "Place inertia where max. lost load in next step."
        
    elif resolve_strategy == "hindsight_concentrate":
        resolve_strategy_str = "Place inertia where max. lost load in next step + concentrate inertia."
    else: 
        raise IOError(f"Resolve equality method '{resolve_strategy}' not known!")
    
    label_text = (f"{resolve_strategy_str}"+
                  f"\n Resolve needed: {(resolve_counter *100/max_iter)}\\%")
    if resolve_counter > 0:
        label_text += f", Random Decisions: {(((still_random_counter/resolve_counter)*100)):.2f}\\%"
    fig.text(0.5, 0.05, label_text, horizontalalignment='center',
             fontsize=16, transform=ax.transAxes)
    
    ax2.set_xlabel("$t_n$")
    ax2.set_ylabel("$N_{\\textrm{splits mit}}/N_{\\textrm{splits}}$")
    ax2_twin.set_ylabel("$L_{\\textrm{mit, loss}}/N_{\\textrm{splits}}$ (--)")
    
    if save_fig:
        fig_path = f"syn_inertia_map_co2lvl{co2_lvl:.2f}_nn{nn}_deltaErot{delta_Erot:.2f}_{resolve_strategy}.png"
        
        fig.savefig(fig_path, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        
    else:
        plt.show()
    
    return

def plot_all_syn_inertia_map(nn=400):
    """Plot all c02 lvls and rot E"""
    
    file_list = glob(f"results/sclopf/syn_inertia_mitigation/*N{nn}*.pklz")
    
    co2_lvl_ls = [float(xx.split("Co2")[1].split(f"_N{nn}")[0]) for xx in file_list]
    rotE_ls = [float(xx.split("rotE")[1].split("_rocof")[0]) for xx in file_list]
    method_ls = [xx.split("_")[-1].split(".pklz")[0] for xx in file_list]
    
    for co2_r, rotE_r, method_r in zip(co2_lvl_ls, rotE_ls, method_ls):
        plot_map_inertia_placement(co2_r, delta_Erot=rotE_r, 
                                   resolve_strategy=method_r, save_fig=True)
    

def plot_mitigate_load_loss_n_splits_over_time_diff_E0(co2_lvl_ls=[.1, .8], deltaE_ls=[1, 5, 20], 
                                                       nn=400, max_iter=10000,
                                                       resolve_method="random",
                                                       save_fig=True):
    """Plot the mitigate 

    Args:
        co2_lvl (_type_): _description_
        nn (int, optional): _description_. Defaults to 400.
        save_fig (bool, optional): _description_. Defaults to True.
    """
    
    linestyle_ls = ["-", "--", ":", "-."]
    
    # Load synthetic inertia placement
    fig, ax = plt.subplots(3, 1, sharex='all', 
                           figsize=(10, 9))
    already_labeled = False
    for co2_idx, co2_lvl_r in enumerate(co2_lvl_ls):
        for E_idx, delta_Erot in enumerate(deltaE_ls):
            fpath_in = ("results/sclopf/syn_inertia_mitigation/"+ 
                        f"synthetic_inertia_placement_Co2{co2_lvl_r:.2f}" + 
                        f"_N{nn}_deltarotE{delta_Erot:.2f}_rocofthres-1.00_lshare0.00" + 
                        f"_maxiter{max_iter}_{resolve_method}.pklz")
            with gzip.open(fpath_in) as fh_in:
                modified_comp_idx, _, inertia_placed_ls, _, _ = pickle.load(fh_in)
                
            inertia_placed_arr = np.array(inertia_placed_ls)
            
            total_dangerous_splits = len(modified_comp_idx)
            inertia_placed_arr = np.array(inertia_placed_ls)
            label_str = f" = {delta_Erot} MWs"
            if not already_labeled:
                label_str = '$\\Delta E_{\\rm rot}$' + label_str
                already_labeled = True
                
            style_dict = dict(color=color_ls[E_idx], linestyle=linestyle_ls[co2_idx])
            if co2_idx == 0:
                style_dict['label'] = label_str
                
            ax[0].plot(inertia_placed_arr[:, 0], (1-inertia_placed_arr[:, 4]/total_dangerous_splits),
                    **style_dict)
            
            ax[1].plot(inertia_placed_arr[:, 0],
                    np.cumsum(inertia_placed_arr[:, 3])/total_dangerous_splits,
                    **style_dict)
            
            ax[2].plot(inertia_placed_arr[:, 0],
                    np.cumsum(inertia_placed_arr[:, 2] * delta_Erot),
                    **style_dict)
            
    [ax[0].plot([], [], color='k', linestyle=linestyle_ls[idx], label=f"CO$_2$={co2_r:.1f}")
     for idx, co2_r in enumerate(co2_lvl_ls)]
    # Aesthetics 
    ax[0].set_ylabel("$N_{\\rm sp, m}/N_{\\textrm{sp}}$")
    ax[1].set_ylabel("$L_{\\rm l, m}/N_{\\textrm{spy}}$")
    ax[2].set_ylabel("$E_{\\rm r, t}$/MWs")
    ax[-1].set_xlabel("$t_n$")
    
    ax[0].set_xlim(left=0, right=max_iter)
    ax[0].legend(loc="upper left", fontsize=14)
    
    fig.tight_layout()
    
    fig.text(.99, 0.01, resolve_method, horizontalalignment="right", fontsize=20)
    
    if save_fig:
        fig_path = f"syn_inertia_co2lvl_over_time_{resolve_method}.png"
        fig.savefig(fig_path)
        fig.clear()
        plt.close(fig)
    
    else:
        plt.show()
    
    return

def plot_all_syn_inertia_over_time():
    
    resolve_method = ["random", "concentrate", "hindsight"]
    [plot_mitigate_load_loss_n_splits_over_time_diff_E0(resolve_method=xx, save_fig=True) for xx in resolve_method]
    
    return

def aggregate_months(casc_dict):
    
    keys_all = list(casc_dict.keys())
    month_str_ls = ["01", "02", "03", "04", "05", "06", 
                    "07", "08", "09", "10", "11", "12"]
    nr_splits_per_month = list()
    for month_r in month_str_ls:
        month_r_keys = [xx for xx in keys_all if xx[5:7] == month_r]
        
        if len(month_r_keys) > 0:
            nr_splits = 0
            for key_r in month_r_keys:
                nr_splits += len(casc_dict[key_r])
            nr_splits_per_month.append([int(month_r), nr_splits])
            
    return np.array(nr_splits_per_month)


def plot_line_mitigation():
    """"""
    
    # Load file
    meta_data_path = "results/sclopf/cascade_results/"
    
    norm_path = meta_data_path + "system_splits_Co2L0.1_n400.pickle"
    with open(norm_path, 'rb') as fh_in_norm:
        norm_dict = pickle.load(fh_in_norm)
    
    norm_splits_per_month = aggregate_months(norm_dict)
    
    small_change_path = meta_data_path + "system_splits_Co2L0.1_n400_lineextension_nnlines10_deltanumpara3.0000.pklz"
    with gzip.open(small_change_path, 'rb') as fh_in_small:
        vulnerable_edges_small, small_dict = pickle.load(fh_in_small)
        
    small_splits_per_month = aggregate_months(small_dict)
    
    large_change_path = meta_data_path + "system_splits_Co2L0.1_n400_lineextension_nnlines10_deltanumpara5.0000.pklz"
    with gzip.open(large_change_path, 'rb') as fh_in_large:
        vulnerable_edges_large, large_dict = pickle.load(fh_in_large)
    
    large_splits_per_month = aggregate_months(large_dict)
    
    return norm_dict, small_dict, large_dict
    return vulnerable_edges_small, vulnerable_edges_large, norm_splits_per_month, small_splits_per_month, large_splits_per_month
