#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the results that came out of the mitigation strategies.
The two mitigation strategies are line extension and inertia placement."""

import os
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

from tqdm import tqdm

from glob import glob

color1 = "#d95f02"
color2 = "#7570b3"
color3 = "#1b9e77"
color4 = "#e7298a"

color_ls = [color1, color2,
            color3, color4]

script_path = os.path.dirname(__file__)


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

def aggregate_months(casc_dict, show_progress=True, to_month=12):
    
    keys_all = list(casc_dict.keys())
    month_str_ls = [f"{xx+1:02d}" for xx in range(to_month)]
    nr_splits_per_month = list()
    for month_r in tqdm(month_str_ls, disable=not show_progress):
        month_r_keys = [xx for xx in keys_all if xx[5:7] == month_r]
        
        if len(month_r_keys) > 0:
            nr_splits = 0
            for key_r in month_r_keys:
                nr_splits += len(casc_dict[key_r])
            nr_splits_per_month.append([int(month_r), nr_splits])
            
    return np.array(nr_splits_per_month)


def data_line_mitigation_diff_nnlines(delta_para=5, save_res=True):
    """"""
    
    # Load file
    meta_data_path = "results/sclopf/cascade_results/"
    
    norm_path = meta_data_path + "system_splits_Co2L0.1_n400.pklz"
    with gzip.open(norm_path, 'rb') as fh_in_norm:
        norm_dict = pickle.load(fh_in_norm)
    
    norm_splits_per_month = aggregate_months(norm_dict)
    
    mitigate_split_ls = list()
    for nn_lines in tqdm([10, 20, 40]):
        small_change_path = (meta_data_path + "system_splits_Co2L0.1_n400_" + 
                             f"lineextension_nnlines{nn_lines}" +
                             f"_deltanumpara{delta_para:.4f}_" + 
                             "stopped_2013-03-01 00:00.pklz")
        with gzip.open(small_change_path, 'rb') as fh_in_r:
            _, vulnerable_edges_r, dict_r = pickle.load(fh_in_r)
        splits_per_month_r = aggregate_months(dict_r)
        
        res_out_r = (nn_lines, vulnerable_edges_r, splits_per_month_r)
        mitigate_split_ls.append(res_out_r)
        
    if save_res:
        path_out = script_path + f"/cascdes_par_months_deltanumpara{delta_para:.4f}_diffnn_lines.pklz"
        with gzip.open(path_out, 'wb') as fh_out:
            pickle.dump((norm_splits_per_month, mitigate_split_ls), fh_out)
    
    return norm_splits_per_month, mitigate_split_ls


def data_line_mitigation_diff_numpara(nn_lines=10, save_res=True):
    """"""
    
    # Load file
    meta_data_path = "results/sclopf/cascade_results/"
    
    norm_path = meta_data_path + "system_splits_Co2L0.1_n400.pklz"
    with gzip.open(norm_path, 'rb') as fh_in_norm:
        norm_dict = pickle.load(fh_in_norm)
    
    norm_splits_per_month = aggregate_months(norm_dict)
    
    mitigate_split_ls = list()
    for delta_para in tqdm([1, 5]):
        small_change_path = (meta_data_path + "system_splits_Co2L0.1_n400_" + 
                             f"lineextension_nnlines{nn_lines}" +
                             f"_deltanumpara{delta_para:.4f}_" + 
                             "stopped_2013-03-01 00:00.pklz")
        with gzip.open(small_change_path, 'rb') as fh_in_r:
            _, vulnerable_edges_r, dict_r = pickle.load(fh_in_r)
        splits_per_month_r = aggregate_months(dict_r, to_month=2)
        
        res_out_r = (delta_para, vulnerable_edges_r, splits_per_month_r)
        mitigate_split_ls.append(res_out_r)
        
    if save_res:
        path_out = script_path + f"/cascdes_par_months_nnlines{nn_lines}_diffnumpara_lines.pklz"
        print(path_out)
        with gzip.open(path_out, 'wb') as fh_out:
            pickle.dump((norm_splits_per_month, mitigate_split_ls), fh_out)
    
    return norm_splits_per_month, mitigate_split_ls


def plot_most_likely_lines_on_map(savefig=True):
    
    # Load network
    pypsa_net = data_handling.load_pypsa_network("data/European_networks_sclopf/sclopf-elec_s_400_ec_lv1.0_Co2L0.1-2920SEG.nc", use_sclopf=True)
    nx_graph = data_handling.build_networkx_graph(pypsa_net, snet_index=0)
    
    # Load file
    path_out = (script_path + 
                    f"/cascdes_par_months_deltanumpara{1:.4f}_diffnn_lines.pklz")
    
    with gzip.open(path_out) as fh_in:
        norm_month_ls, diff_para_ls = pickle.load(fh_in)
        
    vuln_res_dict = dict()
    for ele_r in diff_para_ls[::-1]:
        nn_lines, vulnerable_ls, _ = ele_r
        vuln_res_dict[nn_lines] = vulnerable_ls
    
    # Plot it
    edge_color_ls = list()
    edge_width_ls = list()
    
    for ll in nx_graph.edges():
        
        if ll in vuln_res_dict[10] and ll in vuln_res_dict[20] and ll in vuln_res_dict[40]:
            edge_color_ls.append(color_ls[0])
            edge_width_ls.append(3)
            
        elif ll in vuln_res_dict[40] and ll in vuln_res_dict[20] and ll not in vuln_res_dict[10]:
            edge_color_ls.append(color_ls[1])
            edge_width_ls.append(3)
            
        elif ll in vuln_res_dict[40] and ll not in vuln_res_dict[20] and ll not in vuln_res_dict[10]:
            edge_color_ls.append(color_ls[2])
            edge_width_ls.append(3)
            
        else:
            edge_color_ls.append('gray')
            edge_width_ls.append(1.3)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pos_graph = networkx.get_node_attributes(nx_graph, "pos")
    neti = networkx.draw_networkx_edges(nx_graph, pos_graph, ax=ax,
                                 edge_color=edge_color_ls,
                                width=edge_width_ls)
    
    plt.legend([matplotlib.lines.Line2D([0, 1], [0, 1], color=color_ls[idx], lw=3) for idx in range(3)],
               [10, 20, 40])
    
    ax.axis('off')
    
    if savefig:
        fig_path = script_path + "/lines_on_map.png"
        fig.savefig(fig_path, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
    
    return
    

def plot_bar_histograms_diff_nnlines(save_fig=False):
    
    # Load data
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for idx_para, delta_para in enumerate([1, 5]):
        path_out = (script_path + 
                    f"/cascdes_par_months_deltanumpara{delta_para:.4f}_diffnn_lines.pklz")
    
        with gzip.open(path_out) as fh_in:
            norm_month_ls, diff_para_ls = pickle.load(fh_in)
            
        
        
        width_bar = 1/(len(diff_para_ls)+1) - 0.05
        
        norm_month_arr = np.array(norm_month_ls)
        
        ax[idx_para].bar(norm_month_arr[:, 0] - width_bar, norm_month_arr[:, 1]*1e-3, width_bar,
            label="Normal", color=color_ls[0])
        
        for idx, ele_r in enumerate(diff_para_ls):
            nn_lines, _, splits_per_month = ele_r

            ax[idx_para].bar(splits_per_month[:, 0] + idx * width_bar,
                splits_per_month[:, 1]*1e-3, width_bar, label="$N_{\\rm lines}=" +
                f"{nn_lines}" + "$", color=color_ls[idx+1])

        title_str = "Diff. number of lines for $\\Delta C_{\\rm para}=" + f"{delta_para}$"
        ax[idx_para].set_title(title_str)
        
    # Aesthetics
    for ax_r in ax: 
        ax_r.set_xlabel("Months", fontsize=18)
        ax_r.set_ylabel("\\# Splits $/ 10^3$", fontsize=18)
    
    ax[0].legend(ncols=1, fontsize=12)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = script_path + f"/nr_of_cascades_diffnnlines.png"
        fig.savefig(fig_path, bbox_inches='tight')
        
        fig.clear()
        plt.close(fig)
        
    else:
        plt.show()
    
    
    return


def plot_bar_histograms_diff_numpara(save_fig=False):
    
    # Load data
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for idx_li, nn_lines in enumerate([10, 20, 40]):
        path_out = (script_path + 
                    f"/cascdes_par_months_nnlines{nn_lines}_diffnumpara_lines.pklz")
    
        with gzip.open(path_out) as fh_in:
            norm_month_ls, diff_para_ls = pickle.load(fh_in)
            
        
        
        width_bar = 1/(len(diff_para_ls)+1) - 0.05
        
        norm_month_arr = np.array(norm_month_ls)
        
        ax[idx_li].bar(norm_month_arr[:, 0] - width_bar, norm_month_arr[:, 1]*1e-3, width_bar,
            label="Normal", color=color_ls[0])
        
        
        
        for idx, ele_r in enumerate(diff_para_ls):
            delta_para, _, splits_per_month = ele_r

            ax[idx_li].bar(splits_per_month[:, 0] + idx * width_bar,
                splits_per_month[:, 1]*1e-3, width_bar, label="$\\Delta {C_{\\rm para}=" +
                f"{delta_para:.1f}" + "}$", color=color_ls[idx+1])

        title_str = "Diff. capacities for $N_{\\rm lines}=" + f"{nn_lines}$"
        ax[idx_li].set_title(title_str)
        
    # Aesthetics
    for ax_r in ax: 
        ax_r.set_xlabel("Months", fontsize=18)
        ax_r.set_ylabel("\\# Splits $/ 10^3$", fontsize=18)
    
    ax[0].legend(ncols=1, fontsize=12)
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = script_path + f"/nr_of_cascades_diffnumpara.png"
        fig.savefig(fig_path, bbox_inches='tight')
        
        fig.clear()
        plt.close(fig)
        
    else:
        plt.show()
    
    return
