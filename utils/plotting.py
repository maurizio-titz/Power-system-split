import os
os.chdir("..")
from utils.config import path_to_cascade_results, path_to_pypsa_network, path_to_statistics
from utils import data_handling
from utils import cascade_simulation
import pypsa
import pickle
import numpy as np
import importlib
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
%load_ext autoreload
%autoreload 2


import sys
import pickle
import copy

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.cluster import KMeans
from utils.config import path_to_evaluation_results
from config import *

import cartopy.crs as ccrs
import cartopy
from utils.config import path_to_pypsa_network, path_to_cascade_results


from tqdm.notebook import tqdm
from sklearn.metrics import silhouette_score

from utils import data_handling, cascade_simulation

from matplotlib import colors
from utils.config import path_to_evaluation_results

def plot_clusters(indicator_name, co2l, samples_per_centroid, centroids, failed_edges=None, save_dir=None, cmap="seismic", n_subplots=16):
    
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l
        
    index_by_sample_numbers = np.argsort(samples_per_centroid)[::-1][:n_subplots]
    
    maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    mins = [min(centroids[i]) for i in index_by_sample_numbers]
    vmax = max(maxes)
    vmin =  min(mins)
    
    if "clipped_tanh" in indicator_name:
        vmax = 0
        vmin = -1.0
    elif "clipped" in indicator_name:
        vmax = 0
        vmin = vmin
    elif "tanh" in indicator_name or "sign"in indicator_name:
        vmax = 1.0
        vmin = -1.0
    
    

    n_subplots = min(n_subplots, len(centroids))
    # first_n_centroids = centroids
    ncols = 4
    n_rows = int(np.ceil(n_subplots/ncols))
# fig, axes = plt.subplots(n_rows, ncols, figsize=(10, n_rows*5))
    fig = plt.figure(figsize=(ncols*3, n_rows*3))
    gs = GridSpec(n_rows, ncols, figure=fig)
# axes = [fig.add_subplot(ax) for ax in gs]
# axes = [fig.add_subplot(gs[:3, i]) for i in range(4)]
# axs2 = [f.add_subplot(gs[3:6, i]) for i in range(4)]
    fig.subplots_adjust(hspace=-0.1, wspace=0.0)
# mpl.style.use('default')
    plt.rc('text', usetex=False)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

##### setup parameters #####
    labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']
    vmaxvals = []
    vminvals = []

#### Primary failures ######

# cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
# cmap.set_under('gainsboro', 1.0)
    # cmap = copy.copy(mpl.cm.get_cmap("coolwarm"))

# for count, co2l in enumerate(selected_co2ls[::-1]):
# for controid,samples_in_centriod,gs_iter in zip(centroids,samples_per_centroid,gs):

    plot_count = 0
    for ind, gs_iter in zip(index_by_sample_numbers, gs):
        if plot_count == n_subplots:
            break
        plot_count += 1
        
        controid,samples_in_centriod = centroids[ind],samples_per_centroid[ind]

        ax = fig.add_subplot(gs_iter)
        level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    # c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
    # c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

    # print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    # print('Max val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))


        nodes = nx.draw_networkx_nodes(nx_graph,
                                pos=pos,
                                ax=ax,
                                node_color=controid,
                                cmap=cmap,
                                vmax=vmax,
                                vmin=vmin,
                                node_size=7)
        nodes.set_edgecolor('black')
        nodes.set_linewidth(0.2)
    
        edges = nx.draw_networkx_edges(nx_graph,
                                pos=pos,
                                ax=ax,
                                edge_color="black",
                                width=0.5,
                                edge_cmap=cmap,
                                # edge_color=failed_edges_prob,
                                # edge_vmin=np.log10(vmin),
                                # edge_vmax=np.log10(vmax)
                                )
        # ax.text(0.0, 0.85, s=f"{samples_in_centriod} splits", ha="left")
        # ax.set_ylabel(f"{round(samples_in_centriod/10**6, ndigits=1)}mio splits", ha="left")

        ax.axis('off')
        ax.set_title(f"{round(samples_in_centriod/10**6, ndigits=2)}mio splits", y=0.9)


    cbar_ax = fig.add_axes([0.95, 0.375, 0.005, 0.25]) # fig.add_axes([0.89, 0.35, 0.005, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=26, width=1.0, which='both')
    # cb.ax.set_yticks([-1,0,1])
    cb.ax.set_yticks([vmin,vmax])
    cb.ax.set_ylabel(indicator_name.replace("_", " "), fontsize=26, rotation=270, labelpad=30)
# ax.text(0.95, 0.95,
#             type.replace("_", " "),
#             fontsize=25,
#             weight='bold',
#             verticalalignment='center',
#             # transform=axs_clust[3].transAxes
#             )
    # fig.tight_layout()
    fig.subplots_adjust(hspace=-0.05, wspace=-0.05)
    if save_dir != None:
        fig.savefig(save_dir + f"clusters_{indicator_name}_co2l{co2l_string}_k{len(centroids)}.pdf", bbox_inches='tight')
        

def load_lost_load_share():
    
    co2l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    path = path_to_evaluation_results + f"masks_all_n400.pklz"
    with gzip.open(path, 'rb') as out:
            masks = pickle.load(out)

    vectors = []
    for co2l,mask in zip(co2l_list,masks):
        path = path_to_evaluation_results + f"/total_lost_load_share_Co2L{co2l}_n{n_nodes}.pklz"
        with gzip.open(path, 'rb') as out:
            vectors.append(pickle.load(out)[mask])

    return np.concatenate(vectors)

def load_masked_indicator_vectors(indicator_type, co2l_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] ,n_nodes=400):
    if "component" in indicator_type:
        path = path_to_evaluation_results + f"components_masks_all_n{n_nodes}.pklz"
    else:
        path = path_to_evaluation_results + f"masks_all_n{n_nodes}.pklz"
    
    with gzip.open(path, 'rb') as out:
            masks = pickle.load(out)

    vectors = []
    for co2l,mask in zip(co2l_list,masks):
        # path = path_to_evaluation_results + f"/total_lost_load_share_Co2L{co2l}_n{n_nodes}.pklz"
        path = path_to_indicator_vectors + f"/indicator_vector_{indicator_type}_Co2l{co2l}_n{n_nodes}.pklz"
        with gzip.open(path, 'rb') as out:
            vectors.append(pickle.load(out)[-1][mask])
            
            # with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            #     results.append(pickle.load(out)[-1])

    return np.concatenate(vectors)

def map_components_to_original(values, original_ind_to_components_ind):
    """map values defined on splits to new indicator vectors defined on components

    Args:
        values (np.array): array to be mapped, e.g. failed edges, lost load etc.
        original_ind_to_components_ind (array): original index to component index mapping
    """
    
    path = path_to_evaluation_results + f"masks_all_n{n_nodes}.pklz"
    with gzip.open(path, 'rb') as out:
            masks = pickle.load(out)
    
    components_ind_to_original_ind = get_components_ind_to_original_ind(original_ind_to_components_ind[masks])
    return values[components_ind_to_original_ind]

def plot_clusters_lost_load(indicator_name, co2l, samples_per_centroid, centroids, labels, failed_edges=None, save_dir=None, cmap="seismic", n_subplots=16, total_lost_load_share=None, centroid_mean_distance=None, edge_cmap="inferno", original_ind_to_components_ind=None, cbar_label=None):
    
    # edge_cmap = mpl.cm.get_cmap(edge_cmap)
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l
        
    index_by_sample_numbers = np.argsort(samples_per_centroid)[::-1][:n_subplots]
    
        
    if total_lost_load_share is None:
        total_lost_load_share = load_lost_load_share()
    if original_ind_to_components_ind is not None:
        total_lost_load_share = map_components_to_original(total_lost_load_share, original_ind_to_components_ind)
        if failed_edges is not None:
            failed_edges = map_components_to_original(failed_edges, original_ind_to_components_ind)
        
    if centroid_mean_distance != None:
        normalized_centroid_mean_distance = centroid_mean_distance/max(centroid_mean_distance)
        
        
    # mean_lost_load_share_per_centroid = np.array([np.mean(total_lost_load_share[labels==i]) for i in range(len(centroids))])
    sum_lost_load_share_per_centroid = np.array([np.sum(total_lost_load_share[labels==i]) for i in range(len(centroids))])
    
    index_by_lost_load = np.argsort(sum_lost_load_share_per_centroid)[::-1][:n_subplots]
    
    maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    mins = [min(centroids[i]) for i in index_by_sample_numbers]
    vmax = max(maxes)
    vmin =  min(mins)
    
    if "clipped_tanh" in indicator_name:
        vmax = 0
        vmin = -1.0
    elif "clipped" in indicator_name:
        vmax = 0
        vmin = vmin
    elif "tanh" in indicator_name or "sign"in indicator_name:
        vmax = 1.0
        vmin = -1.0
    elif "lshare" in indicator_name:
        vmax = 1.0
        vmin = 0.0
        if "main" in cbar_label:
            vmin = 0.001
            centroids = np.abs(centroids-1)
        
    
    vmin_edge = 1e-3
    vmax_edge = 1
    
    if cbar_label is None:
        cbar_label = indicator_name.replace("_", " ")

    n_subplots = min(n_subplots, len(centroids))
    # first_n_centroids = centroids
    ncols = 4
    n_rows = int(np.ceil(n_subplots/ncols))
# fig, axes = plt.subplots(n_rows, ncols, figsize=(10, n_rows*5))
    fig = plt.figure(figsize=(ncols*3, n_rows*3))
    gs = GridSpec(n_rows, ncols, figure=fig)
# axes = [fig.add_subplot(ax) for ax in gs]
# axes = [fig.add_subplot(gs[:3, i]) for i in range(4)]
# axs2 = [f.add_subplot(gs[3:6, i]) for i in range(4)]
    fig.subplots_adjust(hspace=-0.1, wspace=0.0)
# mpl.style.use('default')
    plt.rc('text', usetex=False)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

##### setup parameters #####
    # labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']
    # vmaxvals = []
    # vminvals = []

#### Primary failures ######

# cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
# cmap.set_under('gainsboro', 1.0)
    # cmap = copy.copy(mpl.cm.get_cmap("coolwarm"))

# for count, co2l in enumerate(selected_co2ls[::-1]):
# for controid,samples_in_centriod,gs_iter in zip(centroids,samples_per_centroid,gs):

    plot_count = 0
    for ind, gs_iter in zip(index_by_lost_load, gs):
    # for ind, gs_iter in zip(index_by_sample_numbers, gs):
        if plot_count == n_subplots:
            break
        plot_count += 1
        
        controid,samples_in_centriod = centroids[ind],samples_per_centroid[ind]
        if not (failed_edges is None):
            failed_edges_prob = failed_edges[labels==ind].mean(axis=0)

        ax = fig.add_subplot(gs_iter)
        level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    # c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
    # c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

    # print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    # print('Max val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))


        nodes = nx.draw_networkx_nodes(nx_graph,
                                pos=pos,
                                ax=ax,
                                node_color=controid,
                                cmap=cmap,
                                vmax=vmax,
                                vmin=vmin,
                                node_size=10)
        nodes.set_edgecolor('black')
        nodes.set_linewidth(0.2)
    
        if failed_edges is None:
            edges = nx.draw_networkx_edges(nx_graph,
                                pos=pos,
                                ax=ax,
                                edge_color="black",
                                width=0.5,
                                # edge_cmap=cmap,
                                )
        else:
            failed_edges_prob_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in failed_edges_prob])
            edges = nx.draw_networkx_edges(nx_graph,
                                    pos=pos,
                                    ax=ax,
                                    # edge_color="black",
                                    width=1,
                                    edge_cmap=edge_cmap,
                                    edge_color=failed_edges_prob_log,
                                    # edge_vmin=vmin_edge,
                                    # edge_vmax=vmax_edge
                                    edge_vmin=np.log10(vmin_edge),
                                    edge_vmax=np.log10(vmax_edge)
                                    )

        ax.axis('off')
        
        
        if centroid_mean_distance != None:
            ax.set_title(f"cum. l.l.={round(sum_lost_load_share_per_centroid[ind]/ sum_lost_load_share_per_centroid.sum()*100, ndigits=1)}%\n"
                         f"n={round(samples_in_centriod/10**6, ndigits=2)}mio\n"
                         f"d={round(normalized_centroid_mean_distance[ind],ndigits=1)}", y=0.7, loc="left")
            # loc="left")
        else:
            ax.set_title(f"cum. l.l.={round(sum_lost_load_share_per_centroid[ind]/ sum_lost_load_share_per_centroid.sum()*100, ndigits=1)}%\n"
                         f"n={round(samples_in_centriod/10**6, ndigits=2)}mio", y=0.75, loc="left")
        # sum_lost_load_share_per_centroid.max()


    if failed_edges is None:
        cbar_ax = fig.add_axes([0.375, 0.05, 0.25, 0.03])
    else:
        cbar_ax = fig.add_axes([0.25, 0.05, 0.2, 0.02])
        # cbar_ax = fig.add_axes([0.0, 0.05, 0.3, 0.02])
        cbar_ax_edge = fig.add_axes([0.55, 0.05, 0.2, 0.02])
        # cbar_ax_edge = fig.add_axes([0.7, 0.05, 0.3, 0.02])
        # sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=vmin_edge, vmax=vmax_edge))
        sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge))
        cb_edge = fig.colorbar(sm_edge, cax=cbar_ax_edge, orientation='horizontal')
        cb_edge.ax.tick_params(labelsize=16, width=1.0, which='both')
        # cb.ax.set_yticks([-1,0,1])
        # cb_edge.ax.set_xticks([vmin_edge,vmax_edge])
        cb_edge.ax.set_xlabel("edge failure prob", fontsize=18, rotation=0)#, labelpad=-10)
        cb_edge.ax.xaxis.set_label_position('top')

    # cbar_ax = fig.add_axes([0.9, 0.375, 0.005, 0.125]) # fig.add_axes([0.89, 0.35, 0.005, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=18, width=1.0, which='both')
    if "main" in cbar_label:
        cb.ax.set_xticks([vmin,0.5,1], labels=[vmin, 0.5, 1])
    # cb.ax.set_xticks([vmin,vmax])
    cb.ax.set_xlabel(cbar_label, fontsize=18, rotation=0)
    cb.ax.xaxis.set_label_position('top')
    
    # fig.tight_layout()
    fig.subplots_adjust(hspace=-0.15, wspace=-0.05)
    if save_dir != None:
        if failed_edges is not None:
            fig.savefig(save_dir + f"clusters_lines_lost_load_{indicator_name}_co2l{co2l_string}_k{len(centroids)}_vmin{vmin}.pdf", bbox_inches='tight')
        else:
            fig.savefig(save_dir + f"clusters_lost_load_{indicator_name}_co2l{co2l_string}_k{len(centroids)}_vmin{vmin}.pdf", bbox_inches='tight')
            
    plt.close()
        
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    if float(a) == 1:
        return "10^" + str(b)
    else:
        return a + " * 10^{" + str(b) + "}"

sci_notation(0.001, sig_fig=5)