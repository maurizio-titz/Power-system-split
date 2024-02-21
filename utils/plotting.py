import os

os.chdir("..")
# from utils.config import path_to_cascade_results, path_to_pypsa_network, path_to_statistics
import pypsa
import pickle
import numpy as np
import importlib
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd

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
import gzip

import cartopy.crs as ccrs
import cartopy

from tqdm.notebook import tqdm
from sklearn.metrics import silhouette_score

from matplotlib import colors

# from utils.config import path_to_pypsa_network, path_to_cascade_results
from utils.config import path_to_evaluation_results, path_to_indicator_vectors

# from utils import data_handling, cascade_simulation


def plot_clusters(
    nx_graph,
    pos,
    indicator_name,
    co2l,
    samples_per_centroid,
    centroids,
    failed_edges=None,
    save_dir=None,
    cmap="seismic",
    n_subplots=16,
):
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l

    index_by_sample_numbers = np.argsort(samples_per_centroid)[::-1][:n_subplots]

    maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    mins = [min(centroids[i]) for i in index_by_sample_numbers]
    vmax = max(maxes)
    vmin = min(mins)

    if "clipped_tanh" in indicator_name:
        vmax = 0
        vmin = -1.0
    elif "clipped" in indicator_name:
        vmax = 0
        vmin = vmin
    elif "tanh" in indicator_name or "sign" in indicator_name:
        vmax = 1.0
        vmin = -1.0
    if "share" in indicator_name:
        vmax = 1
        vmin = 0

    n_subplots = min(n_subplots, len(centroids))
    # first_n_centroids = centroids
    ncols = 4
    n_rows = int(np.ceil(n_subplots / ncols))
    # fig, axes = plt.subplots(n_rows, ncols, figsize=(10, n_rows*5))
    fig = plt.figure(figsize=(ncols * 3, n_rows * 3))
    gs = GridSpec(n_rows, ncols, figure=fig)
    # axes = [fig.add_subplot(ax) for ax in gs]
    # axes = [fig.add_subplot(gs[:3, i]) for i in range(4)]
    # axs2 = [f.add_subplot(gs[3:6, i]) for i in range(4)]
    fig.subplots_adjust(hspace=-0.1, wspace=0.0)
    # mpl.style.use('default')
    plt.rc("text", usetex=False)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    ##### setup parameters #####
    labels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}"]
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

        controid, samples_in_centriod = centroids[ind], samples_per_centroid[ind]

        ax = fig.add_subplot(gs_iter)

        # Load likelihoods as dictionary and transform into array
        # c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
        # c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

        # print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
        # print('Max val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))

        nodes = nx.draw_networkx_nodes(
            nx_graph,
            pos=pos,
            ax=ax,
            node_color=controid,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            node_size=7,
        )
        nodes.set_edgecolor("black")
        nodes.set_linewidth(0.2)

        edges = nx.draw_networkx_edges(
            nx_graph,
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

        ax.axis("off")
        ax.set_title(f"{sci_notation(samples_in_centriod, sig_fig=1)} splits", y=0.9)

    cbar_ax = fig.add_axes(
        [0.95, 0.375, 0.005, 0.25]
    )  # fig.add_axes([0.89, 0.35, 0.005, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=26, width=1.0, which="both")
    # cb.ax.set_yticks([-1,0,1])
    cb.ax.set_yticks([vmin, vmax])
    cb.ax.set_ylabel(
        indicator_name.replace("_", " "), fontsize=26, rotation=270, labelpad=30
    )
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
        fig.savefig(
            save_dir
            + f"clusters_{indicator_name}_co2l{co2l_string}_k{len(centroids)}.pdf",
            bbox_inches="tight",
        )


def plot_indicator_vectors(
    nx_graph,
    pos,
    indicator_name,
    co2l,
    indicator_vectors,
    save_dir=None,
    cmap="seismic",
    n_subplots=16,
    shuffle=False,
):
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l

    vmax = 1
    vmin = 0

    if "clipped_tanh" in indicator_name:
        vmax = 0
        vmin = -1.0
    elif "clipped" in indicator_name:
        vmax = 0
        vmin = vmin
    elif "tanh" in indicator_name or "sign" in indicator_name:
        vmax = 1.0
        vmin = -1.0
    if "share" in indicator_name:
        vmax = 1
        vmin = 0

    max_val = indicator_vectors.max()
    if max_val > 1:
        print(f"setting vmax to {max_val}")
        vmax = max_val

    n_subplots = min(n_subplots, len(indicator_vectors))
    if shuffle:
        rand_inds = np.random.choice(
            np.arange(len(indicator_vectors)), size=n_subplots, replace=False
        )
    ncols = 8
    n_rows = int(np.ceil(n_subplots / ncols))
    fig = plt.figure(figsize=(ncols * 3, n_rows * 3))
    gs = GridSpec(n_rows, ncols, figure=fig)
    fig.subplots_adjust(hspace=-0.1, wspace=0.0)
    plt.rc("text", usetex=False)

    plot_count = 0
    for ind, gs_iter in enumerate(gs):
        if plot_count == n_subplots:
            break
        plot_count += 1

        if shuffle:
            ind = rand_inds[ind]
        indicator_vector = indicator_vectors[ind]

        ax = fig.add_subplot(gs_iter)

        nodes = nx.draw_networkx_nodes(
            nx_graph,
            pos=pos,
            ax=ax,
            node_color=indicator_vector,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
        )
        nodes.set_edgecolor("black")
        nodes.set_linewidth(0.2)

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edge_color="black",
            width=0.5,
            edge_cmap=cmap,
            # edge_color=failed_edges_prob,
            # edge_vmin=np.log10(vmin),
            # edge_vmax=np.log10(vmax)
        )

    # cbar_ax = fig.add_axes([0.95, 0.375, 0.005, 0.25])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # cb = fig.colorbar(sm, cax=cbar_ax)
    cb = fig.colorbar(sm)
    fig.subplots_adjust(hspace=-0.05, wspace=-0.05)
    if save_dir != None:
        fig.savefig(
            save_dir + f"split_plots_{indicator_name}_co2l{co2l_string}.pdf",
            bbox_inches="tight",
        )

    return fig


def plot_centroid_with_failures(
    nx_graph,
    pos,
    group_failed_edges,
    cmap,
    edge_cmap,
    vmax,
    vmin,
    vmin_edge,
    vmax_edge,
    controid,
    failed_edges_prob,
    ax,
):
    nodes = nx.draw_networkx_nodes(
        nx_graph,
        pos=pos,
        ax=ax,
        node_color=controid,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        node_size=10,
    )
    nodes.set_edgecolor("black")
    nodes.set_linewidth(0.2)

    if group_failed_edges is None:
        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edge_color="black",
            width=0.5,
            # edge_cmap=cmap,
        )
    else:
        failed_edges_prob_log = np.array(
            [np.log10(x) if x > 1e-12 else -np.inf for x in failed_edges_prob]
        )
        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            # edge_color="black",
            width=1,
            edge_cmap=edge_cmap,
            edge_color=failed_edges_prob_log,
            # edge_vmin=vmin_edge,
            # edge_vmax=vmax_edge
            edge_vmin=np.log10(vmin_edge),
            edge_vmax=np.log10(vmax_edge),
        )


def load_lost_load_share(n_nodes=400, mask=None):
    co2l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if mask is None:
        path = path_to_evaluation_results + f"masks_all_n400.pklz"
        with gzip.open(path, "rb") as out:
            mask = pickle.load(out)

    vectors = []
    for co2l, mask in zip(co2l_list, mask):
        path = (
            path_to_evaluation_results
            + f"/total_lost_load_share_Co2L{co2l}_n{n_nodes}.pklz"
        )
        with gzip.open(path, "rb") as out:
            vectors.append(pickle.load(out)[mask])

    return np.concatenate(vectors)


def load_masked_indicator_vectors(
    indicator_type,
    co2l_list=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    n_nodes=400,
    masks=None,
):
    if masks is None:
        if "component" in indicator_type:
            path = path_to_evaluation_results + f"components_masks_all_n{n_nodes}.pklz"
        else:
            path = path_to_evaluation_results + f"masks_all_n{n_nodes}.pklz"

        with gzip.open(path, "rb") as out:
            masks = pickle.load(out)

    vectors = []
    for co2l, mask in zip(co2l_list, masks):
        # path = path_to_evaluation_results + f"/total_lost_load_share_Co2L{co2l}_n{n_nodes}.pklz"
        path = (
            path_to_indicator_vectors
            + f"/indicator_vector_{indicator_type}_Co2l{co2l}_n{n_nodes}.pklz"
        )
        with gzip.open(path, "rb") as out:
            vectors.append(pickle.load(out)[-1][mask])

            # with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            #     results.append(pickle.load(out)[-1])

    return np.concatenate(vectors)


def map_components_to_original(values, original_ind_to_components_ind, n_nodes=400):
    """map values defined on splits to new indicator vectors defined on components

    Args:
        values (np.array): array to be mapped, e.g. failed edges, lost load etc.
        original_ind_to_components_ind (array): original index to component index mapping
    """

    path = path_to_evaluation_results + f"masks_all_n{n_nodes}.pklz"
    with gzip.open(path, "rb") as out:
        masks = pickle.load(out)

    components_ind_to_original_ind = get_components_ind_to_original_ind(
        original_ind_to_components_ind[masks]
    )
    return values[components_ind_to_original_ind]


def get_components_ind_to_original_ind(ind_to_ind):
    n_components_indicator_vectors = max(ind_to_ind[-1])
    ind_to_ind_reverse = np.empty(n_components_indicator_vectors + 1, dtype=int)
    for original_ind, comp_inds in enumerate(ind_to_ind):
        ind_to_ind_reverse[comp_inds] = original_ind
    return ind_to_ind_reverse


def plot_clusters_lost_load(
    nx_graph,
    pos,
    indicator_name,
    co2l,
    samples_per_centroid,
    centroids,
    labels,
    edge_centroids=None,
    save_dir=None,
    cmap="seismic",
    n_subplots=16,
    total_lost_load_share=None,
    centroid_mean_distance=None,
    edge_cmap="inferno",
    # original_ind_to_components_ind=None,
    cbar_label=None,
    mask=None,
    custom_order=None,
    ncols=4,
    titles=None,
    group_affiliation=None,
):
    # edge_cmap = mpl.cm.get_cmap(edge_cmap)
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l

    index_by_sample_numbers = np.argsort(samples_per_centroid)[::-1][:n_subplots]

    if total_lost_load_share is None:
        total_lost_load_share = load_lost_load_share(mask=mask)
    # if original_ind_to_components_ind is not None:
    #     total_lost_load_share = map_components_to_original(
    #         total_lost_load_share, original_ind_to_components_ind
    #     )
    #     if failed_edges is not None:
    #         failed_edges = map_components_to_original(
    #             failed_edges, original_ind_to_components_ind
    #         )
    # if edge_centroids is None:
    #     failed_edges = load_masked_indicator_vectors("failed_edges", masks=mask)

    if centroid_mean_distance != None:
        normalized_centroid_mean_distance = centroid_mean_distance / max(
            centroid_mean_distance
        )

    sum_lost_load_share_per_centroid = np.array(
        [np.sum(total_lost_load_share[labels == i]) for i in range(len(centroids))]
    )
    if custom_order is None:
        plot_order = np.argsort(sum_lost_load_share_per_centroid)[::-1][:n_subplots]
        order_string = ""
    else:
        plot_order = custom_order
        order_string = "_custom_order"
        # if group_affiliation is not None:
        #     group_affiliation = group_affiliation[custom_order]

    maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    mins = [min(centroids[i]) for i in index_by_sample_numbers]
    vmax = max(maxes)
    vmin = min(mins)

    # if "clipped_tanh" in indicator_name:
    #     vmax = 0
    #     vmin = -1.0
    # elif "clipped" in indicator_name:
    #     vmax = 0
    #     vmin = vmin
    # elif "tanh" in indicator_name or "sign" in indicator_name:
    #     vmax = 1.0
    #     vmin = -1.0
    # elif "lshare" in indicator_name:
    #     vmax = 1.0
    #     vmin = 0.0
    #     if "main" in cbar_label:
    #         vmin = 0.001
    #         centroids = np.abs(centroids - 1)

    vmax = 1.0
    vmin = 10**-3
    centroids = np.abs(centroids - 1)

    vmin_edge = 1e-3
    vmax_edge = 1

    if cbar_label is None:
        cbar_label = indicator_name.replace("_", " ")

    n_subplots = min(n_subplots, len(centroids))
    n_rows = int(np.ceil(n_subplots / ncols))
    fig = plt.figure(figsize=(ncols * 3, n_rows * 3))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.5, n_rows * 3], hspace=0.05)
    gs_maps = GridSpecFromSubplotSpec(
        n_rows, ncols, subplot_spec=gs[1], hspace=0.04, wspace=0.0
    )
    gs_cbars = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.0, wspace=0.2)
    ax_node_colorbar = fig.add_subplot(gs_cbars[0])
    ax_edge_colorbar = fig.add_subplot(gs_cbars[1])
    mpl.style.use("default")
    plt.rc("text", usetex=False)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    plot_count = 0
    for ind, gs_iter in zip(plot_order, gs_maps):
        if plot_count == n_subplots:
            break
        plot_count += 1

        centroid, samples_in_centriod = centroids[ind], samples_per_centroid[ind]
        if edge_centroids is not None:
            edge_centroid = edge_centroids[ind]

        ax = fig.add_subplot(gs_iter)
        nodes = nx.draw_networkx_nodes(
            nx_graph,
            pos=pos,
            ax=ax,
            node_color=centroid,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            node_size=10,
        )
        nodes.set_edgecolor("black")
        nodes.set_linewidth(0.2)

        if edge_centroid is None:
            edges = nx.draw_networkx_edges(
                nx_graph,
                pos=pos,
                ax=ax,
                edge_color="black",
                width=0.5,
                # edge_cmap=cmap,
            )
        else:
            failed_edges_prob_log = np.array(
                # [np.log10(x) if x > 1e-12 else -np.inf for x in edge_centroid]
                [np.log10(x) for x in edge_centroid]
            )
            edges = nx.draw_networkx_edges(
                nx_graph,
                pos=pos,
                ax=ax,
                # edge_color="black",
                width=1,
                edge_cmap=edge_cmap,
                edge_color=failed_edges_prob_log,
                # edge_vmin=vmin_edge,
                # edge_vmax=vmax_edge
                edge_vmin=np.log10(vmin_edge),
                edge_vmax=np.log10(vmax_edge),
            )

        ax.axis("off")

        if titles is None:
            number_of_samples_string = sci_notation(samples_in_centriod, sig_fig=1)
            cum_lost_load_symbol = r"$\bar{\sum loss}$"
            # sci_notation(number, sig_fig=2)
            if centroid_mean_distance != None:
                title = (
                    f"{group_affiliation[ind]}\n"
                    + f"{cum_lost_load_symbol}={round(sum_lost_load_share_per_centroid[ind]/ sum_lost_load_share_per_centroid.sum()*100, ndigits=1)}%\n"
                    + f"n=${number_of_samples_string}$\n"
                    # + f"d={round(normalized_centroid_mean_distance[ind],ndigits=1)}\n"
                )
            else:
                title = (
                    f"cum. l.l.={round(sum_lost_load_share_per_centroid[ind]/ sum_lost_load_share_per_centroid.sum()*100, ndigits=1)}%\n"
                    + f"n={number_of_samples_string}"
                )
        else:
            title = titles[ind]
        ax.set_title(
            title,
            y=0.73,
            loc="left",
        )

    # if edge_centroid is None:
    #     # cbar_ax = fig.add_axes([0.375, 0.05, 0.25, 0.03])
    #     pass
    # else:
    #     # cbar_ax = fig.add_axes([0.25, 0.05, 0.2, 0.02 / n_rows * 3])
    #     # # cbar_ax = fig.add_axes([0.0, 0.05, 0.3, 0.02/n_rows*3])
    #     # cbar_ax_edge = fig.add_axes([0.55, 0.05, 0.2, 0.02 / n_rows * 3])
    #     # # cbar_ax_edge = fig.add_axes([0.7, 0.05, 0.3, 0.02/n_rows*3])
    #     # # sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=vmin_edge, vmax=vmax_edge))
    sm_edge = plt.cm.ScalarMappable(
        cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
    )
    cb_edge = fig.colorbar(
        sm_edge, cax=ax_edge_colorbar, orientation="horizontal", aspect=1000
    )
    cb_edge.ax.tick_params(labelsize=14, width=1.0, which="both")
    # cb.ax.set_yticks([-1,0,1])
    # cb_edge.ax.set_xticks([vmin_edge,vmax_edge])
    cb_edge.ax.set_xlabel(
        # r'$\langle p_{\ell}}\rangle$', fontsize=14, rotation=0
        "edge failure prob",
        fontsize=14,
        rotation=0,
    )  # , labelpad=-10)
    cb_edge.ax.xaxis.set_label_position("top")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
    )
    cb = fig.colorbar(sm, cax=ax_node_colorbar, orientation="horizontal", aspect=1000)
    cb.ax.tick_params(labelsize=14, width=1.0, which="both")
    # if "main" in cbar_label:
    #     cb.ax.set_xticks([vmin, 0.5, 1], labels=[vmin, 0.5, 1])
    # cb.ax.set_xticks([vmin,vmax])
    cb.ax.set_xlabel(cbar_label, fontsize=14, rotation=0)
    cb.ax.xaxis.set_label_position("top")

    if save_dir != None:
        if edge_centroid is not None:
            fig.savefig(
                save_dir
                + f"clusters_lines_lost_load_{indicator_name}_co2l{co2l_string}_k{len(centroids)}_vmin{vmin}{order_string}.pdf",
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                save_dir
                + f"clusters_lost_load_{indicator_name}_co2l{co2l_string}_k{len(centroids)}_vmin{vmin}{order_string}.pdf",
                bbox_inches="tight",
            )
        plt.close()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    if float(a) == 1:
        return "10^" + str(b)
    else:
        return a + " \cdot 10^{" + str(b) + "}"


def plot_group_lost_load_hist_by_co2_single(
    group,
    centroid_inds,
    labels,
    lost_loads,
    masks_sig_to_co2,
    co2l_inds_hist,
    co2_lvls_hist,
    weights,
    n_failures_weighted,
    # colors,
    ax=None,
):
    """plots the lost load distribution for a group of centroids

    Args:
        group (string): group name
        centroid_inds (array): array of centroid indices
        labels (array): array of labels describing which centroid each split belongs to
        lost_loads (array): array containing the lost load for each split
        masks_sig_to_co2 (array): binary mask indicating which splits belong to which co2 level
        co2l_inds_hist (list): indices of co2 levels to plot
        co2_lvls_hist (list): co2 levels to plot
        weights (arra): array containing the weights for each split
        n_failures_weighted (int): weighted number of failures in each co2 lvl
        # colors (list): list of colors to use for the plots
    """
    # cmap="cvidis"
    cmap = plt.get_cmap("cividis_r")
    co2ls = np.arange(0.1, 0.81, 0.1).round(1)
    group_mask = np.any([labels == i for i in centroid_inds], axis=0)
    lost_loads_lvl = [
        lost_loads[i][group_mask[masks_sig_to_co2[co2l_inds_hist[i]]]]
        for i in range(len(co2_lvls_hist))
    ]
    if ax is None:
        fig, ax = plt.subplots(
            1, len(co2_lvls_hist), figsize=(len(co2_lvls_hist) * 3, 3), sharey=True
        )
    # fig, ax = plt.subplots(len(co2_lvls_hist),1, figsize=(3, 3), sharey=True)
    for i, co2l_ind in enumerate(co2l_inds_hist):
        co2 = co2_lvls_hist[i]
        ax.hist(
            np.array(lost_loads_lvl[i]) * 100,
            weights=weights[group_mask & masks_sig_to_co2[co2l_ind]]
            / (n_failures_weighted),
            label=f"CO2={int(co2*100)}%",
            bins=np.arange(0, 100, 5),
            alpha=0.8,
            log=True,
            color=cmap(np.where(co2ls == co2)[0][0] / (len(co2ls) - 1)),
            histtype="step",
            #  label='{} \%'.format(int(level*100)),
            linewidth=3,
            #  color=cmap(ind/len(co2ls)),
        )
    ax.set_ylim(0.5 * 10**-8, 5 * 10**-4)
    ax.set_yticks([10**-i for i in range(4, 9)])
    # ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=5))
    # plt.hist(lost_loads_lvl, bins=np.arange(101)/100) # , label=f"CO2={co2*100}%"
    # ax.legend()
    ax.grid()
    ax.legend()
    # ax.set_title(f"CO2={co2*100}%", pad=-10)
    # if i!=0:
    #     ax.set_yticks([])
    # ax[1].set_title(group)
    ax.set_xlabel("total lost load share [%]")
    ax.set_ylabel("share of events")


def plot_group_lost_load_hist_by_co2(
    group,
    centroid_inds,
    labels,
    lost_loads,
    masks_sig_to_co2,
    co2l_inds_hist,
    co2_lvls_hist,
    weights,
    n_snapshots_weighted,
    colors,
    axes=None,
):
    """plots the lost load distribution for a group of centroids

    Args:
        group (string): group name
        centroid_inds (array): array of centroid indices
        labels (array): array of labels describing which centroid each split belongs to
        lost_loads (array): array containing the lost load for each split
        masks_sig_to_co2 (array): binary mask indicating which splits belong to which co2 level
        co2l_inds_hist (list): indices of co2 levels to plot
        co2_lvls_hist (list): co2 levels to plot
        weights (arra): array containing the weights for each split
        n_snapshots_weighted (int): weighted number of snapshots in each co2 lvl
        colors (list): list of colors to use for the plots
    """

    group_mask = np.any([labels == i for i in centroid_inds], axis=0)
    lost_loads_lvl = [
        lost_loads[i][group_mask[masks_sig_to_co2[co2l_inds_hist[i]]]]
        for i in range(len(co2_lvls_hist))
    ]
    if axes is None:
        fig, axes = plt.subplots(
            1, len(co2_lvls_hist), figsize=(len(co2_lvls_hist) * 3, 3), sharey=True
        )
    # fig, axes = plt.subplots(len(co2_lvls_hist),1, figsize=(3, 3), sharey=True)
    for i, (co2l_ind, ax) in enumerate(zip(co2l_inds_hist, axes[::-1])):
        co2 = co2_lvls_hist[i]
        ax.hist(
            np.array(lost_loads_lvl[i]) * 100,
            weights=weights[group_mask & masks_sig_to_co2[co2l_ind]]
            / n_snapshots_weighted
            / 100,
            label=f"CO2={co2*100}%",
            bins=np.arange(0, 100, 5),
            alpha=0.5,
            log=True,
            color=colors[i],
        )
        # plt.hist(lost_loads_lvl, bins=np.arange(101)/100) # , label=f"CO2={co2*100}%"
        # ax.legend()
        ax.grid()
        ax.set_title(f"CO2={co2*100}%", pad=-10)
    if i != 0:
        ax.set_yticks([])
    # axes[1].set_title(group)
    axes[1].set_xlabel("total lost load share [%]")
    axes[0].set_ylabel("share of events [%]")
    if axes is None:
        fig.subplots_adjust(wspace=0.05)
