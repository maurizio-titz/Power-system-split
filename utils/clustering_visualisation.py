import os

from utils.clustering.data_handling import load_clustering

os.chdir("..")
import copy
import gzip
import pickle
from typing import Optional

import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from utils.data_handling import get_actual_co2_level
from utils.config import (
    path_to_clustering_results_sclopf,
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_indicator_vectors_sclopf,
)


def plot_clusters_old(
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


def plot_grid(
    nx_graph,
    pos,
    indicator_vector,
    ax=None,
    save_dir=None,
    cmap="seismic",
    vmax=1,
    vmin=0,
    node_size=150,
):

    plt.rc("text", usetex=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    nodes = nx.draw_networkx_nodes(
        nx_graph,
        pos=pos,
        ax=ax,
        node_color=indicator_vector,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        node_size=node_size,
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
    # sm = plt.cm.ScalarMappable(cmap=cmap)
    # cb = fig.colorbar(sm, cax=cbar_ax)
    cb = plt.colorbar(sm, ax=ax)
    # if save_dir != None:
    #     fig.savefig(
    #         save_dir + f"split_plots_{indicator_name}_co2l{co2l_string}.pdf",
    #         bbox_inches="tight",
    #     )

    return ax.figure


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
    vmax=1,
    vmin=0,
):
    if co2l == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        co2l_string = "all"
    else:
        co2l_string = co2l

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

    cmap = mpl.cm.get_cmap(cmap)
    cmap.set_under("gainsboro", 0.0)

    plot_count = 0
    for ind, gs_iter in enumerate(gs):
        if plot_count == n_subplots:
            break
        plot_count += 1

        if shuffle:
            ind = rand_inds[ind]
        indicator_vector = indicator_vectors[ind, :]

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
    cmap,
    edge_cmap,
    vmax,
    vmin,
    vmin_edge,
    vmax_edge,
    centroid,
    failed_edges_prob,
    ax,
    colors_classes=None,
    node_size=10,
    edge_width=1,
    radius=0.5,
    pie_nodes=True,
    edge_log_scale=True,
):
    if len(centroid.shape) > 1:
        if colors_classes is None:
            raise ValueError("colors_classes must be provided for pie plot")
        # majority_class = np.argmax(centroid, axis=0)
        # probs_majority_class = np.max(centroid, axis=0)
        # c = [
        #     cmap[class_ind][proba]
        #     for class_ind, proba in zip(majority_class, probs_majority_class)
        # ]
        if pie_nodes:
            for i_node, node in enumerate(nx_graph.nodes()):
                ax.pie(
                    centroid[i_node, :],
                    colors=colors_classes,
                    radius=radius,
                    center=(pos[node][0], pos[node][1]),
                    # zorder=2,
                )
            pos_arr = np.array(list(pos.values()))
            ax.set_xlim(pos_arr[:, 0].min() - 1, pos_arr[:, 0].max() + 1)
            ax.set_ylim(pos_arr[:, 1].min() - 1, pos_arr[:, 1].max() + 1)
        else:
            for i_node, node in enumerate(nx_graph.nodes()):
                for class_ind in np.argsort(centroid[i_node, :])[::-1]:

                    a = ax.scatter(
                        pos[node][0],
                        pos[node][1],
                        color=colors_classes[class_ind],
                        s=centroid[i_node, class_ind] * node_size,
                        zorder=2,
                    )

    else:
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

    # if len(centroid.shape) == 1:
    if failed_edges_prob is None:
        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edge_color="black",
            width=edge_width,
            # edge_cmap=cmap,
        )
    else:
        if edge_log_scale:
            failed_edges_prob = np.array(
                [np.log10(x) if x > 1e-12 else -np.inf for x in failed_edges_prob]
            )
            vmin_edge = np.log10(vmin_edge)
            vmax_edge = np.log10(vmax_edge)

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            # edge_color="black",
            width=edge_width,
            edge_cmap=edge_cmap,
            edge_color=failed_edges_prob,
            edge_vmin=vmin_edge,
            edge_vmax=vmax_edge,
        )
        edges.set_zorder(0)
    # else:
    #     for i_edge, edge in enumerate(nx_graph.edges()):
    #         ax.plot(
    #             [pos[edge[0]][0], pos[edge[1]][0]],
    #             [pos[edge[0]][1], pos[edge[1]][1]],
    #             color=edge_cmap(failed_edges_prob[i_edge]),
    #             linewidth=edge_width,
    #             zorder=-1,
    #         )


def load_lost_load_share_broken(
    n_nodes=600, mask=None, co2l_list=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
):
    if mask is None:
        path = path_to_evaluation_results_sclopf + f"masks_all_n400.pklz"
        with gzip.open(path, "rb") as out:
            mask = pickle.load(out)

    vectors = []
    for co2l, mask in zip(co2l_list, mask):
        path = (
            path_to_evaluation_results_sclopf
            + f"/total_lost_load_share_Co2L{co2l}_n{n_nodes}.pklz"
        )
        with gzip.open(path, "rb") as out:
            vectors.append(pickle.load(out)[mask])

    return np.concatenate(vectors)


def get_lost_load_share(
    split_properties,
    lost_load_type="total",
    dir=None,
):
    if dir is not None:
        with gzip.open(f"{dir}/lost_load_{lost_load_type}_share.pklz", "rb") as out:
            return pickle.load(out)

    lost_load_all = []
    for co2l, mask in zip(co2l_list, masks):
        split_properties = pd.read_csv(
            path_to_evaluation_results_sclopf
            + f"split_props_Co2L{co2l}_n{n_nodes}.csv",
            index_col=0,
        )
        lost_load = split_properties[f"lost_load_{lost_load_type}_share"].values
        lost_load_all.append(lost_load[mask])

    if dir is not None:
        with gzip.open(f"{dir}/lost_load_{lost_load_type}_share.pklz", "wb") as out:
            pickle.dump(np.concatenate(lost_load_all), out)

    return np.concatenate(lost_load_all)


def load_indicator_vectors_only(
    indicator_type,
    n_nodes=600,
    co2lvls=(),
    use_sclopf=True,
):
    vectors = {}
    for co2l in co2lvls:
        if use_sclopf:
            path = (
                path_to_evaluation_results_sclopf
                + f"/{indicator_type}_indicator_vector_Co2L{co2l}_n{n_nodes}.pklz"
            )
        else:
            path = (
                path_to_evaluation_results_lopf
                + f"/{indicator_type}_indicator_vector_Co2L{co2l}_n{n_nodes}.pklz"
            )
        with gzip.open(path, "rb") as out:
            vectors[co2l] = pickle.load(out)[-1]

    return vectors


def load_masked_indicator_vectors(
    indicator_type,
    sub_dir=None,
    n_nodes=600,
    masks_dict={},
    return_weights=False,
    transformation=None,
    n_nodes_split=None,
    lost_load_share=None,
    use_sclopf=True,
):
    if (return_weights or masks_dict is None) and sub_dir is None:
        raise NotImplementedError

        # sub_dir = get_path_to_clustering_dir(
        #     n_nodes=n_nodes,
        #     co2l=co2l,
        #     indicator_type=indicator_type,
        #     transformation=transformation,
        #     n_nodes_split=n_nodes_split,
        #     lost_load_share=lost_load_share,
        # )

    if masks_dict is None:
        raise NotImplementedError
        # path = sub_dir + "masks.pklz"

        # with gzip.open(path, "rb") as out:
        #     masks = pickle.load(out)

    vectors = []
    for co2l, mask in masks_dict.items():
        if use_sclopf:
            path = (
                path_to_evaluation_results_sclopf
                + f"/{indicator_type}_indicator_vector_Co2L{co2l}_n{n_nodes}.pklz"
            )
        else:
            path = (
                path_to_evaluation_results_lopf
                + f"/{indicator_type}_indicator_vector_Co2L{co2l}_n{n_nodes}.pklz"
            )
        with gzip.open(path, "rb") as out:
            vectors.append(pickle.load(out)[-1][mask])

    if return_weights:
        with gzip.open(f"{sub_dir}/weights.pklz", "rb") as fh_in:
            weights = pickle.load(fh_in)
        return np.concatenate(vectors), weights

    return np.concatenate(vectors)


def map_components_to_original(values, original_ind_to_components_ind, n_nodes=600):
    """map values defined on splits to new indicator vectors defined on components

    Args:
        values (np.array): array to be mapped, e.g. failed edges, lost load etc.
        original_ind_to_components_ind (array): original index to component index mapping
    """

    path = path_to_evaluation_results_sclopf + f"masks_all_n{n_nodes}.pklz"
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
    centroids,
    centroids_df,
    edge_centroids=None,
    save_dir=None,
    cmap="seismic",
    n_subplots=16,
    edge_cmap="inferno",
    cbar_label=None,
    custom_order=None,
    ncols=4,
    titles=None,
    show_ind=False,
    show=True,
    algorithm_name="",
    ignore_labels=(),
):
    centroids_df = centroids_df.copy()

    if isinstance(ignore_labels, int):
        ignore_labels = [ignore_labels]

    has_ungrouped = -1 in centroids_df.index

    if custom_order is None:
        if has_ungrouped:
            assert list(centroids_df.index) == list(np.arange(-1, len(centroids) - 1))
        else:
            assert list(centroids_df.index) == list(np.arange(len(centroids)))
        centroids_df.reset_index(drop=False, inplace=True)
        centroids_df.sort_values(by="weighted_lost_load", ascending=False, inplace=True)
        idx_ungrouped = (
            centroids_df.index[centroids_df.label == -1][0] if has_ungrouped else None
        )
        order_string = ""
    else:
        raise NotImplementedError("custom_order is not implemented yet")
        centroids_df = centroids_df.loc[custom_order]
        order_string = "_custom_order"

    assert np.allclose(
        centroids_df.weighted_lost_load.values,
        (centroids_df.mean_load_loss_share * centroids_df.n_samples).values,
        rtol=1e-8,
        atol=1e-12,
    ), "n_samples, mean load loss share and weighted lost load share do not match"

    # maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    # mins = [min(centroids[i]) for i in index_by_sample_numbers]
    # vmax = max(maxes)
    # vmin = min(mins)

    vmax = 1.0
    vmin = 10**-3

    vmin_edge = 1e-3
    vmax_edge = 1

    if cbar_label is None:
        cbar_label = indicator_name.replace("_", " ")

    n_subplots = min(n_subplots, len(centroids))
    n_rows = int(np.ceil(n_subplots / ncols))
    fig = plt.figure(figsize=(ncols * 3, n_rows * 3))
    gs = GridSpec(
        2, 1, figure=fig, height_ratios=[0.5, n_rows * 3], hspace=0.5 / n_rows
    )

    suptitle = f"num clusters={centroids_df.shape[0]}"
    if has_ungrouped:
        share_ungrouped = (
            centroids_df.loc[idx_ungrouped, "n_samples"] / centroids_df.n_samples.sum()
        )
        loss_share_ungrouped = (
            centroids_df.loc[idx_ungrouped, "weighted_lost_load"]
            / centroids_df.weighted_lost_load.sum()
        )
        suptitle += f", ungrouped={share_ungrouped:.0%}, ungrouped loss share={loss_share_ungrouped:.2%}\n"

    fig.suptitle(
        suptitle,
        fontsize=18,
        y=1 - 0.06 * 4 / n_rows * 3,
    )

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
    for iterrow, gs_iter in zip(centroids_df.iterrows(), gs_maps):
        ind = iterrow[0]
        label = iterrow[1]["label"]
        n_samples = iterrow[1]["n_samples"]
        total_lost_load_share = iterrow[1]["weighted_lost_load"]
        mean_lost_load_share_per_centroid = iterrow[1]["mean_load_loss_share"]

        assert np.isclose(
            total_lost_load_share,
            mean_lost_load_share_per_centroid * n_samples,
            rtol=1e-8,
            atol=1e-12,
        ), "n_samples, mean load loss share and weighted lost load share do not match"

        if label in ignore_labels:
            print(f"Skipping label {label} in plot_clusters_lost_load")
            continue

        if plot_count == n_subplots:
            break
        plot_count += 1

        centroid = centroids[label]

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

        if edge_centroids is None:
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
                [np.log10(x) for x in edge_centroids[label]]
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
            number_of_samples_string = sci_notation(n_samples, sig_fig=1)
            mean_lost_load_share_string = (
                ""
                if mean_lost_load_share_per_centroid is None
                else (f"mean R$={mean_lost_load_share_per_centroid*100:.0f}\\%$")
            )
            total_lost_load_share_string = (
                ""
                if total_lost_load_share is None
                else (f"total R$={sci_notation(total_lost_load_share, sig_fig=1)}\\%$")
            )
            # cum_lost_load_symbol = r"$\bar{r}$"

            title = (
                f"{int(label)} "
                + r"$\bar{R}$"
                + f"={round(total_lost_load_share / centroids_df.weighted_lost_load.sum()*100, ndigits=1)}%\n"
                + f"n=$"
                + number_of_samples_string
                + "$\n"
                + mean_lost_load_share_string
                + "\n"
                + total_lost_load_share_string
            )
            # if show_ind:
            #     title = f"{ind}: " + title
        else:
            title = titles[ind]
        ax.set_title(
            title,
            y=0.76,
            loc="left",
        )

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

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=ax_node_colorbar, orientation="horizontal", aspect=1000)
    cb.ax.tick_params(labelsize=14, width=1.0, which="both")
    # if "main" in cbar_label:
    #     cb.ax.set_xticks([vmin, 0.5, 1], labels=[vmin, 0.5, 1])
    # cb.ax.set_xticks([vmin,vmax])
    cb.ax.set_xlabel(cbar_label, fontsize=14, rotation=0)
    cb.ax.xaxis.set_label_position("top")

    if save_dir is not None:
        fname = ""
        if edge_centroids is not None:
            fname = f"{algorithm_name}_clustering_all_vmin{vmin}{order_string}"
        else:
            fname = f"{algorithm_name}_clustering_all_vmin{vmin}{order_string}"
        if show_ind:
            fname += "_inds"
        fig.savefig(
            save_dir + fname + ".pdf",
            bbox_inches="tight",
        )
    if show:
        plt.close()


def plot_clusters(
    nx_graph: nx.Graph,
    pos,
    indicator_name: str,
    co2l: list,
    samples_per_centroid: Optional[np.ndarray],
    centroids,
    edge_centroids=None,
    save_dir=None,
    cmap: str = "seismic",
    n_subplots: int = 16,
    centroid_mean_distance=None,
    edge_cmap: str = "inferno",
    cbar_label: str = "",
    ncols=4,
    titles=None,
    sort_by_sample_number: bool = False,
):
    """Plots the node indicator vector centroids as a map.

    Args:
        nx_graph (nx.Graph): _description_
        pos (_type_): _description_
        indicator_name (str): _description_
        co2l (list): _description_
        samples_per_centroid (Optional[np.ndarray]): _description_
        centroids (_type_): _description_
        labels (_type_): _description_
        edge_centroids (_type_, optional): _description_. Defaults to None.
        save_dir (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "seismic".
        n_subplots (int, optional): _description_. Defaults to 16.
        centroid_mean_distance (_type_, optional): _description_. Defaults to None.
        edge_cmap (str, optional): _description_. Defaults to "inferno".
        cbar_label (str, optional): _description_. Defaults to "".
        mask (_type_, optional): _description_. Defaults to None.
        ncols (int, optional): _description_. Defaults to 4.
        titles (_type_, optional): _description_. Defaults to None.
    """
    co2l_string = str(max(co2l)) + "-" + str(min(co2l))
    if (not samples_per_centroid is None) and sort_by_sample_number:
        index_by_sample_numbers = np.argsort(samples_per_centroid)[::-1][:n_subplots]
    else:
        index_by_sample_numbers = np.arange(n_subplots)

    maxes = [max(centroids[i]) for i in index_by_sample_numbers]
    mins = [min(centroids[i]) for i in index_by_sample_numbers]
    vmax = max(maxes)
    vmin = min(mins)

    vmax = 1.0
    vmin = 10**-3
    if "main" in cbar_label:
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
    for ind, gs_iter in zip(index_by_sample_numbers, gs_maps):
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
            )
        else:
            failed_edges_prob_log = np.array([np.log10(x) for x in edge_centroid])
            edges = nx.draw_networkx_edges(
                nx_graph,
                pos=pos,
                ax=ax,
                width=1,
                edge_cmap=edge_cmap,
                edge_color=failed_edges_prob_log,
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
                    ""
                    f"i={ind}\n"
                    # + f"{cum_lost_load_symbol}={round(total_lost_load_share[ind]/ total_lost_load_share.sum()*100, ndigits=1)}%\n"
                    + f"n=${number_of_samples_string}$\n"
                    # + f"d={round(normalized_centroid_mean_distance[ind],ndigits=1)}\n"
                )
            else:
                title = (
                    # f"cum. l.l.={round(total_lost_load_share[ind]/ total_lost_load_share.sum()*100, ndigits=1)}%\n"
                    # +
                    f"n={number_of_samples_string}"
                )
        else:
            title = titles[ind]
        ax.set_title(
            title,
            y=0.73,
            loc="left",
        )

    sm_edge = plt.cm.ScalarMappable(
        cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
    )
    cb_edge = fig.colorbar(
        sm_edge, cax=ax_edge_colorbar, orientation="horizontal", aspect=1000
    )
    cb_edge.ax.tick_params(labelsize=14, width=1.0, which="both")
    cb_edge.ax.set_xlabel(
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
    cb.ax.set_xlabel(cbar_label, fontsize=14, rotation=0)
    cb.ax.xaxis.set_label_position("top")

    if edge_centroid is not None:
        edge_string = "lines_"
    else:
        edge_string = ""
    if sort_by_sample_number is not None:
        sort_string = "_sorted"
    else:
        sort_string = ""

    if save_dir is not None:
        fig.savefig(
            save_dir
            + f"/clusters_{edge_string}{indicator_name}_co2l{co2l_string}_k{len(centroids)}_vmin{vmin}{sort_string}.pdf",
            bbox_inches="tight",
        )


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
        return a + r" \cdot 10^{" + str(b) + "}"


def get_color_from_cmap(co2, co2ls, cmap):
    try:
        color = cmap(np.where(co2ls == co2)[0][0] / (len(co2ls) - 1))
    except IndexError:
        color = cmap((co2 - min(co2ls)) / (max(co2ls) - min(co2ls)))
    return color


def plot_group_lost_load_hist_by_co2_single(
    group_mask,
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
    co2ls = np.arange(0.0, 0.61, 0.1).round(1)
    # group_mask = np.any([labels == i for i in centroid_inds], axis=0)
    # print(lost_loads.shape)
    # print(list(group_mask.values())[0].shape)
    # print(masks_sig_to_co2[co2l_inds_hist[0]].shape)
    lost_loads_lvl = [
        lost_loads[group_mask & masks_sig_to_co2[co2l_inds_hist[i]]]
        for i in range(len(co2_lvls_hist))
    ]
    if ax is None:
        fig, ax = plt.subplots()
        # fig, ax = plt.subplots(
        #     1, len(co2_lvls_hist), figsize=(len(co2_lvls_hist) * 3, 3), sharey=True
        # )
    # fig, ax = plt.subplots(len(co2_lvls_hist),1, figsize=(3, 3), sharey=True)
    for i, co2l_ind in enumerate(co2l_inds_hist):
        co2 = co2_lvls_hist[i]
        c = get_color_from_cmap(co2, co2ls, cmap)
        ax.hist(
            np.array(lost_loads_lvl[i]) * 100,
            weights=weights[group_mask & masks_sig_to_co2[co2l_ind]]
            / (n_failures_weighted),
            label=f"CO2={get_actual_co2_level(co2, percent=True)}%",
            bins=np.linspace(0, 100, 21),
            alpha=0.8,
            log=True,
            color=c,
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


def plot_clusters_wrapper(
    indicator_type: str,
    transformation: str,
    n_nodes: int,
    n_clusters: int,
    nx_graph: nx.Graph,
    masks: list,
    pos,
    co2l=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    clustering_results_dir=None,
    save_dir=None,
    sort_by_sample_number=False,
    use_sclopf=True,
):
    """plot all clusters for a given indicator type and transformation. Plots the node centroid and the corresponding edge centroid.

    Args:
        indicator_type (str): _description_
        transformation (str): _description_
        n_nodes (str): _description_
        n_clusters (int): _description_
        nx_graph (nx.Graph): _description_
        masks (list): _description_
        pos (_type_): _description_
        co2l (tuple, optional): _description_. Defaults to (0.1, 0.2, 0.3, 0.4, 0.5, 0.6).
        clustering_results_dir (str, optional): _description_.
    """

    if transformation == None:
        indicator_name = indicator_type
    else:
        indicator_name = indicator_type + "_" + transformation

    edge_cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
    if "not_zero" in transformation:
        node_cmap = plt.get_cmap("viridis")
        node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
    if indicator_type == "rocof" and transformation != "blackout":
        node_cmap = plt.get_cmap("seismic")
        node_cmap = truncate_colormap(node_cmap, 0.15, 0.85, 1000)
        if transformation == "clipped":
            node_cmap = plt.get_cmap("viridis")
            node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
    elif indicator_type == "lshare" or transformation == "blackout":
        node_cmap = plt.get_cmap("plasma_r")
        node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
        node_cmap.set_under("gainsboro", 1.0)
        edge_cmap = copy.copy(mpl.cm.get_cmap("cividis_r"))

    edge_cmap.set_under("gainsboro", 1.0)
    if "main" in transformation:
        node_cbar_label = "split off main component prob"
    elif "blackout" in transformation:
        node_cbar_label = "blackout prob"
    else:
        node_cbar_label = None

    labels, centroids, samples_per_centroid, centroid_mean_distance, _, _ = (
        load_clustering(
            n_nodes=n_nodes,
            co2l=co2l,
            n_clusters=n_clusters,
            indicator_type=indicator_type,
            load_dir=clustering_results_dir,
            transformation=transformation,
            use_sclopf=use_sclopf,
        )
    )
    os.makedirs(save_dir, exist_ok=True)
    failed_edges_indicator_vectors, weights = load_masked_indicator_vectors(
        "failed_edges",
        masks_dict=masks,
        return_weights=True,
        sub_dir=clustering_results_dir,
        use_sclopf=use_sclopf,
        n_nodes=n_nodes,
    )
    edge_centroids = np.array(
        [
            np.average(
                failed_edges_indicator_vectors[labels == i],
                axis=0,
                weights=weights[labels == i],
            )
            for i in range(n_clusters)
        ]
    )
    plot_clusters(
        nx_graph,
        pos,
        indicator_name,
        co2l,
        samples_per_centroid,
        centroids,
        cmap=node_cmap,
        n_subplots=n_clusters,
        centroid_mean_distance=centroid_mean_distance,
        edge_centroids=edge_centroids,
        edge_cmap=edge_cmap,
        save_dir=save_dir,
        cbar_label=node_cbar_label,
        sort_by_sample_number=sort_by_sample_number,
    )
