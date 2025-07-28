# %%
import os
import sys

import numpy as np
import pypsa
import pandas as pd
import networkx as nx

root_path = "./"
sys.path.append(root_path)

from utils.visualization import get_co2_levels
import numpy as np
import matplotlib.pyplot as plt

import gzip
from utils.config import path_to_clustering_results_sclopf, path_to_vis_results_sclopf
from utils.clustering_visualisation import *
from utils import data_handling
from utils.config import path_to_pypsa_network_sclopf, path_to_figures_sclopf
from utils import cascade_simulation
from utils.clustering import load_clustering

# Load network graph and node positions, the same for all CO2 levels
snet_index = 0
n_nodes = 600
network = data_handling.load_pypsa_network(0.6, n_nodes, True)
nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
pos = nx.get_node_attributes(nx_graph, "pos")
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
    nx_graph
)

# Determine number of simulations,
# i.e., total number of initial failures over the simulated period of one year
bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph), nx_graph)
double_line_failures = cascade_simulation.calc_possible_double_line_failures(
    num_parallels, ignored_idxs=bridge_idxs
)
num_snapshots_weighted = network.snapshot_weightings.objective.sum()
num_failures_weighted = len(double_line_failures) * num_snapshots_weighted

n_nodes_split, lost_load_share = None, 0.05
co2l_list = get_co2_levels(
    n_nodes=n_nodes,
)
indicator_type = "rocof"
transformation = "blackout"
# max_dist = 3
# decay = 1


########### loading data ###########

save_dir = get_path_to_clustering_dir(
    n_nodes=n_nodes,
    co2l=co2l_list,
    indicator_type=indicator_type,
    transformation=transformation,
    n_nodes_split=n_nodes_split,
    lost_load_share=lost_load_share,
)
path_to_clustering_results = save_dir
# %%
split_properties_all = pd.read_hdf(
    path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
)

# %%
from utils.config import path_to_vis_results_sclopf

split_properties_filtered = pd.read_hdf(
    save_dir + f"data_filtered_{n_nodes}.h5", index_col=0
)

# %%
masks_fpath = save_dir + f"/masks_dict.pklz"
with gzip.open(masks_fpath, "rb") as out:
    masks_dict = pickle.load(out)
weights_fpath = save_dir + f"/weights_filtered_dict.pklz"
with gzip.open(weights_fpath, "rb") as out:
    weights_dict = pickle.load(out)

if transformation == None:
    indicator_name = indicator_type
else:
    indicator_name = indicator_type + "_" + transformation


# %%
for co2 in co2l_list:
    print(
        f"CO2 Level: {co2}: {masks_dict[co2].sum() == (split_properties_filtered.co2l==co2).sum()}"
    )


# %%
masks_fpath = save_dir + f"/masks_dict.pklz"
with gzip.open(masks_fpath, "rb") as out:
    masks_dict = pickle.load(out)

# %%
with gzip.open(save_dir + f"unique_blackout_vecs_n{n_nodes}.pklz", "rb") as out:
    unique_blackout_dict = pickle.load(out)
unique_blackout_vecs = np.array(list(unique_blackout_dict.keys()))

# %%
unique_blackout_vecs.sum(axis=1).min()

# %%
with gzip.open(
    save_dir + f"blackout_vectors_filtered_dict_{n_nodes}.pklz", "rb"
) as out:
    blackout_vectors_filtered_dict = pickle.load(out)

failed_edges_indicator_vectors_filtered = load_masked_indicator_vectors(
    "failed_edges", masks_dict=masks_dict
)

# %%
failed_edges_indicator_vectors_filtered = failed_edges_indicator_vectors_filtered
# %%
co2l_list = list(masks_dict.keys())

# %%
from utils.visualization import get_actual_co2_level

co2l_masks = [(split_properties_filtered.co2l == co2l).values for co2l in co2l_list]
actual_co2l_list = get_actual_co2_level(lvls=co2l_list, n_nodes=n_nodes)

clustering_res_info = {}


node_cmap = plt.get_cmap("plasma_r")
node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
node_cmap.set_under("gainsboro", 1.0)
# node_cbar_label = "split off main component prob"
node_cbar_label = "blackout probability"

edge_cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
edge_cmap = copy.copy(mpl.cm.get_cmap("cividis_r"))
edge_cmap.set_under("gainsboro", 1.0)

plotting_dir = path_to_clustering_results + indicator_name + "/"
plotting_dir = save_dir

# %%
split_lost_load = split_properties_filtered.lost_load_share_blackout
weights = np.concatenate([weights_dict[co2l] for co2l in co2l_list])
split_weighted_lost_load = split_lost_load * weights
# %%
for vec in unique_blackout_dict.keys():
    unique_blackout_dict[vec]["total_weight"] = unique_blackout_dict[vec][
        "weight"
    ].sum()

blackout_vectors_filtered = np.concatenate(
    list(blackout_vectors_filtered_dict.values())
)
blackout_vectors_filtered.shape

# %%
list(masks_dict.values())[0].shape

# %%
mask_all = np.concatenate(
    list(masks_dict.values()),
    axis=0,
)


########### loading clustering results ###########
for fname in os.listdir(save_dir)[::-1]:
    clustering_res_path = os.path.join(save_dir, fname)
    if not "fitted" in fname:
        # print(f"Skipping {fname}, not a fitted clustering result.")
        continue
    else:
        print(f"Processing {fname}")
    with gzip.open(clustering_res_path, "rb") as out:
        clustering_res = pickle.load(out)
    print(clustering_res_path, clustering_res.labels_.shape)

    labels, counts = np.unique(clustering_res.labels_, return_counts=True)
    plt.hist(counts, bins=100, log=True)
    plt.title(f"Clustering results for {fname}")
    plt.xlabel("Number of nodes in cluster")
    plt.ylabel("Frequency")
    plt.savefig(clustering_res_path.replace("fitted.pklz", "hist.png"))
    # plt.show()
    plt.close()

    with gzip.open(clustering_res_path, "rb") as out:
        clustering_res = pickle.load(out)

    print(clustering_res.labels_.shape)
    labels, counts = np.unique(clustering_res.labels_, return_counts=True)

    # %%
    labels_unique_vecs = clustering_res.labels_

    # with gzip.open(clustering_res_path.replace(".pklz", "_labels_all.npy"), "rb") as out:
    labels_all = np.load(
        clustering_res_path.replace("fitted.pklz", "labels_all.npy"), allow_pickle=True
    )

    with gzip.open(
        clustering_res_path.replace("fitted.pklz", "group_masks.pklz"), "rb"
    ) as out:
        group_masks = pickle.load(out)

    # %%

    # %% [markdown]
    # ### calculate centroid group properties

    # %%
    # co2l_key = "all"
    # labels, centroids, samples_per_centroid, centroid_mean_distance, inertia, silhouette_avg = load_clustering(n_nodes, co2l_list, n_clusters, indicator_type, save_dir, transformation=transformation)
    # %%
    labels_all = np.load(
        clustering_res_path.replace("fitted.pklz", "labels_all.npy"), allow_pickle=True
    )

    # %%
    n_clusters = len(np.unique(labels))
    centroid_weighted_lost_load = np.array(
        [np.sum(split_weighted_lost_load[labels_all == i]) for i in labels]
    )
    centroid_lost_load_share = (
        centroid_weighted_lost_load / centroid_weighted_lost_load.sum()
    )

    centroids = [
        blackout_vectors_filtered[group_masks[label]].T
        @ weights[group_masks[label]]
        / group_masks[label].sum()
        for label in labels
        # if label != -1
    ]

    # %%
    edge_centroids = [
        failed_edges_indicator_vectors_filtered[group_masks[label]].T
        @ weights[group_masks[label]]
        / group_masks[label].sum()
        for label in labels
        # if label != -1
    ]

    # %%
    len(centroid_weighted_lost_load)

    # %%
    centroid_groups_lost_load_share = {
        label: centroid_weighted_lost_load[label] for label in labels  # if label != -1
    }

    # %%
    samples_per_centroid = np.array(
        [weights[group_masks[label]].sum() for label in labels]
        # [group_masks[label].sum() for label in labels if label != -1]
    )
    assert (
        sum(samples_per_centroid) == weights.sum()
    )  # - weights[group_masks[-1]].sum()

    mean_load_loss_share_per_centroid = (
        centroid_weighted_lost_load / samples_per_centroid
    )
    assert all(mean_load_loss_share_per_centroid >= 0) and all(
        mean_load_loss_share_per_centroid <= 1
    )

    # %%
    if transformation == None:
        indicator_name = indicator_type
    else:
        indicator_name = indicator_type + "_" + transformation

    # if "main" in transformation:
    #     node_cbar_label="split off main component prob"
    # else:
    #     node_cbar_label=None
    node_cbar_label = "blackout probability"
    # Add numerical error tolerance for floating point comparison
    assert np.allclose(
        centroid_weighted_lost_load,
        mean_load_loss_share_per_centroid * samples_per_centroid,
        rtol=1e-8,
        atol=1e-12,
    ), "total_lost_load_share must be equal to mean_lost_load_share_per_centroid * samples_per_centroid (within tolerance)"

    labels[
        centroid_weighted_lost_load
        != mean_load_loss_share_per_centroid * samples_per_centroid
    ]

    centroids_df = pd.DataFrame(
        index=labels,
        data=np.array(
            [
                samples_per_centroid,
                centroid_weighted_lost_load,
                mean_load_loss_share_per_centroid,
            ]
        ).T,
        columns=[
            "n_samples",
            "weighted_lost_load",
            "mean_load_loss_share",
        ],
    )
    centroids_df.index.name = "label"
    centroids_df.to_csv(clustering_res_path.replace("fitted.pklz", "centroids.csv"))

    clustering_res_info[fname] = {
        "n_clusters": centroids_df.shape[0],
        "share_ungrouped_unique_events": round(
            centroids_df.n_samples[0] / centroids_df.n_samples.sum(), 2
        ),
        "share_loss_ungrouped": round(
            centroids_df.weighted_lost_load[0] / centroids_df.weighted_lost_load.sum(),
            2,
        ),
    }
    # custom_order = np.concatenate(list(centroid_groups_dict.values())).astype(int)
    # group_affiliation = np.array([None for i in range(n_clusters)])
    # for group, centroid_indices in centroid_groups_dict.items():
    #     group_affiliation[centroid_indices] = group

    print(indicator_name, n_clusters)
    os.makedirs(plotting_dir, exist_ok=True)
    plot_clusters_lost_load(
        nx_graph,
        pos,
        indicator_name,
        centroids,
        centroids_df=centroids_df,
        cmap=node_cmap,
        n_subplots=200,
        edge_centroids=edge_centroids,
        edge_cmap=edge_cmap,
        cbar_label=node_cbar_label,
        # mask=masks,
        # custom_order=custom_order,
        ncols=4,
        # group_affiliation=group_affiliation,
        save_dir=save_dir,
        show_ind=False,
        show=True,
        algorithm_name=fname.split("_fitted")[0],
        # algorithm_name="test",
    )

    mpl.style.use("default")
    plt.rc("text", usetex=False)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    group_weighted_samples_list = samples_per_centroid
    mean_centroids = centroids
    group_failed_edges = edge_centroids
    original_ind_to_components_ind = (None,)
    for sort_by in ["loss", "number"]:
        if sort_by == "loss":
            custom_order = np.argsort(list(centroid_groups_lost_load_share.values()))[
                ::-1
            ]
        elif sort_by == "number":
            custom_order = np.argsort(group_weighted_samples_list)[::-1]

        node_cmap = node_cmap
        n_subplots = n_clusters
        edge_cmap = edge_cmap
        # node_cbar_label = "split off main component prob"
        sum_lost_load_share_per_centroid = np.array(
            list(centroid_groups_lost_load_share.values())
        )
        group_names = np.arange(len(centroids))
        centroid_inds = group_names

        if sum_lost_load_share_per_centroid is None:
            sum_lost_load_share_per_centroid = np.array(
                [
                    np.sum(split_lost_load[labels == i])
                    for i in range(len(mean_centroids))
                ]
            )

        vmax = 1.0
        vmin = 0.001
        if "main" in transformation:
            mean_centroids = np.abs(mean_centroids - 1)
        co2l_list = np.array(co2l_list)

        vmin_edge = 1e-3
        vmax_edge = 1.0

        if node_cbar_label is None:
            node_cbar_label = indicator_name.replace("_", " ")

        n_subplots = min(n_subplots, len(mean_centroids))
        ncols = 3
        n_rows = 4
        fig_scaling = 4
        fig = plt.figure(figsize=(ncols * fig_scaling, n_rows * fig_scaling * 1.3))

        gs = GridSpec(2, 1, figure=fig, hspace=0.15, height_ratios=[0.05, n_rows][::-1])
        gs_legend_colorax = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0])
        gs_group = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs[0, 0], hspace=0.2)
        plt.rc("text", usetex=False)

        plot_count = 0
        ind = 0
        custom_order_iter = iter(custom_order)

        co2_lvls_hist = [0.6, 0.2, 0.0]
        co2l_inds_hist = [
            np.where(np.array(co2l_list) == co2l)[0][0] for co2l in co2_lvls_hist
        ]

        for row in range(n_rows):

            gs_row = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_group[row, 0], hspace=-0.05, height_ratios=[3, 1]
            )
            gs_row_map = GridSpecFromSubplotSpec(
                1, ncols, subplot_spec=gs_row[0, 0], wspace=0.0
            )
            gs_row_hist = GridSpecFromSubplotSpec(
                1, ncols, subplot_spec=gs_row[1, 0], wspace=0.18, hspace=0.5
            )

            for column in range(ncols):

                ind = next(custom_order_iter)
                # print(len(list(custom_order_iter)))

                if plot_count == n_subplots:
                    break
                plot_count += 1

                # while centroids_df"ungrouped" in group_names[ind]  or group_names[ind] == "none":
                #     print(f"skipping {group_names[ind]}")
                #     ind+=1

                centroid, samples_in_centriod = (
                    mean_centroids[ind],
                    group_weighted_samples_list[ind],
                )
                failed_edges_prob = group_failed_edges[ind]

                ax = fig.add_subplot(gs_row_map[0, column])

                plot_centroid_with_failures(
                    nx_graph,
                    pos,
                    group_failed_edges,
                    node_cmap,
                    edge_cmap,
                    vmax,
                    vmin,
                    vmin_edge,
                    vmax_edge,
                    centroid,
                    failed_edges_prob,
                    ax,
                )

                ax.axis("off")

                subtitle = f"={round(sum_lost_load_share_per_centroid[ind]/ sum_lost_load_share_per_centroid.sum()*100, ndigits=1)}%\nn={round(samples_in_centriod/10**6, ndigits=2)}mio"
                subtitle = r"$\bar{R}$" + subtitle
                ax.set_title(
                    group_names[ind],
                    y=0.90,
                    x=-0.05,
                    fontsize=14,
                    loc="left",
                    # horizontalalignment="left",
                )
                ax.text(
                    -15,
                    52.5,
                    subtitle,
                    fontsize=14,
                )
                # sum_lost_load_share_per_centroid.max()

                # add gridspec for histograms to gs
                ax_hist = fig.add_subplot(gs_row_hist[0, column])
                plot_group_lost_load_hist_by_co2_single(
                    group_names[ind],
                    group_masks[labels[ind]],
                    labels,
                    split_lost_load,
                    co2l_masks,
                    co2l_inds_hist,
                    co2_lvls_hist,
                    weights,
                    n_failures_weighted=num_failures_weighted,
                    ax=ax_hist,
                )
                if column != 0:
                    ax_hist.set_ylabel("")
                ax_hist.tick_params(axis="y", which="major", pad=0)
                ax_hist.set_xlabel(ax_hist.get_xlabel(), rotation=0, labelpad=0)
                h, l = ax_hist.get_legend_handles_labels()
                ax_hist.legend().set_visible(False)

            ax_hist_legend = fig.add_subplot(gs_legend_colorax[0])
            GridSpec(n_rows, 1, figure=fig, hspace=0.2)
            ax_hist_legend.axis("off")
            ax_hist_legend.legend(h, l, loc="center", fontsize=15, ncols=1)

            cbar_ax_node = fig.add_subplot(gs_legend_colorax[1])
            cbar_ax_edge = fig.add_subplot(gs_legend_colorax[2])
            sm_edge = plt.cm.ScalarMappable(
                cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge, vmax=vmax_edge)
            )
            cb_edge = fig.colorbar(sm_edge, cax=cbar_ax_edge, orientation="horizontal")
            cb_edge.ax.tick_params(labelsize=14, width=1.0, which="both")
            cb_edge.ax.set_xlabel("line failure prob", fontsize=14, rotation=0)
            cb_edge.ax.xaxis.set_label_position("top")

            sm_node = plt.cm.ScalarMappable(
                cmap=node_cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax)
            )
            cb_node = fig.colorbar(sm_node, cax=cbar_ax_node, orientation="horizontal")
            cb_node.ax.tick_params(labelsize=14, width=1.0, which="both")
            # if "main" in node_cbar_label:
            #     cb_node.ax.set_xticks([vmin, 0.5, 1], labels=[vmin, 0.5, 1])
            cb_node.ax.set_xlabel(node_cbar_label, fontsize=14, rotation=0)
            cb_node.ax.xaxis.set_label_position("top")

        fig.savefig(
            clustering_res_path.split("_fitted")[0]
            + f"_co2l{co2_lvls_hist}_{ncols}cols_{n_rows}rows_sort{sort_by.capitalize()}.pdf",
            bbox_inches="tight",
        )
clustering_res_info = pd.DataFrame.from_dict(clustering_res_info)
clustering_res_info = clustering_res_info.T
clustering_res_info.to_csv(save_dir + "clustering_res_info.csv")
