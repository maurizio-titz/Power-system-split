"""clusters the filtered indicator vectors using kmeans and saves the results to disk"""

from datetime import datetime
import gzip
import os
import pickle
import sys

from matplotlib import pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from filter_splits import split_mask
from utils.clustering_uncertainty import (
    get_unique_vectors,
    typed_katz_centrality_batch,
    uncertainty_distance_wrapper_concat,
)
from utils.indicator_utils import load_indicator_vectors

sys.path.append("./")
from utils import data_handling
from utils.clustering import (
    cluster_kmeans,
    get_path_to_clustering_dir,
    transform_indicator_vectors,
)
from utils.clustering_visualisation import (
    plot_clusters_wrapper,
    plot_grid,
    plot_indicator_vectors,
)
from loguru import logger
from utils.config import (
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_lopf,
    path_to_vis_results_sclopf,
)


use_sclopf = True

if use_sclopf:
    path_to_evaluation_results = path_to_evaluation_results_sclopf
    path_to_pypsa_network = path_to_pypsa_network_sclopf
    path_to_vis_results = path_to_vis_results_sclopf
else:
    path_to_evaluation_results = path_to_evaluation_results_lopf
    path_to_pypsa_network = path_to_pypsa_network_lopf
    path_to_vis_results = path_to_vis_results_lopf

indicator_type_transformation = [
    # ("blackout"),
    ("rocof", "blackout"),
    # ("lshare", "main_comp_most_frequent_lshare"),
]

n_nodes = 600
n_clusters_list = [16, 64, 128, 256]

# ignore splits with less than 1% lost load share
n_nodes_split, lost_load_share = None, 0.01

co2l_list = [0.05, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][::-1]
# co2l_list = [0.6]

for indicator_type, transformation in indicator_type_transformation:

    if transformation is None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""
    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l="all",
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=n_nodes_split,
        lost_load_share=lost_load_share,
        use_sclopf=use_sclopf,
    )
    os.makedirs(save_dir, exist_ok=True)

    logger_num = logger.add(f"{save_dir}/log.txt")

    logger.info(f"using {indicator_type} indicator type")
    logger.info(f"transforming with {transformation}")

    # get masks
    try:
        logger.info("loading masks")
        with gzip.open(save_dir + "/masks.pklz", "rb") as fh_in:
            masks = pickle.load(fh_in)
    except FileNotFoundError:
        logger.info("######## masks not found, calculating masks ########")

        split_props = pd.read_hdf(
            path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
        )
        split_props.lost_load_share_blackout = (
            split_props.lost_load_share_blackout.astype(float)
        )
        masks = [
            split_props[
                split_props.index.get_level_values("co2l") == co2l
            ].lost_load_share_blackout
            > lost_load_share
            for co2l in co2l_list
        ]

        split_properties_all = []
        # for co2l in co2l_list:
        #     split_properties_df = pd.read_csv(
        #         path_to_vis_results
        #         + f"split_properties_Co2L{co2l}_n{n_nodes}.csv",
        #         index_col=0,
        #     )
        #     split_properties_df["co2l"] = co2l
        #     split_properties_all.append(split_properties_df)
        # split_properties_df = pd.concat(split_properties_all)
        masks = [
            split_mask(
                # split_properties_df[split_properties_df.co2l == co2l],
                split_props[split_props.index.get_level_values("co2l") == co2l],
                lost_load_share,
                # n_nodes_split,
                ignore_shedding=(transformation == "blackout"),
            )
            for co2l in co2l_list
        ]
        logger.info(
            f"keeping {sum([mask.sum() for mask in masks])} splits out of {len(masks) * len(masks[0])} total splits"
        )

        masks_fpath = save_dir + "/masks.pklz"
        with gzip.open(masks_fpath, "wb") as out:
            pickle.dump(masks, out)

    ############ calculate typed katz centrality indicator vectors for each co2l and indicator type
    def get_adjacency_matrix_from_nx_graph(nx_graph):
        """returns the adjacency matrix of the nx graph"""
        I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
            nx_graph
        )
        I_m = np.absolute(I_m)
        adjacency_matrix = I_m @ I_m.T
        # Remove self-loops for sparse matrices
        if hasattr(adjacency_matrix, "setdiag"):
            adjacency_matrix.setdiag(0)
        else:
            np.fill_diagonal(adjacency_matrix, 0)  # remove self-loops
        return adjacency_matrix

    # I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
    #     nx_graph
    # )
    # I_m = np.absolute(I_m)
    # adjacency_matrix = I_m @ I_m.T
    # # Remove self-loops for sparse matrices
    # if hasattr(adjacency_matrix, "setdiag"):
    #     adjacency_matrix.setdiag(0)
    # else:
    #     np.fill_diagonal(adjacency_matrix, 0)  # remove self-loops

    save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_n{n_nodes}.pklz"
    try:
        with gzip.open(save_path_unique_vecs, "rb") as fh_in:
            unique_vecs_dict = pickle.load(fh_in)
    # if not os.path.exists(save_path_unique_vecs):
    except FileNotFoundError:
        blackout_indicator_vectors_all = []
        for i, (mask, co2l) in enumerate(zip(masks, co2l_list)):
            logger.info(f"Co2 level {co2l:.2f}")
            logger.info(
                f"current time: {datetime.now().strftime('%H:%M')} - loading indicator vectors for co2l {co2l} and indicator type {indicator_type}"
            )
            network = data_handling.load_pypsa_network(
                co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
            )
            snapshot_weights = network.snapshot_weightings.objective

            indicator_vectors = load_indicator_vectors(
                n_nodes,
                co2l,
                indicator_type,
                path_to_indicator_vectors=path_to_evaluation_results,
                mask=mask,
                weights=snapshot_weights,
            )
            blackout_indicator_vectors_all.append(
                transform_indicator_vectors(
                    indicator_vectors, transformation, indicator_type
                )
            )
        blackout_indicator_vectors_all = np.concatenate(
            blackout_indicator_vectors_all, axis=0
        )

        logger.info(
            f"current time: {datetime.now().strftime('%H:%M')} - finding duplicate rows in indicator vectors"
        )
        unique_vecs_dict = get_unique_vectors(blackout_indicator_vectors_all)
        num_duplicates = sum(
            [len(duplicat_idxs) for duplicat_idxs in unique_vecs_dict.values()]
        )
        logger.info(
            f"found {num_duplicates} duplicate rows from {blackout_indicator_vectors_all.shape[0]} total rows and {len(unique_vecs_dict)} unique vectors"
        )
        with gzip.open(save_path_unique_vecs, "wb") as fh_out:
            pickle.dump(unique_vecs_dict, fh_out)

        # # os.makedirs(safe_path, exist_ok=True)
        # # with gzip.open(f"{safe_path}/weights.pklz", "wb") as fh_out:
        # #     pickle.dump(weights, fh_out)

        # transform indicator vector
        # all_unique_vecs = []
        # #! fix, concatenating all unique vectors from all co2l levels creates duplicates!
        # for i, (mask, co2l) in enumerate(zip(masks, co2l_list)):
        #     save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_Co2L{co2l:.1f}_n{n_nodes}.pklz"
        #     with gzip.open(save_path_unique_vecs, "rb") as fh_in:
        #         unique_vecs_dict = pickle.load(fh_in)
        #     unique_vectors = np.array(list(unique_vecs_dict.keys()))
        #     all_unique_vecs.append(unique_vectors)
        # all_unique_vecs = np.concatenate(all_unique_vecs, axis=0)
        # np.savez_compressed(save_path_all_unique_vecs, unique_vectors=all_unique_vecs)

    # logger.info(f"current time: {datetime.now().strftime('%H:%M')} - calculating typed katz centralities for {len(all_unique_vecs)} unique vectors")
    unique_vecs = np.array(list(unique_vecs_dict.keys()))

    typed_katz_centralities_dict = {}

    katz_param_grid = ParameterGrid(
        {
            "decay_factor": [1, 1.2, 1.5, 2],
            "max_distance": [1, 2, 3],
        }
    )
    for katz_params in katz_param_grid:
        decay_factor = katz_params["decay_factor"]
        max_distance = katz_params["max_distance"]

        save_path_centralities = (
            save_dir + f"indicator_vector_katz_maxD{max_distance}_decay{decay_factor}"
        )
        try:
            typed_katz_centralities = np.load(save_path_centralities + ".npy")
        except FileNotFoundError:
            logger.info(
                f"current time: {datetime.now()} - calculating typed katz centralities with decay factor {decay_factor} and max distance {max_distance}"
            )

            snet_index = 0
            # Check if 'network' variable exists, otherwise load it
            if "network" not in locals():
                network = data_handling.load_pypsa_network(
                    co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
                )

            nx_graph = data_handling.build_networkx_graph(
                network, snet_index=snet_index
            )
            pos = nx.get_node_attributes(nx_graph, "pos")
            snapshot_weights = network.snapshot_weightings.objective
            adjacency_matrix = get_adjacency_matrix_from_nx_graph(nx_graph)

            typed_katz_centralities = typed_katz_centrality_batch(
                adjacency_matrix=adjacency_matrix,
                node_class_vectors=unique_vecs,
                decay_factor=decay_factor,
                max_distance=max_distance,
            )
            np.save(save_path_centralities, typed_katz_centralities)
        typed_katz_centralities_dict[(decay_factor, max_distance)] = (
            typed_katz_centralities
        )

    # # calculate distance matrix
    # from utils.clustering_uncertainty import calculate_distance_matrix
    decay_factor = 1
    max_distance = 3
    typed_katz_centralities = typed_katz_centralities_dict[(decay_factor, max_distance)]
    blackout_katz_centralities = np.concatenate(
        (unique_vecs, typed_katz_centralities), axis=1
    )
    save_path_distance_matrix = (
        save_dir + f"/distance_matrix_n{n_nodes}_maxD{max_distance}_decay{decay_factor}"
    )
    try:
        distance_matrix = np.load(save_path_distance_matrix + ".npy")
    except FileNotFoundError:
        logger.info(
            f"current time: {datetime.now().strftime('%H:%M')} - calculating distance matrix for {len(unique_vecs)} unique vectors"
        )
        distance_matrix = pairwise_distances(
            blackout_katz_centralities, metric=uncertainty_distance_wrapper_concat
        )
        np.save(save_path_distance_matrix, distance_matrix)

    # logger.info(
    #     "current time: "
    #     f"{datetime.now().strftime('%H:%M')} - plotting indicator vectors and katz centralities"
    # )
    # # set seed for reproducibility
    for plot_count in tqdm(range(3)):
        random_idx = [10000, 20000, -1000][plot_count]
        n_rows = len(katz_param_grid.param_grid[0]["decay_factor"])
        n_cols = len(katz_param_grid.param_grid[0]["max_distance"])
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_rows * 8, n_cols * 8),
        )

        for i_decay_factor in []:  # range(n_rows):
            for i_max_distance in range(n_cols):
                decay_factor = katz_param_grid.param_grid[0]["decay_factor"][
                    i_decay_factor
                ]
                max_distance = katz_param_grid.param_grid[0]["max_distance"][
                    i_max_distance
                ]
                ax = axs[i_decay_factor, i_max_distance]
                # too_small = True
                # while too_small:
                # random_idx = np.random.randint(0, len(blackout_katz_centralities))
                # blackout_katz_centrality = blackout_katz_centralities[random_idx]
                # n_n = int(blackout_katz_centrality.shape[0] / 2)
                # too_small = blackout_vec.sum() < 15
                katz_centrality = typed_katz_centralities_dict[
                    (decay_factor, max_distance)
                ][random_idx, :]

                # if i == 1:
                #     cmap = "cividis"
                # else:
                # cmap = matplotlib.colors.ListedColormap(["y", "black"])
                vmin = katz_centrality.min()
                vmax = katz_centrality.max()
                plot_grid(
                    nx_graph,
                    pos,
                    katz_centrality,
                    ax=ax,
                    save_dir=None,
                    cmap="cividis",
                    vmax=vmax,
                    vmin=vmin,
                    node_size=100,
                )
                # axs[0].set_title(f"blackout vector")
                ax.set_title(f"decay {decay_factor}, max dist {max_distance}")
                # axs[1].set_title(f"katz centrality vector")
                # logger.info(
                #     f"current time: {datetime.now().strftime('%H:%M')} - saving blackout vector and katz centrality plot to {save_dir}/blackout_vec_katz_centrality_{i}.png"
                # )
            fig.savefig(
                save_dir + f"/katz_centrality_{plot_count}.png",
            )

            blackout_vec = unique_vecs[random_idx, :]
            f = plot_grid(
                nx_graph,
                pos,
                blackout_vec,
                # ax=ax,
                save_dir=None,
                cmap=matplotlib.colors.ListedColormap(["y", "black"]),
                # vmax=vmax,
                # vmin=vmin,
            )
            f.savefig(
                save_dir + f"/blackout_vec_{plot_count}.png",
            )

        # fig = plot_indicator_vectors(
        #     nx_graph,
        #     pos,
        #     "",
        #     None,
        #     np.stack((blackout_vec, katz_centrality), axis=0),
        #     save_dir=None,
        #     cmap="cividis",
        #     n_subplots=16,
        #     shuffle=False,
        #     vmax=katz_centrality.max(),
        #     vmin=0,
        # )
        # fig.savefig(
        #     save_dir + f"/blackout_vec_katz_centrality_unique_{i}.png",
        # )
        # if i == 20:
        #     break
    # logger.info(f"clustering indicator vectors, {datetime.now()}")
    # for n_clusters in n_clusters_list:
    #     logger.info(f"performing clustering {n_clusters}")
    #     cluster_kmeans(
    #         n_nodes,
    #         co2l_list,
    #         n_clusters,
    #         indicator_type,
    #         path_to_evaluation_results,
    #         save_dir,
    #         transformation=transformation,
    #         mask=masks,
    #         weights=snapshot_weights,
    #     )

    #     logger.info("plotting clusters")
    #     plot_clusters_wrapper(
    #         indicator_type,
    #         transformation,
    #         n_nodes,
    #         n_clusters,
    #         nx_graph,
    #         masks,
    #         pos,
    #         co2l=co2l_list,
    #         clustering_results_dir=save_dir,
    #         save_dir=save_dir,
    #         sort_by_sample_number=True,
    #         use_sclopf=use_sclopf,
    #     )

    logger.info("performing OPTICS clustering")
    for min_samples in tqdm([10, 20, 50, 100]):
        logger.info(f"performing optics clustering with min_samples {min_samples}")
        distance_matrix = distance_matrix[
            :10000, :10000
        ]  # limit to first 1000 vectors for optics clustering
        opt = OPTICS(
            metric="precomputed",
            min_samples=min_samples,
            cluster_method="xi",
            n_jobs=-1,
        )
        opt.fit(distance_matrix)
        # save clustering results
        with gzip.open(
            save_dir
            + f"/optics_clustering_n{n_nodes}_maxD{max_distance}_decay{decay_factor}_minS{min_samples}.pklz",
            "wb",
        ) as fh_out:
            pickle.dump(opt, fh_out)

    logger.info("performing agglomerative clustering")
    agglomerative_param_grid = ParameterGrid(
        {
            "n_clusters": n_clusters_list,
            "linkage": ["average", "complete", "single"],
        }
    )
    for params in tqdm(agglomerative_param_grid):
        n_clusters = params["n_clusters"]
        linkage = params["linkage"]
        logger.info(
            f"performing agglomerative clustering with {n_clusters} clusters and {linkage} linkage"
        )
        agg = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        agg.fit(distance_matrix)

        # save clustering results
        with gzip.open(
            save_dir
            + f"/agglo_clustering_n{n_nodes}_maxD{max_distance}_decay{decay_factor}_ncl{n_clusters}_{linkage}_link.pklz",
            "wb",
        ) as fh_out:
            pickle.dump(agg, fh_out)
