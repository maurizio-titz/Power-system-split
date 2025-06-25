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
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from filter_splits import split_mask
from utils.clustering_uncertainty import (
    get_unique_vectors_with_weights,
    prepare_clusters_for_analysis,
    typed_katz_centrality_batch,
    uncertainty_distance_wrapper_concat,
)
from utils.indicator_utils import load_indicator_vectors
from utils.visualization import get_co2_levels

sys.path.append("./")
from utils import data_handling
from utils.clustering import (
    cluster_kmeans,
    get_path_to_clustering_dir,
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

n_clusters_list = [64, 128, 256]
clustering_params = {
    "optics": {
        "min_samples": [20, 50, 100],
        "cluster_method": ["xi"],
        "n_jobs": [-1],
        "metric": ["precomputed"],
    },
    "agglomerative": {
        "n_clusters": n_clusters_list,
        "linkage": ["average", "complete", "single"],
        "metric": ["precomputed"],
    },
    "kmedoids": {
        "n_clusters": n_clusters_list,
        "metric": ["precomputed"],
        "method": ["alternate"],
    },
    "dbscan": {
        "eps": [0.1, 0.2, 0.3, 0.4, 0.5],
        "min_samples": [20, 50, 100],
        "metric": ["precomputed"],
        "n_jobs": [-1],
    },
}

if __name__ == "__main__":

    indicator_type_transformation = [
        # ("blackout"),
        ("rocof", "blackout"),
        # ("lshare", "main_comp_most_frequent_lshare"),
    ]

    n_nodes = 600

    # ignore splits with less than 5% lost load share
    min_n_nodes_split, min_lost_load_share = None, 0.05

    co2l_list = get_co2_levels(n_nodes, ignore_lvls=0.05)
    print(f"co2l_list: {co2l_list}")
    print("ignoring co2l level 0.05!!!!!!!!!!!!!!!!!!!!!!")
    # co2l_list = [0.05, 0.6]

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
            n_nodes_split=min_n_nodes_split,
            lost_load_share=min_lost_load_share,
            use_sclopf=use_sclopf,
        )
        os.makedirs(save_dir, exist_ok=True)

        logger_num = logger.add(f"{save_dir}/log.txt")

        logger.info(f"save_dir: {save_dir}, n_nodes: {n_nodes}, co2l_list: {co2l_list}")
        logger.info(f"using {indicator_type} indicator type")
        logger.info(f"transforming with {transformation}")

        # get masks
        try:
            logger.info("loading masks")
            with gzip.open(save_dir + "/masks.pklz", "rb") as fh_in:
                masks_dict = pickle.load(fh_in)
        except FileNotFoundError:
            logger.info("######## masks not found, calculating masks ########")

            split_props = pd.read_hdf(
                path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
            )
            split_props.lost_load_share_blackout = (
                split_props.lost_load_share_blackout.astype(float)
            )

            masks_dict = {
                co2l: split_mask(
                    # split_properties_df[split_properties_df.co2l == co2l],
                    split_props[split_props.index.get_level_values("co2l") == co2l],
                    min_lost_load_share,
                    # n_nodes_split,
                    ignore_shedding=True,
                )
                for co2l in co2l_list
            }
            logger.info(
                f"keeping {sum([mask.sum() for mask in masks_dict.values()])} splits out of {sum([len(mask) for mask in masks_dict.values()])} total splits"
            )

            masks_dict_fpath = save_dir + "/masks_dict.pklz"
            with gzip.open(masks_dict_fpath, "wb") as fh_out:
                pickle.dump(masks_dict, fh_out)

        save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_n{n_nodes}.pklz"
        try:
            with gzip.open(save_path_unique_vecs, "rb") as fh_in:
                unique_vecs_dict = pickle.load(fh_in)
        except FileNotFoundError:
            blackout_vectors_filtered = []
            logger.info(
                f"current time: {datetime.now().strftime('%H:%M')} - loading indicator vectors, indicator type {indicator_type}"
            )
            network = data_handling.load_pypsa_network(
                co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
            )
            snapshot_weights = network.snapshot_weightings.objective

            if "split_props" not in locals():
                split_props = pd.read_hdf(
                    path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
                )
            (
                blackout_vectors_filtered_dict,
                weights_filtered_dict,
                split_props_filtered,
            ) = load_indicator_vectors(
                n_nodes,
                indicator_type,
                transformation,
                path_to_indicator_vectors=path_to_evaluation_results,
                mask=masks_dict,
                weights=snapshot_weights,
                split_props=split_props,
                co2_list=co2l_list,
            )
            with gzip.open(f"{save_dir}/weights_filtered_dict.pklz", "wb") as fh_out:
                pickle.dump(weights_filtered_dict, fh_out)
            split_props_filtered.to_hdf(
                f"{save_dir}/data_filtered_{n_nodes}.h5", key="split_props"
            )

            with gzip.open(
                f"{save_dir}/blackout_vectors_filtered_dict_{n_nodes}.pklz", "wb"
            ) as fh_out:
                pickle.dump(blackout_vectors_filtered_dict, fh_out)
            with gzip.open(
                f"{save_dir}/weights_filtered_dict_{n_nodes}.pklz", "wb"
            ) as fh_out:
                pickle.dump(weights_filtered_dict, fh_out)
            blackout_vectors_filtered = np.concatenate(
                list(blackout_vectors_filtered_dict.values())
            )
            weights_filtered = np.concatenate(list(weights_filtered_dict.values()))

            logger.info(
                f"current time: {datetime.now().strftime('%H:%M')} - finding duplicate rows in indicator vectors"
            )
            unique_vecs_dict = get_unique_vectors_with_weights(
                blackout_vectors_filtered, weights_filtered
            )
            num_duplicates = sum(
                [len(duplicat_idxs) for duplicat_idxs in unique_vecs_dict.values()]
            )
            logger.info(
                f"found {num_duplicates} duplicate rows from {blackout_vectors_filtered.shape[0]} total rows and {len(unique_vecs_dict)} unique vectors"
            )
            with gzip.open(save_path_unique_vecs, "wb") as fh_out:
                pickle.dump(unique_vecs_dict, fh_out)

        unique_vecs = np.array(list(unique_vecs_dict.keys()))
        weights_filtered = [d["weight"] for d in unique_vecs_dict.values()]

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
                save_dir
                + f"indicator_vector_katz_maxD{max_distance}_decay{decay_factor}"
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
                adjacency_matrix = data_handling.get_adjacency_matrix_from_nx_graph(
                    nx_graph
                )

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
        typed_katz_centralities = typed_katz_centralities_dict[
            (decay_factor, max_distance)
        ]
        blackout_katz_centralities = np.concatenate(
            (unique_vecs, typed_katz_centralities), axis=1
        )
        save_path_distance_matrix = (
            save_dir
            + f"/distance_matrix_n{n_nodes}_maxD{max_distance}_decay{decay_factor}"
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
            # for plot_count in tqdm(range(3)):
            #     random_idx = [10000, 20000, -1000][plot_count]
            #     n_rows = len(katz_param_grid.param_grid[0]["decay_factor"])
            #     n_cols = len(katz_param_grid.param_grid[0]["max_distance"])
            #     fig, axs = plt.subplots(
            #         nrows=n_rows,
            #         ncols=n_cols,
            #         figsize=(n_rows * 8, n_cols * 8),
            #     )

            #     for i_decay_factor in []:  # range(n_rows):
            #         for i_max_distance in range(n_cols):
            #             decay_factor = katz_param_grid.param_grid[0]["decay_factor"][
            #                 i_decay_factor
            #             ]
            #             max_distance = katz_param_grid.param_grid[0]["max_distance"][
            #                 i_max_distance
            #             ]
            #             ax = axs[i_decay_factor, i_max_distance]
            #             # too_small = True
            #             # while too_small:
            #             # random_idx = np.random.randint(0, len(blackout_katz_centralities))
            #             # blackout_katz_centrality = blackout_katz_centralities[random_idx]
            #             # n_n = int(blackout_katz_centrality.shape[0] / 2)
            #             # too_small = blackout_vec.sum() < 15
            #             katz_centrality = typed_katz_centralities_dict[
            #                 (decay_factor, max_distance)
            #             ][random_idx, :]

            #             # if i == 1:
            #             #     cmap = "cividis"
            #             # else:
            #             # cmap = matplotlib.colors.ListedColormap(["y", "black"])
            #             vmin = katz_centrality.min()
            #             vmax = katz_centrality.max()
            #             plot_grid(
            #                 nx_graph,
            #                 pos,
            #                 katz_centrality,
            #                 ax=ax,
            #                 save_dir=None,
            #                 cmap="cividis",
            #                 vmax=vmax,
            #                 vmin=vmin,
            #                 node_size=100,
            #             )
            #             # axs[0].set_title(f"blackout vector")
            #             ax.set_title(f"decay {decay_factor}, max dist {max_distance}")
            #             # axs[1].set_title(f"katz centrality vector")
            #             # logger.info(
            #             #     f"current time: {datetime.now().strftime('%H:%M')} - saving blackout vector and katz centrality plot to {save_dir}/blackout_vec_katz_centrality_{i}.png"
            #             # )
            #         fig.savefig(
            #             save_dir + f"/katz_centrality_{plot_count}.png",
            #         )

            #         blackout_vec = unique_vecs[random_idx, :]
            #         f = plot_grid(
            #             nx_graph,
            #             pos,
            #             blackout_vec,
            #             # ax=ax,
            #             save_dir=None,
            #             cmap=matplotlib.colors.ListedColormap(["y", "black"]),
            #             # vmax=vmax,
            #             # vmin=vmin,
            #         )
            #         f.savefig(
            #             save_dir + f"/blackout_vec_{plot_count}.png",
            #         )

            # Define clustering parameters in a dictionary
        logger.info(
            f"distance matrix shape: {distance_matrix.shape}, "
            f"unique vectors shape: {unique_vecs.shape}, "
        )
        if not "split_props_filtered" in locals():
            split_props_filtered = pd.read_hdf(
                f"{save_dir}/data_filtered_{n_nodes}.h5", key="split_props"
            )

        logger.info("performing OPTICS clustering")
        OPTICS_param_grid = ParameterGrid(clustering_params["optics"])
        for params in tqdm(OPTICS_param_grid):
            min_samples = params["min_samples"]
            save_path = (
                save_dir
                + f"/optics_clustering_n{n_nodes}_maxD{max_distance}_decay{decay_factor}_minS{min_samples}.pklz"
            )
            if not os.path.exists(save_path):
                logger.info(
                    f"performing optics clustering with min_samples {min_samples}"
                )
                # distance_matrix_subset = distance_matrix[
                #     :10000, :10000
                # ]  # limit to first 10000 vectors for optics clustering
                opt = OPTICS(**params)
                opt.fit(distance_matrix)
                logger.info("opt.labels_shape: " + str(opt.labels_.shape))
                # save clustering results
                with gzip.open(
                    save_path,
                    "wb",
                ) as fh_out:
                    pickle.dump(opt, fh_out)
            prepare_clusters_for_analysis(save_path, unique_vecs_dict, split_props)

        logger.info("performing agglomerative clustering")
        agglomerative_param_grid = ParameterGrid(clustering_params["agglomerative"])
        for params in tqdm(agglomerative_param_grid):
            n_clusters = params["n_clusters"]
            linkage = params["linkage"]

            agglomerative_clustering_save_path = (
                save_dir
                + f"/agglo_clustering_n{n_nodes}_maxD{max_distance}_decay{decay_factor}_ncl{n_clusters}_{linkage}.pklz"
            )

            if not os.path.exists(agglomerative_clustering_save_path):
                logger.info(
                    f"performing agglomerative clustering with {n_clusters} clusters and {linkage} linkage"
                )
                agg = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage=linkage,
                )
                agg.fit(distance_matrix)
                logger.info(
                    f"agg.labels_shape: {agg.labels_.shape}, "
                    f"agg.n_connected_components_: {agg.n_connected_components_}"
                )

                # save clustering results
                with gzip.open(
                    agglomerative_clustering_save_path,
                    "wb",
                ) as fh_out:
                    pickle.dump(agg, fh_out)
            prepare_clusters_for_analysis(
                agglomerative_clustering_save_path, unique_vecs_dict, split_props
            )

        logger.info("performing kmedoids clustering")
        medoids_param_grid = ParameterGrid(clustering_params["kmedoids"])
        for params in tqdm(medoids_param_grid):
            n_clusters = params["n_clusters"]
            method = params["method"]
            metric = params["metric"]

            kmedoids_clustering_save_path = (
                save_dir
                + f"/kmedoids_clustering_n{n_nodes}_maxD{max_distance}_decay{decay_factor}_ncl{n_clusters}.pklz"
            )

            if not os.path.exists(kmedoids_clustering_save_path):
                logger.info(f"performing medoids clustering with {n_clusters} clusters")
                k_med = KMedoids(
                    n_clusters=n_clusters,
                    metric=metric,
                    method=method,
                )
                k_med.fit(distance_matrix)
                logger.info(f"k_med.labels_shape: {k_med.labels_.shape}, ")

                # save clustering results
                with gzip.open(
                    kmedoids_clustering_save_path,
                    "wb",
                ) as fh_out:
                    pickle.dump(k_med, fh_out)

            prepare_clusters_for_analysis(
                kmedoids_clustering_save_path, unique_vecs_dict, split_props
            )

        DBSCAN_param_grid = ParameterGrid(clustering_params["dbscan"])
        for params in tqdm(DBSCAN_param_grid):
            eps = params["eps"]
            min_samples = params["min_samples"]
            metric = params["metric"]
            n_jobs = params["n_jobs"]

            dbscan_save_path = (
                save_dir
                + f"/dbscan_clustering_n{n_nodes}_maxD{max_distance}_eps{eps}_minS{min_samples}.pklz"
            )

            if not os.path.exists(dbscan_save_path):
                logger.info(
                    f"performing DBSCAN clustering with eps {eps} and min_samples {min_samples}"
                )
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric=metric,
                    n_jobs=n_jobs,
                )
                dbscan.fit(distance_matrix, sample_weight=weights_filtered)

                # save clustering results
                with gzip.open(
                    dbscan_save_path,
                    "wb",
                ) as fh_out:
                    pickle.dump(dbscan, fh_out)

            prepare_clusters_for_analysis(
                dbscan_save_path, unique_vecs_dict, split_props
            )
