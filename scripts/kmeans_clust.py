"""clusters the filtered indicator vectors using kmeans and saves the results to disk"""

import gzip
import os
import pickle
import sys

import networkx as nx
import pandas as pd
from filter_splits import split_mask

sys.path.append("./")
from utils import data_handling
from utils.clustering import cluster_kmeans, get_path_to_clustering_dir
from utils.clustering_visualisation import plot_clusters_wrapper
from utils.config import (
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_lopf,
    path_to_vis_results_sclopf,
)

use_sclopf = False

if use_sclopf:
    path_to_evaluation_results = path_to_evaluation_results_sclopf
    path_to_pypsa_network = path_to_pypsa_network_sclopf
    path_to_vis_results = path_to_vis_results_sclopf
else:
    path_to_evaluation_results = path_to_evaluation_results_lopf
    path_to_pypsa_network = path_to_pypsa_network_lopf
    path_to_vis_results = path_to_vis_results_lopf

indicator_type_transformation = [
    ("rocof", "blackout"),
    # ("lshare", "main_comp_most_frequent_lshare"),
]

n_nodes = 800
n_clusters_list = [2, 16, 64, 128, 256]

n_nodes_split, lost_load_share = 800, 0.005

# ignore 0.7 and 0.8 because these scenarios are almost identical to 0.6. (because optimization is brownfield)
co2l_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for indicator_type, transformation in indicator_type_transformation:
    print(indicator_type)
    print(transformation)

    if transformation is None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""
    save_dir = get_path_to_clustering_dir(
        n_nodes=n_nodes,
        co2l=co2l_list,
        indicator_type=indicator_type,
        transformation=transformation,
        n_nodes_split=n_nodes_split,
        lost_load_share=lost_load_share,
        use_sclopf=False
    )
    os.makedirs(save_dir, exist_ok=True)

    # get masks
    print("loading masks")
    split_props = pd.read_hdf(
        path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
    )
    masks = [
        split_props[
            split_props.index.get_level_values("co2l") == co2l
        ].lost_load_share_blackout
        > lost_load_share
        for co2l in co2l_list
    ]

    # split_properties_all = []
    # for co2l in co2l_list:
    #     split_properties_df = pd.read_csv(
    #         path_to_vis_results
    #         + f"split_properties_Co2L{co2l}_n{n_nodes}.csv",
    #         index_col=0,
    #     )
    #     split_properties_df["co2l"] = co2l
    #     split_properties_all.append(split_properties_df)
    # split_properties_df = pd.concat(split_properties_all)
    # masks = [
    #     split_mask(
    #         split_properties_df[split_properties_df.co2l == co2l],
    #         n_nodes_split,
    #         lost_load_share,
    #         ignore_shedding=(transformation == "blackout"),
    #     )
    #     for co2l in co2l_list
    # ]
    masks_fpath = save_dir + "/masks.pklz"
    with gzip.open(masks_fpath, "wb") as out:
        pickle.dump(masks, out)

    snet_index = 0
    network = data_handling.load_pypsa_network_wrapper(
        co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
    )

    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    pos = nx.get_node_attributes(nx_graph, "pos")
    snapshot_weights = network.snapshot_weightings.objective

    for n_clusters in n_clusters_list:
        print(f"performing clustering {n_clusters}")
        cluster_kmeans(
            n_nodes,
            co2l_list,
            n_clusters,
            indicator_type,
            path_to_evaluation_results,
            save_dir,
            transformation=transformation,
            mask=masks,
            weights=snapshot_weights,
        )

        print("plotting clusters")
        plot_clusters_wrapper(
            indicator_type,
            transformation,
            n_nodes,
            n_clusters,
            nx_graph,
            masks,
            pos,
            co2l=co2l_list,
            clustering_results_dir=save_dir,
            save_dir=save_dir,
            sort_by_sample_number=True,
            use_sclopf=False,
        )
