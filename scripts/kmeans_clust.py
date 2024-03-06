"""clusters the filtered indicator vectors using kmeans and saves the results to disk"""

import gzip
import pickle
import sys

import pandas as pd
from filter_splits import split_mask
from tqdm import tqdm

from utils.clustering import cluster_kmeans
from utils.config import path_to_evaluation_results

sys.path.append("./")
from utils import data_handling
from utils.config import (
    path_to_clustering_results,
    path_to_indicator_vectors,
    path_to_pypsa_network,
)

types = [
    # "rocof_component",
    # "failed_edges",
    "lshare",
]
transformations = [
    "main_comp_most_frequent_lshare",
    #    "sign", "tanh", None, "clipped_tanh", "clipped"
]
indicator_type_transformation = [("lshare", "main_comp_most_frequent_lshare"), ("rocof", "blackout")]
# n_clusters_list = [8, 16, 32, 48, 64, 96, 128]
n_clusters_list = [16, 128]
NUM_NODES = 400

n_nodes_split, lost_load_share = 5, 0.005

co2l_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# get masks
print("loading masks")
split_properties_all = []
for co2l in co2l_list:
    split_properties_df = pd.read_csv(
        path_to_evaluation_results + f"split_properties_Co2L{co2l}_n{NUM_NODES}.csv",
        index_col=0,
    )
    split_properties_df["co2l"] = co2l
    split_properties_all.append(split_properties_df)
split_properties_df = pd.concat(split_properties_all)
masks = [
    split_mask(
        split_properties_df[split_properties_df.co2l == co2l],
        n_nodes_split,
        lost_load_share,
    )
    for co2l in co2l_list
]
# path = path_to_evaluation_results + f"/masks_all_n{n_nodes}.pklz"
path = path_to_evaluation_results + f"/masks_0.1-0.8_n{NUM_NODES}.pklz"
with gzip.open(path, "wb") as out:
    pickle.dump(masks, out)
# with gzip.open(path, 'rb') as out:
#     masks = pickle.load(out)


snet_index = 0
network = data_handling.load_pypsa_network(0.5, NUM_NODES, path_to_pypsa_network)
nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
snapshot_weights = network.snapshot_weightings.objective

print("performing clustering")
for indicator_type, transformation in indicator_type_transformation:
    print(indicator_type)
    print(transformation)
    for n_clusters in tqdm(n_clusters_list):
        print(n_clusters)
        # split_properties_df = pd.read_csv(path_to_evaluation_results + f'split_properties_Co2L{co2l}_n{n_nodes}.csv', index_col=0)
        # mask = split_mask(split_properties_df, n_nodes, lost_load_share)
        cluster_kmeans(
            NUM_NODES,
            co2l_list,
            n_clusters,
            indicator_type,
            path_to_indicator_vectors,
            path_to_clustering_results,
            transformation=transformation,
            mask=masks,
            weights=snapshot_weights,
        )
