#!usr/bin/env python
# -*- coding: utf-8 -*-


import pickle
import sys
import warnings
from glob import glob

warnings.simplefilter(action="ignore", category=FutureWarning)

import gzip

import networkx as nx
import numpy as np

sys.path.append("./")
from utils import cascade_simulation, data_handling
from utils import visualization as vis
from utils.config import *
from utils.indicator_utils import load_indicator_vectors

# from utils.plotting import plot_clusters

# Setup paths
# path_to_pypsa_network   = data_path + 'European_networks_sclopf/'
# path_to_cascade_results   = results_path + 'sclopf/cascade_results/'
path_to_eval_results = path_to_evaluation_results_sclopf
# save_path = results_path + 'sclopf/split_visualization/'
save_path = path_to_cascade_results_sclopf

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid.
snet_index = 0

# Choose fraction of simulations for agglomerative clustering
# (the rest is classified via NN classifier)
subset_size_for_clustering = 0.01

# Load arguments
n_nodes = 400

# Choose minimum distance between seperated cluster
# (the node-criterion only selects splits with >10 nodes,
# so we have to resolve a least hamming distance between the cluster of >9)
min_cluster_dist_nodes_list = np.array([5, 7, 9, 11]) / n_nodes


# Setup co2 levels
# TODO also include CO_2=0
# glob_pypsa_search_str = ("data/European_networks_sclopf/" +
#                          "sclopf-elec_s_{0}*.nc".format(n_nodes))
# pypsa_file_ls = glob(glob_pypsa_search_str)
# co2l_list = [float(xx.split("Co2L")[-1].split("-")[0])
#              for xx in pypsa_file_ls if float(xx.split("Co2L")[-1].split("-")[0]) > 0]
# print("Available CO2 Levels:")
# print(sorted(co2l_list))

# # Load PyPSA network and the graph of the subnetwork
# network = data_handling.load_pypsa_network(0.0, n_nodes, path_to_pypsa_network)
# nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
# I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)

#### Cluster split components ####
print("\nClustering split components...\n")

co2l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
indicator_type = "rocof"
n_nodes_split = 5
lost_load_share = 0.005

# affinity="dice"

masks = []
for co2 in co2l_list:
    file_name = (
        f"split_mask_Co2L{co2}_n{n_nodes}_n{n_nodes_split}_l{lost_load_share}.pklz"
    )
    with gzip.open(path_to_indicator_vectors_sclopf + "/" + file_name, "rb") as infile:
        mask = pickle.load(infile)
    masks.append(mask)

print("loading indicator vectors...")
indicator_vectors = load_indicator_vectors(
    n_nodes, co2l_list, indicator_type, path_to_indicator_vectors_sclopf, masks
)
if not isinstance(indicator_vectors, np.ndarray):
    indicator_vectors = np.array(indicator_vectors)
indicator_vectors = np.tanh(indicator_vectors)
indicator_type = "rocof_tanh"

# for affinity in ["dice", "hamming"]:
for affinity in ["euclidean"]:
    min_cluster_dist_nodes_list = [6, 7, 8, 9, 10][::-1]
    for min_cluster_dist_nodes in min_cluster_dist_nodes_list:
        # # Initialize results
        # component_props = pd.DataFrame()
        # indicator_vectors = np.empty((0, nx_graph.number_of_nodes()), int)

        # # Append all indicator vectors and components props
        # print('Concatenate components...')
        # for co2l in tqdm(co2l_list):
        # indicator_vec_level = np.load(path_to_eval_results + f'indicator_vectors_Co2L{co2l}_n{n_nodes}.npy')
        #     component_props_level = pd.read_hdf(path_to_eval_results + f'component_properties_Co2L{co2l}_n{n_nodes}.h5')

        #     indicator_vectors = np.concatenate([indicator_vectors, indicator_vec_level])
        #     component_props_level.loc[:, 'co2l'] = co2l
        #     component_props = component_props.append(component_props_level, ignore_index=True)
        # indicator_vectors = np.load(path_to_indicator_vectors + f'indicator_vectors_all_n{n_nodes}.npy')
        print(affinity, min_cluster_dist_nodes)

        n_cluster, cluster_labels = vis.cluster_indicator_vectors_combined(
            indicator_vectors,
            min_cluster_distance=min_cluster_dist_nodes,
            subset=subset_size_for_clustering,
            affinity=affinity,
            indicator_type=indicator_type,
            save_dir=path_to_clustering_results_sclopf + f"agglom_{indicator_type}/",
        )
        # with open(path_to_clustering_results + f'cluster_labels_all_n{n_nodes}_agglom_{affinity}_minDist{min_cluster_dist_nodes}.pklz', 'wb') as outfile:
        #     pickle.dump(cluster_labels, outfile)
        # component_props.loc[:, 'cluster_label'] = cluster_labels


# #### Cluster splits ####
# print('\n### Extracting splits ###\n')

# split_groups = component_props.groupby(['co2l', 'time_stamp', 'split_number'])
# split_props = pd.DataFrame(index=split_groups.groups.keys(),
#                            columns=['n_components', 'lost_load',
#                                     'lost_load_share', 'category'])
# split_props.index=split_props.index.rename(['co2l','time_stamp','split_number'])
# split_vectors = np.empty((split_groups.ngroups,
#                          nx_graph.number_of_nodes()), float)

# for i, (name, split) in enumerate(tqdm(split_groups)):
#     split_props.loc[name,'n_components'] = split.shape[0]

#     split_props.loc[name,'lost_load_share'] = (split.rocof.abs()>1).mul(split.load_share).sum()
#     split_props.loc[name,'lost_load'] = (split.rocof.abs()>1).mul(split.load).sum()

#     if split_props.loc[name,'lost_load_share']>0.99:
#         split_props.loc[name,'category'] = 'global_blackout'
#     elif split_props.loc[name,'lost_load_share']>0:
#         split_props.loc[name,'category'] = 'local_blackout'
#     elif ((split.rocof.abs()<1) & (split.load_share>0.99)).any():
#         split_props.loc[name,'category'] = 'negligible'
#     else:
#         split_props.loc[name,'category'] = 'no_blackout'

#     for idx, component_props_of_idx in split.iterrows():
#         split_vectors[i, indicator_vectors[idx].astype(bool)] = component_props_of_idx.rocof


# # Save clustering results
# np.save(save_path + f'indicator_vectors_all_n{n_nodes}.npy', indicator_vectors)
# component_props.to_hdf(save_path + f'component_properties_all_n{n_nodes}.h5', key='df', mode= 'w')
# np.save(save_path + f'split_vectors_all_n{n_nodes}.npy', split_vectors)
# split_props=split_props.astype(dtype=dict(zip(split_props.columns,[int,float,float,str])))
# split_props.to_hdf(save_path + f'split_properties_all_n{n_nodes}.h5', key='df', mode= 'w')
