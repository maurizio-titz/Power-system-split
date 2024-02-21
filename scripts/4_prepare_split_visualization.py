#!usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import pickle

from glob import glob

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('./')
from utils import cascade_simulation, data_handling
from utils import visualization as vis
from utils.config import *

# Setup paths 
path_to_pypsa_network   = data_path + 'European_networks_sclopf/'
path_to_cascade_results   = results_path + 'sclopf/cascade_results/'
path_to_eval_results= results_path + 'sclopf/evaluation_results/'
save_path = results_path + 'sclopf/split_visualization/'

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Choose fraction of simulations for agglomerative clustering
# (the rest is classified via NN classifier)
subset_size_for_clustering = 0.01 

# Choose minimum distance between seperated cluster
# (the node-criterion only selects splits with >10 nodes, 
# so we have to resolve a least hamming distance between the cluster of >9)
min_cluster_dist_nodes = 9

# Load arguments
n_nodes = int(sys.argv[1])

# Setup co2 levels
# TODO also include CO_2=0
# glob_pypsa_search_str = ("data/European_networks_sclopf/" +
#                          "sclopf-elec_s_{0}*.nc".format(n_nodes))    
# pypsa_file_ls = glob(glob_pypsa_search_str)
# co2l_list = [float(xx.split("Co2L")[-1].split("-")[0]) 
#              for xx in pypsa_file_ls if float(xx.split("Co2L")[-1].split("-")[0]) > 0]
print("Available CO2 Levels:")
# co2l_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
co2l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(sorted(co2l_list))
# Load PyPSA network and the graph of the subnetwork 
network = data_handling.load_pypsa_network(0.0, n_nodes, path_to_pypsa_network)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)

#### Cluster split components #### 
print('\nClustering split components...\n')

# Initialize results
component_props = pd.DataFrame()
# indicator_vectors = np.empty((0, nx_graph.number_of_nodes()), int)

# Append all indicator vectors and components props
print('Concatenate components...')
for co2l in tqdm(co2l_list):

    # indicator_vec_level = np.load(path_to_eval_results + f'indicator_vectors_Co2L{co2l}_n{n_nodes}.npy')
    component_props_level = pd.read_hdf(path_to_eval_results + f'component_properties_Co2L{co2l}_n{n_nodes}.h5')
    
    # indicator_vectors = np.concatenate([indicator_vectors, indicator_vec_level])
    component_props_level.loc[:, 'co2l'] = co2l
    component_props = component_props.append(component_props_level, ignore_index=True)
    
# n_cluster, cluster_labels = vis.cluster_indicator_vectors_combined(indicator_vectors,
#                                                                    min_cluster_distance=min_cluster_dist_nodes/len(nx_graph),
#                                                                    subset_ratio=subset_size_for_clustering)
# component_props.loc[:, 'cluster_label'] = cluster_labels


#### Cluster splits #### 
print('\n### Extracting splits ###\n')

split_groups = component_props.groupby(['co2l', 'time_stamp', 'split_number']) 
split_props = pd.DataFrame(index=split_groups.groups.keys(),
                           columns=['n_components', 'lost_load', 
                                    'lost_load_share', 'category'])
split_props.index=split_props.index.rename(['co2l','time_stamp','split_number'])
split_vectors = np.empty((split_groups.ngroups,
                         nx_graph.number_of_nodes()), float)

for i, (name, split) in enumerate(tqdm(split_groups)):
    split_props.loc[name,'n_components'] = split.shape[0]
    
    split_props.loc[name,'lost_load_share'] = (split.rocof.abs()>1).mul(split.load_share).sum()
    split_props.loc[name,'lost_load'] = (split.rocof.abs()>1).mul(split.load).sum()
    
    if split_props.loc[name,'lost_load_share']>0.99:
        split_props.loc[name,'category'] = 'global_blackout'
    elif split_props.loc[name,'lost_load_share']>0:
        split_props.loc[name,'category'] = 'local_blackout'
    elif ((split.rocof.abs()<1) & (split.load_share>0.99)).any():
        split_props.loc[name,'category'] = 'negligible'
    else:
        split_props.loc[name,'category'] = 'no_blackout'
    
    # for idx, component_props_of_idx in split.iterrows():
    #     split_vectors[i, indicator_vectors[idx].astype(bool)] = component_props_of_idx.rocof



# Save clustering results 
# np.save(save_path + f'indicator_vectors_all_n{n_nodes}.npy', indicator_vectors)
component_props.to_hdf(save_path + f'component_properties_all_n{n_nodes}.h5', key='df', mode= 'w')
np.save(save_path + f'split_vectors_all_n{n_nodes}.npy', split_vectors)
split_props=split_props.astype(dtype=dict(zip(split_props.columns,[int,float,float,str])))
split_props.to_hdf(save_path + f'split_properties_all_n{n_nodes}.h5', key='df', mode= 'w')
        
        
#### Calculate likelihoods #####

print('\nCalculate likelihoods of primary/secondary failures...\n')
likelihoods_primary = dict(keys=co2l_list)
likelihoods_secondary = dict(keys=co2l_list)

bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph),
                                                           nx_graph)
n_2_failures = cascade_simulation.calc_possible_double_line_failures(num_parallels,
                                                                     ignored_idxs=bridge_idxs)
# incorporating the weighting of the snapshots
number_of_simulations = len(n_2_failures) * network.snapshot_weightings.generators.sum()
print(number_of_simulations)

for co2l in co2l_list[::-1]:
    
    print('Co2 level %.2f' % co2l)
    
    # Load results
    splitting_cascades = pickle.load(open(path_to_cascade_results + 
                                      f'system_splits_Co2L{co2l}_n{n_nodes}.pickle' ,'rb'))
    
    # Calculate likelihood of edge to be primary or secondary failure
    l_primary, l_secondary = vis.calc_likelihood_failure(nx_graph, 
                                                         splitting_cascades, 
                                                         number_of_simulations, 
                                                         network.snapshot_weightings.generators)
    
    likelihoods_primary[co2l] = l_primary
    likelihoods_secondary[co2l] = l_secondary
    

with open(save_path + f'edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle' , 'wb') as handle:
    pickle.dump(likelihoods_primary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(save_path + f'edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle' , 'wb') as handle:
    pickle.dump(likelihoods_secondary, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
