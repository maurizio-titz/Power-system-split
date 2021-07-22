import sys
import pickle5 as pickle 

import networkx as nx
import numpy as np
import pypsa
import pandas as pd


path = '/media/fkaiser/'

sys.path.append('../power_system_split/')
import utils,visualisation

#### Evaluate component cluster for CO$_2$ levels together ###


# Setup paths and parameters
path_to_pypsa_network   = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/'
results_path = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/Results/'
cluster_results_path = '/media/jkruse/Projects/system_splits/power-system-split/results/'

criterion = 'nodes' # or 'load'
snet_index = 0 # only AC grid of CE
co2l_list = co2ls = np.arange(0.0,0.99,0.05)
n_subset_solutions = -1 # number of random sampled time steps from each co2 level simulation, -1 means all

# Load network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') # +year+'/elec_s_306_ec_lv1.0_1H.nc')#
G = utils.build_networkx_graph(network, snet_index = snet_index)

# Determine number of simulations
bridges = list(nx.bridges(nx.Graph(G)))
non_bridges = list(set(list(G.edges())) - set(bridges))
n_initial_failures = len(non_bridges)
if n_subset_solutions==-1:
    n_time_stamps = network.snapshots.size
else:
    n_time_stamps = n_subset_solutions
number_of_simulations = n_initial_failures*n_time_stamps


# Indicator vector construction 
# Construct one data set of split components from all CO$_2$ level simulations

component_props = pd.DataFrame(columns= ['co2l','time_stamp', 'number_of_split', 'inertia_proxy', 'load_imbalance',
                                         'available_flexible_generation'])
indicator_vectors = np.empty((0, len(G)))

for co2l in co2l_list:
    print('\n','CO2 level ', co2l)
    
    file = results_path+'system_splits_Co2L{:.2}.pickle'.format(co2l)
    print(file)
    splitting_cascades = pickle.load(open(file,'rb'))
    file = results_path+'Europe_Co2L{:.2}_split_evaluation_'.format(co2l)+criterion+'_based_snet_%i'%snet_index+'.pickle'
    print(file)
    solution_dict = pickle.load(open(file,'rb'))
   
    if n_subset_solutions==-1:
        indicator_vectors_level, component_props_level = visualisation.indicator_vectors_from_rocof_solution(solution_dict,
                                                                                                             splitting_cascades,
                                                                                                             G, criterion=criterion)
    else:
        subset_of_keys = np.random.choice(list(solution_dict.keys()), n_subset_solutions, replace=False)
        subset_solution_dict = {key: solution_dict[key] for key in subset_of_keys}
        indicator_vectors_level, component_props_level = visualisation.indicator_vectors_from_rocof_solution(subset_solution_dict,
                                                                                                             splitting_cascades,
                                                                                                             G, criterion=criterion)
    indicator_vectors = np.concatenate([indicator_vectors, indicator_vectors_level])
    component_props_level.loc[:, 'co2l'] = co2l
    component_props = component_props.append(component_props_level, ignore_index=True)
    
indicator_vectors = indicator_vectors.astype(bool)


#indicator_vectors = np.load(cluster_results_path + 'indicator_vectors_all_co2level_{}_based.npy'.format(criterion))
#component_props = pd.read_hdf(cluster_results_path + 'split_component_properties_all_co2level_{}_based.h5'.format(criterion))


# Cluster split components 
#...this can take a few hours for all time steps and co2 levels

#Agglomerative clustering
#(the node-criterion only selects splits with >10 nodes, so we have to resolve a hamming distance between the cluster of >19)
min_cluster_distance = 19./ len(G) 
n_cluster, cluster_labels = visualisation.cluster_indicator_vectors_agglomerative(indicator_vectors,
                                                                                  min_cluster_distance=min_cluster_distance,
                                                                                  cluster_distance_type='single')
component_props.loc[:, 'cluster_label_agg_mdist{:.3}'.format(min_cluster_distance)] = cluster_labels

# # DBSCAN
# neighbor_max_dist = 19./ len(G) 
# n_cluster, n_noise, cluster_labels = visualisation.cluster_indicator_vectors_dbscan(indicator_vectors,
#                                                                                     neighbor_max_dist=neighbor_max_dist)
# component_props.loc[:, 'cluster_label_db_mdist{:.3}'.format(neighbor_max_dist)] = cluster_labels

# %%
component_props.to_hdf(cluster_results_path + 'split_component_properties_all_co2level_{}_based.h5'.format(criterion), key='df')
np.save(cluster_results_path + 'indicator_vectors_all_co2level_{}_based.npy'.format(criterion), indicator_vectors)

