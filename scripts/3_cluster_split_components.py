import sys
import pickle

import networkx as nx
import pandas as pd
import numpy as np
import pypsa

from sklearn.neighbors import RadiusNeighborsClassifier


path = '/media/fkaiser/'

#sys.path.append(path+'Code/power-system-split/power_system_split/')
sys.path.append('../power_system_split/')
#sys.path.append('./')
import utils,visualisation


# Setup paths 
path_to_pypsa_network   = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/'
evaluation_result_path = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/Results/'
cluster_results_path = '/media/jkruse/Projects/system_splits/power-system-split/results/'

# Setup parameters
criterion = 'nodes' # or 'load'
snet_index = 0 # only AC grid of CE
co2l_list = np.arange(0.0,0.99,0.05)
subset_size_for_clustering = 0.01
# the node-criterion only selects splits with >10 nodes, 
# so we have to resolve a least hamming distance between the cluster of >9
min_cluster_distance_in_nodes = 9


# Load network
# TODO: the graph G should be independent of co2l right? so that we can load it out of the loop!?
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') 
G = utils.build_networkx_graph(network, snet_index = snet_index)

# Determine number of simulations,
# i.e., total number of initial failures over the simulated period of one year
bridges = list(nx.bridges(nx.Graph(G)))
non_bridges = list(set(list(G.edges())) - set(bridges))
n_initial_failures = len(non_bridges)
n_time_stamps = network.snapshots.size
number_of_simulations = n_initial_failures*n_time_stamps


# Indicator vector construction 
# (Construct one data set of split components from all CO$_2$ level simulations)

component_props = pd.DataFrame(columns= ['co2l','time_stamp', 'number_of_split', 'inertia_proxy',
                                         'load_imbalance', 'available_flexible_generation'])
indicator_vectors = np.empty((0, len(G)),dtype = 'int')

for co2l in co2l_list[::-1]:
    print('\n','CO2 level ', co2l)
    level = np.round(co2l,2)
    level_string = f'Co2L{level}'
    print(level_string)
    splitting_cascades = pickle.load(open(evaluation_result_path+f'system_splits_Co2L{level}.pickle' ,'rb'))
    fname = f'Europe_Co2L{level}_split_evaluation_'+criterion+'_based_snet_%i'%snet_index+'_w_hvdc.pickle'
    solution_dict = pickle.load(open(evaluation_result_path + fname,'rb'))
   

    indicator_vec_level, comp_props_level = visualisation.indicator_vectors_from_rocof_solution(solution_dict,
                                                                                                splitting_cascades,
                                                                                                G,
                                                                                                criterion=criterion)

    indicator_vectors = np.concatenate([indicator_vectors, indicator_vec_level])
    comp_props_level.loc[:, 'co2l'] = co2l
    component_props = component_props.append(comp_props_level, ignore_index=True)
    
indicator_vectors = indicator_vectors.astype('int')
component_props.co2l = component_props.co2l.round(2)


# Optionally: Save and load intermediate indicator vector results

#np.save(cluster_results_path + 'indicator_vectors_all_co2level_{}_based_w_hvdc.npy'.format(criterion),indicator_vectors)
#component_props.to_hdf(cluster_results_path + 'split_component_properties_all_co2level_{}_based_w_hvdc.h5'.format(criterion), key='df')
#indicator_vectors = np.load(cluster_results_path + 'indicator_vectors_all_co2level_{}_based_w_hvdc.npy'.format(criterion))
#component_props = pd.read_hdf(cluster_results_path + 'split_component_properties_all_co2level_{}_based_w_hvdc.h5'.format(criterion))


# Cluster split components 

# Select subset of components for random time stemps and co2 levels
indices_all_components = np.arange(indicator_vectors.shape[0])
np.random.shuffle(indices_all_components)
subset_indices = indices_all_components[:int(subset_size_for_clustering*indicator_vectors.shape[0])]
remaining_indices = indices_all_components[int(subset_size_for_clustering*indicator_vectors.shape[0]):]
print(subset_indices.size, 'components will be used.')
print(remaining_indices.size, 'components will be assigned via NN classification.')


# Identify cluster in subset with agglomerative clustering
min_c_distance = min_cluster_distance_in_nodes/ len(G) 
n_cluster, subset_c_labels = visualisation.cluster_indicator_vectors_agglomerative(indicator_vectors[subset_indices], 
                                                                                  min_cluster_distance = min_c_distance,
                                                                                  cluster_distance_type = 'average')
component_props.loc[:, 'cluster_label'] = np.nan
component_props.loc[subset_indices, 'cluster_label'] = subset_c_labels
n_cluster

# Assign remaining vectors to clusters via Nearest-Neighbor classifier
rnc = RadiusNeighborsClassifier(radius= min_c_distance, metric='hamming', outlier_label=-1, n_jobs=20)
rnc.fit(indicator_vectors[subset_indices], subset_c_labels)
remaining_cluster_labels = rnc.predict(indicator_vectors[remaining_indices])
component_props.loc[remaining_indices,'cluster_label'] = remaining_cluster_labels

print(component_props.shape)
print(indicator_vectors.shape)
print(np.unique(component_props["cluster_label"]))


# Save results
np.save(cluster_results_path +\
        'indicator_vectors_all_co2level_{}_based_w_hvdc.npy'.format(criterion),indicator_vectors)
component_props.to_hdf(cluster_results_path +\
                       'split_component_properties_all_co2level_{}_based_w_hvdc.h5'.format(criterion),
                       key='df')

