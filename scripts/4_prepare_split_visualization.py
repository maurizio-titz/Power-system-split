import pickle
import sys

import numpy as np
import pypsa
import pandas as pd 

sys.path.append('./power_system_split/')
import utils
import visualisation as vis



# Setup paths 
path_to_pypsa_network   = './data/European_networks/'
path_to_cascades   = './results/cascade_results/'
path_to_evaluation = './results/evaluation_results/'
save_path = './results/split_visualization/'

# Setup parameters for network
criterion = 'nodes' 
snet_index = 0 # only AC grid of CE
co2l_list = np.arange(0.0,0.99,0.05)

# Choose fraction of simulations for agglomerative clustering
# (the rest is classified via NN classifier)
subset_size_for_clustering = 0.1 

# Choose minimum distance between seperated cluster
# (the node-criterion only selects splits with >10 nodes, 
# so we have to resolve a least hamming distance between the cluster of >9)
min_cluster_dist_nodes = 9

# Files to save clustering results
ind_vec_file = 'ind_vec_all_co2level_{}_based_w_hvdc_dist_{}.npy'.format(criterion,
                                                                         min_cluster_dist_nodes)
props_file = 'split_comp_props_all_co2level_{}_based_w_hvdc_dist_{}.h5'.format(criterion,
                                                                               min_cluster_dist_nodes)

# Load network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') 
G = utils.build_networkx_graph(network, snet_index = snet_index)


##### Indicator vector construction for split clustering #####
# (Construct one data set of split components from all CO$_2$ level simulations)

print('\nConstructing indicator vectors of split components...\n')
component_props = pd.DataFrame(columns= ['co2l','time_stamp', 'number_of_split',
                                         'inertia_proxy', 'load_imbalance' ])
indicator_vectors = np.empty((0, len(G)),dtype = 'int')

# Iterate through all Co2 levels
for co2l in co2l_list[::-1]:
    
    print('Co2 level %.2f' % co2l)
    level = np.round(co2l,2)
    
    # Load results
    splitting_cascades = pickle.load(open(path_to_cascades+f'system_splits_Co2L{level}.pickle' ,'rb'))
    fname = f'Europe_Co2L{level}_split_evaluation_'+criterion+'_based_snet_%i'%snet_index+'_w_hvdc.pickle'
    solution_dict = pickle.load(open(path_to_evaluation + fname,'rb'))
   
    # Construct vectors and their properties
    indicator_vec_level, comp_props_level = vis.indicator_vectors_from_rocof_solution(solution_dict, splitting_cascades,
                                                                                      G, criterion=criterion)
    # Append to set of vectors and properties
    indicator_vectors = np.concatenate([indicator_vectors, indicator_vec_level])
    comp_props_level.loc[:, 'co2l'] = co2l
    component_props = component_props.append(comp_props_level, ignore_index=True)
    
indicator_vectors = indicator_vectors.astype('int')
component_props.co2l = component_props.co2l.round(2)

# Alternatively: Load old indicator vector results
#indicator_vectors = np.load(save_path + ind_vec_file)
#component_props = pd.read_hdf(save_path + props_file)

#### Cluster split components #### 

print('\nClustering indicator vectors...\n')
n_cluster, cluster_labels = vis.cluster_indicator_vectors_combined(indicator_vectors,
                                                                   min_cluster_distance=min_cluster_dist_nodes/len(G),
                                                                   subset_ratio=subset_size_for_clustering)
component_props.loc[:, 'cluster_label'] = cluster_labels

# Save cluster results 
np.save(save_path + ind_vec_file, indicator_vectors)
component_props.to_hdf(save_path + props_file, key='df')


     
#### Calculate likelihoods #####

print('\nCalculate likelihoods of primary/secondary failures...\n')
likelihoods_primary = dict(keys=co2l_list.round(2))
likelihoods_secondary = dict(keys=co2l_list.round(2))

number_of_snapshots = network.snapshots.shape[0]

for co2l in co2l_list[::-1]:
    
    print('Co2 level %.2f' % co2l)
    level = np.round(co2l,2)
    
    # Load results
    splitting_cascades = pickle.load(open(path_to_cascades+f'system_splits_Co2L{level}.pickle' ,'rb'))
    fname = f'Europe_Co2L{level}_split_evaluation_'+criterion+'_based_snet_%i'%snet_index+'_w_hvdc.pickle'
    solution_dict = pickle.load(open(path_to_evaluation + fname,'rb'))
    
    # Calculate likelihood of edge to be primary or secondary failure
    l_primary, l_secondary = vis.calc_likelihood_failure(G,splitting_cascades, number_of_snapshots,
                                                         solution_dict=solution_dict)
    
    likelihoods_primary[level] = l_primary
    likelihoods_secondary[level] = l_secondary
    
    
with open(save_path + 'edge_likelihoods_primary_all_co2ls.pickle' , 'wb') as handle:
    pickle.dump(likelihoods_primary, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open(save_path + 'edge_likelihoods_secondary_all_co2ls.pickle' , 'wb') as handle:
    pickle.dump(likelihoods_secondary, handle, protocol = pickle.HIGHEST_PROTOCOL)   
    
