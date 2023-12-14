import sys

import numpy as np
import pypsa
import pandas as pd
import networkx as nx 

root_path = './'
sys.path.append(root_path)
from . import visualization as vis 


# Setup paths 
path_to_pypsa_network   = root_path + 'data/European_networks/'
path_to_vis_results = root_path + 'results/split_visualization/'
save_path = root_path + 'results/inertia_placement/'

# Setup parameters for network
criterion = 'nodes' 
snet_index = 0 # only AC grid of CE
co2l_list = np.arange(0.0,0.99,0.05)

# Load network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network + 
                           'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') 
G = build_networkx_graph(network, snet_index = snet_index)

# Get number of split simulations
bridges = list(nx.bridges(nx.Graph(G)))
non_bridges = list(set(list(G.edges())) - set(bridges))
n_initial_failures = len(non_bridges)
n_time_stamps = network.snapshots.size
number_of_simulations = n_initial_failures*n_time_stamps 

# Load split component properties and their indicator vectors
component_props = pd.read_hdf(path_to_vis_results + 'split_comp_props_all_co2level_{}_based_w_hvdc_dist_9.h5'.format(criterion))
indicator_vectors = np.load(path_to_vis_results + 'ind_vec_all_co2level_{}_based_w_hvdc_dist_9.npy'.format(criterion))

# assume that synthetic inertia corresponding to
# a generator with x MW is added
additional_synthetic_inertia_per_step = 1*1000 # [MW]

# Quantile for value at risk (var) estimation
q=0.995

# Relative deviation from reference value that we allow
epsilon=0.

# Calculate reference var from reference level
ref_level = 0.55
comp_props_level_co2lref = component_props[component_props.co2l==ref_level].copy()
comp_props_level_co2lref['rocof'] = vis.calc_rocof(comp_props_level_co2lref.load_imbalance,
                                             comp_props_level_co2lref.inertia_proxy)
var_co2lref = vis.val_at_risk(comp_props_level_co2lref, number_of_simulations, q, method='abs')


for co2l in co2l_list:

    level = round(co2l, 2)
    print('\n ### CO2 level ', level, ' ###\n')
    
    indicator_vectors_level = indicator_vectors[component_props.co2l==level].copy()
    comp_props_level = component_props[component_props.co2l==level].copy()
    comp_props_level['rocof'] = vis.calc_rocof(comp_props_level.load_imbalance,
                                               comp_props_level.inertia_proxy)


    comp_props_new, inertia_placement_info = inertia_placement(comp_props_level,
                                                                     indicator_vectors_level,
                                                                     var_co2lref,
                                                                     additional_synthetic_inertia_per_step,
                                                                     q,
                                                                     number_of_simulations,
                                                                     list(G.nodes()),
                                                                     epsilon)


    
        
    # Save results for co2 level
    comp_props_level['new_inertia'] = comp_props_new.inertia_proxy
    comp_props_level['new_rocof'] =  vis.calc_rocof(comp_props_level.load_imbalance,
                                                    comp_props_level.new_inertia)
    inertia_placement_info.to_hdf(save_path+'inertia_placement_steps_co2l{:02.0f}.h5'.format(level*100), key='df')
    comp_props_level.to_hdf(save_path+'component_props_with_syn_inertia_co2l{:02.0f}.h5'.format(level*100), key='df')
