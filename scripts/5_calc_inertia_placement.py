import sys

import numpy as np
import pypsa
import pandas as pd
import networkx as nx 

sys.path.append('./power_system_split/')
import utils
import visualisation as vis 



# Setup paths 
path_to_pypsa_network   = './data/European_networks/'
path_to_vis_results = './results/split_visualization/'
save_path = './results/inertia_placement/'

# Setup parameters for network
criterion = 'nodes' 
snet_index = 0 # only AC grid of CE
co2l_list = np.arange(0.0,0.99,0.05)

# Load network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') 
G = utils.build_networkx_graph(network, snet_index = snet_index)

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
# a generator with 1 GW is added
additional_synthetic_inertia_per_step = 10000 # [MW]

# Quantile for value at risk (var) estimation
q= 0.999

# Calculate reference var from 95% Co2 system
comp_props_co2l95 = component_props[component_props.co2l==0.95].copy()
comp_props_co2l95['rocof'] = 50*comp_props_co2l95.load_imbalance /  (comp_props_co2l95.inertia_proxy*2*6)
var_co2l95 = vis.val_at_risk_rocof(comp_props_co2l95, number_of_simulations, q)


for co2l in co2l_list:

    level = round(co2l, 2)
    print('\n ### CO2 level ', level, ' ###\n')
    
    # Extract rocof without additional inertia
    indicator_vectors_level = indicator_vectors[component_props.co2l==level].copy()
    comp_props_level = component_props[component_props.co2l==level].copy()
    comp_props_level['old_rocof'] = 50*comp_props_level.load_imbalance /  (comp_props_level.inertia_proxy*2*6)

    # Initialize values for inertia placement
    step=1
    new_rocof = comp_props_level.old_rocof.values
    new_var = vis.val_at_risk_rocof(comp_props_level, number_of_simulations, q,
                                    'old_rocof')
    new_inertia = comp_props_level.inertia_proxy.values
    old_inertia = comp_props_level.inertia_proxy.values
    inertia_placement_steps = pd.DataFrame(columns=['new_var', 'added_node_index'], dtype=np.float)
    inertia_placement_steps.loc[0, 'added_node_index'] = np.nan
    inertia_placement_steps.loc[0, 'new_var'] = new_var
    potential_comp_props = comp_props_level.copy()
    
    
    # Continue placing inertia until the 95% reference value is reached
    while new_var > var_co2l95 :

        # Try all potential nodes 
        for count, node in enumerate(list(G.nodes())):

            # identify splits where this node is involved
            indices = np.where(indicator_vectors_level[:, count])[0]

            # assign additional_synthetic_inertia to all split components
            synthetic_inertia = np.zeros(comp_props_level.shape[0])
            synthetic_inertia[indices] += additional_synthetic_inertia_per_step

            # add new synthetic inertia
            potential_inertia = old_inertia + synthetic_inertia
            potential_rocof = 50*comp_props_level.load_imbalance.values / (potential_inertia*2*6)
            potential_comp_props.loc[:, 'rocof'] = potential_rocof
            
            potential_var = vis.val_at_risk_rocof(potential_comp_props, number_of_simulations, q)
                    
            # Accept inertia placement if it improves the var
            if potential_var < new_var:
                
                new_rocof = potential_rocof
                new_var = potential_var
                new_inertia = potential_inertia
                best_step_node_index = count

        print(step, ' New var:', new_var, 'Target Var:', var_co2l95)
        
        # Save the best inertia allocation of this placement step      
        inertia_placement_steps.loc[step, 'added_node_index'] = best_step_node_index
        inertia_placement_steps.loc[step, 'new_var'] = new_var

        
        # Go to next placement step
        old_inertia = new_inertia
        step+=1
        
    # Save results for co2 level
    comp_props_level['new_inertia'] = new_inertia
    comp_props_level['new_rocof'] =  new_rocof
    comp_props_level['new_inertia'] = new_inertia   
    inertia_placement_steps.to_hdf(save_path+'inertia_placement_steps_co2l{:02.0f}.h5'.format(level*100), key='df')
    comp_props_level.to_hdf(save_path+'component_props_with_syn_inertia_co2l{:02.0f}.h5'.format(level*100), key='df')
    print(inertia_placement_steps)