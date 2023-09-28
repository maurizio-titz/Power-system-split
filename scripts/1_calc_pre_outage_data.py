import sys
sys.path.append('./')

import cartopy.geodesic as gd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from utils import data_handling, subgraph_evaluation
from utils.config import results_path, data_path

# Setup paths 
path_to_pypsa_network = data_path + 'European_networks_sclopf/'
save_path = results_path + 'sclopf/pre_outage_data/'

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Setup co2 levels
co2l_list = np.arange(0.0,0.81,0.1).round(1)

# Load arguments
n_nodes = int(sys.argv[1])

# Get number of time steps and graph
network = data_handling.load_pypsa_network(0.0, n_nodes, path_to_pypsa_network)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
n_time_steps = network.snapshots.shape[0]

### Extract pre-outage inertia data #####

inertia_time_series = np.zeros((co2l_list.shape[0], n_time_steps))
nodal_inertia_min_max = np.zeros((co2l_list.shape[0], 2,
                                  nx_graph.number_of_nodes()))

for i, co2l in enumerate(co2l_list):
    print('Co2 level %.2f' % co2l)
    
    network = data_handling.load_pypsa_network(co2l, n_nodes, path_to_pypsa_network)
    nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
    
    for t_count,timestamp in enumerate(network.snapshots):
        obs = subgraph_evaluation.evaluate_observables_for_subgraphs([nx_graph],
                                                                     network,
                                                                     timestamp,
                                                                     snet = 0)
        inertia_time_series[i,t_count] = obs.rot_energy
    
    
    # Select timestamps with min and max total inertia     
    t_largest = network.snapshots[np.argmax(inertia_time_series[i])]
    t_smallest = network.snapshots[np.argmin(inertia_time_series[i])]

    for t_count,timestamp in enumerate([t_smallest, t_largest]):
        for nodecount,node in enumerate(nx_graph.nodes()):
            subgraph = nx.Graph()
            subgraph.add_node(node)
            obs = subgraph_evaluation.evaluate_observables_for_subgraphs([subgraph],
                                                                network,
                                                                timestamp,
                                                                snet = 0)
            
            nodal_inertia_min_max[i, t_count, nodecount] = obs.rot_energy
    
np.save(save_path + 'inertia_time_series_all_co2ls.npy',
        inertia_time_series)
np.save(save_path + 'min_max_nodal_inertia_generation_all_co2ls.npy',
        nodal_inertia_min_max)
     
     
#### Calculate dipole vectors ####

pos = nx.get_node_attributes(nx_graph, 'pos')
dipole_vector = np.zeros((len(co2l_list),2,len(network.snapshots)))
mean_consumption_vector = np.zeros((len(co2l_list),nx_graph.number_of_nodes()))
weighted_mean_consumption_vector = np.zeros((len(co2l_list),nx_graph.number_of_nodes()))
graph_net_mismatch = np.zeros((len(co2l_list),len(network.snapshots)))

for i,co2l in enumerate(co2l_list):
    
    print('Co2 level %.2f' % co2l)
    
    network = data_handling.load_pypsa_network(co2l, n_nodes, path_to_pypsa_network)
    
    ### NOTE: Here, the mean position is subtracted from the coordinates
    position_vector = np.array([pos[n] for n in nx_graph.nodes()])
    position_vector[:,1] -= np.mean(position_vector[:,1])
    position_vector[:,0] -= np.mean(position_vector[:,0])    
    
    nodal_balance = network.generators_t.p.mul(network.generators.sign).T.groupby(network.generators["bus"]).sum()
    nodal_balance = nodal_balance.add(network.storage_units_t.p.T.groupby(network.storage_units["bus"]).sum(),
        fill_value=0).add(-network.loads_t.p.T.groupby(network.loads["bus"]).sum(),
                          fill_value=0)
    
    ## add hvdc link subtraction/addition
    nodal_balance = nodal_balance.add(-network.links_t.p0.T.groupby(network.links["bus0"]).sum(), fill_value=0)
    nodal_balance = nodal_balance.add(-network.links_t.p1.T.groupby(network.links["bus1"]).sum(), fill_value=0)
    
    nodal_balance = -nodal_balance
    
    for nodecount,node in enumerate(nx_graph.nodes()):
        mean_consumption_vector[i,nodecount] = nodal_balance.mean(axis = 1).loc[node]
        weighted_mean_consumption_vector[i,nodecount] = nodal_balance.mul(network.snapshot_weightings.generators, axis="columns").mean(axis = 1).loc[node]
        graph_net_mismatch[i] += nodal_balance.loc[node]
        dipole_vector[i] += np.outer(position_vector[nodecount],nodal_balance.loc[node].to_numpy())
        if np.any(np.isnan(dipole_vector)):
            raise ValueError('SPI coordinates are not valid!')
    
np.save(save_path + 'dipole_vector_time_series_all_co2ls.npy', dipole_vector)
np.save(save_path + 'mean_nodal_consumption_all_co2ls.npy', mean_consumption_vector)
np.save(save_path + 'weighted_mean_nodal_consumption_all_co2ls.npy', weighted_mean_consumption_vector)
np.save(save_path + 'graph_net_power_mismatch_time_series_all_co2ls.npy', graph_net_mismatch)

#### Calculate spatial power inhomogeneity ####
# (To calculate the spatial power inhomogeneity (spi) we rescale the spi vector
# by the below scale factor, since it has the units of power and positions in long and lat
# We then evaluate the geodesic distance between the mean position (given in logitude and latitude)
# and the end point of the rescaled spi vector)

print('\nCalculate spatial power inhomogeneity...')
vec_norm = np.zeros((len(co2l_list),len(network.snapshots)))
positions = np.array([pos[n] for n in nx_graph.nodes()])
mean_pos = np.array([np.mean(positions[:,0]),np.mean(positions[:,1])])
scale_factor = 1e6

k = gd.Geodesic() 

for i,co2l in enumerate(co2l_list):
    print('Co2 level %.2f' % co2l)

    for j in range(len(network.snapshots)):
        shapely_pos = Point(dipole_vector[i,:,j]/scale_factor+mean_pos)
        distance = k.inverse(shapely_pos.coords, mean_pos)[0,0]/1000
        vec_norm[i,j] = distance
        if np.any(np.isnan(vec_norm)):
            raise ValueError('SPI coordinates are not valid!')
np.save(save_path + 'spi_time_series_all_co2ls.npy', vec_norm)
