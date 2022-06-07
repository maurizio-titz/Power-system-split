import sys

import cartopy.geodesic as gd
import networkx as nx
import numpy as np
import pypsa
from shapely.geometry import Point

sys.path.append('./')
from power_system_split import utils

# Setup paths 
path_to_pypsa_network   = './data/European_networks_scopf/'
save_path = './results/sclopf/pre_outage_data/'

# Setup parameters for network
criterion = 'nodes' 
snet_index = 0 # only AC grid of CE
co2l_list = np.arange(0.0,0.99,0.05)

# Load network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L0.5-3H.nc') 
G = utils.build_networkx_graph(network,snet_index = snet_index)
pos = nx.get_node_attributes(G,'pos')

### Calculate (total) inertia time series #####

print('\nCalculate total inertia time series ...')
inertia_time_series = np.zeros((len(co2l_list),2920))

for i, co2l in enumerate(co2l_list):
    print('Co2 level %.2f' % co2l)
    level = np.round(co2l,2)
    network = pypsa.Network()
    network.import_from_netcdf(path_to_pypsa_network+f'elec_s_800_ec_lv1.0_Co2L{level}-3H.nc')  
    inertia_time_series[i] = utils.calc_system_inertia_over_time(network,snet_index = 0)

np.save(save_path + 'inertia_time_series_all_co2ls.npy', inertia_time_series)


#### Calculate inertia generation time series per node for two Co2 levels ####

print('\nCalculate inertia generation time series per node...')
target_levels = [0.95,0.00]
inertia_generation = np.zeros((len(target_levels),2,len(G.nodes())))

for j,level in enumerate(target_levels):
    
    # Load PyPSA network for co2 level
    network = pypsa.Network()
    network.import_from_netcdf(path_to_pypsa_network+f'elec_s_800_ec_lv1.0_Co2L{np.round(level,2)}-3H.nc') 
    
    # Select timestamps with min and max total inertia     
    level_index = np.where(np.round(co2l_list,2) == level)[0][0]
    sorted_inertia_time_index = np.argsort(inertia_time_series[level_index])
    t_largest_inertia = network.snapshots[sorted_inertia_time_index[-1]]
    t_smallest_inertia = network.snapshots[sorted_inertia_time_index[0]]
    timestamps = [t_smallest_inertia, t_largest_inertia]
    
    # Calculate inertia generation per node
    load_shedding_indices = network.generators[network.generators.carrier.isin(['load'])].index
    for i,timestamp in enumerate(timestamps):
        current_generation = network.generators_t.p.loc[timestamp].copy()
        current_generation[load_shedding_indices] /= 1e3
        current_storage    = network.storage_units_t.p.loc[timestamp]
        for nodecount,node in enumerate(G.nodes()):
            subgraph = nx.Graph()
            subgraph.add_node(node)
            inertia_generation[j,i,nodecount] = utils.get_inertia_gen_subgraph(subgraph,
                                                                               network.generators,
                                                                               current_generation,
                                                                               network.storage_units,
                                                                               current_storage,
                                                                               use_pnom = True)
            
np.save(save_path + 'min_max_nodal_inertia_generation_co2l_95_and_00.npy', inertia_generation)
     
     
#### Calculate dipole vectors ####

print('\nCalculate dipole vectors...')
dipole_vector = np.zeros((len(co2l_list),2,len(network.snapshots)))

mean_consumption_vector = np.zeros((len(co2l_list),len(G.nodes())))
graph_net_mismatch = np.zeros((len(co2l_list),len(network.snapshots)))

for i,co2l in enumerate(co2l_list):
    
    print('Co2 level %.2f' % co2l)
    
    network = pypsa.Network()
    level = np.round(co2l,2)
    network.import_from_netcdf(path_to_pypsa_network+f'elec_s_800_ec_lv1.0_Co2L{level}-3H.nc')  
    
    ### NOTE: Here, the mean position is subtracted from the coordinates
    position_vector = np.array([pos[n] for n in G.nodes()])
    position_vector[:,1] -= np.mean(position_vector[:,1])
    position_vector[:,0] -= np.mean(position_vector[:,0])    
    
    nodal_balance = (network.generators_t.p.T.groupby(network.generators["bus"]).sum()-\
                    network.loads_t.p.T.groupby(network.loads["bus"]).sum()+\
                    network.storage_units_t.p.T.groupby(network.storage_units["bus"]).sum()).copy()
    
    ## add hvdc link subtraction/addition
    nodal_balance = nodal_balance.add(-network.links_t.p0.T.groupby(network.links["bus0"]).sum(), fill_value=0)
    nodal_balance = nodal_balance.add(-network.links_t.p1.T.groupby(network.links["bus1"]).sum(), fill_value=0)
    
    nodal_balance = -nodal_balance
    
    for nodecount,node in enumerate(G.nodes()):
        mean_consumption_vector[i,nodecount] = nodal_balance.mean(axis = 1).loc[node]
        graph_net_mismatch[i] += nodal_balance.loc[node]
        dipole_vector[i] += np.outer(position_vector[nodecount],nodal_balance.loc[node].to_numpy())
        
np.save(save_path + 'dipole_vector_time_series_all_co2ls.npy', dipole_vector)
np.save(save_path + 'mean_nodal_consumption_all_co2ls.npy', mean_consumption_vector)
np.save(save_path + 'graph_net_power_mismatch_time_series_all_co2ls.npy', graph_net_mismatch)

#### Calculate spatial power inhomogeneity ####

# (To calculate the spatial power inhomogeneity (spi) we rescale the spi vector
# by the below scale factor, since it has the units of power and positions in long and lat
# We then evaluate the geodesic distance between the mean position (given in logitude and latitude)
# and the end point of the rescaled spi vector)

print('\nCalculate spatial power inhomogeneity...')
vec_norm = np.zeros((len(co2l_list),len(network.snapshots)))
positions = np.array([pos[n] for n in G.nodes()])
mean_pos = np.array([np.mean(positions[:,0]),np.mean(positions[:,1])])
scale_factor = 1e6

k = gd.Geodesic() 

for i,co2l in enumerate(co2l_list):
    print('Co2 level %.2f' % co2l)

    for j in range(len(network.snapshots)):
        shapely_pos = Point(dipole_vector[i,:,j]/scale_factor+mean_pos)
        distance = k.inverse(shapely_pos, Point(mean_pos))[0,0]/1000
        vec_norm[i,j] = distance

np.save(save_path + 'spi_time_series_all_co2ls.npy', vec_norm)
