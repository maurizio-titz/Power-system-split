#!/usr/bin/python3
# -*- coding: utf-8 -*

import pickle
import gzip
import sys
import networkx as nx
from tqdm import tqdm

sys.path.append('./') 
from utils import cascade_simulation, data_handling

# Setup paths to solved PyPSA networks and results of this script
path_to_pypsa_network = './data/European_networks_sclopf/'
save_path =  './results/sclopf/cascade_results/'

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Load arguments
co2l = float(sys.argv[1]) 
n_nodes = int(sys.argv[2])
if len(sys.argv) > 3: 
    save_all_cascades = bool(sys.argv[3])
else:
    save_all_cascades = False

# Load PyPSA network, the graph of the subnetwork and its matrices
network = data_handling.load_pypsa_network(co2l, n_nodes, path_to_pypsa_network)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)

# Calculate possible N-1 and N-2 failures (using non-bridges)
bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph),
                                                           nx_graph)
n_2_failures = cascade_simulation.calc_possible_double_line_failures(num_parallels, ignored_idxs=bridge_idxs)
n_1_failures = cascade_simulation.calc_possible_single_line_failures(num_parallels, ignored_idxs=bridge_idxs)

splitting_cascades = {}

### Check N-1 stability ###
print('\n#### N-1 failures: Co2 level', co2l, ' | #Nodes:', n_nodes,' ####')
for snapshot in tqdm(network.snapshots):

    P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

    for initial_failure in n_1_failures:

        failing_links, system_split = cascade_simulation.simulate_cascade(I_m, B_d, P_0, line_limits,
                                                                          num_parallels, initial_failure,
                                                                          max_cascade_length=1)
        if len(failing_links)>1:
            raise(RuntimeError('PyPSA networks are not N-1 stable!'))


### Simulation of N-2 failures ###
print('\n#### N-2 failures: Co2 level', co2l, ' | #Nodes:', n_nodes, ' ####')
for snapshot in tqdm(network.snapshots):
    
    splitting_cascades[snapshot.strftime('%Y-%m-%d %H:00')] = {}

    P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

    for initial_failure in n_2_failures:

        failing_links, system_split = cascade_simulation.simulate_cascade(I_m, B_d, P_0, line_limits,
                                                                          num_parallels, initial_failure)
        
        if save_all_cascades:
            splitting_cascades[snapshot.strftime('%Y-%m-%d %H:00')][tuple(initial_failure)] = failing_links
        
        elif system_split:
            splitting_cascades[snapshot.strftime('%Y-%m-%d %H:00')][tuple(initial_failure)] = failing_links 
    
    fpath_out = save_path + f"system_splits_Co2L{co2l}_n{n_nodes}"
    if save_all_cascades:
        fpath_out += "_allcascades" 
    with gzip.open( fpath_out + ".pklz", 'wb') as handle:
        pickle.dump(splitting_cascades, handle, protocol = pickle.HIGHEST_PROTOCOL)
