import pickle
import sys
import networkx as nx
from tqdm import tqdm

sys.path.append('./') 
from power_system_split import cascade_simulation, data_handling

# Setup paths to solved PyPSA networks and results of this script
save_path =  './results/sclopf/cascade_results/'

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Load co2 level
co2l = 0.0 #float(sys.argv[1]) #TODO include sys arg again!

# Load PyPSA network, the graph of the subnetwork and its matrices
network = data_handling.load_pypsa_network(co2l)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)

# Calculate possible N-1 and N-2 failures (using non-bridges)
bridges_indices = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph),
                                                           nx_graph)
n_2_failures = cascade_simulation.calc_possible_double_line_failures(num_parallels, bridges_indices)
n_1_failures = cascade_simulation.calc_possible_single_line_failures(num_parallels, bridges_indices)

splitting_cascades = {}

#TODO: remove this when using all snapshots
current_snapshots = network.snapshots

#TODO: one cascade simulation (one trigger) is 2x faster on my laptop than on Otter. Why? 

### Check N-1 stability ###

for snapshot in current_snapshots:

    print('#### N-1 failures: Co2 level', co2l, ' snapshot ', snapshot.strftime('%Y-%m-%d %H:00'), ' ####')

    P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

    for initial_failure in tqdm(n_1_failures):

        failing_links, system_split = cascade_simulation.simulate_cascade(I_m, B_d, P_0, line_limits,
                                                                          num_parallels, initial_failure,
                                                                          max_cascade_length=1)
        if len(failing_links)>1:
            raise(RuntimeError('PyPSA networks are not N-1 stable!'))


### Simulation of N-2 failures ###

for snapshot in current_snapshots:

    print('#### N-2 failures: Co2 level', co2l, ' snapshot ', snapshot.strftime('%Y-%m-%d %H:00'), ' ####')
    
    splitting_cascades[snapshot.strftime('%Y-%m-%d %H:00')] = {}


    P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

    for initial_failure in tqdm(n_2_failures):

        failing_links, system_split = cascade_simulation.simulate_cascade(I_m, B_d, P_0, line_limits,
                                                                          num_parallels, initial_failure)
        if system_split:
            splitting_cascades[snapshot.strftime('%Y-%m-%d %H:00')][tuple(initial_failure)] = failing_links 

    with open(save_path + f'system_splits_Co2L{co2l}.pickle' , 'wb') as handle:
        pickle.dump(splitting_cascades, handle, protocol = pickle.HIGHEST_PROTOCOL)
