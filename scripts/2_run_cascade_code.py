import pickle
import sys

import networkx as nx
import numpy as np
import pypsa
from tqdm import tqdm

sys.path.append('./power_system_split/')
import utils

# Setup paths to solved PyPSA networks and results of this script
path_to_pypsa_network   = './data/European_networks/'
save_path =  './results/cascade_results/'

# Load co2 level
co2l = float(sys.argv[1])

# Load PyPSA network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+f'elec_s_800_ec_lv1.0_Co2L{co2l}-3H.nc')
network.determine_network_topology()

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# Otherwise you can choose None or -1.
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Build graph for subnetwork
G = utils.build_networkx_graph(network, snet_index= snet_index)

# Get line index and line limits for cascade simulation
# Line index is a unique number to identify the given edge.
# It is directly extracted from pypsa
indices = nx.get_edge_attributes(G, 'line_index')
line_limits = nx.get_edge_attributes(G, 's_nom')

# Extract time steps
current_snapshots = network.snapshots

# Iterate only over non-bridge edges for now
bridges = list(nx.bridges(nx.Graph(G)))
non_bridges = list(set(list(G.edges())) - set(bridges))

splitting_cascades = {}

# Iterate over each time step in data set
for snapshot in tqdm(current_snapshots[:10]):

    splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')] = []

    initial_loading = {e: np.sum([network.lines_t.p0.loc[snapshot].loc[index] for index in index_list]) for
                       e, index_list in indices.items()}
    P0 = utils.get_effective_injections(G, initial_loading)

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    # Simulate cascade for every (non-bridge) edge in the (sub)network
    for edge in non_bridges:

        failing_links, system_split = utils.simulate_cascade_PTDF_based_edge_based_reduced(G,
                                                                                           trigger_links=[edge],
                                                                                           line_limits=line_limits,
                                                                                           P0=P0)
        if system_split:
            splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')].append(failing_links)

with open(save_path + f'system_splits_Co2L{co2l}.pickle' , 'wb') as handle:
    pickle.dump(splitting_cascades, handle, protocol = pickle.HIGHEST_PROTOCOL)
