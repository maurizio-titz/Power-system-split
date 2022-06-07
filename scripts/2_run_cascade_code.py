import pickle
import sys

import networkx as nx
import numpy as np
import pypsa
from tqdm import tqdm
import pandas as pd

sys.path.append('./')
from power_system_split import utils

# Setup paths to solved PyPSA networks and results of this script
path_to_pypsa_network   = './data/European_networks_sclopf/'
save_path =  './results/sclopf/cascade_results/'

# Load co2 level
co2l = 0.1 # float(sys.argv[1])

# Load PyPSA network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+f'sclopf-elec_s_800_ec_lv1.0_Co2L{co2l}-3H.nc')

# The following line is needed to remove the outage lines used for SCLOPF. When not removed, multiple edges 
# are added to the network graph
outage_lines = network.lines[network.lines.index.str[-6:]=='outage'].index
network.mremove('Line', outage_lines)
network.determine_network_topology()


######
#TODO: remove this line if the networks contain timestamp snapshots
network.snapshots = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))

network.snapshot_weightings.index = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))

for component in network.all_components:
    pnl = network.pnl(component)
    attrs = network.components[component]["attrs"]

    for k,default in attrs.default[attrs.varying].iteritems():
        pnl[k].index = pd.date_range(start='2013', freq='3H', periods=len(network.snapshots))
######



# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# Otherwise you can choose None or -1.
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Build graph for subnetwork
G = utils.build_networkx_graph(network, snet_index= snet_index)

# Get line index 
# Line index is a unique number to identify the given edge.
# It is directly extracted from pypsa
indices = nx.get_edge_attributes(G, 'line_index')

# Extract time steps
current_snapshots = network.snapshots[:1]

# Calculate possible double line failure (using non-bridges)
possible_failures = utils.calc_possible_double_line_failures(G)

splitting_cascades = {}

# Iterate over each time step in data set
for snapshot in current_snapshots:

    splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')] = []

    initial_loading = {e: np.sum([network.lines_t.p0.loc[snapshot].loc[index] for index in index_list]) for
                       e, index_list in indices.items()}
    P0 = utils.get_effective_injections(G, initial_loading)

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    # Simulate cascade for every tuple of trigger links
    for initial_failure in tqdm(possible_failures):

        failing_links, system_split = utils.simulate_cascade_PTDF_based_edge_based_reduced(G,
                                                                                           trigger_links=initial_failure,
                                                                                           P0=P0)
        if system_split:
            splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')].append(failing_links)

with open(save_path + f'system_splits_Co2L{co2l}.pickle' , 'wb') as handle:
    pickle.dump(splitting_cascades, handle, protocol = pickle.HIGHEST_PROTOCOL)
