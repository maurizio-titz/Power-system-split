import sys
import pickle,itertools
import requests, json
import copy

import networkx as nx
import numpy as np
import pypsa

#date_string = datetime.datetime.now().strftime("%I_%M_%B_%d")

path = '/media/fkaiser/'

sys.path.append(path+'Code/power-system-split/power_system_split/')
import utils,visualisation

path_to_pypsa_network   = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/'
save_path = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/Results/'

current_month = int(sys.argv[1])
co2l = float(sys.argv[2])

network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+'elec_s_800_ec_lv1.0_Co2L%.1f-3H.nc'%co2l) 
network.determine_network_topology()

snet_index = 0
snet = network.sub_networks['obj'][snet_index]

branches = snet.branches()
branches = branches[["bus0","bus1","x_pu_eff","s_nom"]]

G = utils.build_networkx_graph(branches)

indices = nx.get_edge_attributes(G, 'line_index')
line_limits = nx.get_edge_attributes(G, 's_nom')

current_snapshots = network.snapshots[network.snapshots.month == current_month]

### iterate only over non-bridge edges for now
bridges = list(nx.bridges(nx.Graph(G)))

non_bridges = list(set(list(G.edges())) - set(bridges))

splitting_cascades = {}

for snapshot in current_snapshots:

    splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')] = []

    initial_loading = {e: np.sum([network.lines_t.p0.loc[snapshot].loc[index] for index in index_list]) for
                       e, index_list in indices.items()}
    P0 = utils.get_effective_injections(G, initial_loading)

    l_copy = initial_loading.copy()
    for key in l_copy.keys():
        initial_loading[key[::-1]] = initial_loading[key]

    for edge in non_bridges:

        failing_links, system_split = utils.simulate_cascade_PTDF_based_edge_based_reduced(G,
                                                                                           trigger_links=[edge],
                                                                                           line_limits=line_limits,
                                                                                           P0=P0)
        if system_split:
            splitting_cascades[snapshot.strftime('%d_%m_%Y_%H')].append(failing_links)

with open(save_path + 'system_splits_%i_Co2L%.1f.pickle' % (current_month, co2l), 'wb') as handle:
    pickle.dump(splitting_cascades, handle, protocol = pickle.HIGHEST_PROTOCOL)
