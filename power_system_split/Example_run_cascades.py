#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains an example on how to run cascade simulations for
an existing solved pypsa network

it should be called with arguments

year month number_of_nodes

and is for now tailored specifically for analyses of German networks that
were obtained in 10.1109/EEM49802.2020.9221886
"""
import os

import pypsa
import networkx as nx
import numpy as np
import pickle

from tqdm import tqdm

import sys

script_path = os.path.dirname(os.path.realpath('__file__'))

if not os.path.exists(script_path + '/Results'):
    os.makedirs(script_path + '/Results')

sys.path.append('power-system-split/power_system_split')
import utils

current_year = int(sys.argv[1])
current_month = int(sys.argv[2])
number_of_nodes = int(sys.argv[3])

save_path = 'Results/'
load_path = '%i/networks/' % current_year

network = pypsa.Network()
network.import_from_netcdf(load_path + 'elec_s_%i_ec_lv1.0_1H.nc' % number_of_nodes)
network.determine_network_topology()

branches = network.branches()
branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom"]]

G = utils.build_networkx_graph(branches)

current_snapshots = network.snapshots[(network.snapshots.month == current_month) & (network.snapshots.day == 1)]

indices = nx.get_edge_attributes(G, 'line_index')
line_limits = nx.get_edge_attributes(G, 's_nom')

### iterate only over non-bridge edges for now
bridges = list(nx.bridges(nx.Graph(G)))

non_bridges = list(set(list(G.edges())) - set(bridges))

splitting_cascades = {}

for snapshot in tqdm(current_snapshots):

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

with open(save_path + 'system_splits_%i_%i_%i.pickle' % (current_year, current_month, number_of_nodes), 'wb') as handle:
    pickle.dump(splitting_cascades, handle, protocol=pickle.HIGHEST_PROTOCOL)
