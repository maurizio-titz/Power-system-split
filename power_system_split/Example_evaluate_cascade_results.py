#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains an example on how to evaluate the load imbalance
and inertia for each subgraph and point in time for an existing cascade
result/system split simulation

it is for now tailored specifically for analyses of German networks that
were obtained in 10.1109/EEM49802.2020.9221886 """

import pypsa
import pickle

import sys
import os

from tqdm import tqdm

sys.path.append('power-system-split/power_system_split')
import utils

script_path = os.path.dirname(os.path.realpath('__file__'))

path_to_pypsa_network = script_path
path_to_cascade_results = script_path + '/Results'

save_path = script_path + '/Results'

year = 2015
number_of_nodes = 306
month = 1

network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network + '/' +
                           str(year) +
                           '/networks/elec_s_' +
                           str(number_of_nodes) +
                           '_ec_lv1.0_1H.nc')

network.determine_network_topology()

branches = network.branches()

branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom"]]
Graph = utils.build_networkx_graph(branches)

use_pnom = True
criterion = 'load'

cascade_res_path = "{0}/system_splits_{1:d}_{2:d}_{3:d}.pickle".format(path_to_cascade_results,
                                                                       year, month, number_of_nodes)
splitting_cascades = pickle.load(open(cascade_res_path, 'rb'))

solution_dict = {}

for key in tqdm(splitting_cascades.keys()):
    timestamp = utils.solution_key_to_pandas_timestamp(key)
    solution_dict[key] = []
    splits = splitting_cascades[key]

    for i, split in enumerate(splits):
        results = utils.evaluate_split_observables(split,
                                                   network,
                                                   timestamp,
                                                   use_pnom=use_pnom,
                                                   criterion='load')
        if results['inertia_proxy']:
            solution_dict[key].append(i)
            solution_dict[key].append(results)

pickle.dump(solution_dict,
            open(save_path + '/Germany_' +
                 str(number_of_nodes) +
                 '_split_evaluation_' +
                 str(year) +
                 '.pickle', 'wb'))
