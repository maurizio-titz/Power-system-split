#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains an example on how to evaluate the load imbalance
and inertia for each subgraph and point in time for an existing cascade
result/system split simulation

it is for now tailored specifically for analyses of German networks that
were obtained in 10.1109/EEM49802.2020.9221886 """

import sys
import os
import pickle

from tqdm import tqdm
import pypsa


sys.path.append('power-system-split/power_system_split')
import utils

script_path = os.path.dirname(os.path.realpath('__file__'))

path_to_pypsa_network = script_path + '/data/Germany/'
path_to_cascade_results = script_path + '/data/Germany/'

save_path = script_path + '/data/Germany/'

year = 2015
number_of_nodes = 306
month = 1

network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network +
                           str(year) +
                           '/elec_s_' +
                           str(number_of_nodes) +
                           '_ec_lv1.0_1H.nc')

Graph = utils.build_networkx_graph(network)

use_pnom = True
criterion = 'load'

splitting_cascades = pickle.load(open(path_to_cascade_results+\
                                      str(year)+\
                                      '/system_splits_'+\
                                      str(year)+\
                                      '_'+\
                                      str(number_of_nodes)+\
                                      '_edge_based.pickle','rb'))

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
            open(save_path +\
                str(year) +\
                '/Germany_' +\
                str(number_of_nodes) +\
                '_split_evaluation_' +\
                str(year) +\
                '.pickle', 'wb'))
