#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains an example on how to visualise system splits

it is for now tailored specifically for analyses of German networks that
were obtained in 10.1109/EEM49802.2020.9221886 """

import os
import sys
import pickle

import pypsa


sys.path.append('power-system-split/power_system_split')
import utils, visualisation

script_path = os.path.dirname(os.path.realpath('__file__'))
path_to_pypsa_network = script_path + '/data/Germany/'
path_to_cascade_results = script_path + '/data/Germany/'

year = 2015
number_of_nodes = 306
month = 1

network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network + 
                           str(year) +
                           '/elec_s_' +
                           str(number_of_nodes) +
                           '_ec_lv1.0_1H.nc')

network.determine_network_topology()

branches = network.branches()

branches = branches[["bus0", "bus1", "x_pu_eff", "s_nom"]]
nx_graph = utils.build_networkx_graph(branches)

solution_dict = pickle.load(open(path_to_cascade_results +\
                                 str(year) +\
                                 '/Germany_' +\
                                 str(number_of_nodes) +\
                                 '_split_evaluation_' +\
                                 str(year) +\
                                 '.pickle', 'rb'))


number_of_snapshots = len(network.snapshots)

splitting_cascades = pickle.load(open(path_to_cascade_results+\
                                      str(year)+\
                                      '/system_splits_'+\
                                      str(year)+\
                                      '_'+\
                                      str(number_of_nodes)+\
                                      '_edge_based.pickle','rb'))


adjacencies = visualisation.get_split_adjacencies_from_rocof_solutions(solution_dict,
                                                                       splitting_cascades,
                                                                       nx_graph)
##############################
#### Do stuff with adjacencies
##############################


##############################
# primary secondary likelihood of failures


likelihood_primary, likelihood_secondary = visualisation.calc_likelihood_failure(nx_graph,
                                                                                 splitting_cascades,
                                                                                 number_of_snapshots,
                                                                                 solution_dict=solution_dict)
