#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains an example on how to visualise system splits

it is for now tailored specifically for analyses of German networks that
were obtained in 10.1109/EEM49802.2020.9221886 """

import pypsa
import networkx as nx
import numpy as np
import pickle

import sys
sys.path.append('power-system-split/power_system_split')
import utils,visualisation

path_to_pypsa_network   = ''
path_to_rocof_solutions = ''

year            = 2016
number_of_nodes = 306

network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+\
                           str(year)+\
                           '/networks/elec_s_'+\
                           str(number_of_nodes)+\
                           '_ec_lv1.0_1H.nc')

network.determine_network_topology()

branches = network.branches()

branches = branches[["bus0","bus1","x_pu_eff","s_nom"]]
nx_graph = utils.build_networkx_graph(branches)

solution_dict =  pickle.load(open(save_path+'Germany_'+\
                  str(number_of_nodes)+\
                  '_split_evaluation_'+\
                  str(year)+\
                  '.pickle','rb'))

adjacencies = visualisation.get_split_adjacencies_from_rocof_solutions(solution_dict,
                                                                       nx_graph)
