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
results_path = path + 'Data/system_split/European_Networks/New_networks_CO2_levels/Results/'

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

use_pnom  = True
criterion = 'nodes'

splitting_cascades = pickle.load(open(results_path+
                                      'system_splits_%i_Co2L%.1f.pickle' % (current_month, co2l),'rb'))

solution_dict = {}

for key in splitting_cascades.keys():
    timestamp = utils.solution_key_to_pandas_timestamp(key)
    solution_dict[key] = []
    splits = splitting_cascades[key]
    for i,split in enumerate(splits):
        results = utils.evaluate_split_observables(split,
                                                   network,
                                                   timestamp,
                                                   use_pnom = use_pnom,
                                                   criterion = criterion)
        if results['inertia_proxy']:
            solution_dict[key].append(i)
            solution_dict[key].append(results)

with open(results_path + 'Europe_%i_Co2L%.1f_split_evaluation_' % (current_month, co2l)+ criterion +'_based.pickle' , 'wb') as handle:
    pickle.dump(solution_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

