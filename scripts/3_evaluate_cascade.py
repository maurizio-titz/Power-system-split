import pickle
import sys

import pypsa
import pandas as pd
from tqdm import tqdm

sys.path.append('./')
from power_system_split import utils

# Setup paths to solved PyPSA networks and results own scripts
path_to_pypsa_network   = './data/European_networks_lopf/'
path_to_cascade_results  = './results/lopf/cascade_results/'
save_path =  './results/sclopf/evaluation_results/'

# Load co2 level
co2l = 0.9# float(sys.argv[1])

# Load PyPSA network
network = pypsa.Network()
network.import_from_netcdf(path_to_pypsa_network+f'elec_s_800_ec_lv1.0_Co2L{co2l}-3H.nc')
#network.import_from_netcdf(path_to_pypsa_network+f'sclopf-elec_s_800_ec_lv1.0_Co2L{co2l}-3H.nc')

# The following line is needed to remove the outage lines used for SCLOPF. When not removed, multiple edges 
# are added to the network graph
outage_lines = network.lines[network.lines.index.str[-6:]=='outage'].index
network.mremove('Line', outage_lines)
network.determine_network_topology()

######
#TODO: add function for this. Adding time stamps is necessary because they are missing in the pypsa networks
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

# "Node" criterion means that only split components 
# with at least 10 nodes are evaluated
use_pnom  = True
criterion = 'all'

# Load cascade results
splitting_cascades = pickle.load(open(path_to_cascade_results+
                                      f'system_splits_Co2L{co2l}.pickle' ,'rb'))

solution_dict = {}

# Iterate over time stamps
for key in tqdm(splitting_cascades.keys()):
    timestamp = utils.solution_key_to_pandas_timestamp(key)
    solution_dict[key] = []
    splits = splitting_cascades[key]
    
    # Iterate over splits occuring for that time stamp
    for i,split in enumerate(splits):
        results = utils.evaluate_split_observables(split,
                                                   network,
                                                   timestamp,
                                                   use_pnom = use_pnom,
                                                   criterion = criterion,
                                                   snet_index= snet_index)
        if results['inertia_proxy']:
            # save the index of the split that was evaluated (i)
            # and the results dict that contains the inertia proxy
            # and the load imbalance for each split component that fulfills the criterion
            # NOTE: to get the Rocof from load imbalance and the inertia proxy
            # you need to multiply by a factor of 50Hz/(2*inertia_constant), where
            # we typically set inertia_constant = 6s^{-1}
            solution_dict[key].append(i)
            solution_dict[key].append(results)

with open(save_path + f'Europe_Co2L{co2l}_split_evaluation_' + criterion +f'_based_snet_{snet_index}_w_hvdc.pickle' ,
          'wb') as handle:
    pickle.dump(solution_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

