import pickle
import sys

import pypsa
from tqdm import tqdm

sys.path.append('./power_system_split/')
import utils

# Setup paths to solved PyPSA networks and results own scripts
path_to_pypsa_network   = './data/European_networks/'
path_to_cascade_results   = './results/cascade_results/'
save_path =  './results/evaluation_results/'

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

# "Node" criterion means that only split components 
# with at least 10 nodes are evaluated
use_pnom  = True
criterion = 'nodes'

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

