import pickle
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np


sys.path.append('./')
from utils import data_handling, subgraph_evaluation

# Setup paths to solved PyPSA networks and results own scripts
path_to_pypsa_network = './data/European_networks_sclopf/'
path_to_cascade_results  = './results/sclopf/cascade_results/'
save_path =  './results/sclopf/evaluation_results/'

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid. 
snet_index = 0

# Load arguments
co2l = float(sys.argv[1]) 
n_nodes = int(sys.argv[2])

# Load PyPSA network, the graph of the subnetwork and its matrices
network = data_handling.load_pypsa_network(co2l, n_nodes, path_to_pypsa_network)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)

# Load cascade results
splitting_cascades = pickle.load(open(path_to_cascade_results+
                                      f'system_splits_Co2L{co2l}_n{n_nodes}.pickle' ,'rb'))

# Initialize results
component_props = pd.DataFrame(columns=['time_stamp', 'init_failure_0','init_failure_1',
                                        'split_number','rot_energy', 'power_imbalance',
                                        'load', 'rocof', 'load_share'],
                               index=[], dtype=float)
indicator_vectors = np.empty((0, nx_graph.number_of_nodes()), int)

for timestamp, splits in tqdm(splitting_cascades.items()):
    for i,(init_failure, cascade) in enumerate(splits.items()):

        subgraphs = data_handling.get_subgraphs_from_edges(cascade,nx_graph)
        
        observables = subgraph_evaluation.evaluate_observables_for_subgraphs(subgraphs, network,
                                                                             timestamp)
        observables['init_failure_0'] = init_failure[0]
        observables['init_failure_1'] = init_failure[1]
        observables['split_number'] = i
        component_props = component_props.append(observables, ignore_index=True)
        
        # Append properties and vectors such that component_props.iloc[i] refers to 
        # indicator_vectors[i]
        indicator_vec = subgraph_evaluation.get_indicator_vectors_of_subgraphs(subgraphs, nx_graph)
        indicator_vectors = np.append(indicator_vectors, indicator_vec, axis=0)

    
    np.save(save_path + f'indicator_vectors_Co2L{co2l}_n{n_nodes}.npy', indicator_vectors)
    component_props.to_hdf(save_path + f'component_properties_Co2L{co2l}_n{n_nodes}.h5',
                           key='df', mode= 'w')

