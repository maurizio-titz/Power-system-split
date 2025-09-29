#!usr/bin/env python
# -*- coding: utf-8 -*-


import gzip
import pickle
import sys
import warnings


# from glob import glob

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("./")

from utils.data_handling import get_co2_levels
from utils import cascade_simulation, data_handling
from utils import visualization as vis
from utils.config import (
    path_to_cascade_results_lopf,
    path_to_cascade_results_sclopf,
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_vis_results_lopf,
    path_to_vis_results_sclopf,
)

use_sclopf = True
# Setup paths
if use_sclopf:
    save_path = path_to_vis_results_sclopf
    path_to_cascade_results = path_to_cascade_results_sclopf
else:
    save_path = path_to_vis_results_lopf
    path_to_cascade_results = path_to_cascade_results_lopf

os.makedirs(save_path, exist_ok=True)

# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid.
snet_index = 0

# Load arguments
n_nodes = 600

from utils.config import path_to_sclopf_data
import datetime

print("Available CO2 Levels:")
co2l_list = get_co2_levels(n_nodes)

print(sorted(co2l_list))
# Load PyPSA network and the graph of the subnetwork
network = data_handling.load_pypsa_network(
    co2lvl=0.0, n_nodes=n_nodes, use_sclopf=use_sclopf
)
nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
    nx_graph
)

#### Cluster split components ####
print("\nClustering split components...\n")

# Initialize results
component_props = pd.DataFrame()
# indicator_vectors = np.empty((0, nx_graph.number_of_nodes()), int)

# Append all indicator vectors and components props
print("Concatenate components...")
for co2l in tqdm(co2l_list):

    # indicator_vec_level = np.load(path_to_evaluation_results + f'indicator_vectors_Co2L{co2l}_n{n_nodes}.npy')
    if use_sclopf:
        component_props_level = pd.read_hdf(
            path_to_evaluation_results_sclopf
            + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
        )
    else:
        component_props_level = pd.read_hdf(
            path_to_evaluation_results_lopf
            + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
        )

    component_props_level.loc[:, "co2l"] = co2l
    component_props = pd.concat(
        [component_props, component_props_level], ignore_index=True
    )

#### Cluster splits ####
print("\n### Extracting splits ###\n")
print("Current time:", datetime.datetime.now())

split_groups = component_props.groupby(["co2l", "time_stamp", "split_number"])
split_props = pd.DataFrame(
    index=split_groups.groups.keys(),
    # columns=["n_components"],
)
split_props.index = split_props.index.rename(
    ["co2l", "time_stamp", "split_number_snapshot"]
)

if use_sclopf:
    triggers_0 = split_groups.init_failure_0.unique()
    triggers_0 = [ii[0] for ii in triggers_0.values]
    triggers_1 = split_groups.init_failure_1.unique()
    triggers_1 = [ii[0] for ii in triggers_1.values]

    split_props["init_failure_0"] = triggers_0
    split_props["init_failure_1"] = triggers_1
else:
    triggers = split_groups.init_failure.unique()
    triggers = [ii[0] for ii in triggers.values]
    split_props["init_failure"] = triggers

split_props["lost_load_share_shedding"] = (
    split_groups.shedding_load_loss_share.sum().astype(float)
)
split_props["lost_load_share_blackout"] = (
    split_groups.blackout_load_loss_share.sum().astype(float)
)
split_props["lost_load_share_total"] = split_groups.total_load_loss_share.sum().astype(
    float
)
split_props["n_components"] = split_groups.size().astype(int)
# split_props["load_share_split_off"] = 1 - split_groups.load_share.max()
split_props["largest_component_load_share"] = split_groups.load_share.max().astype(
    float
)
split_props["load"] = split_groups.load.sum().astype(float)
split_props["snapshot_weighting"] = network.snapshot_weightings.generators.loc[
    split_props.index.get_level_values("time_stamp")
].values.astype(int)
split_props["co2l"] = split_props.index.get_level_values("co2l")


component_props.to_hdf(
    save_path + f"component_properties_all_n{n_nodes}.h5", key="df", mode="w"
)
split_props.to_hdf(
    save_path + f"split_properties_all_n{n_nodes}.h5", key="df", mode="w"
)
for lvl in split_props["co2l"].unique():
    split_props_lvl = split_props[split_props["co2l"] == lvl]
    split_props_lvl.to_csv(save_path + f"split_props_Co2L{lvl}_n{n_nodes}.csv")


### Calculate likelihoods #####

print("\nCalculate likelihoods of primary/secondary failures...\n")
likelihoods_primary = dict(keys=co2l_list)
likelihoods_secondary = dict(keys=co2l_list)
likelihoods_total = dict(keys=co2l_list)

bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph), nx_graph)
if use_sclopf:
    n_2_failures = cascade_simulation.calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )
    # incorporating the weighting of the snapshots
    number_of_simulations = (
        len(n_2_failures) * network.snapshot_weightings.generators.sum()
    )
else:
    number_of_simulations = (
        nx_graph.number_of_edges() - len(bridge_idxs)
    ) * network.snapshot_weightings.generators.sum()
print(number_of_simulations)

for co2l in co2l_list[::-1]:

    print("Co2 level %.2f" % co2l)

    if use_sclopf:
        full_path_to_cascades = (
            path_to_cascade_results_sclopf + f"system_splits_Co2L{co2l}_n{n_nodes}.pklz"
        )
    else:
        full_path_to_cascades = (
            path_to_cascade_results_lopf
            + f"system_splits_singlelinefailures_Co2L{co2l}_n{n_nodes}_lopf.pklz"
        )
    # Load results
    with gzip.open(
        full_path_to_cascades,
        "rb",
    ) as infile:
        splitting_cascades = pickle.load(infile)

    # Calculate likelihood of edge to be primary or secondary failure
    l_primary, l_secondary, l_total = vis.calc_likelihood_failure(
        nx_graph,
        splitting_cascades,
        number_of_simulations,
        network.snapshot_weightings.generators,
    )

    likelihoods_primary[co2l] = l_primary
    likelihoods_secondary[co2l] = l_secondary
    likelihoods_total[co2l] = l_total


with open(
    save_path + f"edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle", "wb"
) as handle:
    pickle.dump(likelihoods_primary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(
    save_path + f"edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle", "wb"
) as handle:
    pickle.dump(likelihoods_secondary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(
    save_path + f"edge_likelihoods_total_all_co2ls_n{n_nodes}.pickle", "wb"
) as handle:
    pickle.dump(likelihoods_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
