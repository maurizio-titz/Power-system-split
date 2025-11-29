import pickle
import sys
import os

import numpy as np
import gzip
import networkx as nx

sys.path.append("./")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from utils import cascade_simulation, data_handling
from utils.config import use_extensions
from utils.data_handling import get_co2_levels

n_nodes = 600

overwrite = True
if overwrite:
    print("Overwrite is enabled: Existing files will be replaced.")

co2ls = get_co2_levels(n_nodes)
snet_index = 0

for co2l in co2ls:
    if use_extensions:
        # if extensions are used, we need to build the graph from the LOPF network
        # because it contains the information about the original and updated s_nom values
        # we need the original s_nom values to update the other line limits
        network_LOPF = data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=False, lopt=use_extensions
        )
        nx_graph = data_handling.build_networkx_graph(
            network_LOPF, snet_index=snet_index, update_lines=use_extensions
        )
        network = data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
        )
    else:
        network = data_handling.load_pypsa_network(
            n_nodes=600, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
        )
        nx_graph = data_handling.build_networkx_graph(
            network, snet_index=snet_index, update_lines=False
        )
    data_handling.save_networkx_graph(
        nx_graph, snet_index=snet_index, co2lvl=co2l, overwrite=overwrite
    )
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    bridge_idxs = data_handling.nx_edges_to_matrix_indices(
        nx.bridges(nx_graph), nx_graph
    )
    n_2_failures = cascade_simulation.calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )
    file_path = data_handling.path_to_grid_data + f"n_2_failures_co2lvl{co2l}.pklz"
    with gzip.open(file_path, "wb") as fh:
        pickle.dump(n_2_failures, fh, protocol=pickle.HIGHEST_PROTOCOL)

    data_handling.save_grid_matrices(
        snet_index=snet_index,
        co2lvl=co2l,
        I_m=I_m,
        B_d=B_d,
        num_parallels=num_parallels,
        line_limits=line_limits,
        overwrite=overwrite,
    )

    effective_injections = {
        snapshot: data_handling.get_effective_injections(network, snapshot, nx_graph)
        for snapshot in network.snapshots
    }
    data_handling.save_effective_injections(
        co2lvl=co2l, effective_injections=effective_injections, overwrite=overwrite
    )
    snapshots = np.array(list(network.snapshots))
    data_handling.save_snapshot_list(snapshots, co2lvl=co2l, overwrite=overwrite)
