import sys
import os

sys.path.append("./")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from utils import data_handling
from utils.config import use_extensions
from utils.data_handling import get_co2_levels

n_nodes = 600


co2ls = get_co2_levels(n_nodes)
snet_index = 0

for co2l in co2ls:
    network = data_handling.load_pypsa_network(
        n_nodes=600, co2lvl=co2l, use_sclopf=False, lopt=use_extensions
    )
    nx_graph = data_handling.build_networkx_graph(
        network, snet_index=snet_index, update_lines=use_extensions
    )
    data_handling.save_networkx_graph(nx_graph, snet_index=snet_index, co2lvl=co2l)
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    data_handling.save_grid_matrices(
        snet_index=snet_index,
        co2lvl=co2l,
        I_m=I_m,
        B_d=B_d,
        num_parallels=num_parallels,
        line_limits=line_limits,
    )

    effective_injections = {
        snapshot: data_handling.get_effective_injections(network, snapshot, nx_graph)
        for snapshot in network.snapshots
    }
    data_handling.save_effective_injections(
        co2lvl=co2l, effective_injections=effective_injections
    )
    snapshots = list(network.snapshots)
    data_handling.save_snapshot_list(snapshots, co2lvl=co2l)
