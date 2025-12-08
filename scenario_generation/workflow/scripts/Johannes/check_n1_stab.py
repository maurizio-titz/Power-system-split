from datetime import datetime as dt

import networkx as nx
from tqdm import tqdm

from .utils import cascade_simulation, data_handling

# import .utils.cascade_simulation as cascade_simulation
# import .utils.data_handling as data_handling


def check_n1_stab(path_to_pypsa_network, use_sclopf, snet_index=0, stop_timestamp_str=None):
    """_summary_

    Args:
        network (_type_): _description_
        use_sclopf (bool): in the sclopf case we use different lookup tables for the line outages than for lopf
        snet_index (int, optional): The index of the synchronouse sub_network we want to use, e.g. we want to exclude great britain etc because we don't care about it. Should be 0. Defaults to 0.
        stop_timestamp_str (str, optional): test only snapshots before stop_timestamp_str. Defaults to None.
    """
    print("\n---------------------------------------- \nStarting contingency test\n---------------------------------------- \n")

    network = data_handling.load_pypsa_network(path_to_pypsa_network, use_sclopf)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    # Calculate possible N-1 and N-2 failures (using non-bridges)
    bridge_idxs = data_handling.nx_edges_to_matrix_indices(
        nx.bridges(nx_graph), nx_graph
    )
    n_1_failures = cascade_simulation.calc_possible_single_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )
    for snapshot in tqdm(network.snapshots):
        key_now = snapshot.strftime("%Y-%m-%d %H:00")

        P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

        for initial_failure in n_1_failures:
            failing_links, system_split = cascade_simulation.simulate_cascade(
                I_m,
                B_d,
                P_0,
                line_limits,
                num_parallels,
                initial_failure,
                max_cascade_length=1,
                use_sclopf=use_sclopf,
            )
            if len(failing_links) > 1:
                raise (
                    RuntimeError(
                        f"PyPSA networks are not N-1 stable! Failing lines: {failing_links}"
                    )
                )
        if (
            stop_timestamp_str is not None
            and dt.strptime(stop_timestamp_str, "%Y-%m-%d %H:00") <= snapshot
        ):
            break
        
    snapshot = network.snapshots[-1]
    key_now = snapshot.strftime("%Y-%m-%d %H:00")
    print(f"\n---------------------------------------- \nContingency test sucessful until {key_now}\n---------------------------------------- \n")

