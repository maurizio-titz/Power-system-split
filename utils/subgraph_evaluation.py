#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation of observables in subgraphs of PyPSA network
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


import numpy as np
import pandas as pd

# Inertia constants based on
# https://eepublicdownloads.azureedge.net/clean-documents/SOC%20documents/Inertia%20and%20RoCoF_v17_clean.pdf
INERTIA_CONSTANTS = pd.Series(
    {
        "coal": 4.2,
        "oil": 4.3,
        "biomass": 3.3,
        "solar": 0,
        "CCGT": 4.2,
        "lignite": 3.8,
        "geothermal": 3.5,
        "OCGT": 4.2,
        "onwind": 0,
        "offwind-ac": 0,
        "offwind-dc": 0,
        "nuclear": 5.9,
        "hydro": 3.7,
        "PHS": 3.5,
        "ror": 2.7,
        "H2": 0,
        "battery": 0,
        "load": 0,
    }
)


def get_power_imbalance_subgraph(
    subgraph,
    generators,
    current_generation,
    storages,
    current_storage,
    loads,
    current_load,
    HVDC_transport,
):
    """Calculate power imbalance due to subgraph"""

    # Get PyPSA components that are in subgraph
    gen_mask = generators["bus"].isin(subgraph.nodes())
    load_mask = loads["bus"].isin(subgraph.nodes())
    store_mask = storages["bus"].isin(subgraph.nodes())
    hvdc_mask = HVDC_transport.index.isin(subgraph.nodes())

    # Calculate power imbalance
    power_imbalance = (
        current_generation[gen_mask].sum()
        + current_storage.loc[store_mask].sum()
        - current_load[load_mask].sum()
    )
    power_imbalance -= HVDC_transport[hvdc_mask].sum()

    return power_imbalance


def get_load_subgraph(subgraph, loads, current_load):
    """Calculate load in subgraph (ingnoring HVDC and storage consumption)"""

    load_mask = loads["bus"].isin(subgraph.nodes())
    # stores = storages[storages["bus"].isin(list(subgraph.nodes()))]
    # HVDC =...

    subgraph_load = current_load.loc[load_mask].sum()

    return subgraph_load


def get_inertia_gen_subgraph(
    subgraph,
    generators,
    current_generation,
    storages,
    current_storage,
    participation_threshold=0.05,
):
    """Calculate total rotational energy (in GWs) in the subgraph.

    Args:
        subgraph (networkx graph): Subgraph of power system.
        generators (pandas.DataFrame): PyPSA generators
        current_generation (pandas.Series): Entry of PyPSA generation time series
        storages (pandas.DataFrame): PyPSA storages
        current_storage (pandas.Series):  Entry of PyPSA storage time series
        participation_threshold (float, optional): Share of nominal power. Above the threshold, a
        generator is considered to be online. Defaults to 0.05.

    Returns:
        float: rotational energy
    """

    # Calc nominal power of generators
    gens = generators.loc[:, ["carrier", "p_nom", "p_max_pu"]]
    gens.loc[:, "nominal_power"] = gens.p_max_pu * gens.p_nom

    # Get generators that are in subgraph and online in current timestamp
    in_subgraph = generators["bus"].isin(subgraph.nodes())
    is_online = (
        current_generation[gens.index] > participation_threshold * gens.nominal_power
    )
    gens = gens[in_subgraph & is_online]

    # Calc nominal power of storage units
    stores = storages.loc[:, ["carrier", "p_nom", "p_max_pu"]]
    stores.loc[:, "nominal_power"] = stores.p_max_pu * stores.p_nom

    # Get storage units that are in subgraph and online in current timestamp
    in_subgraph = storages["bus"].isin(subgraph.nodes())
    is_online = (
        current_storage[stores.index] > participation_threshold * stores.nominal_power
    )
    stores = stores[in_subgraph & is_online]

    # Calc rotational energy
    total_gens = pd.concat([gens, stores])
    total_gens = total_gens.reset_index(names=["index"]).set_index(["index", "carrier"])
    rot_energy = total_gens.nominal_power.mul(INERTIA_CONSTANTS, level=1).sum()

    return rot_energy


def get_indicator_vectors_of_subgraphs(subgraphs, nx_graph):
    """
    Construct boolean indicator vector for a subgraph of the nx_graph
    """

    list_of_nodes = list(nx_graph)
    indicator_vec = np.empty((len(subgraphs), nx_graph.number_of_nodes()), int)

    for ii, subgraph in enumerate(subgraphs):

        indicator_vec[ii] = np.isin(list_of_nodes, list(subgraph))

    return indicator_vec


def evaluate_observables_for_subgraphs(
    subgraphs: list, network, timestamp: str, snet=0, flows=None
) -> np.array:
    """Evaluate power imbalance, rotational energy and load of subgraphs for a time stamp.

    Args:
        subgraphs (list): List of networkx graphs
        network (pypsa.network): PyPSA network
        timestamp (string): Format '%Y-%m-%d %H:00'
        snet (int, optional): Index of subnetwork under investigation. Defaults to 0.

    Returns:
        np.array: array, first dimension iterates over subgraphs, second holds observables in the order [rot_energy, power_imbalance, load, rocof, load_share, line_momentum]
    """

    if flows is None:
        warnings.warn(
            "Flows not provided",
            UserWarning,
        )

    # Extract current power injections
    current_generation = (
        network.generators_t.p.loc[timestamp].mul(network.generators.sign).copy()
    )
    current_storage = network.storage_units_t.p.loc[timestamp]
    current_load = network.loads_t.p.loc[timestamp]
    HVDC_transport = (
        network.links_t.p0.loc[timestamp].groupby(network.links["bus0"]).sum()
    )
    HVDC_transport = HVDC_transport.add(
        network.links_t.p1.loc[timestamp].groupby(network.links["bus1"]).sum(),
        fill_value=0,
    )
    total_current_load_subgraph = current_load[
        network.buses.sub_network.astype(int) == snet
    ].sum()

    # Check if network is balanced
    if (
        np.abs(current_generation.sum() + current_storage.sum() - current_load.sum())
        > 1e-1
    ):
        print("Warning: Power imbalance for", timestamp, "is non-zero:")
        print(
            np.abs(
                current_generation.sum() + current_storage.sum() - current_load.sum()
            )
        )

    if np.abs(HVDC_transport.sum()) > 1e-1:
        print("Warning: HVDC transport for", timestamp, "does not sum to zero:")
        print(np.abs(HVDC_transport.sum()))

    # results = pd.DataFrame(columns=['time_stamp', 'rot_energy', 'power_imbalance',
    #                                'load', 'rocof', 'load_share'],
    #                       index=range(len(subgraphs)), dtype=float)

    # results array with columns
    # [rot_energy', 'power_imbalance', 'load', 'rocof', 'load_share']
    results_arr = np.empty((len(subgraphs), 6), dtype=float)

    for ii, subgraph in enumerate(subgraphs):

        results_arr[ii, 0] = get_inertia_gen_subgraph(
            subgraph,
            network.generators,
            current_generation,
            network.storage_units,
            current_storage,
        )
        results_arr[ii, 1] = get_power_imbalance_subgraph(
            subgraph,
            network.generators,
            current_generation,
            network.storage_units,
            current_storage,
            network.loads,
            current_load,
            HVDC_transport,
        )
        results_arr[ii, 2] = get_load_subgraph(subgraph, network.loads, current_load)

        results_arr[ii, 3] = 50 * results_arr[ii, 1] / ((results_arr[ii, 0] + 1e-8) * 2)

        results_arr[ii, 4] = results_arr[ii, 2] / total_current_load_subgraph

        results_arr[ii, 5] = get_inertia_flow_subgraph(
            subgraph=subgraph, lines=network.lines, flows=flows
        )

    return results_arr


import networkx as nx

from utils import data_handling
from utils.cascade_simulation import solve_lpf


def get_flows(
    network,
    snapshot,
    snet_index: int = 0,
):
    """Run the cascade simulation for single line failures.

    Args:
        co2l (float): CO2 level of the previously simulated PyPSA networks.
        n_nodes (int): Number of nodes fo the PyPSA networks.
        epsilon (float): Limit above smax needed for a line to fail. Default 1e-4.
        save_all_cascades (bool, optional): If ''. Defaults to False.
        snet_index (int, optional): _description_. Defaults to 0.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network and load this data set. Defaults to False.
        initial_remove_all (bool): If 'True' remove all initial circuits and not use look_up_table.
    """

    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    P_0 = data_handling.get_effective_injections(network, snapshot, nx_graph)

    flows = solve_lpf(P_0, B_d, I_m)

    return flows


def get_inertia_flow_subgraph(
    subgraph,
    lines,
    flows,
):
    """Calculate total rotational energy (in GWs) in the subgraph.

    Args:
        subgraph (networkx graph): Subgraph of power system.
        generators (pandas.DataFrame): PyPSA generators
        current_generation (pandas.Series): Entry of PyPSA generation time series
        storages (pandas.DataFrame): PyPSA storages
        current_storage (pandas.Series):  Entry of PyPSA storage time series
        participation_threshold (float, optional): Share of nominal power. Above the threshold, a
        generator is considered to be online. Defaults to 0.05.

    Returns:
        float: rotational energy
    """
    raise NotImplementedError(
        "electro-magnetic inertia not implemented yet")
    
    if subgraph.number_of_edges() == 0:
        return 0

    # Get lines that are in subgraph and online in current timestamp
    data_handling.check_if_edges_sorted(subgraph)
    
    lines["from_to"] = list(zip(lines["bus0"], lines["bus1"]))
    lines_snet = lines[lines.sub_network.astype(int) == 0]
    edge_orientations = nx.get_edge_attributes(subgraph, "orientation").values()
    lines_subgraph = lines_snet[lines_snet["from_to"].isin(edge_orientations)]
    # lines_in_subnet0 = network.lines[(network.buses.sub_network.loc[network.lines.bus0].astype(int)==0).values & (network.buses.sub_network.loc[network.lines.bus1].astype(int)==0).values]

    line_lengths = lines_subgraph.length
    from operator import itemgetter
    line_momentum = np.sum(abs(np.array(itemgetter(*lines_subgraph.from_to.values)(flows)) * line_lengths))
    
    # assert subgraph.number_of_edges() == len(line_momentum), (
    #     "Number of edges in subgraph does not match number of caculated line_momentums subgraph"
    # )

    return line_momentum
