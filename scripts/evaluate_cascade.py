#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the results of the cascade experiments to determine
the properties of the system splits."""

import gzip
import os
import pickle
import re
import sys
from datetime import datetime as dt
import networkx

import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append("./")
# Post messages to mattermost
from utils.data_handling import get_co2_levels
from utils.cascade_simulation import solve_lpf
import utils.config as cfg
from utils import data_handling, send_mattermost_messages, subgraph_evaluation
from utils.alternative_split_indicator_vectors import (  # run_all_co2_lvl_node_based,
    extract_nodal_rocof_and_load_share_in_split_from_old_results,
    find_failed_edge_indicator_vector_for_cascade_results,
    run_all_co2_lvl_edge_based,
)
from utils.config import (
    path_to_cascade_results_lopf,
    path_to_cascade_results_sclopf,
    path_to_evaluation_results_lopf,
    path_to_evaluation_results_sclopf,
    path_to_pypsa_network_lopf,
    path_to_pypsa_network_sclopf,
)

save_path_sclopf = path_to_evaluation_results_sclopf
save_path_lopf = path_to_evaluation_results_lopf

# Dummy decorator to not get stuck on @profile
if "profile" not in globals():

    def profile(func):
        return func


@profile
def evaluate_cascade(
    co2l: float,
    n_nodes: int,
    snet_index: int = 0,
    start_time_str=None,
    end_time_str=None,
    eval_indicator_vectors: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    use_sclopf=True,
    overrwrite: bool = False,
    # calc_split_indicator_vectors:bool=False
):
    """Find the properties of the splits (i.e., RoCoF or lost load) and
    indicator vectors describing the network.

    Args:
        co2l (float): Target CO2 level in PyPSA Optimization.
        n_nodes (int): Number of nodes in PyPSA network.
        snet_index (int, optional): Select a particular subnetwork for calculations (if the pypsa network has different ones).
                    For our data set, "0" indicates the Continental European AC grid. Defaults to 0.
        start_time_str, end_time_str (str Format "YYYY-mm-dd HH:MM"): If either or both are not None, the evaluation will only be
                    done between start_time and end_time.
    """

    calc_split_indicator_vectors = False  # does not work currently

    if start_time_str is not None or end_time_str is not None:
        raise NotImplementedError(
            "Not implemented anymore. comp_props is referenced before assignement..."
        )

    def check_format_time_str(time_str):

        correct_len_str = len("xxxx-xx-xx xx:xx")
        has_correct_len = len(time_str) == correct_len_str

        rr = re.compile(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}")
        does_match = rr.match(time_str) is not None

        return has_correct_len, does_match

    # Check if correct format of start or end time has been given
    if start_time_str is not None:
        check_format_r = check_format_time_str(start_time_str)
        assert check_format_r, "Provided start time does not have correct format. "
        start_time_dt = dt.strptime(start_time_str, "%Y-%m-%d %H:%M")

    if end_time_str is not None:
        check_format_r = check_format_time_str(end_time_str)
        assert check_format_r, "Provided end time does not have correct format."
        end_time_dt = dt.strptime(end_time_str, "%Y-%m-%d %H:%M")

    # Load PyPSA network, the graph of the subnetwork and its matrices
    if verbose:
        print("Loading PyPSA Network and converting it to NetworkX Graph.\n")

    if use_sclopf:
        full_path_to_cascades = (
            path_to_cascade_results_sclopf + f"system_splits_Co2L{co2l}_n{n_nodes}.pklz"
        )
        save_path = save_path_sclopf
    else:
        full_path_to_cascades = (
            path_to_cascade_results_lopf
            + f"system_splits_singlelinefailures_Co2L{co2l}_n{n_nodes}_lopf.pklz"
        )
        save_path = save_path_lopf
    print(f"saving results to {save_path}")

    save_df_path = save_path + f"component_properties_Co2L{co2l}_n{n_nodes}"
    if not overrwrite and os.path.exists(save_df_path + ".h5"):
        raise FileExistsError(
            f"Results already exist at {save_df_path}.h5, skipping evaluation."
        )

    os.makedirs(save_path, exist_ok=True)

    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)

    data_handling.check_if_edges_sorted(nx_graph)

    # Load cascade results
    with gzip.open(full_path_to_cascades, "rb") as fh_casc:
        splitting_cascades = pickle.load(fh_casc)

    # Initialize results
    if use_sclopf:
        comp_cols = [
            "time_stamp",
            "init_failure_0",
            "init_failure_1",
            "split_number",
            "rot_energy",
            "power_imbalance",
            "load",
            "rocof",
            "load_share",
            "shedding_load_loss_share",
            "blackout_load_loss_share",
            "total_load_loss_share",
            # "line_momentum",
        ]
    else:
        comp_cols = [
            "time_stamp",
            "init_failure",
            "split_number",
            "rot_energy",
            "power_imbalance",
            "load",
            "rocof",
            "load_share",
            "shedding_load_loss_share",
            "blackout_load_loss_share",
            "total_load_loss_share",
            # "line_momentum",
        ]

    splitting_cascades_dtkeys = {
        dt.strptime(key, "%Y-%m-%d %H:%M"): value
        for key, value in splitting_cascades.items()
    }

    if verbose:
        print("Starting evaluation of System Splits.\n")

    if eval_indicator_vectors:
        component_indicator_vectors_ls = list()

        ### this is not used in the current implementation, since the alternative split indicator functions work just fine
        if calc_split_indicator_vectors:
            rocof_indicator_vectors_ls = list()
            lshare_indicator_vectors_ls = list()

            total_nr_splits = sum(len(vv) for vv in splitting_cascades.values())

            indicator_vector_rocof = np.full(
                (total_nr_splits, nx_graph.number_of_nodes), np.nan, dtype=float
            )

    component_props_dict = dict()
    out_dict_key = 0
    # idx_indi_vec = 0

    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    data_handling.check_if_edges_sorted(nx_graph)
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    split_number_total = 0
    split_numbers = []

    for timestamp, splits in tqdm(
        splitting_cascades_dtkeys.items(), disable=not show_progress
    ):

        # the following lines are were added when implementing line momentum, which is not used yet
        # P_0 = data_handling.get_effective_injections(network, timestamp, nx_graph)
        # lines_in_subnet0 = network.lines[(network.buses.sub_network.loc[network.lines.bus0].astype(int)==0).values & (network.buses.sub_network.loc[network.lines.bus1].astype(int)==0).values]
        # list(zip(lines_in_subnet  0.bus0, lines_in_subnet0.bus1)) == networkx.get_edge_attributes(network, "orientation").values()
        # flows = solve_lpf(P_0, B_d, I_m)
        # flows_dict = dict(zip(networkx.get_edge_attributes(nx_graph, "orientation").values(), flows))

        for split_number_snapshot, (init_failure, cascade) in enumerate(
            # tqdm(splits.items(), leave=False, disable=not show_progress)
            splits.items()
        ):
            split_number_total += 1

            subgraphs = data_handling.get_subgraphs_from_edges(cascade, nx_graph)

            observables_split_components = (
                subgraph_evaluation.evaluate_observables_for_subgraphs(
                    subgraphs=subgraphs,
                    network=network,
                    timestamp=timestamp,  # flows=flows_dict
                )
            )
            # each entry holds: [rot_energy, power_imbalance, load, rocof, load_share]

            for observables_single_component in observables_split_components:
                load_shedded = (
                    abs(min(0, observables_single_component[1]))
                    / observables_single_component[2]
                    if observables_single_component[2] != 0
                    else 0
                )
                blackout_load_loss = (
                    int(observables_single_component[3] < -1)
                    * observables_single_component[4]
                )
                total_load_loss_share = max(load_shedded, blackout_load_loss)
                if use_sclopf:
                    dict_out_ele = [
                        timestamp,
                        init_failure[0],
                        init_failure[1],
                        split_number_snapshot,
                        *observables_single_component,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]
                else:
                    dict_out_ele = [
                        timestamp,
                        init_failure,
                        split_number_snapshot,
                        *observables_single_component,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]
                component_props_dict[out_dict_key] = dict_out_ele
                out_dict_key += 1

                split_numbers.append(split_number_total)

            # Component indicator vectors:
            # Append properties and vectors such that component_props.iloc[ii] refers to
            # indicator_vectors[ii]
            if eval_indicator_vectors:
                indi_vec_r = subgraph_evaluation.get_indicator_vectors_of_subgraphs(
                    subgraphs, nx_graph
                )

                component_indicator_vectors_ls.extend(indi_vec_r)

            # Split-wise indicator vectors
            if calc_split_indicator_vectors:
                rocof_vector = np.empty((n_nodes, 1), dtype=float)
                for split_number_snapshot in range(len(subgraphs)):
                    rocof_vector[indi_vec_r[split_number_snapshot]] = (
                        observables_split_components[split_number_snapshot][3]
                    )
                rocof_indicator_vectors_ls.append(rocof_vector)

                lshare_vector = np.empty((n_nodes, 1), dtype=float)
                for split_number_snapshot in range(len(subgraphs)):
                    lshare_vector[indi_vec_r[split_number_snapshot]] = (
                        observables_split_components[split_number_snapshot][4]
                    )
                lshare_indicator_vectors_ls.append(lshare_vector)

            # only one timestamp for testing
            # break
        # break

    # Convert dictionary to component props pdDataFrame and save
    if eval_indicator_vectors:
        indi_vec_save_path = (
            save_path + f"component_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz"
        )
        with gzip.open(indi_vec_save_path, "wb") as fh_vec_out:
            pickle.dump(np.array(component_indicator_vectors_ls, dtype=int), fh_vec_out)

        if calc_split_indicator_vectors:

            rocof_indi_vec_save_path = (
                save_path + f"rocof_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz"
            )
            with gzip.open(rocof_indi_vec_save_path, "wb") as fh_vec_out:
                pickle.dump(
                    np.array(rocof_indicator_vectors_ls, dtype=float).squeeze(-1),
                    fh_vec_out,
                )

            lshare_indi_vec_save_path = (
                save_path + f"load_share_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz"
            )
            with gzip.open(lshare_indi_vec_save_path, "wb") as fh_vec_out:
                pickle.dump(
                    np.array(lshare_indicator_vectors_ls, dtype=float), fh_vec_out
                )

    component_props = pd.DataFrame.from_dict(
        component_props_dict, orient="index", columns=comp_cols
    )
    component_props["snapshot_weighting"] = network.snapshot_weightings.generators.loc[
        component_props.time_stamp
    ].values
    component_props["split_number"] = split_numbers

    save_df_path = save_path + f"component_properties_Co2L{co2l}_n{n_nodes}"

    component_props.to_hdf(save_df_path + ".h5", key="df", mode="w")

    if cfg.mattermost_url is not None:
        message_text = (
            f"Evaluation of N={n_nodes}, C02_lvl={co2l} finished and results saved"
        )
        send_mattermost_messages.post_message(message_text, cfg.mattermost_url)

    return component_props


if __name__ == "__main__":
    from utils.config import path_to_sclopf_data
    import os

    n_nodes_in = 600
    # co2l_in = get_co2_levels(n_nodes_in)
    co2l_in = [0.05]

    if isinstance(co2l_in, float):
        co2l_in = [co2l_in]
    for co2l_in in co2l_in:
        print(f"###############################################################")
        print(f"Evaluating cascade for CO2 level {co2l_in} and n_nodes {n_nodes_in}")
        print(f"###############################################################")
        try:
            evaluate_cascade(
                co2l_in,
                n_nodes_in,
                use_sclopf=True,
                eval_indicator_vectors=True,
                overrwrite=True,
                # end_time_str="2013-01-02 00:00",
            )
        except FileExistsError as e:
            print(
                f"Skipping evaluation for {co2l_in} and {n_nodes_in} due to existing results."
            )

        print(f"###############################################################")
        print(
            f"Getting edge indicator vecs for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
        )
        print(f"###############################################################")
        find_failed_edge_indicator_vector_for_cascade_results(
            co2l_in,
            n_nodes_in,
            save_res=True,
            verbose=True,
            overwrite=False,
            use_sclopf=True,
        )
        print(f"###############################################################")
        print(
            f"Extracting nodal RoCoF and load share for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
        )
        print(f"###############################################################")
        extract_nodal_rocof_and_load_share_in_split_from_old_results(
            co2l_in,
            n_nodes_in,
            save_res=True,
            verbose=True,
            overwrite=False,
            use_sclopf=True,
        )
        print(f"###############################################################")
        print(
            f"Running all CO2 level edge based for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
        )
        print(f"###############################################################")
        find_failed_edge_indicator_vector_for_cascade_results(
            co2l_in,
            n_nodes_in,
            save_res=True,
            verbose=True,
            overwrite=False,
            use_sclopf=True,
        )
