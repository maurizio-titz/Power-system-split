#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the results of the cascade experiments to determine
the properties of the system splits. Without arguments, all CO2 levels are processed sequentially. This can take several days, so parallel execution is recommended.
"""

import gzip
import os
import pickle
import re
import sys
from datetime import datetime as dt
import networkx

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


sys.path.append("./")
# Post messages to mattermost
from utils.data_handling import get_co2_levels
from utils.cascade_simulation import solve_lpf
import utils.config as cfg
from utils import data_handling, send_mattermost_messages, subgraph_evaluation
from utils.alternative_split_indicator_vectors import (  # run_all_co2_lvl_node_based,
    get_nodal_rocof_vectors,
    find_failed_edge_indicator_vector_for_cascade_results,
)
from utils.config import (
    path_to_cascade_results_sclopf,
    path_to_evaluation_results_sclopf,
    path_to_pypsa_network_sclopf,
)

save_path_sclopf = path_to_evaluation_results_sclopf

# Dummy decorator to not get stuck on @profile
if "profile" not in globals():

    def profile(func):
        return func


load_inertia_constant = 0.4  # in seconds, for load inertia approximation


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
    overwrite: bool = False,
    calc_split_indicator_vectors: bool = True,
    load_inertia: bool = True,
    show_progress_2: bool = False,
    sort_cascades: bool = True,
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
        raise NotImplementedError("LOPF based not supported anymore.")
        full_path_to_cascades = (
            path_to_cascade_results_lopf
            + f"system_splits_singlelinefailures_Co2L{co2l}_n{n_nodes}_lopf.pklz"
        )
        save_path = save_path_lopf

    print(f"saving results to {save_path}")

    save_df_path = save_path + f"component_properties_Co2L{co2l}_n{n_nodes}"
    if os.path.exists(save_df_path + ".h5"):
        if not overwrite:
            raise FileExistsError(
                f"Results already exist at {save_df_path}.h5, skipping evaluation."
            )
        else:
            print(f"Overwriting existing results at {save_df_path}.h5 as requested.")

    os.makedirs(save_path, exist_ok=True)

    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf)
    nx_graph = data_handling.load_networkx_graph(snet_index=snet_index, co2lvl=co2l)

    data_handling.check_if_edges_sorted(nx_graph)

    # Load cascade results
    with gzip.open(full_path_to_cascades, "rb") as fh_casc:
        splitting_cascades = pickle.load(fh_casc)

    if sort_cascades:
        # check if splitting_cascades are sorted by datetime keys, if not, sort and save back
        keys = list(splitting_cascades.keys())
        if not keys == sorted(keys):
            print("Sorting splitting_cascades by datetime keys...")
            # sort splitting_cascades by datetime keys
            splitting_cascades = dict(
                sorted(
                    splitting_cascades.items(),
                    key=lambda x: dt.strptime(x[0], "%Y-%m-%d %H:%M"),
                )
            )
            # save sorted splitting_cascades back to file
            with gzip.open(full_path_to_cascades, "wb") as fh_casc:
                pickle.dump(splitting_cascades, fh_casc)

    # Initialize results
    if use_sclopf:
        if load_inertia:
            comp_cols = [
                "time_stamp",
                "trigger_weighting",
                "init_failure_0",
                "num_par_failure_0",
                "init_failure_1",
                "num_par_failure_1",
                "split_number",
                "rot_energy_gen",
                "power_imbalance",
                "load",
                "rocof_noLoadInertia",
                "load_share",
                "load_inertia",
                "rot_energy",  # this the total inertia, i.e. generation + load inertia
                "rocof",
                "shedding_load_loss_share",
                "blackout_load_loss_share",
                "total_load_loss_share",
            ]
        else:
            comp_cols = [
                "time_stamp",
                "trigger_weighting",
                "init_failure_0",
                "num_par_failure_0",
                "init_failure_1",
                "num_par_failure_1",
                "split_number",
                "rot_energy",
                "power_imbalance",
                "load",
                "rocof",
                "load_share",
                "shedding_load_loss_share",
                "blackout_load_loss_share",
                "total_load_loss_share",
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
            # lshare_indicator_vectors_ls = list()

            # total_nr_splits = sum(len(vv) for vv in splitting_cascades.values())

            # indicator_vector_rocof = np.full(
            #     (total_nr_splits, nx_graph.number_of_nodes()), np.nan, dtype=float
            # )

    component_props_dict = dict()
    out_dict_key = 0
    # idx_indi_vec = 0

    # nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
    # data_handling.check_if_edges_sorted(nx_graph)
    # I_m, B_d, num_parallels, line_limits = data_handling.load_grid_matrices(
    #     snet_index=snet_index, co2lvl=co2l
    # )
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

        for component_number, (init_failures, cascade_weight_tuple) in enumerate(
            tqdm(splits.items(), leave=False, disable=not show_progress_2)
        ):
            split_number_total += 1

            trigger0 = init_failures[0][0]
            num_par_failure0 = init_failures[0][1]
            trigger1 = init_failures[1][0]
            num_par_failure1 = init_failures[1][1]

            cascade = cascade_weight_tuple[0]
            weight = cascade_weight_tuple[1]

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
                rot_energy_gen = observables_single_component[0]
                power_imbalance = observables_single_component[1]
                load = observables_single_component[2]
                rocof = observables_single_component[3]
                load_share = observables_single_component[4]

                if (
                    load_inertia
                ):  # compute load inertia contribution and update rocof accordingly

                    load_inertia = load * load_inertia_constant
                    rot_energy_total = rot_energy_gen + load_inertia
                    rocof_updated = (
                        50 * power_imbalance / ((rot_energy_total + 1e-8) * 2)
                    )
                    observables_single_component[3] = (
                        rocof_updated  # update rocof value
                    )
                    rocof = rocof_updated

                # the shedded and blackout load losses are given in share of total system load!
                if load != 0:
                    load_shedded = abs(min(0, power_imbalance)) / load * load_share
                else:
                    load_shedded = 0
                blackout_load_loss = (
                    int(abs(rocof) > 1)  # if |RoCoF| > 1 Hz/s, consider it a blackout
                    * load_share
                )
                total_load_loss_share = max(load_shedded, blackout_load_loss)
                if use_sclopf:
                    if load_inertia:
                        dict_out_ele = [
                            timestamp,
                            weight,
                            trigger0,
                            num_par_failure0,
                            trigger1,
                            num_par_failure1,
                            component_number,
                            *observables_single_component,
                            load_inertia,
                            rot_energy_total,
                            rocof,
                            load_shedded,
                            blackout_load_loss,
                            total_load_loss_share,
                        ]
                    else:
                        dict_out_ele = [
                            timestamp,
                            weight,
                            trigger0,
                            num_par_failure0,
                            trigger1,
                            num_par_failure1,
                            component_number,
                            *observables_single_component,
                            load_shedded,
                            blackout_load_loss,
                            total_load_loss_share,
                        ]
                else:
                    dict_out_ele = [
                        timestamp,
                        init_failures,
                        component_number,
                        *observables_single_component,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]
                component_props_dict[out_dict_key] = dict_out_ele
                if np.isnan(np.array(dict_out_ele[1:])).any():
                    print(
                        f"Warning: NaN values found in component properties for CO2 level {co2l}, timestamp {timestamp}, component number {component_number}."
                    )
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
                rocof_vector = np.empty(nx_graph.number_of_nodes(), dtype=float)
                for component_number, component_indicator_vec in enumerate(indi_vec_r):
                    rocof_vector[component_indicator_vec.astype(bool)] = (
                        observables_split_components[component_number][3]
                    )
                rocof_indicator_vectors_ls.append(rocof_vector)

                # lshare_vector = np.empty((n_nodes, 1), dtype=float)
                # for split_number_snapshot in range(len(subgraphs)):
                #     lshare_vector[indi_vec_r[split_number_snapshot]] = (
                #         observables_split_components[split_number_snapshot][4]
                #     )
                # lshare_indicator_vectors_ls.append(lshare_vector)

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
                    np.array(rocof_indicator_vectors_ls, dtype=float),
                    fh_vec_out,
                )
            print(f"Saved RoCoF indicator vectors to {rocof_indi_vec_save_path}")

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


def process_timestamp_chunk(
    timestamp_splits_chunk,
    co2l,
    n_nodes,
    snet_index,
    comp_cols,
    load_inertia,
    load_inertia_constant,
    eval_indicator_vectors,
    calc_split_indicator_vectors,
    use_sclopf,
):
    """Process a chunk of timestamps. Module-level function for pickling.

    Loads network and nx_graph once per chunk to avoid redundant I/O.
    """
    # Load objects once per worker/chunk
    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf)
    nx_graph = data_handling.load_networkx_graph(snet_index=snet_index, co2lvl=co2l)

    all_component_props = []
    all_component_indicator_vectors = []
    all_rocof_indicator_vectors = []
    all_split_numbers = []

    for timestamp, splits in tqdm(
        timestamp_splits_chunk, desc="Timestamps in chunk", leave=False
    ):
        component_props_list = []
        component_indicator_vectors_list = []
        rocof_indicator_vectors_list = []
        split_numbers_local = []
        split_number_local = 0

        for component_number, (init_failures, cascade_weight_tuple) in enumerate(
            splits.items()
        ):
            split_number_local += 1

            trigger0 = init_failures[0][0]
            num_par_failure0 = init_failures[0][1]
            trigger1 = init_failures[1][0]
            num_par_failure1 = init_failures[1][1]

            cascade = cascade_weight_tuple[0]
            weight = cascade_weight_tuple[1]

            subgraphs = data_handling.get_subgraphs_from_edges(cascade, nx_graph)

            observables_split_components = (
                subgraph_evaluation.evaluate_observables_for_subgraphs(
                    subgraphs=subgraphs,
                    network=network,
                    timestamp=timestamp,
                )
            )

            for observables_single_component in observables_split_components:
                rot_energy_gen = observables_single_component[0]
                power_imbalance = observables_single_component[1]
                load = observables_single_component[2]
                rocof = observables_single_component[3]
                load_share = observables_single_component[4]

                if load_inertia:
                    load_inertia_val = load * load_inertia_constant
                    rot_energy_total = rot_energy_gen + load_inertia_val
                    rocof_updated = (
                        50 * power_imbalance / ((rot_energy_total + 1e-8) * 2)
                    )
                    observables_single_component[3] = rocof_updated
                    rocof = rocof_updated

                if load != 0:
                    load_shedded = abs(min(0, power_imbalance)) / load * load_share
                else:
                    load_shedded = 0

                blackout_load_loss = int(abs(rocof) > 1) * load_share
                total_load_loss_share = max(load_shedded, blackout_load_loss)

                if use_sclopf:
                    if load_inertia:
                        dict_out_ele = [
                            timestamp,
                            weight,
                            trigger0,
                            num_par_failure0,
                            trigger1,
                            num_par_failure1,
                            component_number,
                            *observables_single_component,
                            load_inertia_val,
                            rot_energy_total,
                            rocof,
                            load_shedded,
                            blackout_load_loss,
                            total_load_loss_share,
                        ]
                    else:
                        dict_out_ele = [
                            timestamp,
                            weight,
                            trigger0,
                            num_par_failure0,
                            trigger1,
                            num_par_failure1,
                            component_number,
                            *observables_single_component,
                            load_shedded,
                            blackout_load_loss,
                            total_load_loss_share,
                        ]
                else:
                    dict_out_ele = [
                        timestamp,
                        init_failures,
                        component_number,
                        *observables_single_component,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]

                if np.isnan(np.array(dict_out_ele[1:])).any():
                    print(
                        f"Warning: NaN values in CO2 level {co2l}, timestamp {timestamp}, "
                        f"component {component_number}."
                    )

                component_props_list.append(dict_out_ele)
                split_numbers_local.append(split_number_local)

            # Indicator vectors
            if eval_indicator_vectors:
                indi_vec_r = subgraph_evaluation.get_indicator_vectors_of_subgraphs(
                    subgraphs, nx_graph
                )
                component_indicator_vectors_list.extend(indi_vec_r)

                if calc_split_indicator_vectors:
                    rocof_vector = np.empty(nx_graph.number_of_nodes(), dtype=float)
                    for idx, component_indicator_vec in enumerate(indi_vec_r):
                        rocof_vector[component_indicator_vec.astype(bool)] = (
                            observables_split_components[idx][3]
                        )
                    rocof_indicator_vectors_list.append(rocof_vector)

        # Collect results for this timestamp
        all_component_props.extend(component_props_list)
        all_split_numbers.extend(split_numbers_local)
        if eval_indicator_vectors:
            all_component_indicator_vectors.extend(component_indicator_vectors_list)
            if calc_split_indicator_vectors:
                all_rocof_indicator_vectors.extend(rocof_indicator_vectors_list)

    return {
        "component_props": all_component_props,
        "split_numbers": all_split_numbers,
        "component_indicator_vectors": all_component_indicator_vectors,
        "rocof_indicator_vectors": all_rocof_indicator_vectors,
    }


def process_timestamp_splits(
    timestamp,
    splits,
    co2l,
    n_nodes,
    snet_index,
    comp_cols,
    load_inertia,
    load_inertia_constant,
    eval_indicator_vectors,
    calc_split_indicator_vectors,
    use_sclopf,
):
    """Process all splits for a single timestamp. Module-level function for pickling.

    Loads network and nx_graph inside the worker to avoid pickling large objects.
    """
    # Load objects inside worker instead of passing them
    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf)
    nx_graph = data_handling.load_networkx_graph(snet_index=snet_index, co2lvl=co2l)

    component_props_list = []
    component_indicator_vectors_list = []
    rocof_indicator_vectors_list = []
    split_numbers_local = []
    split_number_local = 0

    for component_number, (init_failures, cascade_weight_tuple) in enumerate(
        splits.items()
    ):
        split_number_local += 1

        trigger0 = init_failures[0][0]
        num_par_failure0 = init_failures[0][1]
        trigger1 = init_failures[1][0]
        num_par_failure1 = init_failures[1][1]

        cascade = cascade_weight_tuple[0]
        weight = cascade_weight_tuple[1]

        subgraphs = data_handling.get_subgraphs_from_edges(cascade, nx_graph)

        observables_split_components = (
            subgraph_evaluation.evaluate_observables_for_subgraphs(
                subgraphs=subgraphs,
                network=network,
                timestamp=timestamp,
            )
        )

        for observables_single_component in observables_split_components:
            rot_energy_gen = observables_single_component[0]
            power_imbalance = observables_single_component[1]
            load = observables_single_component[2]
            rocof = observables_single_component[3]
            load_share = observables_single_component[4]

            if load_inertia:
                load_inertia_val = load * load_inertia_constant
                rot_energy_total = rot_energy_gen + load_inertia_val
                rocof_updated = 50 * power_imbalance / ((rot_energy_total + 1e-8) * 2)
                observables_single_component[3] = rocof_updated
                rocof = rocof_updated

            if load != 0:
                load_shedded = abs(min(0, power_imbalance)) / load * load_share
            else:
                load_shedded = 0

            blackout_load_loss = int(abs(rocof) > 1) * load_share
            total_load_loss_share = max(load_shedded, blackout_load_loss)

            if use_sclopf:
                if load_inertia:
                    dict_out_ele = [
                        timestamp,
                        weight,
                        trigger0,
                        num_par_failure0,
                        trigger1,
                        num_par_failure1,
                        component_number,
                        *observables_single_component,
                        load_inertia_val,
                        rot_energy_total,
                        rocof,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]
                else:
                    dict_out_ele = [
                        timestamp,
                        weight,
                        trigger0,
                        num_par_failure0,
                        trigger1,
                        num_par_failure1,
                        component_number,
                        *observables_single_component,
                        load_shedded,
                        blackout_load_loss,
                        total_load_loss_share,
                    ]
            else:
                dict_out_ele = [
                    timestamp,
                    init_failures,
                    component_number,
                    *observables_single_component,
                    load_shedded,
                    blackout_load_loss,
                    total_load_loss_share,
                ]

            if np.isnan(np.array(dict_out_ele[1:])).any():
                print(
                    f"Warning: NaN values in CO2 level {co2l}, timestamp {timestamp}, "
                    f"component {component_number}."
                )

            component_props_list.append(dict_out_ele)
            split_numbers_local.append(split_number_local)

        # Indicator vectors
        if eval_indicator_vectors:
            indi_vec_r = subgraph_evaluation.get_indicator_vectors_of_subgraphs(
                subgraphs, nx_graph
            )
            component_indicator_vectors_list.extend(indi_vec_r)

            if calc_split_indicator_vectors:
                rocof_vector = np.empty(nx_graph.number_of_nodes(), dtype=float)
                for idx, component_indicator_vec in enumerate(indi_vec_r):
                    rocof_vector[component_indicator_vec.astype(bool)] = (
                        observables_split_components[idx][3]
                    )
                rocof_indicator_vectors_list.append(rocof_vector)

    return {
        "timestamp": timestamp,
        "component_props": component_props_list,
        "split_numbers": split_numbers_local,
        "component_indicator_vectors": component_indicator_vectors_list,
        "rocof_indicator_vectors": rocof_indicator_vectors_list,
    }


@profile
def evaluate_cascade_parallel(
    co2l: float,
    n_nodes: int,
    snet_index: int = 0,
    start_time_str=None,
    end_time_str=None,
    eval_indicator_vectors: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    use_sclopf=True,
    overwrite: bool = False,
    calc_split_indicator_vectors: bool = True,
    load_inertia: bool = True,
    show_progress_2: bool = False,
    sort_cascades: bool = True,
    n_jobs: int = 8,
):
    """Parallel version of evaluate_cascade that parallelizes over timestamps.

    Find the properties of the splits (i.e., RoCoF or lost load) and
    indicator vectors describing the network, using parallel processing.

    Args:
        co2l (float): Target CO2 level in PyPSA Optimization.
        n_nodes (int): Number of nodes in PyPSA network.
        snet_index (int, optional): Select a particular subnetwork for calculations (if the pypsa network has different ones).
                    For our data set, "0" indicates the Continental European AC grid. Defaults to 0.
        start_time_str, end_time_str (str Format "YYYY-mm-dd HH:MM"): If either or both are not None, the evaluation will only be
                    done between start_time and end_time.
        n_jobs (int): Number of parallel jobs to run. Defaults to 8. Use -1 for all cores, -2 for all but one.
    """
    from joblib import Parallel, delayed

    # if start_time_str is not None or end_time_str is not None:
    #     raise NotImplementedError(
    #         "Not implemented anymore. comp_props is referenced before assignement..."
    #     )

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
        raise NotImplementedError("LOPF based not supported anymore.")
    if end_time_str is not None:
        save_path += f"to{end_time_str}/"

    print(f"saving results to {save_path}")

    save_df_path = save_path + f"component_properties_Co2L{co2l}_n{n_nodes}"
    if os.path.exists(save_df_path + ".h5"):
        if not overwrite:
            raise FileExistsError(
                f"Results already exist at {save_df_path}.h5, skipping evaluation."
            )
        else:
            print(f"Overwriting existing results at {save_df_path}.h5 as requested.")

    os.makedirs(save_path, exist_ok=True)

    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf)
    nx_graph = data_handling.load_networkx_graph(snet_index=snet_index, co2lvl=co2l)

    data_handling.check_if_edges_sorted(nx_graph)

    # Load cascade results
    with gzip.open(full_path_to_cascades, "rb") as fh_casc:
        splitting_cascades = pickle.load(fh_casc)

    if sort_cascades:
        keys = list(splitting_cascades.keys())
        if not keys == sorted(keys):
            print("Sorting splitting_cascades by datetime keys...")
            splitting_cascades = dict(
                sorted(
                    splitting_cascades.items(),
                    key=lambda x: dt.strptime(x[0], "%Y-%m-%d %H:%M"),
                )
            )
            with gzip.open(full_path_to_cascades, "wb") as fh_casc:
                pickle.dump(
                    splitting_cascades, fh_casc, protocol=pickle.HIGHEST_PROTOCOL
                )

    # Initialize results
    if use_sclopf:
        if load_inertia:
            comp_cols = [
                "time_stamp",
                "trigger_weighting",
                "init_failure_0",
                "num_par_failure_0",
                "init_failure_1",
                "num_par_failure_1",
                "split_number",
                "rot_energy_gen",
                "power_imbalance",
                "load",
                "rocof_noLoadInertia",
                "load_share",
                "load_inertia",
                "rot_energy",
                "rocof",
                "shedding_load_loss_share",
                "blackout_load_loss_share",
                "total_load_loss_share",
            ]
        else:
            comp_cols = [
                "time_stamp",
                "trigger_weighting",
                "init_failure_0",
                "num_par_failure_0",
                "init_failure_1",
                "num_par_failure_1",
                "split_number",
                "rot_energy",
                "power_imbalance",
                "load",
                "rocof",
                "load_share",
                "shedding_load_loss_share",
                "blackout_load_loss_share",
                "total_load_loss_share",
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
        ]

    splitting_cascades_dtkeys = {
        dt.strptime(key, "%Y-%m-%d %H:%M"): value
        for key, value in splitting_cascades.items()
    }

    if verbose:
        print(f"Starting evaluation of System Splits with {n_jobs} parallel jobs.\n")

    # Parallel processing over timestamps
    timestamp_list = list(splitting_cascades_dtkeys.items())
    if end_time_str is not None:
        timestamp_list = [
            (ts, splits) for ts, splits in timestamp_list if ts <= end_time_dt
        ]

    if show_progress:
        print(
            f"Processing {len(timestamp_list)} timestamps with {n_jobs} parallel jobs..."
        )

    # Chunk timestamps to match number of workers
    chunk_size = max(1, len(timestamp_list) // n_jobs)
    timestamp_chunks = [
        timestamp_list[i : i + chunk_size]
        for i in range(0, len(timestamp_list), chunk_size)
    ]

    if verbose:
        print(
            f"Split into {len(timestamp_chunks)} chunks of ~{chunk_size} timestamps each"
        )

    # Use tqdm for progress tracking
    results_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_timestamp_chunk)(
            chunk,
            co2l,
            n_nodes,
            snet_index,
            comp_cols,
            load_inertia,
            load_inertia_constant,
            eval_indicator_vectors,
            calc_split_indicator_vectors,
            use_sclopf,
        )
        for chunk in tqdm(
            timestamp_chunks, desc="Processing chunks", disable=not show_progress
        )
    )

    # Merge results from chunks
    component_props_dict = {}
    component_indicator_vectors_ls = []
    rocof_indicator_vectors_ls = []
    split_numbers = []
    out_dict_key = 0
    split_number_offset = 0

    for result in results_list:
        for props in result["component_props"]:
            component_props_dict[out_dict_key] = props
            out_dict_key += 1

        # Renumber split numbers to be globally sequential with offset
        offset_local_splits = [s + split_number_offset for s in result["split_numbers"]]
        split_numbers.extend(offset_local_splits)

        # Update offset for next chunk: add the max split number from this chunk
        if result["split_numbers"]:
            split_number_offset += max(result["split_numbers"])

        if eval_indicator_vectors:
            component_indicator_vectors_ls.extend(result["component_indicator_vectors"])

            if calc_split_indicator_vectors:
                rocof_indicator_vectors_ls.extend(result["rocof_indicator_vectors"])

    if verbose:
        print(
            f"Processed {out_dict_key} components across {len(results_list)} timestamps."
        )

    # Save results
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
                    np.array(rocof_indicator_vectors_ls, dtype=float),
                    fh_vec_out,
                )
            if verbose:
                print(f"Saved RoCoF indicator vectors to {rocof_indi_vec_save_path}")

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
        message_text = f"Evaluation of N={n_nodes}, C02_lvl={co2l} finished and results saved (parallel version)"
        send_mattermost_messages.post_message(message_text, cfg.mattermost_url)

    return component_props


def evaluate_cascade_wrapper(
    co2l: float,
    n_nodes: int,
    snet_index: int = 0,
    start_time_str=None,
    end_time_str=None,
    eval_indicator_vectors: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    use_sclopf=True,
    overwrite: bool = False,
    calc_split_indicator_vectors: bool = True,
    load_inertia: bool = True,
    show_progress_2: bool = False,
    sort_cascades: bool = True,
    parallel: bool = False,
    n_jobs: int = 8,
):
    """Wrapper to run parallel or serial evaluation.

    Set `parallel=True` to use `evaluate_cascade_parallel`, otherwise the
    serial `evaluate_cascade` is used.
    """
    if parallel:
        return evaluate_cascade_parallel(
            co2l=co2l,
            n_nodes=n_nodes,
            snet_index=snet_index,
            start_time_str=start_time_str,
            end_time_str=end_time_str,
            eval_indicator_vectors=eval_indicator_vectors,
            verbose=verbose,
            show_progress=show_progress,
            use_sclopf=use_sclopf,
            overwrite=overwrite,
            calc_split_indicator_vectors=calc_split_indicator_vectors,
            load_inertia=load_inertia,
            show_progress_2=show_progress_2,
            sort_cascades=sort_cascades,
            n_jobs=n_jobs,
        )

    return evaluate_cascade(
        co2l=co2l,
        n_nodes=n_nodes,
        snet_index=snet_index,
        start_time_str=start_time_str,
        end_time_str=end_time_str,
        eval_indicator_vectors=eval_indicator_vectors,
        verbose=verbose,
        show_progress=show_progress,
        use_sclopf=use_sclopf,
        overwrite=overwrite,
        calc_split_indicator_vectors=calc_split_indicator_vectors,
        load_inertia=load_inertia,
        show_progress_2=show_progress_2,
        sort_cascades=sort_cascades,
    )


def sort_comps_and_ind_vecs(co2l, n_nodes=600):
    """if component properties are not sorted by time stamp and split number, sort them
    and save sorted versions of component properties and indicator vectors, while
    keeping the unsorted versions as well.
    """
    raise NotImplementedError("Not implemented anymore.")
    fpath_comp = (
        path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
    )

    component_props_level = pd.read_hdf(fpath_comp)
    component_props_level.reset_index(inplace=True)
    # sort components by time stamp and split number
    component_props_level_sorted = component_props_level.sort_values(
        by=["time_stamp", "split_number"], inplace=False
    )
    if component_props_level_sorted.equals(component_props_level):
        print("Component properties are already sorted.")
        return

    if component_props_level_sorted.trigger_weighting.nunique() == 1:
        raise ValueError("trigger weighting has only one unique value")
    rocof_indi_vec_load_path = (
        path_to_evaluation_results_sclopf
        + f"rocof_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz"
    )
    with gzip.open(rocof_indi_vec_load_path, "rb") as fh_vec_out:
        rocof_indicator_vectors = np.array(pickle.load(fh_vec_out), dtype=float)
    rocof_indicator_vectors_sorted = rocof_indicator_vectors[
        component_props_level_sorted.drop_duplicates(
            subset=["split_number"]
        ).split_number
        - 1
    ]
    os.rename(
        rocof_indi_vec_load_path,
        rocof_indi_vec_load_path.replace(".pklz", "_unsorted.pklz"),
    )
    with open(
        path_to_evaluation_results_sclopf
        + f"rocof_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz",
        "wb",
    ) as fh_vec_out_sorted:
        pickle.dump(
            rocof_indicator_vectors_sorted,
            fh_vec_out_sorted,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    for new_num, old_num in enumerate(
        component_props_level_sorted.split_number.unique()
    ):
        component_props_level_sorted.loc[
            component_props_level_sorted.split_number == old_num, "split_number"
        ] = new_num

    # move file to ""..._unsorted.h5"
    os.rename(
        fpath_comp,
        fpath_comp.replace(".h5", "_unsorted.h5"),
    )
    component_props_level_sorted.to_hdf(
        path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2l}_n{n_nodes}.h5",
        key="df",
        mode="w",
    )


if __name__ == "__main__":
    import argparse
    from utils.config import path_to_sclopf_data
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate cascade for specific CO2 levels"
    )
    parser.add_argument("--co2l", type=float, help="CO2 level to process")
    parser.add_argument("--n_nodes", type=int, default=600, help="Number of nodes")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all CO2 levels sequentially",
        default=True,
    )

    args = parser.parse_args()

    if args.co2l is not None:
        # Process single CO2 level (for parallel execution via bash)
        co2l_in = args.co2l
        n_nodes_in = args.n_nodes

        print(60 * "_")
        print(f"Evaluating cascade for CO2 level {co2l_in} and n_nodes {n_nodes_in}")
        print(60 * "_")
        try:
            evaluate_cascade_wrapper(
                co2l_in,
                n_nodes_in,
                use_sclopf=True,
                eval_indicator_vectors=True,
                overwrite=False,
                load_inertia=True,
                n_jobs=30,
                verbose=True,
                parallel=False,
                # end_time_str="2013-01-03 00:00",
            )
        except FileExistsError as e:
            print(
                f"Skipping evaluation for {co2l_in} and {n_nodes_in} due to existing results."
            )
        # print(60 * "_")
        # print(
        #     f"sorting component properties and indicator vectors for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
        # )
        # sort_comps_and_ind_vecs(co2l_in, n_nodes=n_nodes_in)
        print(60 * "_")
        print(
            f"Running all CO2 level edge based for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
        )
        print(60 * "_")
        try:
            find_failed_edge_indicator_vector_for_cascade_results(
                co2l_in,
                n_nodes_in,
                save_res=True,
                verbose=True,
                overwrite=False,
                use_sclopf=True,
            )
        except FileExistsError as e:
            print(
                f"Skipping edge indicator vector evaluation for {co2l_in} and {n_nodes_in} due to existing results."
            )

        print(f"Completed processing CO2 level {co2l_in}")

    elif args.all:
        # Process all CO2 levels sequentially (original behavior)
        n_nodes_in = args.n_nodes
        co2l_list = get_co2_levels(n_nodes_in)

        if isinstance(co2l_list, float):
            co2l_list = [co2l_list]
        co2l_list = sorted(co2l_list, reverse=False)

        for co2l_in in co2l_list:
            print(60 * "_")
            print(
                f"Evaluating cascade for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
            )
            print(60 * "_")
            try:
                evaluate_cascade_wrapper(
                    co2l_in,
                    n_nodes_in,
                    use_sclopf=True,
                    eval_indicator_vectors=True,
                    overwrite=False,
                    load_inertia=True,
                    n_jobs=30,
                    verbose=True,
                    parallel=True,
                    # end_time_str="2013-01-03 00:00",
                )
            except FileExistsError as e:
                print(
                    f"Skipping evaluation for {co2l_in} and {n_nodes_in} due to existing results."
                )
            print(60 * "_")
            # print(
            #     f"sorting component properties and indicator vectors for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
            # )
            # sort_comps_and_ind_vecs(co2l_in, n_nodes=n_nodes_in)
            # print(60 * "_")
            print(
                f"Getting edge indicator vecs for CO2 level {co2l_in} and n_nodes {n_nodes_in}"
            )
            print(60 * "_")
            try:
                find_failed_edge_indicator_vector_for_cascade_results(
                    co2l_in,
                    n_nodes_in,
                    save_res=True,
                    verbose=True,
                    overwrite=False,
                    use_sclopf=True,
                )
            except FileExistsError as e:
                print(
                    f"Skipping edge indicator vector evaluation for {co2l_in} and {n_nodes_in} due to existing results."
                )
            print(60 * "_")
    else:
        print(
            "Please specify either --co2l <value> for single CO2 level or --all for all levels"
        )
        parser.print_help()
