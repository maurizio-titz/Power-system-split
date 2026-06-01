#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Place synthetic inertia to mitigate dangerous system splits
and lost load. This is done via adding rot energy to individual nodes
iteratively (i.e., greedily to the one mitigating the most lost load)."""

import gzip
import itertools
import multiprocessing
import os
import pickle
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.config as cfg
from utils import send_mattermost_messages

# Send messages to mattermost
from utils.config import (
    path_to_evaluation_results_sclopf,
    path_to_inertia_mitigation_results_sclopf,
    path_to_pypsa_network_sclopf,
    path_to_vis_results_sclopf
)
from utils.data_handling import (
    load_pypsa_network_from_path,
    load_pypsa_network,
    load_split_props,
)

# if use_sclopf:
#     path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_sclopf
# else:
#     path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_lopf
# if not os.path.exists(path_to_inertia_mitigation_results):
#     os.mkdir(path_to_inertia_mitigation_results)


def build_idx_to_node_idx_list_n_reverse(indicator_vec_arr: np.ndarray):
    """Build a dictionary that connects individual nodes to the components
    of each split component.

    Args:
        indicator_vec_arr (np.ndarray): _description_

    Returns:
        split_node_idx_ls, node_split_idx_list: Lists collecting the translation
            from split number to node idx and vice versa. Each entry in split_node_idx_ls holds the indices of the nodes that are part of the split. Each entry in node_split_idx_list holds the indices of the splits that the node is part of.
    """

    split_node_idx_ls = list()
    node_split_idx_ls = list()

    for idx_r, row_r in enumerate(indicator_vec_arr):
        split_node_idx_ls.append(np.argwhere(row_r).flatten())

    for column_r in indicator_vec_arr.T:
        node_split_idx_ls.append(np.argwhere(column_r).flatten())

    return split_node_idx_ls, node_split_idx_ls


def get_lost_load_in_member_components(
    node_arr: np.ndarray,
    component_props_arr: np.ndarray,
    snapshot_weightings_arr: np.ndarray,
    node_to_split_idx_ls: list,
    change_rot_energy: float,
    rocof_abs_threshold: float,
    freq_ref: float = 50.0,
) -> np.ndarray:
    """Get how much load would still be lost after adding an additional Erot at each node in 'node_arr'.

    Args:
        node_arr (np.ndarray): Array with nodes for which the total lost load will be displayed.
        component_props_arr (np.ndarray): Component properties of each split.
        snapshot_weightings_arr (np.ndarray): Array with snapshot weightings showing time
            length of a snapshots.
        node_to_split_idx_ls (list): List translating node to connected split numbers.
        change_rot_energy (float): Change of rotational energy per step.
        rocof_neg_threshold (float): Threshold below in Hz/s
            which a split component is considered to be unrecoverable.
        freq_ref (float, optional): Reference frequency in Hz. Defaults to 50..

    Returns:
        np.ndarray: array with total load load for each member node given in 'node_arr'
            if an additional rot energy would be placed.
    """

    lost_load_member_comps = np.zeros(len(node_arr))

    for idx_ele, node_idx in enumerate(node_arr):
        # Find idxs of splits where node_r is a member
        idx_node_in_split = node_to_split_idx_ls[node_idx]

        cut_snapshow_weighting = snapshot_weightings_arr[idx_node_in_split]

        node_cut_comp_arr = component_props_arr[idx_node_in_split, :].copy()

        # Place inertia and evaluate rocof new
        node_cut_comp_arr[:, 0] += change_rot_energy
        new_rocof = freq_ref * (node_cut_comp_arr[:, 1] / (2 * node_cut_comp_arr[:, 0]))
        node_cut_comp_arr[:, 2] = new_rocof

        # See where rocof exceeds threshold and how much load is therefore lost
        lost_idx = np.abs(node_cut_comp_arr[:, 2]) > rocof_abs_threshold
        lost_load_member_comps[idx_ele] = (
            node_cut_comp_arr[lost_idx, 3] * cut_snapshow_weighting[lost_idx]
        ).sum()

    return lost_load_member_comps


def greedy_inertia_placement_step(
    component_props_arr: np.ndarray,
    indicator_vectors: np.ndarray,
    split_to_node_list: list,
    delta_rot_energy: float,
    rocof_abs_threshold: float,
    load_share_threshold: float,
    freq_ref: float = 50.0,
) -> np.ndarray:
    """Place inertia according to the highest amount of mitigated load.

    Args:
        component_props_arr (np.ndarray): Component properties of each split. columns: ["rot_energy", "power_imbalance", "rocof", "load", "load_share", "total_weighting"]
        indicator_vectors (np.ndarray): Vector having ones in every row k showing
            nodes being in split k.
        snapshot_weightings_arr (np.ndarray): Array with snapshot weightings showing
            time length of a snapshots.
        split_to_node_list (list): List translating split number to node idx
        delta_rot_energy (float): Change in rotational energy in MWs.
        rocof_neg_threshold (float): Threshold below in Hz/s
            which a split component is considered to be unrecoverable.
        load_share_threshold (float): Load threshold for a split component to be considered.
        freq_ref (float, optional): Reference frequency in Hz. Defaults to 50..

    Returns:
        np.ndarray: Array that shows how much load lost can be mitigated for each node.
    """

    # Check if component properties has all the entries needed (e.g., last day and first day)
    nn_nodes = indicator_vectors.shape[1]
    mitigate_load_share_loss_node_arr = np.zeros(nn_nodes)

    # Try which lost load can be mitigated when placing delta_rot_energy at each node
    # to pick the node with the maximum change (i.e., greedy optimization).
    for idx_split, split_r in enumerate(component_props_arr):

        modified_rocof = freq_ref * (split_r[1] / (2 * (split_r[0] + delta_rot_energy)))
        if (
            np.abs(split_r[2]) > rocof_abs_threshold
            and np.abs(modified_rocof) < rocof_abs_threshold
            and split_r[3] > load_share_threshold
        ):
            idx_node_for_split = split_to_node_list[idx_split]
            mitigated_load_share_loss_weighted = (
                split_r[3] * split_r[4]
            )  # [4] is the total event weighting
            if len(split_r) > 5:
                mitigated_load_share_loss_weighted *= split_r[
                    5
                ]  # [5] is the mitigation weighting factor
            mitigate_load_share_loss_node_arr[
                idx_node_for_split
            ] += mitigated_load_share_loss_weighted

    return mitigate_load_share_loss_node_arr


def _resolve_equality_concentrate_inertia(
    inertia_already_placed, idxs_nodes_to_choose, atol=1e-8
):

    added_inertia_candidate_nodes = inertia_already_placed[idxs_nodes_to_choose]
    # select node with max added inertia; if equal, select randomly
    max_inertia_in_candidates = added_inertia_candidate_nodes.max()

    canIdx_node_max_candidates = np.where(
        abs(max_inertia_in_candidates - added_inertia_candidate_nodes) < atol
    )[0]

    if len(canIdx_node_max_candidates) > 1:
        canIdx_node_picked = np.random.choice(canIdx_node_max_candidates)
        still_random = True
    else:
        canIdx_node_picked = canIdx_node_max_candidates[0]
        still_random = False

    idx_node_picked = idxs_nodes_to_choose[canIdx_node_picked]

    assert idx_node_picked in idxs_nodes_to_choose

    return idx_node_picked, still_random


def _resolve_equality_hindsight(
    comp_arr,
    snapshot_weightings_arr,
    node_to_split_ls,
    ch_rot_energy_r,
    rocof_threshold_Hz_s,
    freq_ref,
    idxs_nodes_to_choose,
    inertia_already_placed,
    concentrate_inertia=False,
    atol=1e-8,
):

    # Check for which node placed, the lost load in its components is higher
    # resolve conflict again randomly or by concentrating inertia.
    list_lost_load_of_member_components = get_lost_load_in_member_components(
        idxs_nodes_to_choose,
        comp_arr,
        snapshot_weightings_arr,
        node_to_split_ls,
        ch_rot_energy_r,
        rocof_threshold_Hz_s,
        freq_ref=freq_ref,
    )

    max_val_lost = list_lost_load_of_member_components.max()
    canIdxs_max_lost_load = np.where(
        abs(list_lost_load_of_member_components - max_val_lost) < atol
    )[0]
    idxs_max_lost_load = idxs_nodes_to_choose[canIdxs_max_lost_load]

    still_random = False
    if len(idxs_max_lost_load) > 1:
        # place inertia at node with max added inertia
        if concentrate_inertia:
            idx_node_picked, still_random = _resolve_equality_concentrate_inertia(
                inertia_already_placed, idxs_max_lost_load, atol=atol
            )
        else:
            idx_node_picked = np.random.choice(idxs_max_lost_load)
            still_random = True
    else:
        idx_node_picked = idxs_max_lost_load[0]

    assert idx_node_picked in idxs_nodes_to_choose

    return idx_node_picked, still_random


def run_greedy_inertia_placement(
    component_df: pd.DataFrame,
    indicator_vec_arr: np.ndarray,
    snapshot_weightings_generators: pd.DataFrame,
    delta_rot_energy: float,
    rocof_threshold_Hz_s: float = 1.0,
    max_iter: int = 10000,
    freq_ref: float = 50,
    load_share_threshold: float = 0,
    show_progress: bool = True,
    resolve_equality_method: str = "random",
    atol: float = 1e-8,
    revert_change_fac: bool = False,
    fixed_step_size: bool = False,
    initial_added_inertia_by_node: np.ndarray = None,
) -> tuple:
    """Run inertia placement to reduce the amount of lost load, which is defined as the
    load in a component that suffers an absolute rocof smaller than 'rocof_threshold_Hz_s'.

    Args:
        component_df (pd.DataFrame): Dataframe with information on components of splits.
        indicator_vec_arr (np.ndarray): Vector with entries for each split
        snapshot_weightings_generators (pd.DataFrame): Snapshot weightings to show how many
        delta_rot_energy (float): Change of rotational energy per step.
        rocof_threshold_Hz_s (float, optional): Threshold for absolute RoCoF above which a split component is considered
            to be unrecoverable. Defaults to 1. .
        max_iter (int, optional): Number of iterations. Note, simulation will be stopped if
            every split is below the threshold. Defaults to 10000.
        freq_ref (float, optional): Reference frequency in Hz. Defaults to 50.
        load_share_threshold (float, optional): Load share that splits have to have to be considered. Defaults to 0.
        show_progress (bool, optional): If 'True', show progressbar. Defaults to True.
        resolve_equality_method (str, optional): Method to resolve the situation when inertia could
            be placed at multiple nodes. Either random (choose a random node), hindsight (choose a node)
            that has the higher lost load in the next step), concentrate (concentrate inertia at nodes
            where inertia was placed previously), hindsight_concentrate (mix of hindsight and resolving by
            concentrating inertia). If a conflict remains, random node is chosen.
        atol (float, optional): absolute tolerance for comparison. Defaults to 1e-8.
        revert_change_fac: If 'True', the step size is reset after after placing inertia.
        fixed_step_size: If 'True', the step size is fixed and not increased if no change is detected. Instead an error is raised.

    Returns:
        modified_comp_index, modified_comp_arr,
        inertia_placed_loss_mitigated_ls, componont_mitigated_step,
        resolve_equality_counter, still_used_random_node_choice
    """

    assert (
        rocof_threshold_Hz_s > 0
    ), "rocof_threshold_Hz_s has to be positive. We consider absolute RoCoF."

    # Find out significant splits and reduce comp and ind
    cut_idxs = (component_df["rocof"].abs() > rocof_threshold_Hz_s).values

    # load split properties
    # filter split properties by blackout size threshold
    # filter component_df by filter

    component_df_cut = component_df[cut_idxs]
    componont_mitigated_step = pd.Series(
        index=component_df_cut.index, data=-1
    )  # at which step a component was mitigated

    len_cut_df = len(component_df_cut.index)
    indicator_vec_arr_cut = indicator_vec_arr[cut_idxs]

    # Deal with snapshot weightings
    cut_time_stamps = component_df.time_stamp[cut_idxs].values
    snapshot_weightings_arr_cut = snapshot_weightings_generators[cut_time_stamps].values

    component_df_cut["total_weighting"] = (
        component_df_cut.trigger_weighting * component_df_cut.snapshot_weighting
    )
    # convert component_df to array

    selected_component_props = [
        "rot_energy",
        "power_imbalance",
        "rocof",
        "load_share",
        "total_weighting",
    ]
    if "mitigation_weighting" in component_df_cut.columns:
        selected_component_props.append("mitigation_weighting")

    modified_component_df = component_df_cut.loc[
        :,
        selected_component_props,
    ].copy()
    loss_share_before_mitigation = modified_component_df["load_share"].sum()

    modified_comp_arr = np.array(modified_component_df.values)

    modified_comp_index = modified_component_df.index.values

    inertia_placed_loss_mitigated_ls = list()

    split_to_node_idx_ls, node_to_split_idx_ls = build_idx_to_node_idx_list_n_reverse(
        indicator_vec_arr_cut
    )

    # step, place_added, multi_added
    resolve_equality_counter = 0
    still_used_random_node_choice = 0
    delta_rot_energy_factor = 1.0
    total_placed_inertia = 0.0

    if initial_added_inertia_by_node is None:
        added_inertia_by_node = np.zeros(indicator_vec_arr.shape[1])
        starting_point_string = ""
    else:
        raise NotImplementedError(
            "starting optimization with initial inertia not implemented yet"
        )
        added_inertia_by_node = initial_added_inertia_by_node
        starting_point_string = "_warmStart"

    pbar = tqdm(range(max_iter), disable=not show_progress)
    for idx_step in pbar:
        count_beyond_threshold = np.count_nonzero(
            np.abs(modified_comp_arr[:, 2]) > rocof_threshold_Hz_s
        )
        if idx_step > 1:
            pbar.set_description(
                f"Bey. thres.: {count_beyond_threshold/ len_cut_df*100:.1f}%, placed: {total_placed_inertia/1000:.0f}GWs "
            )
        ch_rot_energy_r = delta_rot_energy_factor * delta_rot_energy

        # Find candidate for largest mitigated lost load
        potential_load_loss_change_by_selected_node = greedy_inertia_placement_step(
            modified_comp_arr,
            indicator_vec_arr_cut,
            split_to_node_idx_ls,
            ch_rot_energy_r,
            rocof_threshold_Hz_s,
            load_share_threshold,
            freq_ref=freq_ref,
        )

        # If no change was detected, more synthetic rot. energy is added
        if (potential_load_loss_change_by_selected_node == 0).all():

            if fixed_step_size:
                # No change detected, but fixed step size is used, so no further inertia is placed.
                print("No change detected, but fixed step size is used. Stopping.")
                break

            delta_rot_energy_factor += 1.0

        else:
            max_change = potential_load_loss_change_by_selected_node.max()
            idx_node = np.where(
                abs(potential_load_loss_change_by_selected_node - max_change) < atol
            )[0]

            # Check if a decision has to be made due to more nodes being options to
            # place inertia
            if len(idx_node) > 1:
                resolve_equality_counter += 1
                if resolve_equality_method == "random":
                    idx_node = np.random.choice(idx_node)
                    still_random = True

                elif resolve_equality_method == "concentrate":
                    idx_node, still_random = _resolve_equality_concentrate_inertia(
                        added_inertia_by_node, idx_node, atol=atol
                    )

                elif resolve_equality_method == "hindsight":
                    idx_node, still_random = _resolve_equality_hindsight(
                        modified_comp_arr,
                        snapshot_weightings_arr_cut,
                        node_to_split_idx_ls,
                        ch_rot_energy_r,
                        rocof_threshold_Hz_s,
                        freq_ref,
                        idx_node,
                        added_inertia_by_node,
                        concentrate_inertia=False,
                        atol=atol,
                    )

                elif resolve_equality_method == "hindsight_concentrate":
                    idx_node, still_random = _resolve_equality_hindsight(
                        modified_comp_arr,
                        snapshot_weightings_arr_cut,
                        node_to_split_idx_ls,
                        ch_rot_energy_r,
                        rocof_threshold_Hz_s,
                        freq_ref,
                        idx_node,
                        added_inertia_by_node,
                        concentrate_inertia=True,
                        atol=atol,
                    )
                else:
                    raise IOError(
                        f"Resolve equality measure '{resolve_equality_method}' not known."
                    )

                if still_random:
                    still_used_random_node_choice += 1

            else:
                idx_node = idx_node[0]

            # Collect change details
            results_out_r = [
                idx_step,
                idx_node,
                delta_rot_energy_factor,
                max_change,
                count_beyond_threshold,
            ]
            inertia_placed_loss_mitigated_ls.append(results_out_r)

            # Modify rest of the components due to the placement of inertia
            idx_node_in_split = node_to_split_idx_ls[idx_node]

            added_inertia_by_node[idx_node] += ch_rot_energy_r
            modified_comp_arr[idx_node_in_split, 0] += ch_rot_energy_r
            new_rocof = freq_ref * (
                modified_comp_arr[idx_node_in_split, 1]
                / (2 * modified_comp_arr[idx_node_in_split, 0])
            )
            modified_comp_arr[idx_node_in_split, 2] = new_rocof
            mitigated_components = modified_comp_index[
                (np.abs(modified_comp_arr[:, 2]) < rocof_threshold_Hz_s)
                & (componont_mitigated_step == -1)
            ]
            componont_mitigated_step.loc[mitigated_components] = idx_step

            if revert_change_fac:
                delta_rot_energy_factor = 1.0

        # Break when all splits are pushed below rocof threshold
        if count_beyond_threshold == 0:
            break
        total_placed_inertia += ch_rot_energy_r

        # current_loss = modified_comp_arr[modified_comp_arr[:, 2] < -1, 3].sum()
        # mitigated_loss = np.array(inertia_placed_loss_mitigated_ls)[:, 3].sum()
        # deviation = (loss_share_before_mitigation - mitigated_loss) / current_loss

    return (
        modified_comp_index,
        modified_comp_arr,
        inertia_placed_loss_mitigated_ls,
        componont_mitigated_step,
        resolve_equality_counter,
        still_used_random_node_choice,
    )


def run_specific_co2lvl_n_size(
    co2_lvl: float,
    nn_nodes: int = 400,
    delta_rot_energy: float = 10,
    max_iter: int = 100000,
    rocof_threshold_Hz_s: float = 1.0,
    lshare_threshold: float = 0.0,
    resolve_equality_method: str = "random",
    save_it: bool = True,
    show_progress: bool = True,
    revert_ch_rotE_fac: bool = False,
    use_sclopf: bool = True,
    blackout_size_threshold: float = 0.0,
) -> tuple:
    """Run the inertia placement for an optimized power system that was analyzed by
    running cascade experiments.

    Args:
        co2_lvl (float): CO2 level of PyPSA network.
        nn_nodes (int, optional): Number of nodes of PyPSA network. Defaults to 800.
        delta_rot_energy (float, optional): Change in rotational energy per step. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10000.
        rocof_threshold_Hz_s (float, optional): Threshold after which a component is counted as
            experiencing a black out aka the load is counted as lost. Defaults to -1..
        lshare_threshold (float, optional): Amount of load share of components that are being considered
            in the greedy mitigation procedure. Defaults to 0, which corresponds to all components
            being considered.
        resolve_equality_method (str, optional): Method to resolve the situation when inertia could
            be placed at multiple nodes. Either random (choose a random node), hindsight (choose a node)
            that has the higher lost load in the next step), concentrate (concentrate inertia at nodes
            where inertia was placed previously), hindsight_concentrate (mix of hindsight and resolving by
            concentrating inertia). If a conflict remains, random node is chosen.
        save_it (bool, optional): If 'True', the results are being pickled and saved. Defaults to True.

    Returns:
        res_tuple: see output of run_greedy_inertia_placement
    """

    # Load files for DataFrame collecting component properties, indicator vectors and PyPSA network.
    if use_sclopf:
        path_to_evaluation_results = path_to_evaluation_results_sclopf
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_sclopf
    else:
        path_to_evaluation_results = path_to_evaluation_results_lopf
        path_to_inertia_mitigation_results = path_to_inertia_mitigation_results_lopf

    fpath_component_in = os.path.join(
        path_to_vis_results_sclopf, f"component_properties_all_n{nn_nodes}.h5"
    )
    component_df = pd.read_hdf(fpath_component_in, key="df")
    component_df = component_df[component_df.co2l == co2_lvl]
    component_df.reset_index(inplace=True, drop=True)
    
    if blackout_size_threshold > 0.0:
        component_df.reset_index(inplace=True, drop=False)
        split_props = load_split_props(
            n_nodes=nn_nodes, co2l=co2_lvl, use_sclopf=use_sclopf
        )
        component_df.reset_index(inplace=True, drop=False)
        #component_df["co2l"] = co2_lvl
        component_df = component_df.set_index(["co2l", "time_stamp", "split_number"])
        component_df["mitigation_weighting"] = 0
        # filter split properties by blackout size threshold
        catastrophic_splits_idx = split_props[
            split_props.lost_load_share_blackout > blackout_size_threshold
        ].index
        component_df.loc[catastrophic_splits_idx, "mitigation_weighting"] = 1.0
        component_df.reset_index(inplace=True, drop=False)
        component_df.index = component_df["index"]
        component_df = component_df.drop(columns=["index"])

    fpath_indicator_vec_in = (
        path_to_evaluation_results
        + f"component_indicator_vectors_Co2L{co2_lvl}_n{nn_nodes}.pklz"
    )
    with gzip.open(fpath_indicator_vec_in, "rb") as fh_in:
        indicator_vec_arr = pickle.load(fh_in)

    pypsa_net = load_pypsa_network(co2_lvl, nn_nodes, use_sclopf=use_sclopf)
    snapshot_weightings_generators = pypsa_net.snapshot_weightings.generators

    res_tuple = run_greedy_inertia_placement(
        component_df,
        indicator_vec_arr,
        snapshot_weightings_generators,
        delta_rot_energy,
        rocof_threshold_Hz_s=rocof_threshold_Hz_s,
        max_iter=max_iter,
        load_share_threshold=lshare_threshold,
        resolve_equality_method=resolve_equality_method,
        show_progress=show_progress,
    )

    if save_it:
        fpath_out = (
            path_to_inertia_mitigation_results
            + f"/synthetic_inertia_placement_Co2{co2_lvl:g}_N{nn_nodes}"
            + f"_deltarotE{delta_rot_energy:g}_rocofthres{rocof_threshold_Hz_s:g}"
            + f"_lshare{lshare_threshold:g}_maxiter{max_iter}_{resolve_equality_method}"
        )
        os.makedirs(path_to_inertia_mitigation_results, exist_ok=True)

        if revert_ch_rotE_fac:
            fpath_out += "_revertfac"
        if blackout_size_threshold > 0.0:
            fpath_out += f"_blackoutthres{blackout_size_threshold:g}"

        with gzip.open(fpath_out + ".pklz", "wb") as fh_out:
            pickle.dump(res_tuple, fh_out)

        message_text = (
            "Finished synthetic inertia placement for  "
            + f"N={nn_nodes}, C02_lvl={co2_lvl} and saved results in '"
            + fpath_out
            + "'."
        )
        print(message_text)
        if cfg.mattermost_url is not None:
            send_mattermost_messages.post_message(message_text, cfg.mattermost_url)

    return res_tuple


def single_call(
    nn_nodes, use_sclopf, max_iter, revert_fac, resolve_method_n_delta_rot_e
):
    """Wrapper to be used in 'run_different_parameters_fo_co2lvl'"""

    resolve_method, delta_rot_e, co2_lvl = resolve_method_n_delta_rot_e

    run_specific_co2lvl_n_size(
        co2_lvl,
        nn_nodes=nn_nodes,
        delta_rot_energy=delta_rot_e,
        show_progress=False,
        save_it=True,
        max_iter=max_iter,
        resolve_equality_method=resolve_method,
        revert_ch_rotE_fac=revert_fac,
        use_sclopf=use_sclopf,
    )

    return


def run_different_parameters_for_co2lvl(
    co2_lvl: Union[float, list],
    nn_nodes: int,
    use_sclopf: bool,
    delta_rot_ls: list,
    max_iter=10000,
    nr_processes: int = 5,
    revert_chrotE_fac: bool = False,
    resolve_equality_method_ls: tuple[str, ...] = (
        "random",
        "concentrate",
        "hindsight",
        "hindsight_concentrate",
    ),
) -> None:
    """Run the function 'run_specific_co2lvl_n_size' for the parameters
    giving delta_rot_energy.

    Args:
        co2_lvl (float): CO2 level
        nn_nodes (int): Number of nodes of PyPSA network.
        delta_rot_ls (list): List with change in rotational energy per step added
        max_iter (int, optional): Number of maximum iterations. Defaults to 10000.
        nr_processes (int, optional): How many processes are being used at the same time. Defaults to 5.
    """

    if isinstance(co2_lvl, float):
        co2_lvl = [co2_lvl]
    resolve_n_deltrotE_ls = list(
        itertools.product(resolve_equality_method_ls, delta_rot_ls, co2_lvl)
    )

    with multiprocessing.get_context("spawn").Pool(processes=nr_processes) as pool:
        partial_func = partial(
            single_call, nn_nodes, use_sclopf, max_iter, revert_chrotE_fac
        )

        [
            xx
            for xx in tqdm(
                pool.imap(partial_func, resolve_n_deltrotE_ls),
                total=len(resolve_n_deltrotE_ls),
            )
        ]

    return
