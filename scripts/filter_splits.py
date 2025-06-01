import sys

sys.path.append("./")
import gzip
import pickle
from itertools import compress

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import data_handling
from utils.config import (
    path_to_evaluation_results_sclopf,
    path_to_indicator_vectors_sclopf,
    path_to_pypsa_network_sclopf,
)


def add_lost_load_to_component_props(component_props):
    """takes component properties and adds columns for lost load due to RoCoF and shedding

    Args:
        component_props (pd.Dataframe): component properties dataframe

    Returns:
        pd.Datafrage: updated properties dataframe
    """
    component_props.time_stamp = pd.to_datetime(component_props.time_stamp)
    component_props = component_props.astype(
        {
            "init_failure_0": "int32",
            "init_failure_1": "int32",
            "split_number": "int32",
        }
    )

    component_props["lost_load_shedding"] = component_props.power_imbalance
    component_props.loc[
        component_props.lost_load_shedding > 0, "lost_load_shedding"
    ] = 0
    component_props.lost_load_shedding = component_props.lost_load_shedding * -1

    component_props["lost_load_rocof"] = 0.0
    component_props.loc[component_props.rocof < -1, "lost_load_rocof"] = (
        component_props.load[component_props.rocof < -1]
    )

    component_props["lost_load_total"] = component_props.loc[
        :, ["lost_load_shedding", "lost_load_rocof"]
    ].max(axis=1)

    return component_props


def create_lshare_df(index_tuple_time_split, indicator_vector_lshare):
    indicator_lshare_df = pd.DataFrame(index=range(len(index_tuple_time_split)))
    indicator_lshare_df["time_stamp"] = [
        index_tuple_time_split[i][0] for i in range(len(index_tuple_time_split))
    ]
    indicator_lshare_df["init_failure_0"] = [
        index_tuple_time_split[i][1][0] for i in range(len(index_tuple_time_split))
    ]
    indicator_lshare_df["init_failure_1"] = [
        index_tuple_time_split[i][1][1] for i in range(len(index_tuple_time_split))
    ]
    indicator_lshare_df["split_number"] = [
        index_tuple_time_split[i][2] for i in range(len(index_tuple_time_split))
    ]
    indicator_lshare_df["indicator_vector_lshare"] = [
        indicator_vector_lshare[i] for i in range(len(index_tuple_time_split))
    ]

    return indicator_lshare_df


def split_mask(
    split_properties_df: pd.DataFrame,
    lost_load_share: float,
    n_nodes:int=None,
    ignore_shedding: bool = False,
):
    """create mask for filtering insignificant splits

    Args:
        n_nodes (int): minimal number of nodes in split-off component for split to be considered significant
        lost_load_share (float): minimal lost load share due to RoCoF and shedding for split to be considered significant
    """
    if not n_nodes is None:
        raise NotImplementedError

    # if ignore_shedding:
    #     index_mask = (split_properties_df.n_nodes_split_off > n_nodes) | (
    #         split_properties_df.lost_load_share_total > lost_load_share
    #     )
    # else:
    #     index_mask = (split_properties_df.n_nodes_split_off > n_nodes) | (
    #         split_properties_df.lost_load_rocof_share > lost_load_share
    #     )

    if ignore_shedding:
        index_mask = split_properties_df.lost_load_share_total > lost_load_share
    else:
        index_mask = split_properties_df.lost_load_rocof_share > lost_load_share
    
    return np.array(index_mask)


def calc_split_props(component_props, indicator_lshare_df, co2l, n_nodes=400):
    """calculate properties of splits from component properties and indicator vectors.

    Args:
        component_props (pd.DataFrame): _description_
        indicator_lshare_df (np.ndarray): _description_
        co2l (float): _description_

    Returns:
        pd.DataFrame: split properties containing loss of load etc.
    """

    split_properties = {}

    ind = 0
    max_ind_current_split = 0

    pbar = tqdm(total=len(component_props) + 1)

    while len(component_props) > 0:
        # if ind > run_n_splits:
        #     break

        current_split_number = component_props.split_number.iloc[0]
        current_time_stamp = component_props.time_stamp.iloc[0]

        max_ind_current_split = 0
        for i in range(len(component_props)):
            if component_props.split_number.iloc[i] == current_split_number:
                max_ind_current_split = i
            else:
                break

        pbar.update(max_ind_current_split + 1)

        current_component_props = component_props.iloc[: max_ind_current_split + 1]
        current_initial_failures = (
            current_component_props.init_failure_0.values[0],
            current_component_props.init_failure_1.values[0],
        )

        total_load = current_component_props.load.sum()
        lost_load_total = current_component_props.lost_load_total.sum()
        lost_load_rocof = current_component_props.lost_load_rocof.sum()
        lost_load_shedding = current_component_props.lost_load_shedding.sum()
        largest_component_load = current_component_props.load.max()

        # get number of nodes that split from main component
        lshare_vec = indicator_lshare_df.loc[ind, "indicator_vector_lshare"]
        n_nodes_split_off = sum(lshare_vec != current_component_props.load_share.max())
        n_components = len(current_component_props)

        split_properties[ind] = {
            "time_stamp": current_time_stamp,
            "init_failures_0": current_initial_failures[0],
            "init_failures_1": current_initial_failures[1],
            "split_number": current_split_number,
            "lost_load_rocof": lost_load_rocof,
            "total_load": total_load,
            "lost_load_shedding": lost_load_shedding,
            "lost_load_total": lost_load_total,
            "largest_component_load": largest_component_load,
            "n_components": n_components,
            "n_nodes_split_off": n_nodes_split_off,
        }

        # remove first max_ind_current_split rows of component_props
        component_props = component_props.iloc[max_ind_current_split + 1 :]
        ind += 1

    split_properties_df = pd.DataFrame.from_dict(split_properties, orient="index")
    split_properties_df["lost_load_rocof_share"] = (
        split_properties_df.lost_load_rocof / split_properties_df.total_load
    )
    split_properties_df["lost_load_shedding_share"] = (
        split_properties_df.lost_load_shedding / split_properties_df.total_load
    )
    split_properties_df["lost_load_share_total"] = (
        split_properties_df.lost_load_total / split_properties_df.total_load
    )
    split_properties_df["co2l"] = co2l

    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf + "sclopf-elec_s_400_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    split_properties_df["snapshot_weighting"] = (
        network.snapshot_weightings.generators.loc[
            split_properties_df.time_stamp
        ].values
    )

    split_properties_df.to_csv(
        path_to_evaluation_results_sclopf
        + f"split_properties_Co2L{co2l}_n{n_nodes}.csv"
    )

    pbar.close()


def split_mask_component_vectors(
    index_mask, split_ind_to_component_ind, n_component_vectors
):
    """project index_mask to component wise indicator vector indices

    Args:
        index_mask (array): mask for filtering insignificant splits
        split_ind_to_component_ind (array): keys are split indices, values are component indices
        n_component_vectors (int): number of component indicator vectors

    Returns:
        np.array: mask for component indicator vectors
    """

    component_mask = np.zeros(n_component_vectors, dtype=bool)
    selected_split_indices = np.where(index_mask)[0]
    print("selected_split_indices")
    print(selected_split_indices.shape)
    print(selected_split_indices[:3])
    print("split_ind_to_component_ind")
    print(type(split_ind_to_component_ind))
    print(split_ind_to_component_ind[:3])
    selected_split_ind_to_component_ind = list(
        compress(split_ind_to_component_ind, index_mask)
    )
    selected_split_ind_to_component_ind = [
        x for xs in selected_split_ind_to_component_ind for x in xs
    ]
    component_mask[selected_split_ind_to_component_ind] = True

    return component_mask


n_nodes_split, lost_load_share = 5, 0.005

if __name__ == "__main__":
    run_n_splits = 5000

    n_nodes = 400
    co2l_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for co2l in [0.0]:
        # for co2l in co2l_list[::-1]:
        print(f"co2l: {co2l}")
        # load component_props
        component_props = pd.read_hdf(
            path_to_evaluation_results_sclopf
            + f"component_properties_Co2L{co2l}_n{n_nodes}.h5",
            key="df",
            #    , mode= 'w'
        )

        component_props = add_lost_load_to_component_props(component_props)

        # load indicator vectors
        file_name = f"indicator_vector_lshare_Co2L{co2l}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors_sclopf + "/" + file_name, "rb") as out:
            index_tuple_time_split, indicator_vector_lshare = pickle.load(out)

        indicator_lshare_df = create_lshare_df(
            index_tuple_time_split, indicator_vector_lshare
        )

        calc_split_props(component_props, indicator_lshare_df, co2l)
