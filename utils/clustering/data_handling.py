import contextlib
import gzip
import os
import pickle
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Optional, Union

import joblib
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils.config import (
    path_to_clustering_results_sclopf,
    path_to_vis_results_sclopf,
)
from utils.indicator_utils import load_indicator_vectors

sys.path.append("./")


def create_string_from_params(params: dict) -> str:
    """Create a string representation of parameters for file naming."""
    param_strs = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            value_str = f"{value:.3f}".replace(".", "p")
        else:
            value_str = str(value)
        param_strs.append(f"{key}{value_str}")
    return "_".join(param_strs)


def transform_indicator_vectors(indicator_vectors, transformation):
    """transforms indicator vectors to allow better clustering

    Args:
        indicator_vectors (np.ndarray): indicator vectors
        transformation (str): name of transformation

    Returns:
        np.ndarray: transformed indicator vectors
    """
    if transformation == "blackout":
        transformed_indicator_vectors = np.zeros_like(indicator_vectors)
        transformed_indicator_vectors[indicator_vectors > 1] = 1
        transformed_indicator_vectors[indicator_vectors < -1] = -1
        transformed_indicator_vectors = transformed_indicator_vectors.astype(int)
    elif transformation == "sign":
        transformed_indicator_vectors = np.sign(indicator_vectors)
    elif transformation == "tanh":
        transformed_indicator_vectors = np.tanh(indicator_vectors)
    # elif transformation == "clipped_tanh":
    #     transformed_indicator_vectors = np.tanh(indicator_vectors)
    #     transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    # elif transformation == "clipped":
    #     transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    # elif transformation == "main_comp_max_lshare":
    #     if indicator_type != "lshare":
    #         raise ValueError(
    #             "is_main_component transformation only works for lshare indicator"
    #         )
    #     transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
    #         indicator_vectors
    #     )
    # elif transformation == "main_comp_most_frequent_lshare":
    #     if indicator_type != "lshare":
    #         raise ValueError(
    #             "is_main_component transformation only works for lshare indicator"
    #         )
    #     transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
    #         indicator_vectors, main_component_by="main_comp_most_frequent_lshare"
    #     )
    # elif transformation == "blackout":
    #     transformed_indicator_vectors = np.array(indicator_vectors < -1, dtype=int)
    # elif transformation == "not_zero":
    #     transformed_indicator_vectors = np.array(indicator_vectors != 0, dtype=int)
    else:
        raise ValueError(f"transformation {transformation} not implemented")

    return transformed_indicator_vectors


def get_split_mask(
    n_nodes,
    co2l_list,
    blackout_size_threshold,
    save_dir,
    split_props=None,
    err_on_missing=False,
    path_to_vis_results=path_to_vis_results_sclopf,
    random_subsample_size=None,
):

    # get masks
    masks_dict_path = save_dir + "/masks_dict.pklz"
    if os.path.exists(masks_dict_path):
        with gzip.open(masks_dict_path, "rb") as fh_in:
            masks_dict = pickle.load(fh_in)
    else:
        if err_on_missing:
            raise FileNotFoundError(
                f"masks_dict_path {masks_dict_path} not found and err_on_missing is True."
            )
        logger.info("Computing masks for CO2 levels...")

        if split_props is None:
            split_props = pd.read_hdf(
                path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
            )
        split_props.lost_load_share_blackout = (
            split_props.lost_load_share_blackout.astype(float)
        )
        split_props = split_props[split_props.co2l.isin(co2l_list)]

        masks_dict = {
            co2l: split_mask_by_blackout_size(
                split_props[split_props.index.get_level_values("co2l") == co2l],
                blackout_size_threshold,
            )
            for co2l in co2l_list
        }

        if (
            isinstance(random_subsample_size, (int, float))
            and random_subsample_size != 1
        ):
            rng = np.random.default_rng(42)
            for co2l in co2l_list:
                selected_indices = masks_dict[co2l].values.nonzero()
                subsample_mask = np.zeros(len(masks_dict[co2l]), dtype=bool)
                n_subsampled = (
                    int(random_subsample_size * len(selected_indices[0]))
                    if random_subsample_size < 1
                    else int(random_subsample_size)
                )
                subsampled_indices = rng.choice(
                    selected_indices[0],
                    size=n_subsampled,
                    replace=False,
                )
                subsample_mask[subsampled_indices] = True
                masks_dict[co2l] = subsample_mask

        with gzip.open(masks_dict_path, "wb") as fh_out:
            pickle.dump(masks_dict, fh_out)
    return masks_dict


def split_mask_by_blackout_size(
    split_properties_df: pd.DataFrame,
    blackout_size_threshold: float,
):
    """create mask for filtering insignificant splits.
    Args:
        split_properties_df (pd.DataFrame): dataframe with split properties
        blackout_size_threshold (float): threshold for blackout size as relative lost load share to filter splits
    """
    index_mask = split_properties_df.lost_load_share_blackout > blackout_size_threshold
    return index_mask


def get_unique_vectors_with_weights(
    row_vectors: np.ndarray, weights: np.ndarray = np.array([])
) -> dict:
    """
    Finds all duplicate row vectors in a binary array and returns their indices as a list of tuples.

    Args:
        row_vectors (np.ndarray): A Nxn array, where N is the number of row vectors and n is the length of each vector.
        weights (np.ndarray): A 1D array of weights corresponding to each row vector.

    Returns:
        dict: A dictionary where keys are unique row vectors (as tuples) and values are lists of indices where these vectors occur in the binary array.
    """
    unique_to_idx_and_weights = OrderedDict()
    if weights.size == row_vectors.shape[0]:
        for idx, (row, weight) in tqdm(enumerate(zip(row_vectors, weights))):
            row_tuple = tuple(row)
            if row_tuple in unique_to_idx_and_weights:
                unique_to_idx_and_weights[row_tuple]["idxs"].append(idx)
                unique_to_idx_and_weights[row_tuple]["weight"] += weight
            else:
                unique_to_idx_and_weights[row_tuple] = {"idxs": [idx], "weight": weight}
    elif weights.size == 0:
        for idx, row in tqdm(enumerate(row_vectors)):
            row_tuple = tuple(row)
            if row_tuple in unique_to_idx_and_weights:
                unique_to_idx_and_weights[row_tuple]["idxs"].append(idx)
            else:
                unique_to_idx_and_weights[row_tuple] = {"idxs": [idx]}
    else:
        raise ValueError(
            "weights must be of the same length as the number of rows in the binary array or empty"
        )

    return unique_to_idx_and_weights


def load_clustering(
    n_nodes=None,
    co2l=None,
    n_clusters=None,
    indicator_type=None,
    load_dir=None,
    transformation=None,
    n_nodes_split=None,
    lost_load_share=None,
    use_sclopf=True,
):
    """load clustering results from zip pickle file. Provide either load_dir or all clustering parameters

    Args:
        n_nodes (int): _description_
        co2l (listlike or float): _description_
        n_clusters (_type_): number of cluster in kmeans
        indicator_type (string): type of indicator vector
        load_dir (_type_): _description_

    Returns:
        (tuple of arrays): labels, centroids, inertia, silhouette_avg
    """

    if load_dir is None:
        load_dir = get_path_to_clustering_dir(
            n_nodes=n_nodes,
            co2l=co2l,
            indicator_type=indicator_type,
            transformation=transformation,
            n_nodes_split=n_nodes_split,
            lost_load_share=lost_load_share,
            use_sclopf=use_sclopf,
        )

    with gzip.open(
        f"{load_dir}/kmeans{n_clusters}.pklz",
        "rb",
    ) as fh_out:
        (
            labels,
            centroids,
            samples_per_centroid,
            centroid_mean_distance,
            inertia,
            silhouette_avg,
        ) = pickle.load(fh_out)
    return (
        labels,
        centroids,
        samples_per_centroid,
        centroid_mean_distance,
        inertia,
        silhouette_avg,
    )


def get_path_to_clustering_dir(
    n_nodes,
    co2l,
    indicator_type,
    transformation,
    lost_load_share,
    n_nodes_split=None,
    use_sclopf=True,
):
    assert n_nodes is not None, "n_nodes must not be None"
    assert co2l is not None, "co2l must not be None"
    assert indicator_type is not None, "indicator_type must not be None"
    assert transformation is not None, "transformation must not be None"
    # assert n_nodes_split is not None, "n_nodes_split must not be None"
    assert lost_load_share is not None, "lost_load_share must not be None"

    if transformation is not None:
        transformation_string = "_" + transformation
    else:
        transformation_string = ""
    if use_sclopf:
        path_to_clustering_results = path_to_clustering_results_sclopf
    else:
        path_to_clustering_results = path_to_clustering_results_lopf
    if n_nodes_split is None:
        n_nodes_split_str = ""
    else:
        n_nodes_split_str = f"_ns{n_nodes_split}"
    if isinstance(co2l, (np.ndarray)):
        co2l = co2l.tolist()
    if isinstance(co2l, (list)):
        co2_string = str(co2l).replace(", ", "_")[1:-1]
    else:
        co2_string = str(co2l)

    return f"{path_to_clustering_results}/{indicator_type}{transformation_string}_Co2L{co2_string}_n{n_nodes}{n_nodes_split_str}_lls{lost_load_share}/"


def split_mask(
    split_properties_df: pd.DataFrame,
    lost_load_share: float,
    n_nodes: int = None,
    ignore_shedding: bool = True,
):
    """create mask for filtering insignificant splits

    Args:
        n_nodes (int): minimal number of nodes in split-off component for split to be considered significant
        lost_load_share (float): minimal lost load share due to RoCoF and shedding for split to be considered significant
    """
    if not n_nodes is None:
        raise NotImplementedError

    if ignore_shedding:
        # index_mask = split_properties_df.lost_load_share_rocof > lost_load_share
        index_mask = split_properties_df.lost_load_share_blackout > lost_load_share
    else:
        index_mask = split_properties_df.lost_load_share_total > lost_load_share

    return np.array(index_mask)
