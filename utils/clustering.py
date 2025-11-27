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

from scripts.filter_splits import split_mask
from utils.config import (
    path_to_clustering_results_lopf,
    path_to_clustering_results_sclopf,
)
from utils.indicator_utils import load_indicator_vectors

sys.path.append("./")


def filter_splits_events(
    use_sclopf,
    path_to_indicator_vectors,
    path_to_vis_results,
    n_nodes,
    min_lost_load_share,
    co2l_list,
    indicator_type,
    transformation,
    save_dir,
    split_props=None,
    err_on_missing=False,
):
    os.makedirs(save_dir, exist_ok=True)
    logger_num = logger.add(f"{save_dir}/log.txt")

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
            co2l: split_mask(
                split_props[split_props.index.get_level_values("co2l") == co2l],
                min_lost_load_share,
                ignore_shedding=True,
            )
            for co2l in co2l_list
        }

        with gzip.open(masks_dict_path, "wb") as fh_out:
            pickle.dump(masks_dict, fh_out)

    save_path_unique_vecs = save_dir + f"/unique_blackout_vecs_n{n_nodes}.pklz"
    if os.path.exists(save_path_unique_vecs):
        with gzip.open(save_path_unique_vecs, "rb") as fh_in:
            unique_vecs_dict = pickle.load(fh_in)
        split_props_filtered = pd.read_hdf(
            f"{save_dir}/data_filtered_{n_nodes}.h5", key="split_props"
        )
    else:
        if err_on_missing:
            raise FileNotFoundError(
                f"save_path_unique_vecs {save_path_unique_vecs} not found and err_on_missing is True."
            )
        logger.info("Loading indicator vectors and computing unique vectors...")

        if split_props is None:
            split_props = pd.read_hdf(
                path_to_vis_results + f"split_properties_all_n{n_nodes}.h5"
            )

        (
            blackout_vectors_filtered_dict,
            weights_filtered_dict,
            split_props_filtered,
        ) = load_indicator_vectors(
            n_nodes,
            indicator_type,
            transformation,
            path_to_indicator_vectors=path_to_indicator_vectors,
            mask=masks_dict,
            split_props=split_props,
            co2_list=co2l_list,
        )

        # Save intermediate results
        with gzip.open(f"{save_dir}/weights_filtered_dict.pklz", "wb") as fh_out:
            pickle.dump(weights_filtered_dict, fh_out)
        split_props_filtered.to_hdf(
            f"{save_dir}/data_filtered_{n_nodes}.h5", key="split_props"
        )
        with gzip.open(
            f"{save_dir}/blackout_vectors_filtered_dict_{n_nodes}.pklz", "wb"
        ) as fh_out:
            pickle.dump(blackout_vectors_filtered_dict, fh_out)
        with gzip.open(
            f"{save_dir}/weights_filtered_dict_{n_nodes}.pklz", "wb"
        ) as fh_out:
            pickle.dump(weights_filtered_dict, fh_out)

        blackout_vectors_filtered = np.concatenate(
            list(blackout_vectors_filtered_dict.values())
        )
        weights_filtered = np.concatenate(list(weights_filtered_dict.values()))

        unique_vecs_dict = get_unique_vectors_with_weights(
            blackout_vectors_filtered, weights_filtered
        )

        with gzip.open(save_path_unique_vecs, "wb") as fh_out:
            pickle.dump(unique_vecs_dict, fh_out)

    unique_vecs = np.array(list(unique_vecs_dict.keys()))
    weights_filtered = [d["weight"] for d in unique_vecs_dict.values()]
    return unique_vecs, split_props_filtered

    # with open(path_group_means, "wb") as out:
    # np.save(out, group_means)
