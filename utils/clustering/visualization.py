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


def prepare_clusters_for_analysis(
    clustering_res_path,
    unique_blackout_dict,
    split_properties_filtered,
    overwrite=False,
):
    """Prepares clustering results for analysis by saving labels and group masks.
    Args:
        clustering_res_path (str): Path to the clustering results file.
        unique_blackout_dict (dict): Dictionary of unique blackout vectors.
        split_properties_filtered (np.ndarray): Filtered properties of the split.
        overwrite (bool): Whether to overwrite existing files.
    Returns:
        None.
        Saves:
        labels_all, attributes each outage vector to cluster.
        group_masks, dictonary which holds a mask for each cluster.
    """

    path_labels_all = clustering_res_path.replace("fitted.pklz", "labels_all.npy")
    path_goup_masks = clustering_res_path.replace("fitted.pklz", "group_masks.pklz")
    # path_group_means = clustering_res_path.replace(".pklz", "_group_means.npy")

    if (
        os.path.exists(path_labels_all)
        and os.path.exists(path_goup_masks)
        and not overwrite
    ):
        print(f"Skipping because labels_all, goup_masks and group_means already exist.")
        return

    print("preparing clusters for analysis...")

    with gzip.open(clustering_res_path, "rb") as out:
        clustering_res = pickle.load(out)

    unique_idx_to_ids = [val["idxs"] for val in unique_blackout_dict.values()]
    unique_blackout_vecs = np.array(list(unique_blackout_dict.keys()))

    labels = clustering_res.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[clustering_res.core_sample_indices_] = True

    class_member_masks = [labels == k for k in unique_labels]
    # group_means = np.array(
    #     [
    #         np.mean(unique_blackout_vecs[labels == k], axis=0)
    #         for k in unique_labels
    #         if k != -1
    #     ]
    # )
    label_to_idxs = {
        k: np.concatenate(
            [
                unique_idx_to_ids[i]
                for i in range(len(unique_idx_to_ids))
                if labels[i] == k
            ]
        )
        for k in unique_labels
    }

    # all_idxs = list(np.concatenate(list(groups.values())))
    # all_idxs.sort()
    labels_all = np.array([None] * split_properties_filtered.shape[0])

    for label, idxs in label_to_idxs.items():
        assert all(
            labels_all[idxs] == None
        ), "labels_all[idxs] must be None before assigning a label. This indicates that the same index is assigned to multiple labels."
        labels_all[idxs] = label

    np.save(path_labels_all, labels_all)

    group_masks = {k: np.array(labels_all == k) for k in unique_labels}
    with gzip.open(path_goup_masks, "wb") as out:
        pickle.dump(group_masks, out)
