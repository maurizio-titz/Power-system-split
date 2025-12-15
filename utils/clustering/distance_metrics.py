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
from pyparsing import List
import scipy
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils.clustering.data_handling import split_mask
from utils.config import (
    path_to_clustering_results_sclopf,
)
from utils.indicator_utils import load_indicator_vectors


def product_weighted_hamming_distance(
    node_classes0,
    node_classes1,
    node_weight0,
    node_weight1,
    order: int = 1,
) -> float:
    """calculates the distance between two indicator vectors, from the hamming distance between the two indicator vectors weighted by the product of the node weights."""

    divs = node_classes0 != node_classes1
    return np.linalg.norm(
        np.multiply(node_weight0[divs], node_weight1[divs]), ord=order
    )


def balanced_overlap_distance_weighted(
    node_classes0: np.ndarray,
    node_classes1: np.ndarray,
    node_weights0: np.ndarray,
    node_weights1: np.ndarray,
) -> float:
    """calculates the balanced overlap distance between two indicator vectors, weighted by the sum of the node uncertainties."""

    assert (
        node_classes0.shape == node_classes1.shape
    ), "node_classes0 and node_classes1 must have the same shape"
    assert (
        node_weights0.shape == node_weights1.shape
    ), "node_weights0 and node_weights1 must have the same shape"
    assert (
        node_classes0.shape == node_weights0.shape
    ), "node_classes0 and node_weights0 must have the same shape"

    assert all(node_weights0 >= 0), "node_weights0 must be non-negative"
    assert all(node_weights1 >= 0), "node_weights1 must be non-negative"

    if len(np.unique(np.concatenate([node_classes0, node_classes1]))) == 1:
        return 0
    node_classes0 = node_classes0.astype(bool)
    node_classes1 = node_classes1.astype(bool)

    dist = (
        1
        - (
            (node_classes0 * node_classes1)
            @ (node_weights0 + node_weights1)
            / (
                np.inner(node_classes0, node_weights0)
                + np.inner(node_classes1, node_weights1)
            )
            + (~node_classes0 * ~node_classes1)
            @ (node_weights0 + node_weights1)
            / (
                np.inner(~node_classes0, node_weights0)
                + np.inner(~node_classes1, node_weights1)
            )
        )
        / 2
    )
    if np.isnan(dist):
        raise ValueError("distance is nan")
    return dist


def multiclass_balanced_distance_weighted(
    node_classes0: np.ndarray,
    node_classes1: np.ndarray,
    node_weights0: np.ndarray,
    node_weights1: np.ndarray,
    dtype=np.float16,
) -> float:
    """calculates the balanced overlap distance between two indicator vectors with an arbitrary number of classes, weighted by the sum of the node uncertainties."""

    assert (
        node_classes0.shape == node_classes1.shape
    ), "node_classes0 and node_classes1 must have the same shape"
    assert (
        node_weights0.shape == node_weights1.shape
    ), "node_weights0 and node_weights1 must have the same shape"
    assert (
        node_classes0.shape == node_weights0.shape
    ), "node_classes0 and node_weights0 must have the same shape"

    assert all(node_weights0 >= 0), "node_weights0 must be non-negative"
    assert all(node_weights1 >= 0), "node_weights1 must be non-negative"

    classes = np.unique(np.concatenate([node_classes0, node_classes1]))
    # print(f"classes: {classes}")

    per_class_score = [
        ((node_classes0 == cls) * (node_classes1 == cls))
        @ (node_weights0 + node_weights1)
        / (
            np.inner(node_classes0 == cls, node_weights0)
            + np.inner(node_classes1 == cls, node_weights1)
        )
        for cls in classes
    ]
    # print(per_class_score)

    distance = 1 - sum(per_class_score) / len(classes)

    # raise error if distance is nan
    if np.isnan(distance):
        raise ValueError("distance is nan")

    if dtype is not None:
        distance = dtype(distance)

    return distance


def multiclass_accuracy_distance_weighted(
    node_classes0: np.ndarray,
    node_classes1: np.ndarray,
    node_weights0: np.ndarray,
    node_weights1: np.ndarray,
    dtype=np.float16,
) -> np.float16:
    """calculates the accuracy distance between two indicator vectors with an arbitrary number of classes, weighted by the sum of the node uncertainties."""

    assert (
        node_classes0.shape == node_classes1.shape
    ), "node_classes0 and node_classes1 must have the same shape"
    assert (
        node_weights0.shape == node_weights1.shape
    ), "node_weights0 and node_weights1 must have the same shape"
    assert (
        node_classes0.shape == node_weights0.shape
    ), "node_classes0 and node_weights0 must have the same shape"

    assert all(node_weights0 >= 0), "node_weights0 must be non-negative"
    assert all(node_weights1 >= 0), "node_weights1 must be non-negative"

    distance = 1 - (node_classes0 == node_classes1) @ (
        node_weights0 * node_weights1
    ) / (node_weights0 @ node_weights1)

    # raise error if distance is nan
    if np.isnan(distance):
        raise ValueError("distance is nan")

    if dtype is not None:
        distance = dtype(distance)

    return distance


def balanced_overlap_distance_weighted_pairwise(
    values1: np.ndarray,  # Shape: (n1, n_nodes)
    values2: np.ndarray,  # Shape: (n2, n_nodes)
    weights1: np.ndarray,  # Shape: (n1, n_nodes)
    weights2: np.ndarray,  # Shape: (n2, n_nodes)
) -> np.ndarray:
    """
    Calculate pairwise distances between all vectors in two matrices.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = values1.shape
    n2, _ = values2.shape

    # Convert to boolean for overlap calculations
    classes1 = values1.astype(bool)  # (n1, n_nodes)
    classes2 = values2.astype(bool)  # (n2, n_nodes)

    # Expand dimensions for broadcasting: (n1, 1, n_nodes) and (1, n2, n_nodes)
    classes1_exp = classes1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    classes2_exp = classes2[np.newaxis, :, :]  # (1, n2, n_nodes)
    weights1_exp = weights1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    weights2_exp = weights2[np.newaxis, :, :]  # (1, n2, n_nodes)

    # Calculate overlaps using broadcasting
    both_true = classes1_exp & classes2_exp  # (n1, n2, n_nodes)
    both_false = ~classes1_exp & ~classes2_exp  # (n1, n2, n_nodes)

    # Weight the overlaps
    weighted_both_true = both_true * weights1_exp * weights2_exp  # (n1, n2, n_nodes)
    weighted_both_false = both_false * weights1_exp * weights2_exp  # (n1, n2, n_nodes)

    # Sum across nodes
    overlap_true = weighted_both_true.sum(axis=2)  # (n1, n2)
    overlap_false = weighted_both_false.sum(axis=2)  # (n1, n2)

    # Calculate denominators
    weighted_sum1 = (classes1_exp * weights1_exp).sum(axis=2)  # (n1, n2)
    weighted_sum2 = (classes2_exp * weights2_exp).sum(axis=2)  # (n1, n2)
    weighted_sum_not1 = ((~classes1_exp) * weights1_exp).sum(axis=2)  # (n1, n2)
    weighted_sum_not2 = ((~classes2_exp) * weights2_exp).sum(axis=2)  # (n1, n2)

    # Calculate balanced overlap distance
    true_overlap_normalized = overlap_true / (weighted_sum1 + weighted_sum2 + 1e-10)
    false_overlap_normalized = overlap_false / (
        weighted_sum_not1 + weighted_sum_not2 + 1e-10
    )

    distance = 1 - (true_overlap_normalized + false_overlap_normalized) / 2

    return distance


def bACC_weighted_pairwise(
    values1: np.ndarray,  # Shape: (n1, n_nodes)
    values2: np.ndarray,  # Shape: (n2, n_nodes)
    weights1: np.ndarray,  # Shape: (n1, n_nodes)
    weights2: np.ndarray,  # Shape: (n2, n_nodes)
    dtype=np.float32,
) -> np.ndarray:
    """
    Calculate pairwise multiclass balanced distances.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = values1.shape
    n2, _ = values2.shape

    # Get all unique classes
    all_classes = np.unique(np.concatenate([values1.ravel(), values2.ravel()]))
    n_classes = len(all_classes)

    distances = np.zeros((n1, n2), dtype=dtype)

    # Calculate balanced accuracy for each class and average
    for class_val in all_classes:
        # Create binary masks for current class
        binary1 = (values1 == class_val).astype(dtype)  # (n1, n_nodes)
        binary2 = (values2 == class_val).astype(dtype)  # (n2, n_nodes)

        # Use broadcasting for pairwise calculation
        binary1_exp = binary1[:, np.newaxis, :]  # (n1, 1, n_nodes)
        binary2_exp = binary2[np.newaxis, :, :]  # (1, n2, n_nodes)
        weights1_exp = weights1[:, np.newaxis, :]  # (n1, 1, n_nodes)
        weights2_exp = weights2[np.newaxis, :, :]  # (1, n2, n_nodes)

        # Calculate weighted true positives, false positives, etc.
        agree = (
            ((binary1_exp == 1) & (binary2_exp == 1)) * (weights1_exp + weights2_exp)
        ).sum(axis=2)
        total_weight = (
            ((binary1_exp == 1) * weights1_exp + (binary2_exp == 1) * weights2_exp)
        ).sum(axis=2)

        # Calculate balanced accuracy
        balanced_acc = np.round((agree + 1e-10) / (total_weight + 1e-10), decimals=6)

        # Convert to distance (1 - accuracy)
        distances += 1 - balanced_acc

    # Average across classes
    distances /= n_classes

    return distances


def hamming_distance_weighted_pairwise(
    values1: np.ndarray,  # Shape: (n1, n_nodes)
    values2: np.ndarray,  # Shape: (n2, n_nodes)
    weights1: np.ndarray,  # Shape: (n1, n_nodes)
    weights2: np.ndarray,  # Shape: (n2, n_nodes)
    dtype=np.float32,
) -> np.ndarray:
    """
    Calculate pairwise multiclass balanced distances.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """

    # Use broadcasting for pairwise calculation
    classes1_exp = values1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    classes2_exp = values2[np.newaxis, :, :]  # (1, n2, n_nodes)
    weights1_exp = weights1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    weights2_exp = weights2[np.newaxis, :, :]  # (1, n2, n_nodes)
    agree = (((classes1_exp == classes2_exp)) * (weights1_exp * weights2_exp)).sum(
        axis=2
    )
    total_weight = (weights1_exp * weights2_exp).sum(axis=2)

    # Calculate balanced accuracy
    acc = np.round((agree + 1e-10) / (total_weight + 1e-10), decimals=6)

    # Convert to distance (1 - accuracy)
    distances = 1 - acc

    return distances


def hamming_distance_pairwise(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Hamming distances between all vectors in two matrices.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = values1.shape
    n2, _ = values2.shape

    # Expand dimensions for broadcasting: (n1, 1, n_nodes) and (1, n2, n_nodes)
    classes1_exp = values1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    classes2_exp = values2[np.newaxis, :, :]  # (1, n2, n_nodes)

    # Calculate Hamming distances using broadcasting
    differences = classes1_exp != classes2_exp  # (n1, n2, n_nodes)

    # Sum across nodes to get Hamming distance
    distance_matrix = differences.sum(axis=2) / n_nodes  # (n1, n2)

    return distance_matrix


def class_distance_pairwise(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise class distances between all vectors in two matrices, i.e. the absolute of the difference of the classes as integers. This doubles the cost for over vs underfrequencies.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = values1.shape
    n2, _ = values2.shape

    # Expand dimensions for broadcasting: (n1, 1, n_nodes) and (1, n2, n_nodes)
    classes1_exp = values1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    classes2_exp = values2[np.newaxis, :, :]  # (1, n2, n_nodes)

    # Calculate Hamming distances using broadcasting
    differences = np.abs(classes1_exp - classes2_exp)  # (n1, n2, n_nodes)

    # Sum across nodes to get Hamming distance
    distance_matrix = differences.sum(axis=2) / n_nodes  # (n1, n2)
    distance_matrix = distance_matrix / 2  # normalizing by maximum difference (2)

    return distance_matrix


def balanced_overlap_distance(
    node_classes0: np.ndarray,
    node_classes1: np.ndarray,
) -> float:
    """calculates the balanced overlap distance between two indicator vectors, weighted by the product of the node uncertainties."""

    assert (
        node_classes0.shape == node_classes1.shape
    ), "node_classes0 and node_classes1 must have the same shape"

    node_classes0 = node_classes0.astype(bool)
    node_classes1 = node_classes1.astype(bool)

    return (
        1
        - (
            (node_classes0 * node_classes1).sum()
            / (node_classes0.sum() + node_classes1.sum())
            + (~node_classes0 * ~node_classes1).sum()
            / (~node_classes0.sum() + ~node_classes1.sum())
        )
        / 2
    )


# def graph_based_clustering():
#     https://graph-tool.skewed.de/


########## combining scores ##########
def geometric_mean(data: List[np.ndarray]) -> np.ndarray:
    """Computes the geometric mean of the similarities. However, input featres are distance matrices."""
    data = [1 - mat for mat in data]  # Convert distances to similarities
    product = data[0]
    for mat in data[1:]:
        product = np.multiply(product, mat)
    return 1 - np.power(product, 1 / len(data))  # Convert back to distances


def geometric_mean_2(data: List[np.ndarray]) -> np.ndarray:
    product = data[0]
    for mat in data[1:]:
        product = np.multiply(product, mat)
    return np.sqrt(product)


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2 * (a * b) / (a + b + 1e-10)
