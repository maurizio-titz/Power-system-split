import contextlib
import gzip
import os
import pickle
import sys
from typing import Callable, Union
from joblib import Parallel, delayed
import joblib
import time

from loguru import logger
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime, timedelta

from scripts.filter_splits import split_mask
from utils.config import (
    path_to_clustering_results_lopf,
    path_to_clustering_results_sclopf,
)
from utils.indicator_utils import load_indicator_vectors


sys.path.append("./")


def typed_katz_centrality_single_vector(
    adjecency_matrix: np.matrix,
    node_class_vectors: np.ndarray,
    decay_factor: float = 0.1,
    max_distance: int = 10,
) -> np.ndarray:
    """calculates the nodewise uncertainty, i.e. how close the node is to nodes of the opposite class."""
    if set(node_class_vectors) == {0, 1}:
        node_class_vectors = 2 * node_class_vectors - 1
    elif set(node_class_vectors) == {-1, 1}:
        pass
    else:
        raise ValueError("node_class_vectors must be binary or -1,1")

    c = np.sum(
        np.array(
            [
                np.linalg.matrix_power(adjecency_matrix, d)
                @ node_class_vectors
                * decay_factor**d
                for d in range(1, max_distance + 1)
            ]
        ),
        axis=0,
    )
    assert c.shape == node_class_vectors.shape, "c.shape != node_class_vectors.shape"

    return c


def typed_katz_centrality_batch(
    adjacency_matrix: np.matrix,
    node_class_vectors: np.ndarray,
    decay_factor: float = 1.5,
    max_distance: int = 10,
) -> np.ndarray:
    """calculates the nodewise uncertainty, i.e. how close the node is to nodes of the opposite class."""
    if set(np.unique(node_class_vectors)) == {0, 1}:
        node_class_vectors = 2 * node_class_vectors - 1
    elif set(node_class_vectors) == {-1, 1}:
        pass
    else:
        raise ValueError("node_class_vectors must be binary, either [0,1] or [-1,1]")

    if scipy.sparse.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.toarray()

    c = np.sum(
        [
            node_class_vectors
            @ normalize(np.linalg.matrix_power(adjacency_matrix, d), axis=0, norm="l1")
            * d ** (-decay_factor)
            for d in range(
                1, max_distance + 1
            )  # +1 because 0 would be the identity matrix
        ],
        axis=0,
    )

    assert c.shape == node_class_vectors.shape, "c.shape != node_class_vectors.shape"

    return abs(c)
    # return c


def neighborhood_homo_batch(
    adjacency_matrix: Union[np.matrix, sparse.csr_matrix],
    node_class_vectors: np.ndarray,
    decay_factor: float = 1.5,
    max_distance: int = 10,
    impurity_type: str = "normalized_shannon",
    neg_val_tolerance: float = 1e-6,
) -> np.ndarray:
    """calculates the nodewise neighbourhood homogeneity, i.e. how close the node is to nodes of the opposite class.
    adjacency_matrix: adjacency matrix of the graph
    node_class_vectors_one_hot: one-hot encoded node class vectors, shape (n_instances, n_nodes, n_classes)
    decay_factor: decay factor for the Katz centrality
    max_distance: maximum distance to consider
    impurity_type: type of impurity to calculate, either "normalized_shannon" or "gini"
    neg_val_tolerance: tolerance for negative impurity values due to numerical errors
    returns: nodewise impurity values, shape (n_instances, n_nodes)
    """

    if len(node_class_vectors.shape) == 3:
        raise NotImplementedError(
            "neighborhood_impurity_batch does not support one hote encoding of node class vectors"
        )
        n_classes = node_class_vectors.shape[2]
        if n_classes > 1:
            node_class_vectors_one_hot = node_class_vectors
        else:
            raise ValueError(
                "node_class_vectors seems to be one-hot encoded but has only one class"
            )
    else:
        classes = np.sort(np.unique(node_class_vectors))
        n_classes = len(classes)
        if n_classes > 1:
            # one-hot encode the node class vectors
            node_class_vectors_one_hot = np.stack(
                [(node_class_vectors == cls).astype(int) for cls in classes],
                axis=0,
            )
        else:
            raise ValueError(
                "node_class_vectors is not one-hot encoded but has only one unique value"
            )

    print(f"{n_classes} classes found in node_class_vectors: {classes}")

    if scipy.sparse.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.toarray()

    c = np.sum(
        [
            node_class_vectors_one_hot
            @ normalize(np.linalg.matrix_power(adjacency_matrix, d), axis=0, norm="l1")
            * d ** (-decay_factor)
            for d in range(
                1, max_distance + 1
            )  # +1 because 0 would be the identity matrix
        ],
        axis=0,
    )
    assert not np.isnan(np.sum(c)), "nan values encountered in impurity calculation"

    c = c / (
        np.sum(c, axis=0, keepdims=True)
    )  # normalize across classes to get probabilities

    assert (
        c.shape == node_class_vectors_one_hot.shape
    ), "c.shape != node_class_vectors.shape"

    if impurity_type == "normalized_shannon":
        c = -np.sum(c * np.log(c + 1e-10), axis=0) / np.log(
            n_classes
        )  # high values mean high heterogeneity
    elif impurity_type == "gini":
        c = 1 - np.sum(c**2, axis=0)  # high values mean high heterogeneity

    c = 1 - c  # convert to homogeneity: high values mean high homogeneity

    if np.min(c) < 0:
        if np.min(c) > -neg_val_tolerance:
            c = np.clip(c, 0, 1)
        else:
            raise ValueError(
                f"negative impurity values encountered, smallest value: {np.min(c)}"
            )
    assert np.isnan(c).sum() == 0, "nan values encountered in impurity calculation"
    return c


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


def uncertainty_distance_wrapper_tuple(
    blackout_centrality_tuple_0,
    blackout_centrality_tuple_1,
    order: int = 1,
) -> float:
    """
    calculates the distance between two indicator vectors. takes a tuple of the node classes and uncertainties for each samples, so it can be used as a distance function in the clustering algorithm.
    """

    node_classes0 = blackout_centrality_tuple_0[0]
    node_uncertainties0 = blackout_centrality_tuple_0[1]
    node_classes1 = blackout_centrality_tuple_1[0]
    node_uncertainties1 = blackout_centrality_tuple_1[1]

    return product_weighted_hamming_distance(
        node_classes0,
        node_classes1,
        node_uncertainties0,
        node_uncertainties1,
        order=order,
    )


def weighted_distance_wrapper(
    blackout_centrality_tuple_0: np.ndarray,
    blackout_centrality_tuple_1: np.ndarray,
    weighted_distance_metric: Callable,
    kwargs: dict = {},
) -> float:
    """
    wraps product_weighted_hamming_distance so it can be used as a distance metric. takes two tuples of blackout centrality vectors, each containing node classes and node weights concatenated.
    """
    assert (
        blackout_centrality_tuple_0.shape == blackout_centrality_tuple_1.shape
    ), "blackout_centrality_tuple_0 and blackout_centrality_tuple_1 must have the same shape"
    assert blackout_centrality_tuple_0.shape[0] % 2 == 0
    n_nodes = int(blackout_centrality_tuple_0.shape[0] / 2)

    node_classes0 = blackout_centrality_tuple_0[:n_nodes]
    node_weights0 = blackout_centrality_tuple_0[n_nodes:]
    node_classes1 = blackout_centrality_tuple_1[:n_nodes]
    node_weights1 = blackout_centrality_tuple_1[n_nodes:]

    return weighted_distance_metric(
        node_classes0,
        node_classes1,
        node_weights0,
        node_weights1,
        **kwargs,
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
    node_classes_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
    node_classes_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
    node_weights_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
    node_weights_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
) -> np.ndarray:
    """
    Calculate pairwise distances between all vectors in two matrices.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = node_classes_matrix1.shape
    n2, _ = node_classes_matrix2.shape

    # Convert to boolean for overlap calculations
    classes1 = node_classes_matrix1.astype(bool)  # (n1, n_nodes)
    classes2 = node_classes_matrix2.astype(bool)  # (n2, n_nodes)

    # Expand dimensions for broadcasting: (n1, 1, n_nodes) and (1, n2, n_nodes)
    classes1_exp = classes1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    classes2_exp = classes2[np.newaxis, :, :]  # (1, n2, n_nodes)
    weights1_exp = node_weights_matrix1[:, np.newaxis, :]  # (n1, 1, n_nodes)
    weights2_exp = node_weights_matrix2[np.newaxis, :, :]  # (1, n2, n_nodes)

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


def multiclass_balanced_distance_weighted_pairwise(
    node_classes_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
    node_classes_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
    node_weights_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
    node_weights_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
    dtype=np.float32,
) -> np.ndarray:
    """
    Calculate pairwise multiclass balanced distances.

    Returns:
        np.ndarray: Distance matrix of shape (n1, n2)
    """
    n1, n_nodes = node_classes_matrix1.shape
    n2, _ = node_classes_matrix2.shape

    # Get all unique classes
    all_classes = np.unique(
        np.concatenate([node_classes_matrix1.ravel(), node_classes_matrix2.ravel()])
    )
    n_classes = len(all_classes)

    distances = np.zeros((n1, n2), dtype=dtype)

    # Calculate balanced accuracy for each class and average
    for class_val in all_classes:
        # Create binary masks for current class
        binary1 = (node_classes_matrix1 == class_val).astype(dtype)  # (n1, n_nodes)
        binary2 = (node_classes_matrix2 == class_val).astype(dtype)  # (n2, n_nodes)

        # Use broadcasting for pairwise calculation
        binary1_exp = binary1[:, np.newaxis, :]  # (n1, 1, n_nodes)
        binary2_exp = binary2[np.newaxis, :, :]  # (1, n2, n_nodes)
        weights1_exp = node_weights_matrix1[:, np.newaxis, :]  # (n1, 1, n_nodes)
        weights2_exp = node_weights_matrix2[np.newaxis, :, :]  # (1, n2, n_nodes)

        # Calculate weighted true positives, false positives, etc.
        tp = (
            ((binary1_exp == 1) & (binary2_exp == 1)) * weights1_exp * weights2_exp
        ).sum(axis=2)
        tn = (
            ((binary1_exp == 0) & (binary2_exp == 0)) * weights1_exp * weights2_exp
        ).sum(axis=2)
        fp = (
            ((binary1_exp == 0) & (binary2_exp == 1)) * weights1_exp * weights2_exp
        ).sum(axis=2)
        fn = (
            ((binary1_exp == 1) & (binary2_exp == 0)) * weights1_exp * weights2_exp
        ).sum(axis=2)

        # Calculate balanced accuracy
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        balanced_acc = (sensitivity + specificity) / 2

        # Convert to distance (1 - accuracy)
        distances += 1 - balanced_acc

    # Average across classes
    distances /= n_classes

    return distances


def compute_distance_matrix_vectorized(
    data_matrix: np.ndarray,  # Shape: (n_samples, n_nodes)
    weights_matrix: np.ndarray,  # Shape: (n_samples, n_nodes)
    metric_func,
    chunk_size: int = 1000,
    test_mode: bool = False,
) -> np.ndarray:
    """
    Compute distance matrix using vectorized operations with chunking to manage memory.

    Returns:
        np.ndarray: Distance matrix of shape (n_samples, n_samples)
    """
    if test_mode:
        data_matrix = data_matrix[:10000, :]
        weights_matrix = weights_matrix[:10000, :]
    n_samples = data_matrix.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    pbar = tqdm(total=(n_samples * (n_samples - 1)) // 2, desc="Distance Calculation")
    # Process in chunks to manage memory
    for i in range(0, n_samples, chunk_size):
        for j in range(i, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            j_end = min(j + chunk_size, n_samples)

            # Extract chunks
            chunk1_data = data_matrix[i:i_end]
            chunk1_weights = weights_matrix[i:i_end]
            chunk2_data = data_matrix[j:j_end]
            chunk2_weights = weights_matrix[j:j_end]

            # Calculate pairwise distances for this chunk
            chunk_distances = metric_func(
                chunk1_data, chunk2_data, chunk1_weights, chunk2_weights
            )

            # Fill the distance matrix
            distance_matrix[i:i_end, j:j_end] = chunk_distances

            # Fill symmetric part (unless it's the diagonal chunk)
            if i != j:
                distance_matrix[j:j_end, i:i_end] = chunk_distances.T
                pbar.update((i_end - i) * (j_end - j))

    pbar.close()
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


def get_order_by_blackout_size(indicator_vectors: np.ndarray):
    """returns the order of the indicator vectors by blackout size"""

    indicator_vectors = np.atleast_2d(indicator_vectors)
    blackout_sizes = indicator_vectors.sum(axis=1)
    # sort the blackout sizes in descending order
    order = np.argsort(blackout_sizes)[::-1]
    indicator_vectors_sorted = indicator_vectors[order]
    blackout_sizes_sorted = blackout_sizes[order]

    return blackout_sizes_sorted, indicator_vectors_sorted, order


def calc_distance_matrix(
    data,
    mode: str = "sequential",
    metric=balanced_overlap_distance_weighted,
    n_jobs=-1,
    show_progress=True,
    test_mode=True,
):
    """Calculate the distance matrix."""
    if mode == "joblib":
        return calc_distance_matrix_memory_efficient(
            data,
            metric=metric,
            n_jobs=n_jobs,
            show_progress=show_progress,
            test_mode=test_mode,
        )
    elif mode == "sequential":
        return calc_distance_matrix_sequential(data, metric=metric, test_mode=test_mode)
    elif mode == "vectorized":
        # split data into classes and weights
        n_nodes = int(data.shape[1] / 2)
        node_classes_matrix = data[:, :n_nodes]
        node_weights_matrix = data[:, n_nodes:]
        return compute_distance_matrix_vectorized(
            node_classes_matrix,
            node_weights_matrix,
            multiclass_balanced_distance_weighted_pairwise,
            chunk_size=500,
            test_mode=test_mode,
        )
    else:
        raise ValueError(f"Unknown distance matrix calculation mode: {mode}")


def calc_distance_matrix_sequential(
    data,
    metric=balanced_overlap_distance_weighted,
    test_mode=False,
):
    """
    Calculate the distance matrix for the given data using the specified metric.

    Parameters:
        data (np.ndarray): The input data for which to calculate distances.
        metric (callable): The distance metric to use.

    Returns:
        np.ndarray: The calculated distance matrix.
    """
    n_samples = data.shape[0]
    if test_mode:
        data = data[:100]  # Limit to first 100 samples for testing

    d = np.zeros((data.shape[0], data.shape[0]), dtype=np.float64)
    with tqdm(
        total=(n_samples * (n_samples - 1)) // 2, desc="Distance Calculation"
    ) as pbar:
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                d[i, j] = metric(data[i], data[j])
                # Store the distance in the appropriate place in the matrix
                d[j, i] = d[i, j]
                pbar.update(1)

    np.fill_diagonal(d, 0)  # Set diagonal to zero
    return d


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def calc_distance_matrix_joblib(
    data,
    metric=balanced_overlap_distance_weighted,
    n_jobs=-1,
    show_progress=True,
    test_mode=True,
):
    """
    Calculate the distance matrix using joblib parallelization.

    Parameters:
        data (np.ndarray): The input data for which to calculate distances.
        metric (callable): The distance metric to use.
        n_jobs (int): The number of parallel jobs to run. Default is -1 (use all available cores).

    Returns:
        np.ndarray: The calculated distance matrix.
    """
    n_samples = data.shape[0]

    def compute_distance(i, j):
        return i, j, metric(data[i], data[j])

    # measure total passed time
    t_now = time.time()
    human_time = datetime.fromtimestamp(t_now).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"Starting distance matrix calculation for {n_samples} samples at {human_time}"
    )

    # Generate all (i,j) pairs where i < j
    pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
    if test_mode:
        n_test_size = 100000
        print(
            f"Test mode: limiting to {n_test_size/(n_samples*(n_samples-1)//2)*100:.1f}% of all pairs"
        )
        pairs = pairs[:n_test_size]  # Limit to first 100 pairs for testing

    # Parallel computation
    if show_progress:
        with tqdm_joblib(
            tqdm(desc="Distance Calculation", total=len(pairs))
        ) as progress_bar:
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(compute_distance)(i, j) for i, j in pairs
            )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(compute_distance)(i, j) for i, j in pairs
        )

    # Fill the distance matrix
    d = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i, j, distance in results:
        d[i, j] = distance
        d[j, i] = distance
    t_end = time.time()
    time_passed = t_end - t_now
    human_time_end = datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S")
    time_str = str(timedelta(seconds=time_passed))
    print(
        f"Finished distance matrix calculation at {human_time_end}, total time: {time_str}"
    )

    np.fill_diagonal(d, 0)
    return d


def compute_distance_chunk(data, pairs_chunk, metric):
    """Compute distances for a chunk of pairs - data is only serialized once per chunk"""
    results = []
    for i, j in pairs_chunk:
        distance = metric(data[i], data[j])
        results.append((i, j, distance))
    return results


def calc_distance_matrix_joblib_chunked(
    data,
    metric=balanced_overlap_distance_weighted,
    n_jobs=-1,
    show_progress=True,
    test_mode=True,
):
    """
    Calculate distance matrix with chunked processing - reduces process overhead.
    """
    n_samples = data.shape[0]

    # measure total passed time
    t_now = time.time()
    human_time = datetime.fromtimestamp(t_now).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"Starting chunked distance matrix calculation for {n_samples} samples at {human_time}"
    )

    # Generate all (i,j) pairs where i < j
    pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
    if test_mode:
        n_test_size = 100000
        print(
            f"Test mode: limiting to {n_test_size/(n_samples*(n_samples-1)//2)*100:.1f}% of all pairs"
        )
        pairs = pairs[:n_test_size]

    chunk_size = len(pairs) // (n_jobs * 4) + 1

    # Split pairs into chunks - this is the key optimization
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    print(f"Processing {len(pairs)} pairs in {len(chunks)} chunks of size {chunk_size}")

    # Parallel computation with chunked processing
    if show_progress:
        with tqdm_joblib(
            tqdm(desc="Distance Calculation", total=len(chunks))
        ) as progress_bar:
            chunk_results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(compute_distance_chunk)(data, chunk, metric) for chunk in chunks
            )
    else:
        chunk_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(compute_distance_chunk)(data, chunk, metric) for chunk in chunks
        )

    # Flatten results
    results = [result for chunk_result in chunk_results for result in chunk_result]

    # Fill the distance matrix
    d = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i, j, distance in results:
        d[i, j] = distance
        d[j, i] = distance

    t_end = time.time()
    time_passed = t_end - t_now
    human_time_end = datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S")
    time_str = str(timedelta(seconds=time_passed))
    print(
        f"Finished distance matrix calculation at {human_time_end}, total time: {time_str}"
    )

    np.fill_diagonal(d, 0)
    return d


def generate_chunks_on_demand(n_samples, chunk_size, test_mode=False):
    """Generator that yields chunks without storing them"""
    pairs_generated = 0
    current_chunk = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if test_mode and pairs_generated >= 100000:
                break

            current_chunk.append((i, j))
            pairs_generated += 1

            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

        if test_mode and pairs_generated >= 100000:
            break

    if current_chunk:
        yield current_chunk


def calc_distance_matrix_memory_efficient(
    data,
    metric=balanced_overlap_distance_weighted,
    n_jobs=-1,
    show_progress=True,
    test_mode=True,
    n_chunks=8,
):
    """Ultra memory-efficient version"""
    n_samples = data.shape[0]

    if n_chunks is None:
        n_cores = n_jobs if n_jobs > 0 else os.cpu_count()
        n_chunks = n_cores * 5

    total_pairs = n_samples * (n_samples - 1) // 2
    if test_mode:
        total_pairs = min(100000, total_pairs)

    chunk_size = max(1, total_pairs // n_chunks)

    # Process chunks one by one to minimize memory
    d = np.zeros((n_samples, n_samples), dtype=np.float64)

    chunk_generator = generate_chunks_on_demand(n_samples, chunk_size, test_mode)

    if show_progress:
        chunk_generator = tqdm(
            chunk_generator, total=n_chunks, desc="Processing chunks"
        )

    # Process chunks sequentially but with parallel distance computation within each chunk
    for chunk in chunk_generator:
        chunk_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(lambda pair: (*pair, metric(data[pair[0]], data[pair[1]])))(pair)
            for pair in chunk
        )

        for i, j, distance in chunk_results:
            d[i, j] = distance
            d[j, i] = distance

    np.fill_diagonal(d, 0)
    return d


def get_unique_vectors_with_weights(
    binary_array: np.ndarray, weights: np.ndarray = np.array([])
) -> dict:
    """
    Finds all duplicate row vectors in a binary array and returns their indices as a list of tuples.

    Args:
        binary_array (np.ndarray): A binary nxm array.

    Returns:
        dict: A dictionary where keys are unique row vectors (as tuples) and values are lists of indices where these vectors occur in the binary array.
    """
    unique_to_idx = OrderedDict()
    if weights.size == binary_array.shape[0]:
        for idx, (row, weight) in tqdm(enumerate(zip(binary_array, weights))):
            row_tuple = tuple(row)
            if row_tuple in unique_to_idx:
                unique_to_idx[row_tuple]["idxs"].append(idx)
                unique_to_idx[row_tuple]["weight"] += weight
            else:
                unique_to_idx[row_tuple] = {"idxs": [idx], "weight": weight}
    elif weights.size == 0:
        for idx, row in tqdm(enumerate(binary_array)):
            row_tuple = tuple(row)
            if row_tuple in unique_to_idx:
                unique_to_idx[row_tuple]["idxs"].append(idx)
            else:
                unique_to_idx[row_tuple] = {"idxs": [idx]}
    else:
        raise ValueError(
            "weights must be of the same length as the number of rows in the binary array or empty"
        )

    return unique_to_idx


def idx_to_unique_vector_idx(unique_to_idx: dict) -> dict:
    """
    Converts a dictionary of unique vectors to a dictionary mapping indices to unique vector indices.

    Args:
        unique_to_idx (dict): A dictionary where keys are unique row vectors (as tuples) and values are lists of indices where these vectors occur in the binary array.

    Returns:
        dict: A dictionary mapping each index in the original array to its corresponding unique vector index.
    """
    idx_to_unique_vector_idx = {}
    for unique_vector_idx, (vector, data) in enumerate(unique_to_idx.items()):
        idxs = np.array(data["idxs"])
        idx_to_unique_vector_idx.update(
            dict(zip(idxs, [unique_vector_idx] * len(idxs)))
        )

    return idx_to_unique_vector_idx


# def graph_based_clustering():
#     https://graph-tool.skewed.de/


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


def get_path_to_clustering_dir(
    n_nodes,
    co2l,
    indicator_type,
    transformation,
    n_nodes_split,
    lost_load_share,
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
