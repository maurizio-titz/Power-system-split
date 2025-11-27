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


def calc_distance_matrix(
    data,
    mode: str = "sequential",
    metric="bACC",
    n_jobs=-1,
    test_mode=True,
    # show_progress=True,
):
    """Calculate the distance matrix."""
    if mode == "joblib":

        raise NotImplementedError(
            "joblib mode is deprecated, use vectorized_joblib mode instead"
        )
    elif mode == "seq":
        if metric == "bACC":
            metric = balanced_overlap_distance_weighted
        elif metric == "ACC":
            metric = multiclass_accuracy_distance_weighted
        else:
            raise ValueError(f"Unknown metric: {metric}")
        metric = partial(
            weighted_distance_wrapper,
            weighted_distance_metric=metric,
        )
        return calc_distance_matrix_sequential(data, metric=metric, test_mode=test_mode)
    elif mode == "vec":
        # split data into classes and weights
        n_nodes = int(data.shape[1] / 2)
        node_classes_matrix = data[:, :n_nodes]
        node_weights_matrix = data[:, n_nodes:]
        return compute_distance_matrix_vectorized(
            node_classes_matrix,
            node_weights_matrix,
            bACC_weighted_pairwise,
            chunk_size=500,
            test_mode=test_mode,
        )
    elif mode == "vec_joblib":
        # split data into classes and weights
        n_nodes = int(data.shape[1] / 2)
        node_classes_matrix = data[:, :n_nodes]
        node_weights_matrix = data[:, n_nodes:]
        return compute_distance_matrix_vectorized_joblib(
            node_classes_matrix,
            node_weights_matrix,
            metric,
            chunk_size=500,
            test_mode=test_mode,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown distance matrix calculation mode: {mode}")


def calc_distance_matrix_sequential(
    data,
    metric=balanced_overlap_distance_weighted,
    test_mode=False,
    dtype=np.float32,
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
    if dtype is not None:
        d = d.astype(dtype)
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


def generate_chunk_coordinates(n_samples, chunk_size):
    """Memory-efficient generator for chunk coordinates"""
    for i in range(0, n_samples, chunk_size):
        for j in range(i, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            j_end = min(j + chunk_size, n_samples)
            yield (i, i_end, j, j_end)


def compute_chunk_distances(
    data_matrix, weights_matrix, metric_func, chunk_coords, dtype=np.float32
):
    """Compute distances for a single chunk"""
    i, i_end, j, j_end = chunk_coords

    # Extract chunks
    chunk1_data = data_matrix[i:i_end]
    chunk1_weights = weights_matrix[i:i_end]
    chunk2_data = data_matrix[j:j_end]
    chunk2_weights = weights_matrix[j:j_end]

    # Calculate pairwise distances for this chunk
    chunk_distances = metric_func(
        chunk1_data, chunk2_data, chunk1_weights, chunk2_weights
    ).astype(dtype)

    return chunk_coords, chunk_distances


def compute_distance_matrix_vectorized_joblib(
    data_matrix: np.ndarray,  # Shape: (n_samples, n_nodes)
    weights_matrix: np.ndarray,  # Shape: (n_samples, n_nodes)
    metric_name: str,
    chunk_size: int = 1000,
    test_mode: bool = False,
    n_jobs: int = -1,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute distance matrix using vectorized operations with parallel chunking.

    Returns:
        np.ndarray: Distance matrix of shape (n_samples, n_samples)
    """
    if test_mode:
        data_matrix = data_matrix[:10000, :]
        weights_matrix = weights_matrix[:10000, :]
    n_samples = data_matrix.shape[0]

    if metric_name == "bACC":
        metric = bACC_weighted_pairwise
    elif metric_name == "ACC":
        metric = ACC_weighted_pairwise
    elif metric_name == "boundary_field_ACC":
        metric = boundary_field_ACC_weighted_pairwise
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    # Generate chunk coordinates - lightweight memory usage
    chunk_coords_gen = generate_chunk_coordinates(n_samples, chunk_size)

    # Convert to list for joblib (still memory efficient for coordinates only)
    chunk_coords_list = list(chunk_coords_gen)
    n_chunks = len(chunk_coords_list)

    print(
        f"Processing {n_samples}x{n_samples} matrix in {n_chunks} chunks of size ~{chunk_size}x{chunk_size}"
    )

    # Parallel processing of chunks
    with tqdm_joblib(tqdm(desc="Distance Calculation", total=n_chunks)) as progress_bar:
        chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_chunk_distances)(
                data_matrix, weights_matrix, metric, coords
            )
            for coords in chunk_coords_list
        )

    # Initialize distance matrix
    distance_matrix = np.zeros((n_samples, n_samples), dtype=dtype)

    # Fill distance matrix with results
    for chunk_coords, chunk_distances in chunk_results:
        i, i_end, j, j_end = chunk_coords

        # Fill the distance matrix
        distance_matrix[i:i_end, j:j_end] = chunk_distances

        # Fill symmetric part (unless it's the diagonal chunk)
        if i != j:
            distance_matrix[j:j_end, i:i_end] = chunk_distances.T

    return distance_matrix
