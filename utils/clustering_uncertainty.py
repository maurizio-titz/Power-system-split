import gzip
import os
import pickle
import sys
from typing import Callable

import numpy as np
import scipy
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import OrderedDict

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
    n_nodes = int(blackout_centrality_tuple_0.shape[0] / 2)
    assert n_nodes % 2 == 0

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

    return (
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


def calc_distance_matrix(
    data,
    metric=weighted_bACC_dist,
):
    """
    Calculate the distance matrix for the given data using the specified metric.

    Parameters:
        data (np.ndarray): The input data for which to calculate distances.
        metric (callable): The distance metric to use.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        np.ndarray: The calculated distance matrix.
    """
    d = np.zeros((data.shape[0], data.shape[0]), dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            d[i, j] = metric(data[i], data[j])
            # Store the distance in the appropriate place in the matrix
            d[j, i] = d[i, j]

    np.fill_diagonal(d, 0)  # Set diagonal to zero
    return d


def sparsify_distance_matrix(
    distance_matrix: np.ndarray,
    blackout_vectors: np.ndarray,
) -> np.ndarray:
    """sparsifies the distance matrix by setting the distance to infinity there is no overlap between blackout or non blackout parts"""
    assert distance_matrix.shape[0] == blackout_vectors.shape[0]

    blackout_vectors = blackout_vectors.astype(bool)
    # set distances to infinity where there is no overlap
    # Vectorized version for speed
    # For each pair (i, j), set distance to inf if there is no overlap in blackout or non-blackout parts
    # Overlap: at least one True in both blackout_vectors[i] & blackout_vectors[j], and at least one False in both

    # Compute overlap matrices
    overlap_blackout = np.dot(blackout_vectors, blackout_vectors.T) > 0
    overlap_non_blackout = np.dot(~blackout_vectors, ~blackout_vectors.T) > 0

    # If either overlap_blackout or overlap_non_blackout is False, set distance to inf
    mask = ~(overlap_blackout & overlap_non_blackout)
    distance_matrix[mask] = 0

    # Convert to sparse matrix format
    distance_matrix = scipy.sparse.csr_matrix(distance_matrix, dtype=float)

    return distance_matrix


def get_order_by_blackout_size(indicator_vectors: np.ndarray):
    """returns the order of the indicator vectors by blackout size"""

    indicator_vectors = np.atleast_2d(indicator_vectors)
    blackout_sizes = indicator_vectors.sum(axis=1)
    # sort the blackout sizes in descending order
    order = np.argsort(blackout_sizes)[::-1]
    indicator_vectors_sorted = indicator_vectors[order]
    blackout_sizes_sorted = blackout_sizes[order]

    return blackout_sizes_sorted, indicator_vectors_sorted, order


def calculate_distance_matrix(
    indicator_vectors: np.ndarray,
    distance_function: Callable,
    max_relative_blackout_size_difference: float,
):
    """ "calculates the distance matrix for the indicator vectors
    Args:
        indicator_vectors (np.ndarray): indicator vectors
        distance_function (Callable): function to calculate the distance between two indicator vectors
        max_relative_blackout_size_difference (float): maximum relative blackout size difference to calculate the distance. If the blackout size of the second indicator vector is smaller the two indicator vectors treated as neighbours in the clustering algorithm, so we don't calculate the distance between them.

    Returns:
        np.array: distance matrix
    """
    blackout_sizes_sorted, indicator_vectors_sorted, order = get_order_by_blackout_size(
        indicator_vectors
    )

    distance_matrix = scipy.sparse.csr_matrix(
        (len(indicator_vectors_sorted), len(indicator_vectors_sorted)), dtype=float
    )

    min_blackout_sizes = (
        blackout_sizes_sorted
        - max_relative_blackout_size_difference * blackout_sizes_sorted
    )

    for i, (indicator_vector, min_blackout_size) in enumerate(
        zip(indicator_vectors_sorted, min_blackout_sizes)
    ):
        # calculate the distance to all other indicator vectors
        for j in range(i + 1, len(indicator_vectors_sorted)):
            # calculate the distance to the other indicator vector
            if blackout_sizes_sorted[j] > min_blackout_size:
                distance_matrix[i, j] = distance_function(
                    indicator_vector, indicator_vectors_sorted[j]
                )
                distance_matrix[i, j] = distance_matrix[j, i]

    return distance_matrix


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

    # with open(path_group_means, "wb") as out:
    # np.save(out, group_means)
