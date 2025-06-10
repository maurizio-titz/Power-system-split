import gzip
import os
import pickle
import sys
from typing import Callable

import numpy as np
import scipy
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import OrderedDict

sys.path.append("./")


def calc_nodewise_uncertainties(
    laplacian: np.matrix,
    node_class_vectors: np.ndarray,
    smothin_order=1,
    save_dir=None,
) -> np.ndarray:
    """calculates the nodewise uncertainty, i.e. how close the node is to nodes of the opposite class.

    Args:
        laplacian (np.matrix): graph laplacian
        node_class_vector (np.ndarray): binary vector of length n_nodes holding the class [0,1] of each node.
        order

    Returns:
        np.ndarray: uncertainty vector
    """
    assert (
        laplacian.shape[0] == laplacian.shape[1] == node_class_vectors.shape[1]
    ), "laplacian and node_class_vector must have the same shape"

    # Compute the shortest path distance dictionary from the Laplacian
    graph = scipy.sparse.csgraph.csgraph_from_dense(laplacian, null_value=0)
    shortest_path_distances = scipy.sparse.csgraph.dijkstra(
        graph, directed=False, return_predecessors=False
    )
    if smothin_order > 1:
        # smooth the shortest path distances
        shortest_path_distances = np.power(
            shortest_path_distances.to_dense(), smothin_order
        )

    closeness = 1 / shortest_path_distances
    np.fill_diagonal(closeness, 0)

    uncertainties = closeness * node_class_vectors
    uncertainties = np.abs(uncertainties)

    # save uncertainties to file
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with gzip.open(f"{save_dir}/uncertainties.pklz", "wb") as fh_out:
            pickle.dump(uncertainties, fh_out)

    return uncertainties


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
        raise ValueError("node_class_vectors must be binary or -1,1")

    if scipy.sparse.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.toarray()
    # decay_factors = np.array([decay_factor**d for d in range(1, max_distance + 1)])

    # adjecency_matrix_powers = np.array([np.linalg.matrix_power(adjacency_matrix, d) for d in range(1, max_distance + 1)])

    # k hop normalization
    # adjecency_matrix_powers_norm = np.array([normalize(A_k, axis=1, norm='l1') for A_k in adjecency_matrix_powers])

    # full neighborhood normalization
    # full_neighborhood_weight = [np.sum(A_k, axis=1)*decay_factor_k for decay_factor_k, A_k in zip(decay_factors,adjecency_matrix_powers)]
    # full_neighborhood_weights = np.sum(adjecency_matrix_powers, axis=1)*decay_factors

    c = np.sum(
        [
            node_class_vectors
            @ normalize(np.linalg.matrix_power(adjacency_matrix, d), axis=0, norm="l1")
            * decay_factor ** (-d)
            for d in range(
                max_distance + 1
            )  # +1 because 0 would be the identity matrix
        ],
        axis=0,
    )
    # assert c.shape == node_class_vectors.shape, "c.shape != node_class_vectors.shape"

    return abs(c)
    # return c


def uncertainty_distance(
    node_classes0,
    node_classes1,
    node_uncertainties0,
    node_uncertainties1,
    order: int = 1,
) -> float:
    """calculates the distance between two indicator vectors."""

    divs = node_classes0 != node_classes1
    return np.linalg.norm(
        np.multiply(node_uncertainties0[divs], node_uncertainties1[divs]), ord=order
    )


def uncertainty_distance_from_tuple(
    node_classes0,
    node_uncertainties0,
    node_classes1,
    node_uncertainties1,
    order: int = 1,
) -> float:
    """calculates the distance between two indicator vectors."""

    divs = node_classes0 != node_classes1
    return np.linalg.norm(
        (node_uncertainties0[divs].dot(node_uncertainties1[divs])), ord=order
    )


def binning_uncertainty_distance(
    node_classes0,
    node_classes1,
    node_uncertainties0,
    node_uncertainties1,
    order: int = 1,
    max_relative_blackout_size_difference: float = 0.1,
) -> float:
    """calculates the distance between two indicator vectors, if the relative black out size difference is below max_relative_blackout_size_difference, otherwise returns infinity."""

    if node_classes0.sum() > node_classes1.sum():
        node_classes0, node_classes1 = (
            node_classes1,
            node_classes0,
        )
    if node_classes0.sum() < node_classes1.sum() * (
        1 - max_relative_blackout_size_difference
    ):
        return np.inf
    else:
        return uncertainty_distance(
            node_classes0,
            node_classes1,
            node_uncertainties0,
            node_uncertainties1,
            order,
        )


def binning_uncertainty_distance_wrapper(
    node_classes_tuple,
    node_uncertainties_tuple,
    order: int = 1,
    max_relative_blackout_size_difference: float = 0.1,
) -> float:
    """
    Not used!
    calculates the distance between two indicator vectors. takes a tuple of the node classes and uncertainties for each samples, so it can be used as a distance function in the clustering algorithm.
    """

    node_classes0 = node_classes_tuple[0]
    node_classes1 = node_classes_tuple[1]
    node_uncertainties0 = node_uncertainties_tuple[0]
    node_uncertainties1 = node_uncertainties_tuple[1]

    return binning_uncertainty_distance(
        node_classes0,
        node_classes1,
        node_uncertainties0,
        node_uncertainties1,
        order=order,
        max_relative_blackout_size_difference=max_relative_blackout_size_difference,
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

    return uncertainty_distance(
        node_classes0,
        node_classes1,
        node_uncertainties0,
        node_uncertainties1,
        order=order,
    )


def uncertainty_distance_wrapper_concat(
    blackout_centrality_tuple_0,
    blackout_centrality_tuple_1,
    order: int = 1,
) -> float:
    """
    calculates the distance between two indicator vectors. takes a tuple of the node classes and uncertainties for each samples, so it can be used as a distance function in the clustering algorithm.
    """
    assert (
        blackout_centrality_tuple_0.shape == blackout_centrality_tuple_1.shape
    ), "blackout_centrality_tuple_0 and blackout_centrality_tuple_1 must have the same shape"
    n_nodes = int(blackout_centrality_tuple_0.shape[0] / 2)
    assert n_nodes % 2 == 0

    node_classes0 = blackout_centrality_tuple_0[:n_nodes]
    node_uncertainties0 = blackout_centrality_tuple_0[n_nodes:]
    node_classes1 = blackout_centrality_tuple_1[:n_nodes]
    node_uncertainties1 = blackout_centrality_tuple_1[n_nodes:]

    return uncertainty_distance(
        node_classes0,
        node_classes1,
        node_uncertainties0,
        node_uncertainties1,
        order=order,
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


def get_neighbour_candidates(
    indicator_vectors: np.ndarray,
    max_relative_blackout_size_difference: float,
):
    blackout_sizes_sorted, indicator_vectors_sorted, order = get_order_by_blackout_size(
        indicator_vectors
    )


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
    for idx, (row, weight) in tqdm(enumerate(zip(binary_array, weights))):
        row_tuple = tuple(row)
        if row_tuple in unique_to_idx:
            unique_to_idx[row_tuple]["idxs"].append(idx)
            unique_to_idx[row_tuple]["weight"] += weight
        else:
            unique_to_idx[row_tuple] = {"idxs": [idx], "weight": weight}

    # Filter out rows that are not duplicates
    # duplicate_indices = {
    #     row: indices for row, indices in duplicates.items() if len(indices) > 1
    # }
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
