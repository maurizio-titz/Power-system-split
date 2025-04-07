import gzip
import os
import pickle
import sys

import numpy as np
import scipy

sys.path.append("./")


def calc_nodewise_uncertainties(
    laplacian: np.matrix,
    node_class_vectors: np.array,
    squared_distance=True,
    save_dir=None,
) -> np.array:
    """calculates the nodewise uncertainty, i.e. how close the node is to nodes of the opposite class.

    Args:
        laplacian (np.matrix): graph laplacian
        node_class_vector (np.array): binary vector of length n_nodes holding the class [0,1] of each node.

    Returns:
        np.array: uncertainty vector
    """
    assert (
        laplacian.shape[0] == laplacian.shape[1] == node_class_vectors.shape[1]
    ), "laplacian and node_class_vector must have the same shape"

    # Compute the shortest path distance dictionary from the Laplacian
    graph = scipy.sparse.csgraph.csgraph_from_dense(laplacian, null_value=0)
    shortest_path_distances = scipy.sparse.csgraph.dijkstra(
        graph, directed=False, return_predecessors=False
    )
    if squared_distance:
        shortest_path_distances = np.square(shortest_path_distances)
    closeness = 1 / shortest_path_distances
    np.fill_diagonal(closeness, 0)

    uncertainties = node_class_vectors @ closeness
    uncertainties = np.abs(uncertainties)

    # save uncertainties to file
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with gzip.open(f"{save_dir}/uncertainties.pklz", "wb") as fh_out:
            pickle.dump(uncertainties, fh_out)

    return uncertainties


def uncertainty_distance(
    node_classes0,
    node_classes1,
    node_uncertainties0,
    node_uncertainties1,
    order: int = 1,
) -> float:
    """calculates the distance between two indicator vectors."""

    divs = node_classes0 - node_classes1
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
    """calculates the distance between two indicator vectors."""

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


def binning_uncertainty_distance(
    node_classes_tuple,
    node_uncertainties_tuple,
    order: int = 1,
    max_relative_blackout_size_difference: float = 0.1,
) -> float:
    """calculates the distance between two indicator vectors."""

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


def get_order_by_blackout_size(indicator_vectors: np.ndarray):
    """returns the order of the indicator vectors by blackout size"""

    blackout_sizes = indicator_vectors.sum(axis=1)
    # sort the blackout sizes in descending order
    order = np.argsort(blackout_sizes)[::-1]
    indicator_vectors_sorted = indicator_vectors[order]
    blackout_sizes_sorted = blackout_sizes[order]

    return blackout_sizes_sorted, indicator_vectors_sorted, order


def calculate_distance_matrix(
    indicator_vectors: np.array,
    distance_function: callable,
    max_relative_blackout_size_difference: float,
):
    """ "calculates the distance matrix for the indicator vectors
    Args:
        indicator_vectors (np.array): indicator vectors
        distance_function (callable): function to calculate the distance between two indicator vectors
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
