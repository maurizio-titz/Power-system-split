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

from utils.clustering.data_handling import split_mask
from utils.config import (
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
    labels: np.ndarray,
    A: Union[np.matrix, sparse.csr_matrix] = None,
    L: Union[np.matrix, sparse.csr_matrix] = None,
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
    from scipy.sparse import diags

    if A is None:
        if L is None:
            raise ValueError("You have to provide either A or L")
        else:
            D = diags(L.diagonal())
            A = D - L

    if len(labels.shape) == 3:
        raise NotImplementedError(
            "neighborhood_impurity_batch does not support one hote encoding of node class vectors"
        )
        n_classes = labels.shape[2]
        if n_classes > 1:
            node_class_vectors_one_hot = labels
        else:
            raise ValueError(
                "node_class_vectors seems to be one-hot encoded but has only one class"
            )
    else:
        classes = np.sort(np.unique(labels))
        n_classes = len(classes)
        if n_classes > 1:
            # one-hot encode the node class vectors
            node_class_vectors_one_hot = np.stack(
                [(labels == cls).astype(int) for cls in classes],
                axis=0,
            )
        else:
            raise ValueError(
                "node_class_vectors is not one-hot encoded but has only one unique value"
            )

    print(f"{n_classes} classes found in node_class_vectors: {classes}")

    if scipy.sparse.issparse(A):
        A = A.toarray()

    c = np.sum(
        [
            node_class_vectors_one_hot
            @ normalize(np.linalg.matrix_power(A, d), axis=0, norm="l1")
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


def boundary_field_UNSTABLE(labels, I, L, tau=1.0):
    raise NotImplementedError(
        "boundary_field_UNSTABLE is deprecated due to instability, yields values >1."
    )
    from scipy.sparse import diags

    # labels: (N, n_nodes)
    # B: incidence (n_edges, n_nodes)    # corrected orientation
    # L: Laplacian (n_nodes, n_nodes)
    # 1) edge cut
    diff = labels @ I  # (N, n_nodes) @ (n_nodes, n_edges)
    c = (diff != 0).astype(float)  # (N, n_edges)

    # 2) map back to nodes
    b = (c @ I.T != 0).astype(float)  # (N, n_nodes)
    # 3) build normalized Laplacian correctly
    D = diags(L.diagonal())
    A = D - L
    deg = np.array(D.diagonal())
    Dinv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    Lsym = Dinv_sqrt @ L @ Dinv_sqrt
    # eigvals = np.linalg.eigvals(Lsym.toarray())
    # assert np.isclose(
    #     eigvals, 1, atol=1e-3
    # ).all(), f"normalized laplacian eigenvalues not in expected range, max deviation: {np.abs(eigvals-1).max():.2g}"

    # 4) stable smoothing
    from scipy.sparse.linalg import expm_multiply

    b_s = np.stack([expm_multiply(-tau * Lsym, row) for row in b])

    return b_s


def boundary_field_rw_smoothing(
    labels: np.ndarray,
    I: Union[np.matrix, sparse.csr_matrix],
    A: Union[np.matrix, sparse.csr_matrix],
    alpha=0.5,
    steps=3,
):
    """
    Compute smoothed boundary fields using a row-stochastic random-walk operator.
    Includes strict sanity checks with error raising.

    Args:
        labels (np.ndarray): shape (N_samples, N_nodes)
        incidence_mat (csr_matrix): node-edge incidence matrix
        A (csr_matrix): adjacency matrix (unweighted or weighted)
        alpha (float): smoothing strength per step
        steps (int): number of smoothing iterations

    Returns:
        np.ndarray of shape (N_samples, N_nodes)
    """

    # -----------------------------
    # 1) Compute edge cuts
    # -----------------------------
    diff = labels @ I
    c = (diff != 0).astype(float)

    if c.min() < 0 or c.max() > 1:
        raise ValueError("Edge cut matrix contains invalid values.")

    # -----------------------------
    # 2) Map cuts back to nodes
    # -----------------------------
    b = ((c @ I.T) != 0).astype(float)

    unique_b = np.unique(b)
    if not np.all(np.isin(unique_b, [0.0, 1.0])):
        raise ValueError("Boundary indicator b must contain only 0 or 1.")

    # -----------------------------
    # 3) Build P = D^{-1} A (row-stochastic)
    # -----------------------------
    deg = np.asarray(A.sum(axis=1)).ravel()

    if deg.min() <= 0:
        raise ValueError(
            "Adjacency matrix contains isolated nodes or bad degree values."
        )

    from scipy.sparse import diags

    Dinv = diags(1.0 / deg)
    P = Dinv @ A

    # Row stochasticity check
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    if np.any(row_sums < 0.999) or np.any(row_sums > 1.001):
        raise ValueError("Transition matrix P is not row-stochastic.")

    # Non-negativity check
    if P.min() < -1e-12:
        raise ValueError("Transition matrix P contains negative entries.")

    # -----------------------------
    # 4) Diffusion: s_new = (1-alpha)*s + alpha*(s @ P.T)
    # -----------------------------
    s = b.copy().astype(float)

    if s.min() < 0 or s.max() > 1:
        raise ValueError("Initial boundary field s must be in [0,1].")

    for k in range(steps):
        s = (1 - alpha) * s + alpha * (s @ P.T)

        if s.min() < -1e-12:
            raise ValueError(
                f"Negative values encountered at step {k}. Min: {s.min():.2g}"
            )

        if s.max() > 1 + 1e-12:
            raise ValueError(
                f"Values above 1 encountered at step {k}. Max: {s.max():.2g}"
            )

    s = np.clip(s, 0.0, 1.0)

    return s


def low_weight_boundary_field(
    labels: np.ndarray,
    I: Union[np.matrix, sparse.csr_matrix],
    A: Union[np.matrix, sparse.csr_matrix],
    alpha: float = 0.5,
    steps: int = 3,
) -> np.ndarray:
    """calculates the boundary field-based weights for each node. This gives high weights to nodes far from class boundaries.
    labels: node class vectors, shape (n_instances, n_nodes)
    I: incidence matrix of the graph
    L: Laplacian matrix of the graph
    tau: smoothing parameter
    returns: nodewise weights, shape (n_instances, n_nodes)
    """

    b_s = boundary_field_rw_smoothing(labels, I=I, A=A, alpha=alpha, steps=steps)

    # Convert to weights: high values for nodes far from boundaries
    weights = 1 - b_s

    if weights.min() < 0:
        raise ValueError("Boundary field weights must be positive.")

    return weights


# def boundary_field_ACC_weighted_pairwise(
#     node_classes_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
#     node_classes_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
#     node_weights_matrix1: np.ndarray,  # Shape: (n1, n_nodes)
#     node_weights_matrix2: np.ndarray,  # Shape: (n2, n_nodes)
#     weighted: bool = True,
#     alpha: float = 0.5,
#     dtype=np.float32,
#     return_components: bool = False,
# ) -> np.ndarray:
#     """
#     Calculate pairwise multiclass balanced distances.

#     Returns:
#         np.ndarray: Distance matrix of shape (n1, n2)
#     """
#     # Use broadcasting for pairwise calculation
#     classes1_exp = node_classes_matrix1[:, np.newaxis, :]  # (n1, 1, n_nodes)
#     classes2_exp = node_classes_matrix2[np.newaxis, :, :]  # (1, n2, n_nodes)
#     weights1_exp = node_weights_matrix1[:, np.newaxis, :]  # (n1, 1, n_nodes)
#     weights2_exp = node_weights_matrix2[np.newaxis, :, :]  # (1, n2, n_nodes)
#     if weighted:
#         agree = (((classes1_exp == classes2_exp)) * (weights1_exp * weights2_exp)).sum(
#             axis=2
#         )
#         total_weight = (weights1_exp * weights2_exp).sum(axis=2)
#         acc = np.round((agree + 1e-10) / (total_weight + 1e-10), decimals=6)
#     else:
#         agree = (classes1_exp == classes2_exp).sum(axis=2)
#         total_weight = classes1_exp.shape[2]
#         acc = np.round((agree + 1e-10) / (total_weight + 1e-10), decimals=6)

#     boundary_field1_exp = 1 - node_weights_matrix1  # (n1, n_nodes)
#     boundary_field2_exp = 1 - node_weights_matrix2  # (n2, n_nodes)

#     # calculate Cosine similarity between boundary fields
#     boundary_field_similarity = cosine_similarity(
#         boundary_field1_exp, boundary_field2_exp
#     )

#     # Convert to distance (1 - accuracy)
#     distances = 1 - (alpha * acc + (1 - alpha) * boundary_field_similarity)
#     # ! try geometric mean

#     if dtype is not None:
#         distances = distances.astype(dtype)

#     if not return_components:
#         return distances
#     else:
#         return distances, acc, boundary_field_similarity
