import gzip
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import path_to_indicator_vectors_sclopf


def load_indicator_vectors(
    n_nodes,
    co2l,
    indicator_type,
    path_to_indicator_vectors=path_to_indicator_vectors_sclopf,
    mask=None,
    weights=None,
):
    if (isinstance(co2l, list) or isinstance(co2l, np.ndarray)) and not isinstance(
        mask, list
    ):
        raise TypeError("if co2l is a list mask has to be a list, too")

    if not (
        (isinstance(co2l, list) or isinstance(co2l, np.ndarray))
        or isinstance(co2l, float)
    ):
        Exception("co2l has to be float or list")

    if weights is None and isinstance(co2l, float):
        all_vectors = load_indicator_vectors_unweighted_single_lvl(
            n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask
        )
        return all_vectors

    elif weights is None and isinstance(co2l, list):
        all_vectors = load_indicator_vectors_unweighted_multiple_lvl(
            n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask
        )
        return all_vectors

    elif weights is not None and isinstance(co2l, float):
        all_vectors, all_weights = load_indicator_vectors_weighted_single_lvl(
            n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights
        )
        return all_vectors, all_weights

    elif weights is not None and isinstance(co2l, list):
        all_vectors, all_weights = load_indicator_vectors_weighted_multiple_lvl(
            n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights
        )
        return all_vectors, all_weights


def load_indicator_vectors_weighted_multiple_lvl(
    n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights
):
    all_vectors = []
    all_weights = []
    for i, co2 in enumerate(co2l):
        file_name = f"{indicator_type}_indicator_vector_Co2L{co2}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            vectors_lvl = pickle.load(out)
        weights_lvl = np.array(
            [int(weights[time_stamp[0]]) for time_stamp in vectors_lvl[0]]
        )
        vectors_lvl = vectors_lvl[-1]

        if mask is not None:
            vectors_lvl = vectors_lvl[np.array(mask[i])]
            weights_lvl = weights_lvl[np.array(mask[i])]

        all_weights.append(weights_lvl)
        all_vectors.append(vectors_lvl)
    all_weights = np.concatenate(all_weights)
    all_vectors = np.concatenate(all_vectors)
    return all_vectors, all_weights


def load_indicator_vectors_weighted_single_lvl(
    n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask, weights
):
    all_weights = []
    file_name = f"indicator_vector_{indicator_type}_Co2L{co2l}_n{n_nodes}.pklz"
    with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
        all_vectors = pickle.load(out)
    all_weights = [weights[time_stamp[0]] for time_stamp in all_vectors[0]]
    all_vectors = all_vectors[-1]
    if mask is not None:
        all_weights = all_weights[mask]
        all_vectors = all_vectors[mask]
    return all_vectors, all_weights


def load_indicator_vectors_unweighted_multiple_lvl(
    n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask
):
    all_vectors = []
    for i, co2 in enumerate(co2l):
        file_name = f"indicator_vector_{indicator_type}_Co2L{co2}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            vectors_lvl = pickle.load(out)
        if isinstance(vectors_lvl, tuple):
            vectors_lvl = vectors_lvl[-1]
        if mask is not None:
            vectors_lvl = vectors_lvl[np.array(mask[i])]
        all_vectors.append(vectors_lvl)
    all_vectors = np.concatenate(all_vectors)
    return all_vectors


def load_indicator_vectors_unweighted_single_lvl(
    n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask
):
    all_vectors = []
    file_name = f"indicator_vector_{indicator_type}_Co2L{co2l}_n{n_nodes}.pklz"
    with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
        all_vectors = pickle.load(out)
    if isinstance(all_vectors, tuple):
        all_vectors = all_vectors[-1]
    if mask is not None:
        all_vectors = all_vectors[mask]
    return all_vectors


def create_component_indicator_vectors(
    n_nodes, co2l, indicator_type, path_to_indicator_vectors, mask=None
):
    """creates component wise indicator vectors from indicator vectors. Loads indicator vectors from path_to_indicator_vectors and saves component indicator vectors to path_to_indicator_vectors. The main component is not considered a split component. Accordingly splits that contain n components are projected onto n-1 component indicator vectors. The mapping from the original indicator vector index to component indicator vector index is saved as a array.

    Args:
        n_nodes (_type_): _description_
        co2l (_type_): _description_
        indicator_type (_type_): _description_
        path_to_indicator_vectors (_type_): _description_
        mask (_type_): _description_
    """

    indicator_vectors = load_indicator_vectors(
        n_nodes,
        co2l,
        indicator_type,
        path_to_indicator_vectors=path_to_indicator_vectors,
        mask=mask,
    )

    component_indicator_vectors = []
    split_ind_to_component_ind_list = []
    component_indicators_ind = 0
    for i, indicator_vector in enumerate(tqdm(indicator_vectors)):
        vals, indices, counts = np.unique(
            indicator_vector, return_inverse=True, return_counts=True
        )
        main_component_val = vals[np.argmax(counts)]
        was_used = False
        for val_count, val in enumerate(vals):
            if val == main_component_val:  # main component is not a split component
                continue
            was_used = True
            inds = indices == val_count
            component_indicator_vals = np.zeros_like(indicator_vector)
            component_indicator_vals[inds] = val
            component_indicator_vectors.append(component_indicator_vals)
        assert was_used, "main component was not used, index={i}"

        if i % 1000 == 0:
            print(component_indicator_vals[10])

        split_ind_to_component_ind_list.append(
            np.arange(
                component_indicators_ind, component_indicators_ind + len(vals) - 1
            )
        )
        component_indicators_ind += (
            len(vals) - 1
        )  # -1 because main component is not a split component

    file_name = (
        f"indicator_vector_{indicator_type}_components_Co2L{co2l}_n{n_nodes}.pklz"
    )
    with gzip.open(path_to_indicator_vectors + "/" + file_name, "wb") as out:
        pickle.dump(
            (split_ind_to_component_ind_list, np.array(component_indicator_vectors)),
            out,
        )


def calc_is_main_component_indicator_vectors(
    lshare_vectors, main_component_by="main_comp_max_lshare"
):
    """returns a list of indicator vectors where each entry in the vectors indicates whether the corresponding node was in the main (1) component or not (0)"""

    indicator_vectors = []
    if main_component_by == "main_comp_max_lshare":
        for lshare in lshare_vectors:
            main_component_lshare = np.unique(lshare)[-1]
            # is_main_component = lshare == main_component_lshare
            indicator_vectors.append(
                np.array(lshare == main_component_lshare, dtype=int)
            )
    elif main_component_by == "main_comp_most_frequent_lshare":
        for lshare in lshare_vectors:
            main_component_lshare = Counter(lshare).most_common(1)[0][0]
            # is_main_component = lshare == main_component_lshare
            indicator_vectors.append(
                np.array(lshare == main_component_lshare, dtype=int)
            )
    else:
        raise ValueError(
            "main_component_by has to be either 'main_comp_max_lshare' or 'main_comp_most_frequent_lshare'"
        )

    return np.array(indicator_vectors)
