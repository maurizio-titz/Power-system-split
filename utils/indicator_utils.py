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
    indicator_type,
    transformation,
    path_to_indicator_vectors=path_to_indicator_vectors_sclopf,
    mask=None,
    split_props=None,
    co2_list=(),
) -> "tuple[dict, dict, pd.DataFrame]":

    assert isinstance(
        split_props, pd.DataFrame
    ), f"split_props must be a DataFrame but is {type(split_props)}"

    vectors_filtered, weights_filtered, split_props_filtered = (
        load_indicator_vectors_weighted_multiple_lvl(
            n_nodes,
            indicator_type,
            transformation,
            path_to_indicator_vectors,
            mask,
            split_props,
            co2list=co2_list,
        )
    )
    return vectors_filtered, weights_filtered, split_props_filtered


def load_indicator_vectors_weighted_multiple_lvl(
    n_nodes,
    indicator_type,
    transformation,
    path_to_indicator_vectors,
    mask,
    split_props: pd.DataFrame,
    co2list=(),
) -> "tuple[dict, dict, pd.DataFrame]":
    if co2list is None or len(co2list) == 0:
        co2list = split_props.co2l.unique()

    all_vectors_dict = {}
    all_weights_dict = {}
    all_split_props = []
    for i, co2lvl in enumerate(co2list):
        if isinstance(co2lvl, str):
            co2lvl = float(co2lvl)
        assert co2lvl <= 0.6 and co2lvl >= 0
        file_name = f"indicator_vector_{indicator_type}_Co2L{co2lvl}_n{n_nodes}.pklz"
        with gzip.open(path_to_indicator_vectors + "/" + file_name, "rb") as out:
            vectors_tuple = pickle.load(out)
        vectors_lvl = vectors_tuple[-1]
        weights_lvl = np.array([index_tuple[2] for index_tuple in vectors_tuple[0]])

        split_props_level = split_props[split_props.co2l == co2lvl]
        if mask is not None:
            vectors_lvl = vectors_lvl[np.array(mask[co2lvl])]
            weights_lvl = weights_lvl[np.array(mask[co2lvl])]
            split_props_level = split_props_level.iloc[np.array(mask[co2lvl])]

        vectors_lvl = transform_indicator_vectors(
            vectors_lvl, transformation, indicator_type
        )

        all_weights_dict[co2lvl] = weights_lvl
        all_vectors_dict[co2lvl] = vectors_lvl
        all_split_props.append(split_props_level)

    split_props_filtered = pd.concat(all_split_props, axis=0)

    return all_vectors_dict, all_weights_dict, split_props_filtered


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


def transform_indicator_vectors(indicator_vectors, transformation, indicator_type):
    """transforms indicator vectors to allow better clustering

    Args:
        indicator_vectors (np.ndarray): indicator vectors
        transformation (str): name of transformation

    Returns:
        np.ndarray: transformed indicator vectors
    """
    if transformation == "sign":
        transformed_indicator_vectors = np.sign(indicator_vectors)
    elif transformation == "tanh":
        transformed_indicator_vectors = np.tanh(indicator_vectors)
    elif transformation == "clipped_tanh":
        transformed_indicator_vectors = np.tanh(indicator_vectors)
        transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    elif transformation == "clipped":
        transformed_indicator_vectors = np.clip(indicator_vectors, None, 0)
    elif transformation == "main_comp_max_lshare":
        if indicator_type != "lshare":
            raise ValueError(
                "is_main_component transformation only works for lshare indicator"
            )
        transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
            indicator_vectors
        )
    elif transformation == "main_comp_most_frequent_lshare":
        if indicator_type != "lshare":
            raise ValueError(
                "is_main_component transformation only works for lshare indicator"
            )
        transformed_indicator_vectors = calc_is_main_component_indicator_vectors(
            indicator_vectors, main_component_by="main_comp_most_frequent_lshare"
        )
    elif transformation == "blackout":
        transformed_indicator_vectors = np.array(indicator_vectors < -1, dtype=int)
    elif transformation == "not_zero":
        transformed_indicator_vectors = np.array(indicator_vectors != 0, dtype=int)
    elif transformation == "overUnder":
        transformed_indicator_vectors = np.zeros_like(indicator_vectors)
        transformed_indicator_vectors[indicator_vectors > 1] = 1
        transformed_indicator_vectors[indicator_vectors < -1] = -1
        transformed_indicator_vectors = transformed_indicator_vectors.astype(int)
    else:
        raise ValueError(f"transformation {transformation} not implemented")

    return transformed_indicator_vectors
