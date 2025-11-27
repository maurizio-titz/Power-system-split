import contextlib
import gzip
import os
import pickle
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from tqdm import tqdm

sys.path.append("../")

from utils.clustering.data_handling import (
    create_split_mask,
    get_path_to_clustering_dir,
    get_split_mask,
    get_unique_outage_vecs,
    get_unique_vectors_with_weights,
    transform_indicator_vectors,
)
from utils.clustering.distance_matrix_calculation import calc_distance_matrix
from utils.clustering.distance_metrics import (
    ACC_weighted_pairwise,
    bACC_weighted_pairwise,
    boundary_field_ACC_weighted_pairwise,
)
from utils.clustering.visualization import prepare_clusters_for_analysis
from utils.config import path_to_indicator_vectors_sclopf, path_to_vis_results_sclopf

data_dir = "./data/"
distance_matrices_dir = "./distance_matrices/"
cluster_dir = "./clustering_results/"


class Clustering(object):
    """Base class for preprocessing, clustering and visualization."""

    n_nodes: int = 600
    co2l_list: List[float]
    indicator_type: str = "rocof"
    transformation: str = "blackout"
    vectors: Optional[dict[float, np.ndarray]]
    split_properties: Optional[pd.DataFrame]
    unique_vector_to_idxs_and_weights: Optional[dict]
    blackout_size_threshold: Optional[float]
    unique_vecs: Optional[np.ndarray]
    weights_filtered: Optional[np.ndarray]
    distance_matrix_path: Optional[List[str]]
    distance_metric: Optional[Callable]
    distance_metric_kwargs: Optional[dict]
    distance_matrix: Optional[np.ndarray]
    distance_matrix_post_processing: Optional[Optional[Callable]]
    clustering_results_dir: Optional[str]
    clustering_algorithm: Optional[Callable]
    clustering_params: Optional[dict]
    save_dir: Optional[str]
    random_subsample_size: Optional[float] = 1

    def __init__(
        self,
        n_nodes,
        co2l,
        indicator_type,
        transformation,
        lost_load_share_min,
    ):
        self.n_nodes = n_nodes
        self.co2l_list = co2l if isinstance(co2l, list) else [co2l]
        self.indicator_type = indicator_type
        self.transformation = transformation
        self.blackout_size_threshold = lost_load_share_min
        self.save_dir = get_path_to_clustering_dir(n_nodes,
        co2l,
        indicator_type,
        transformation,
        lost_load_share_min,)
    
    def load_data(self):
        if self.split_properties is None:
            self.split_properties = pd.read_hdf(
                path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5"
            )
            co2l_mask = self.split_properties["co2_level"].isin(self.co2l_list)
            self.split_properties = self.split_properties[co2l_mask]
        if self.vectors is None:
            vectors_per_lvl = {}
            for co2lvl in self.co2l_list:
                file_name = f"indicator_vector_{self.indicator_type}_Co2L{co2lvl}_n{self.n_nodes}.pklz"
                with gzip.open(
                    path_to_indicator_vectors_sclopf + "/" + file_name, "rb"
                ) as out:
                    vectors_tuple = pickle.load(out)
                vectors_lvl = vectors_tuple[-1]
                # weights_lvl = np.array(
                #     [index_tuple[2] for index_tuple in vectors_tuple[0]]
                # )
                vectors_per_lvl[co2lvl] = vectors_lvl
            self.vectors = vectors_per_lvl

    def subsample_data(self):
        if self.random_subsample_size != 1:
            for co2lvl in self.co2l_list:
                subsample_mask = np.zeros(self.vectors[co2lvl].shape[0], dtype=bool)
                n_subsample = int(
                    self.random_subsample_size * self.vectors[co2lvl].shape[0]
                )
                rng = np.random.default_rng(42)
                selected_indices = rng.choice(
                    self.vectors[co2lvl].shape[0], n_subsample, replace=False
                )
                subsample_mask[selected_indices] = True
                self.vectors[co2lvl] = self.vectors[co2lvl][subsample_mask]
                self.split_properties = self.split_properties.iloc[subsample_mask]

    def transform_vectors(self):
        self.vectors = {
            co2l: transform_indicator_vectors(
                vectors_lvl,
                self.transformation,
            )
            for co2l, vectors_lvl in self.vectors.items()
        }

    def filter_data(self):
        os.makedirs(data_dir, exist_ok=True)
        masks_dict = get_split_mask(
            self.n_nodes,
            self.co2l_list,
            self.blackout_size_threshold,
            save_dir=data_dir,
            split_props=self.split_properties,
            err_on_missing=False,
        )
        for co2l in self.co2l_list:
            index_mask = masks_dict[co2l]
            mask_full = (
                self.split_properties.index.get_level_values("co2_level") != co2l
            )
            mask_full[
                self.split_properties.index.get_level_values("co2_level") == co2l
            ] = index_mask
            self.split_properties = self.split_properties[mask_full]
            self.vectors[co2l] = self.vectors[co2l][index_mask]

        self.unique_vector_to_idxs_and_weights = get_unique_vectors_with_weights(
            np.concatenate([self.vectors[co2l] for co2l in self.co2l_list]),
            np.concatenate(
                [
                    self.split_properties[
                        self.split_properties.index.get_level_values("co2_level")
                        == co2l
                    ].lost_load_share_blackout
                    for co2l in self.co2l_list
                ]
            ),
        )
        self.unique_vecs = np.array(list(self.unique_vector_to_idxs_and_weights.keys()))
        self.weights_filtered = np.array([d["weight"] for d in self.unique_vector_to_idxs_and_weights.values()])

    def set_distance_matrix_path(self):
        self.distance_metric_path = ""

    def set_distance_metric_func(self) -> Callable:
        if self.distance_metric_kwargs.get("distance_metric") == "bACC":
            return bACC_weighted_pairwise
        elif self.distance_metric_kwargs.get("distance_metric") == "ACC":
            return ACC_weighted_pairwise
        elif self.distance_metric_kwargs.get("distance_metric") == "boundary_field_ACC":
            return boundary_field_ACC_weighted_pairwise
        else:
            raise ValueError(
                f"Unknown distance metric: {self.distance_metric_kwargs.get('distance_metric')}"
            )

    def get_distance_matrix(self):
        self.set_distance_metric_func()
        self.set_distance_matrix_path()
        if os.path.exists(self.distance_matrix_fpath):
            distance_matrix = self.load_distance_matrix()
            print("Distance matrix loaded from file.")
        else:
            print("Calculating distance matrix...")
            distance_matrix = calc_distance_matrix(
                data=self.,
                **self.distance_metric_kwargs,
            )
            with gzip.open(self.distance_matrix_path, "wb") as fh_out:
                pickle.dump(distance_matrix, fh_out)

    def load_distance_matrix(self):
        with gzip.open(self.distance_matrix_path, "rb") as fh_in:
            distance_matrix = pickle.load(fh_in)
        return distance_matrix

    def post_process_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        if self.distance_matrix_post_processing is not None:
            distance_matrix = self.distance_matrix_post_processing(distance_matrix)
        return distance_matrix

    def fit_clusters(self):
        alg_params = self.clustering_params["alg_params"]
        fit_params = self.clustering_params["fit_params"]
        
        alg = self.clustering_algorithm(**alg_params)
        
        alg.fit(self.distance_matrix, **fit_params)
        with gzip.open(
            self.save_dir + f"/clustering_{self.clustering_algorithm.__name__}_fitted.pklz",
            "wb",
        ) as fh_out:
            pickle.dump(alg, fh_out)
    
    def prepare_visualize_clusters(self):
        prepare_clusters_for_analysis(
            clustering_res_path=self.save_dir
            + f"/clustering_{self.clustering_algorithm.__name__}_fitted.pklz", self.unique_vector_to_idxs_and_weights, self.split_properties, overwrite=False)

    def plot_clusters(self):
        pass
