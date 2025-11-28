import contextlib
import gzip
import os
import pickle
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Callable, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from loguru import logger
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import diags

from utils.data_handling import load_grid_matrices

sys.path.append("../")

from scripts.plots.plot_clustering_analysis import create_clustering_analysis_plot
from utils.clustering.data_handling import (
    get_path_to_clustering_dir,
    get_split_mask,
    get_unique_vectors_with_weights,
    transform_indicator_vectors,
)
from utils.clustering.distance_matrix_calc import calc_distance_matrix
from utils.clustering.distance_metrics import (
    ACC_weighted_pairwise,
    bACC_weighted_pairwise,
)
from utils.clustering.visualization import prepare_clusters_for_analysis
from utils.config import path_to_indicator_vectors_sclopf, path_to_vis_results_sclopf


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
    distance_matrix_path: Optional[str]
    distance_metric: Optional[Callable]
    distance_metric_kwargs: Optional[dict]
    distance_matrix: Optional[np.ndarray]
    distance_matrix_post_processing: Optional[Optional[Callable]]
    component_distance_matrices: Optional[dict[str, np.ndarray]]
    component_distance_matrix_paths: Optional[dict[str, str]]
    clustering_results_dir: Optional[str]
    clustering_result_filename: Optional[str]
    clustering_algorithm: Optional[Callable]
    clustering_params: Optional[dict]
    save_dir: Optional[str]
    data_dir: Optional[str]
    distance_matrices_dir: Optional[str]
    cluster_dir: Optional[str]
    plot_dir: Optional[str]
    random_subsample_size: Optional[float] = 1
    plotting_params: Optional[dict[str, Any]] = {}
    I: Optional[sparse.csr_matrix]
    B: Optional[sparse.csr_matrix]
    L: Optional[sparse.csr_matrix]
    D: Optional[sparse.csr_matrix]
    A: Optional[sparse.csr_matrix]

    def __init__(
        self,
        n_nodes,
        co2l,
        indicator_type,
        transformation,
        blackout_size_threshold,
        **kwargs,
    ):
        self.n_nodes = n_nodes
        self.co2l_list = co2l if isinstance(co2l, list) else [co2l]
        self.indicator_type = indicator_type
        self.transformation = transformation
        self.blackout_size_threshold = blackout_size_threshold
        self.save_dir = get_path_to_clustering_dir(
            n_nodes,
            co2l,
            indicator_type,
            transformation,
            self.blackout_size_threshold,
        )
        self.data_dir = self.save_dir + "data/"
        self.distance_matrices_dir = self.save_dir + "distance_matrices/"
        self.cluster_dir = self.save_dir + "clustering_results/"
        self.plot_dir = self.save_dir + "clustering_plots/"

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_grid_matrices(self, snet_index=0, co2lvl=0.0):
        self.I, self.B, self.num_par, self.line_limits = load_grid_matrices(
            snet_index=snet_index, co2lvl=co2lvl
        )
        self.L = self.I.dot(self.B).dot(self.I.T)
        self.D = diags(self.L.diagonal())
        self.A = self.D - self.L

    def load_data(self):
        if not "split_properties" in self.__dict__:
            self.split_properties = pd.read_hdf(
                path_to_vis_results_sclopf + f"split_properties_all_n{self.n_nodes}.h5"
            )
            co2l_mask = self.split_properties.index.get_level_values("co2l").isin(
                self.co2l_list
            )
            self.split_properties = self.split_properties[co2l_mask]
        if not "vectors" in self.__dict__:
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

    def transform_vectors(self):
        self.vectors = {
            co2l: transform_indicator_vectors(
                vectors_lvl,
                self.transformation,
            )
            for co2l, vectors_lvl in self.vectors.items()
        }

    def filter_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.masks_dict = get_split_mask(
            self.n_nodes,
            self.co2l_list,
            self.blackout_size_threshold,
            save_dir=self.data_dir,
            split_props=self.split_properties,
            err_on_missing=False,
            random_subsample_size=self.random_subsample_size,
        )

        for co2l in self.co2l_list:
            index_mask = self.masks_dict[co2l]
            mask_full = self.split_properties.index.get_level_values("co2l") != co2l
            mask_full[self.split_properties.index.get_level_values("co2l") == co2l] = (
                index_mask
            )
            self.split_properties = self.split_properties[mask_full]
            self.vectors[co2l] = self.vectors[co2l][index_mask]
        self.split_properties.to_hdf(
            f"{self.data_dir}/data_filtered.h5",
            key="split_props",
        )
        self.unique_vector_to_idxs_and_weights = get_unique_vectors_with_weights(
            np.concatenate([self.vectors[co2l] for co2l in self.co2l_list]),
            np.concatenate(
                [
                    self.split_properties[
                        self.split_properties.index.get_level_values("co2l") == co2l
                    ].lost_load_share_blackout
                    for co2l in self.co2l_list
                ]
            ),
        )
        self.unique_vecs = np.array(list(self.unique_vector_to_idxs_and_weights.keys()))
        self.weights_filtered = np.array(
            [d["weight"] for d in self.unique_vector_to_idxs_and_weights.values()]
        )
        os.makedirs(self.data_dir, exist_ok=True)
        with gzip.open(
            self.data_dir + f"blackout_vectors_filtered_dict.pklz",
            "wb",
        ) as fh_in:
            pickle.dump(self.vectors, fh_in)

    # def subsample_data(self):
    #     if self.random_subsample_size != 1:
    #         rng = np.random.default_rng(42)
    #         for co2l in self.co2l_list:
    #             selected_indices = self.masks_dict[co2l].values.nonzero()
    #             subsample_mask = np.zeros(self.vectors[co2l].shape[0], dtype=bool)
    #             subsampled_indices = rng.choice(
    #                 selected_indices[0],
    #                 int(self.random_subsample_size * len(selected_indices[0])),
    #                 replace=False,
    #             )
    #             subsample_mask[subsampled_indices] = True
    #             self.masks_dict[co2l] = subsample_mask
    #     masks_dict_path = self.data_dir + "masks_dict.pklz"
    #     with gzip.open(masks_dict_path, "wb") as fh_out:
    #         pickle.dump(self.masks_dict, fh_out)

    def _generate_distance_matrix_path(self, metric_kwargs: dict) -> str:
        """Generate a path for a distance matrix based on metric configuration."""
        metric_name = metric_kwargs.get("name", "unknown")
        # Create a hash of the full kwargs to ensure uniqueness
        import hashlib

        kwargs_str = str(sorted(metric_kwargs.items()))
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]

        return (
            +self.distance_matrices_dir
            + f"/distance_matrix_{self.indicator_type}_{self.transformation}_{metric_name}_{kwargs_hash}.pklz"
        )

    def set_distance_matrix_path(self):
        if self.distance_metric_kwargs.get("name") == "composite":
            # For composite metrics, generate a combined path
            metric_names = "_".join(
                [m["name"] for m in self.distance_metric_kwargs["metrics"]]
            )
            import hashlib

            kwargs_str = str(sorted(self.distance_metric_kwargs.items()))
            kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
            self.distance_matrix_path = (
                self.data_dir
                + f"/distance_matrix_{self.indicator_type}_{self.transformation}_composite_{metric_names}_{kwargs_hash}.pklz"
            )
        else:
            # Single metric path (backward compatible)
            self.distance_matrix_path = (
                self.data_dir
                + f"/distance_matrix_{self.indicator_type}_{self.transformation}.pklz"
            )

    def set_distance_metric_func(self) -> Callable:
        if self.distance_metric_kwargs.get("name") == "bACC":
            distance_metric = bACC_weighted_pairwise
        elif self.distance_metric_kwargs.get("name") == "ACC":
            distance_metric = ACC_weighted_pairwise
        # elif self.distance_metric_kwargs.get("name") == "boundary_field_ACC":
        #     return boundary_field_ACC_weighted_pairwise
        else:
            raise ValueError(
                f"Unknown distance metric: {self.distance_metric_kwargs.get('distance_metric')}"
            )

    def get_single_distance_matrix(self, metric_kwargs: dict) -> np.ndarray:
        """Calculate or load a single distance matrix for the given metric configuration.

        Args:
            metric_kwargs: Dictionary containing metric configuration including 'name' and parameters

        Returns:
            Distance matrix as numpy array
        """
        matrix_path = self._generate_distance_matrix_path(metric_kwargs)

        if os.path.exists(matrix_path):
            print(f"Loading distance matrix for {metric_kwargs.get('name')}...")
            with gzip.open(matrix_path, "rb") as fh_in:
                distance_matrix = pickle.load(fh_in)
        else:
            print(f"Calculating distance matrix for {metric_kwargs.get('name')}...")
            distance_matrix = calc_distance_matrix(
                data=self.unique_vecs,
                **{k: v for k, v in metric_kwargs.items() if k != "name"},
            )
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            with gzip.open(matrix_path, "wb") as fh_out:
                pickle.dump(distance_matrix, fh_out)

        return distance_matrix

    def get_composite_distance_matrix(self):
        """Calculate composite distance matrix from multiple component metrics.

        Uses the metric configurations in distance_metric_kwargs['metrics'] to
        calculate individual distance matrices, then combines them using the
        combiner function specified in distance_metric_kwargs['combiner'].
        """
        metrics_list = self.distance_metric_kwargs["metrics"]
        combiner = self.distance_metric_kwargs["combiner"]
        cache_components = self.distance_metric_kwargs.get("cache_components", True)

        # Initialize storage for component matrices
        self.component_distance_matrices = {}
        if cache_components:
            self.component_distance_matrix_paths = {}

        # Calculate or load each component matrix
        component_matrices = []
        for metric_kwargs in metrics_list:
            metric_name = metric_kwargs.get("name", "unknown")
            matrix = self.get_single_distance_matrix(metric_kwargs)

            if cache_components:
                self.component_distance_matrices[metric_name] = matrix
                self.component_distance_matrix_paths[metric_name] = (
                    self._generate_distance_matrix_path(metric_kwargs)
                )

            component_matrices.append(matrix)

        # Combine the matrices
        print("Combining distance matrices...")
        self.distance_matrix = combiner(component_matrices)

        # Save the combined matrix
        os.makedirs(os.path.dirname(self.distance_matrix_path), exist_ok=True)
        with gzip.open(self.distance_matrix_path, "wb") as fh_out:
            pickle.dump(self.distance_matrix, fh_out)

        print("Composite distance matrix created and saved.")

    def get_distance_matrix(self):
        """Calculate or load distance matrix.

        Automatically detects whether to use single or composite metric
        based on distance_metric_kwargs['name'].
        """
        self.set_distance_matrix_path()

        # Check if this is a composite metric
        if self.distance_metric_kwargs.get("name") == "composite":
            # Check if composite matrix already exists
            if os.path.exists(self.distance_matrix_path):
                self.load_distance_matrix()
                print("Composite distance matrix loaded from file.")
            else:
                # Calculate composite matrix from components
                self.get_composite_distance_matrix()
        else:
            # Single metric (original behavior)
            self.set_distance_metric_func()
            if os.path.exists(self.distance_matrix_path):
                self.load_distance_matrix()
                print("Distance matrix loaded from file.")
            else:
                print("Calculating distance matrix...")
                self.distance_matrix = calc_distance_matrix(
                    data=self.unique_vecs,
                    **{
                        k: v
                        for k, v in self.distance_metric_kwargs.items()
                        if k != "name"
                    },
                )
                os.makedirs(os.path.dirname(self.distance_matrix_path), exist_ok=True)
                with gzip.open(self.distance_matrix_path, "wb") as fh_out:
                    pickle.dump(self.distance_matrix, fh_out)

    def load_distance_matrix(self):
        with gzip.open(self.distance_matrix_path, "rb") as fh_in:
            distance_matrix = pickle.load(fh_in)
        self.distance_matrix = distance_matrix

    def post_process_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        if self.distance_matrix_post_processing is not None:
            distance_matrix = self.distance_matrix_post_processing(distance_matrix)
        return distance_matrix

    def fit_clusters(self):
        alg_params = self.clustering_params["alg_params"]
        fit_params = self.clustering_params.get("fit_params", {})

        alg = self.clustering_algorithm(**alg_params)

        alg.fit(self.distance_matrix, **fit_params)
        os.makedirs(self.cluster_dir, exist_ok=True)
        self.clustering_result_filename = (
            f"clustering_{self.clustering_algorithm.__name__}_fitted.pklz"
        )
        with gzip.open(
            self.cluster_dir + self.clustering_result_filename,
            "wb",
        ) as fh_out:
            pickle.dump(alg, fh_out)

    def prepare_visualize_clusters(self):
        prepare_clusters_for_analysis(
            self.cluster_dir + self.clustering_result_filename,
            self.unique_vector_to_idxs_and_weights,
            self.split_properties,
            overwrite=False,
        )

    def plot_cluster(self, **kwargs):
        create_clustering_analysis_plot(
            clustering_res_full_path=self.cluster_dir + self.clustering_result_filename,
            plot_dir=self.plot_dir,
            **self.plotting_params,
            co2l_list=self.co2l_list,
            n_nodes=self.n_nodes,
            **kwargs,
        )

    def run_all(self):
        self.load_data()
        self.transform_vectors()
        self.filter_data()
        # self.subsample_data()
        self.get_distance_matrix()
        self.fit_clusters()
        self.prepare_visualize_clusters()
        self.plot_cluster()
