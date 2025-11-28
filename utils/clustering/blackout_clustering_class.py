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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.model_selection import ParameterSampler
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
    hamming_distance_pairwise,
)
from utils.clustering.boundary import (
    boundary_field,
    low_weight_boundary_field,
    neighborhood_homo_batch,
)
from utils.clustering.visualization import prepare_clusters_for_analysis
from utils.config import path_to_indicator_vectors_sclopf, path_to_vis_results_sclopf

# Preprocessing method registry
PREPROCESSING_METHODS = {
    "boundary_field": boundary_field,
    "neighborhood_homo_batch": neighborhood_homo_batch,
    "low_weight_boundary_field": low_weight_boundary_field,
}
CLUSTERING_ALGORITHMS = {
    "agg": AgglomerativeClustering,
    "dbscan": DBSCAN,
    "optics": OPTICS,
}


# Module-level function for parallel execution (must be picklable)
def _fit_clustering_worker(
    distance_matrix, alg_func, params, fit_params, calc_silhouette=False
):
    """Worker function for parallel clustering execution.

    This is defined at module level to be picklable by joblib.
    """
    model = alg_func(**params)
    model.fit(distance_matrix, **fit_params)

    silhouette_avg = None
    if calc_silhouette:
        from sklearn.metrics import silhouette_score

        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(
            distance_matrix, cluster_labels, metric="precomputed"
        )
        print(f"Silhouette Score: {silhouette_avg:.4f}")

    return model, silhouette_avg


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
    # weights_filtered: Optional[np.ndarray]
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
    adjacency_matrix: Optional[np.ndarray]
    incidence_matrix: Optional[np.ndarray]
    laplacian: Optional[np.ndarray]
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

    def set_attribute(
        self,
        key: str,
        value: Any,
    ):
        """Set an attribute of the class."""
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
        # self.weights_filtered = np.array(
        #     [d["weight"] for d in self.unique_vector_to_idxs_and_weights.values()]
        # )
        os.makedirs(self.data_dir, exist_ok=True)
        with gzip.open(
            self.data_dir + f"blackout_vectors_filtered_dict.pklz",
            "wb",
        ) as fh_in:
            pickle.dump(self.vectors, fh_in)

    def _get_preprocessing_config_hash(self, preprocessing_config: dict = None) -> str:
        """Generate hash from preprocessing config for cache naming."""
        import hashlib

        if preprocessing_config is None:
            preprocessing_config = self.distance_metric_kwargs.get("preprocessing", {})
        config_str = str(sorted(preprocessing_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_preprocessed_data(self, preprocessing_config: dict = None) -> dict:
        """Get preprocessed vectors if preprocessing is configured, otherwise return original.

        Args:
            preprocessing_config: Optional preprocessing configuration. If None, checks self.distance_metric_kwargs

        Returns:
            Dict with 'classes' and 'weights' keys containing the data for distance matrix calculation.
            If no preprocessing, returns original unique_vecs as classes and weights_filtered as weights.
        """
        if preprocessing_config is None:
            preprocessing_config = self.distance_metric_kwargs.get("preprocessing")

        if not preprocessing_config:
            return {"classes": self.unique_vecs}

        config = preprocessing_config
        method_name = config["method"]
        target = config.get(
            "target", "classes"
        )  # Default to 'classes' for backward compatibility

        # Validate target
        if target not in ["classes", "weights"]:
            raise ValueError(
                f"Invalid target '{target}'. Must be 'classes' or 'weights'"
            )

        # Generate cache path
        preprocessing_hash = self._get_preprocessing_config_hash(preprocessing_config)
        cache_path = f"{self.distance_matrices_dir}/preprocessed_data_{method_name}_{target}_{preprocessing_hash}.pklz"

        # Try to load from cache
        if os.path.exists(cache_path):
            print(
                f"Loading preprocessed data from cache: {method_name} (target={target})"
            )
            with gzip.open(cache_path, "rb") as f:
                return pickle.load(f)

        # Compute preprocessing
        print(f"Computing preprocessing: {method_name} (target={target})")
        method = PREPROCESSING_METHODS[method_name]
        params = {k: v for k, v in config.items() if k not in ["method", "target"]}

        # Ensure grid matrices are loaded
        self.set_grid_matrices()

        # Inspect method signature and add matching self attributes as kwargs
        import inspect

        sig = inspect.signature(method)
        method_kwargs = {}
        for param_name in sig.parameters:
            if hasattr(self, param_name):
                method_kwargs[param_name] = getattr(self, param_name)

        # Merge with user-provided params (user params take precedence)
        method_kwargs.update(params)

        # Call preprocessing method with matched arguments
        preprocessed = method(self.unique_vecs, **method_kwargs)

        # Apply preprocessing based on target
        result = {}
        if target == "classes":
            result["classes"] = preprocessed
        elif target == "weights":
            result["classes"] = self.unique_vecs
            result["weights"] = preprocessed

        # Save to disk (separate file, doesn't overwrite unique_vecs)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with gzip.open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Preprocessed data saved to: {cache_path}")

        return result

    def _generate_distance_matrix_path(self, metric_kwargs: dict) -> str:
        """Generate a path for a distance matrix based on metric configuration."""
        metric_name = metric_kwargs.get("name", "unknown")
        # Create a hash of the full kwargs to ensure uniqueness
        import hashlib

        kwargs_str = str(sorted(metric_kwargs.items()))
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]

        return (
            self.distance_matrices_dir
            + f"/distance_matrix_{self.indicator_type}_{self.transformation}_{metric_name}_{kwargs_hash}.pklz"
        )

    def set_distance_matrix_path(self):
        import hashlib

        # Create combined config for hashing (includes preprocessing)
        config_for_hash = {
            "metric": {
                k: v
                for k, v in self.distance_metric_kwargs.items()
                if k != "preprocessing"
            },
            "preprocessing": self.distance_metric_kwargs.get("preprocessing", {}),
        }
        config_str = str(sorted(config_for_hash.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        if self.distance_metric_kwargs.get("name") == "composite":
            # For composite metrics, generate a combined path
            metric_names = "_".join(
                [m["name"] for m in self.distance_metric_kwargs["metrics"]]
            )
            self.distance_matrix_path = (
                self.distance_matrices_dir
                + f"distance_matrix_{self.indicator_type}_{self.transformation}_composite_{metric_names}_{config_hash}.pklz"
            )
        else:
            # Single metric path
            metric_name = self.distance_metric_kwargs.get("name", "unknown")
            self.distance_matrix_path = (
                self.distance_matrices_dir
                + f"distance_matrix_{self.indicator_type}_{self.transformation}_{metric_name}_{config_hash}.pklz"
            )

    def get_distance_metric_func(self, metric) -> Callable:
        if isinstance(metric, str):
            if metric == "bACC":
                return bACC_weighted_pairwise
            elif metric == "ACC":
                return ACC_weighted_pairwise
            elif metric == "cosine_distance":
                from sklearn.metrics.pairwise import cosine_distances

                return cosine_distances
            elif metric == "hamming":
                return hamming_distance_pairwise

            # elif self.distance_metric_kwargs.get("name") == "boundary_field_ACC":
            #     return boundary_field_ACC_weighted_pairwise
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
        else:
            return metric

    def get_single_distance_matrix(self, metric_kwargs: dict) -> np.ndarray:
        """Calculate or load a single distance matrix for the given metric configuration.

        Args:
            metric_kwargs: Dictionary containing metric configuration including 'name', 'preprocessing', and other parameters

        Returns:
            Distance matrix as numpy array
        """
        matrix_path = self._generate_distance_matrix_path(metric_kwargs)

        if os.path.exists(matrix_path):
            print(f"Loading distance matrix for {metric_kwargs.get('name')}...")
            with gzip.open(matrix_path, "rb") as fh_in:
                distance_matrix = pickle.load(fh_in)
        else:
            # Handle preprocessing if specified in metric_kwargs
            preprocessing_config = metric_kwargs.get("preprocessing")
            data_dict = self._get_preprocessed_data(preprocessing_config)

            # Extract classes and weights from the dict
            samples = data_dict["classes"]
            weights = data_dict.get("weights")

            print(f"Calculating distance matrix for {metric_kwargs.get('name')}...")
            distance_metric = self.get_distance_metric_func(metric_kwargs["name"])
            print("Calculating distance matrix...")
            distance_matrix = calc_distance_matrix(
                samples=samples,
                metric=distance_metric,
                weights=weights,
                **{
                    k: v
                    for k, v in metric_kwargs.items()
                    if k not in ["name", "preprocessing"]
                },
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

        Each component metric can have its own preprocessing configuration.
        """
        metrics_list = self.distance_metric_kwargs["metrics"]
        combiner = self.distance_metric_kwargs["combiner"]
        cache_components = self.distance_metric_kwargs.get("cache_components", True)

        # Initialize storage for component matrices
        self.component_distance_matrices = {}
        if cache_components:
            self.component_distance_matrix_paths = {}

        # Calculate or load each component matrix (each handles its own preprocessing)
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
        """Calculate or load distance matrix with optional preprocessing.

        Automatically detects whether to use single or composite metric
        based on distance_metric_kwargs['name'].

        For composite metrics, each component can have its own preprocessing.
        """
        self.set_distance_matrix_path()

        # Check if distance matrix already exists
        if os.path.exists(self.distance_matrix_path):
            self.load_distance_matrix()
            print("Distance matrix loaded from file.")
            return

        # Check if this is a composite metric
        if self.distance_metric_kwargs.get("name") == "composite":
            # Calculate composite matrix from components (each handles its own preprocessing)
            self.get_composite_distance_matrix()
        else:
            # Single metric - handle preprocessing here
            data_dict = self._get_preprocessed_data()
            samples = data_dict["classes"]
            weights = data_dict["weights"]

            self.distance_matrix = self.get_single_distance_matrix(
                self.distance_metric_kwargs
            )

    def load_distance_matrix(self):
        with gzip.open(self.distance_matrix_path, "rb") as fh_in:
            distance_matrix = pickle.load(fh_in)
        self.distance_matrix = distance_matrix

    def post_process_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        if self.distance_matrix_post_processing is not None:
            distance_matrix = self.distance_matrix_post_processing(distance_matrix)
        return distance_matrix

    def _fit_single_clustering(
        self, alg_func, params, fit_params, calc_silhouette=False
    ) -> tuple[Any, Optional[float]]:
        """Helper to fit a single clustering model.

        Args:
            alg_func: Clustering algorithm class
            params: Hyperparameters for the algorithm
            fit_params: Parameters for the fit method

        Returns:
            Fitted clustering model
        """
        model = alg_func(**params)
        model.fit(self.distance_matrix, **fit_params)

        silhouette_avg = None
        if calc_silhouette:
            from sklearn.metrics import silhouette_score

            cluster_labels = model.labels_
            silhouette_avg = silhouette_score(
                self.distance_matrix, cluster_labels, metric="precomputed"
            )
            print(f"Silhouette Score: {silhouette_avg:.4f}")

        return model, silhouette_avg

    def fit_clusters(self):
        """Fit clustering models with hyperparameter search.

        Saves each fitted model with unique filename based on algorithm name and
        parameter hash. Also creates an index file listing all results.

        Returns:
            List of dicts containing filepath, algorithm name, and params for each result
        """
        all_results = []

        for alg, alg_params in self.clustering_params.items():
            alg_params = alg_params.copy()
            n_iter = alg_params.pop("n_iter", 1)
            n_jobs = alg_params.pop("n_jobs", 1)
            calc_silhouette = alg_params.pop("calc_silhouette", False)
            fit_params = alg_params.pop("fit_params", {})
            HP_params = list(
                ParameterSampler(alg_params["HPs"], n_iter=n_iter, random_state=42)
            )

            if isinstance(alg, str):
                alg_func = CLUSTERING_ALGORITHMS[alg]
                alg_name = alg
            else:
                alg_func = alg
                alg_name = alg.__name__

            # Fit models in parallel and collect results
            fitted_models_and_silhouettes = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                max_nbytes=None,
                temp_folder=None,
            )(
                delayed(_fit_clustering_worker)(
                    self.distance_matrix,
                    alg_func,
                    params,
                    fit_params,
                    calc_silhouette=calc_silhouette,
                )
                for params in tqdm(HP_params, desc=alg_name)
            )

            # Save each result with unique filename
            os.makedirs(self.cluster_dir, exist_ok=True)
            for i, ((model, silhouette_avg), params) in enumerate(
                zip(fitted_models_and_silhouettes, HP_params)
            ):
                result = {
                    "model": model,
                    "algorithm": alg_name,
                    "params": params,
                    "fit_params": fit_params,
                }
                if silhouette_avg is not None:
                    result["silhouette_score"] = silhouette_avg

                # Create unique filename with algorithm name and parameter hash
                import hashlib

                param_str = str(sorted(params.items()))
                param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

                filename = f"{alg_name}_params_{param_hash}.pklz"
                filepath = os.path.join(self.cluster_dir, filename)

                with gzip.open(filepath, "wb") as fh_out:
                    pickle.dump(result, fh_out)

                all_results.append(
                    {
                        "filepath": filepath,
                        "filename": filename,
                        "algorithm": alg_name,
                        "params": params,
                    }
                )
                if silhouette_avg is not None:
                    all_results[-1]["silhouette_score"] = silhouette_avg

        # Save index of all results
        index_path = os.path.join(self.cluster_dir, "clustering_results_index.pklz")
        with gzip.open(index_path, "wb") as f:
            pickle.dump(all_results, f)

        print(f"\nSaved {len(all_results)} clustering results to {self.cluster_dir}")
        print(f"Results index saved to: {index_path}")

        # Set the last result as the default for backward compatibility
        if all_results:
            self.clustering_result_filename = all_results[-1]["filename"]

        return all_results

    def load_clustering_results_index(self):
        """Load the index of all clustering results.

        Returns:
            List of dicts containing filepath, algorithm, and params for each result
        """
        index_path = os.path.join(self.cluster_dir, "clustering_results_index.pklz")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Results index not found at {index_path}")

        with gzip.open(index_path, "rb") as f:
            return pickle.load(f)

    def load_clustering_result(self, filename):
        """Load a specific clustering result.

        Args:
            filename: Name of the result file to load

        Returns:
            Dict containing 'model', 'algorithm', 'params', and 'fit_params'
        """
        filepath = os.path.join(self.cluster_dir, filename)
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)

    def asses_clustering_results(self):
        raise NotImplementedError
        #                 unique_labels = np.unique(cluster_labels)
        #         n_clusters = len(unique_labels)

        #         # Remove noise label (-1) if present for cluster count
        #         if -1 in unique_labels:
        #             n_clusters -= 1

        #         silhouette_avg = None

        #         if n_clusters > 1 and len(cluster_labels) > 1:
        #             silhouette_avg = silhouette_score(
        #                 distance_matrix, cluster_labels, metric="precomputed"
        #             )
        #         else:
        #             logger.warning(
        #                 f"Cannot calculate silhouette score for {file}: insufficient clusters (n_clusters={n_clusters})"
        #             )

        #         # Store results
        #         result_info = {
        #             "filename": file,
        #             "algorithm_name": algorithm_name,
        #             "n_clusters": n_clusters,
        #             "n_samples": len(cluster_labels),
        #             "silhouette_score": silhouette_avg,
        #             "has_noise": -1 in unique_labels,
        #             "n_noise_points": (
        #                 np.sum(cluster_labels == -1) if -1 in unique_labels else 0
        #             ),
        #         }

    def prepare_visualize_clusters(self, result_filename=None):
        """Prepare clustering results for visualization.

        Args:
            result_filename: Specific result file to prepare. If None, uses self.clustering_result_filename
        """
        if result_filename is None:
            result_filename = self.clustering_result_filename

        prepare_clusters_for_analysis(
            os.path.join(self.cluster_dir, result_filename),
            self.unique_vector_to_idxs_and_weights,
            self.split_properties,
            overwrite=False,
        )

    def plot_cluster(self, result_filename=None, **kwargs):
        """Plot clustering results.

        Args:
            result_filename: Specific result file to plot. If None, uses self.clustering_result_filename
            **kwargs: Additional arguments passed to plotting function
        """
        if result_filename is None:
            result_filename = self.clustering_result_filename

        create_clustering_analysis_plot(
            clustering_res_full_path=os.path.join(self.cluster_dir, result_filename),
            plot_dir=self.plot_dir,
            **self.plotting_params,
            co2l_list=self.co2l_list,
            n_nodes=self.n_nodes,
            **kwargs,
        )

    def plot_cluster_multiple(
        self, result_filenames: List[str] = [], n_best=None, **plot_kwargs
    ):
        """Plot multiple clustering results.

        Args:
            n_best: Number of best results to plot based on silhouette score.
            result_filenames: List of specific result files to plot.
            **kwargs: Additional arguments passed to plotting function
        """
        # Load all results index
        all_results = self.load_clustering_results_index()
        if not result_filenames:
            if "silhouette_score" not in all_results[0]:
                raise ValueError("No silhouette scores found in results index.")
            else:
                # Sort results by silhouette score in descending order
                sorted_results = sorted(
                    all_results,
                    key=lambda x: x.get("silhouette_score", -1),
                    reverse=True,
                )
                result_filenames = [res["filename"] for res in sorted_results]
                if n_best is not None:
                    result_filenames = result_filenames[:n_best]
                else:
                    print(
                        "No n_best specified, plotting all results sorted by silhouette score."
                    )
        for filename in result_filenames:
            self.prepare_visualize_clusters(result_filename=filename)
            self.plot_cluster(result_filename=filename, **plot_kwargs)

    def run_all(self):
        self.load_data()
        self.transform_vectors()
        self.filter_data()
        self.get_distance_matrix()
        self.fit_clusters()
        self.prepare_visualize_clusters()
        self.plot_cluster()
