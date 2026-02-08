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
    hamming_distance_weighted_pairwise,
    bACC_weighted_pairwise,
    hamming_distance_pairwise,
)
from utils.clustering.boundary import (
    boundary_field_rw_smoothing,
    low_weight_boundary_field,
    neighborhood_homo_batch,
)
from utils.clustering.visualization import prepare_clusters_for_analysis
from utils.config import (
    path_to_indicator_vectors_sclopf,
    path_to_vis_results_sclopf,
    path_to_evaluation_results_sclopf,
)

# Preprocessing method registry
PREPROCESSING_METHODS = {
    "boundary_field": boundary_field_rw_smoothing,
    "neighborhood_homo_batch": neighborhood_homo_batch,
    "low_weight_boundary_field": low_weight_boundary_field,
}
# # CLUSTERING_ALGORITHMS = {
#     "agg": AgglomerativeClustering,
#     "dbscan": DBSCAN,
#     "optics": OPTICS,
# "hdbscan": HDBSCAN,
# }


# Module-level function for parallel execution (must be picklable)
def _fit_clustering_worker(
    distance_matrix, alg_name, params, fit_params, calc_silhouette=False
):
    """Worker function for parallel clustering execution.

    This is defined at module level to be picklable by joblib.
    All imports must be done inside the function to ensure they're available in worker processes.

    Args:
        distance_matrix: Precomputed distance matrix
        alg_name: String name of algorithm ('agg', 'dbscan', 'optics')
        params: Algorithm hyperparameters
        fit_params: Parameters for fit method
        calc_silhouette: Whether to calculate silhouette score
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS

    # Resolve algorithm from name
    alg_map = {
        "agg": AgglomerativeClustering,
        "dbscan": DBSCAN,
        "optics": OPTICS,
    }
    alg_func = alg_map.get(
        alg_name, alg_name
    )  # fallback to alg_name if it's already a class

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
    random_subsample_size: Optional[float] = None
    plotting_params: Optional[dict[str, Any]] = {}
    distance_matrix_dtype: Optional[type] = None  # Cast to this dtype in memory
    clustering_worker_func: Optional[Callable] = (
        None  # External worker function for pickling
    )
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
        self.co2l_list = sorted(self.co2l_list)
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
        self.I, self.B_values, self.num_par, self.line_limits = load_grid_matrices(
            snet_index=snet_index, co2lvl=co2lvl
        )
        self.L = self.I.dot(self.I.T)
        if np.min(self.L.todense()) != -1:
            raise ValueError("Laplacian L must have -1 off-diagonal entries")
        if np.min(self.L.diagonal()) < 1:
            raise ValueError("Degree matrix D must have positive diagonal entries")
        self.D = diags(self.L.diagonal())
        self.A = self.D - self.L
        if set(np.unique(np.array(self.A.todense().flatten().squeeze()))) != {0, 1}:
            raise ValueError("Incidence matrix I must be binary (0/1)")

    def load_data(self):
        if not "split_properties" in self.__dict__:
            self.split_properties = pd.read_hdf(
                path_to_vis_results_sclopf + f"split_properties_all_n{self.n_nodes}.h5"
            )
            co2l_mask = self.split_properties.index.get_level_values("co2l").isin(
                self.co2l_list
            )
            self.split_properties = self.split_properties[co2l_mask]
            if (
                self.split_properties.index.get_level_values("co2l").unique().tolist()
                != self.co2l_list
            ):
                raise ValueError(
                    "Order of CO2 levels in split_properties does not match the specified co2l_list."
                )
        if not "vectors" in self.__dict__:
            vectors_per_lvl = {}
            for co2lvl in self.co2l_list:
                file_name = f"{self.indicator_type}_indicator_vectors_Co2L{co2lvl}_n{self.n_nodes}.pklz"
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
        print(
            f"Number of unique vectors after filtering: {len(self.unique_vector_to_idxs_and_weights)}"
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
        if target == "weights" and (preprocessed < 0).any():
            raise ValueError(
                f"Preprocessing method '{method_name}' produced negative weights."
            )

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
            elif metric == "hamming":
                return hamming_distance_pairwise
            elif metric == "hamming_weighted":
                return hamming_distance_weighted_pairwise
            elif metric == "cosine_distance":
                from sklearn.metrics.pairwise import cosine_distances

                return cosine_distances

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
            # Check for NaNs
            if np.isnan(distance_matrix).any():
                raise ValueError(
                    f"Distance matrix loaded from {matrix_path} contains NaN values. "
                    "This indicates corrupted data or calculation errors."
                )
            # Cast to specified dtype in memory if configured
            if self.distance_matrix_dtype is not None:
                distance_matrix = distance_matrix.astype(self.distance_matrix_dtype)
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
            # Check for NaNs after calculation
            if np.isnan(distance_matrix).any():
                raise ValueError(
                    f"Calculated distance matrix for {metric_kwargs.get('name')} contains NaN values. "
                    "Check your distance metric implementation and input data."
                )
            # Save at full precision
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            with gzip.open(matrix_path, "wb") as fh_out:
                pickle.dump(distance_matrix, fh_out)
            # Cast to specified dtype in memory if configured
            if self.distance_matrix_dtype is not None:
                distance_matrix = distance_matrix.astype(self.distance_matrix_dtype)

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
        combined_matrix = combiner(component_matrices)

        # Check for NaNs in combined matrix
        if np.isnan(combined_matrix).any():
            raise ValueError(
                "Combined distance matrix contains NaN values. "
                "Check your combiner function and component matrices."
            )

        # Save the combined matrix at full precision
        os.makedirs(os.path.dirname(self.distance_matrix_path), exist_ok=True)
        with gzip.open(self.distance_matrix_path, "wb") as fh_out:
            pickle.dump(combined_matrix, fh_out)

        # Cast to specified dtype in memory if configured
        if self.distance_matrix_dtype is not None:
            combined_matrix = combined_matrix.astype(self.distance_matrix_dtype)
        self.distance_matrix = combined_matrix

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

            self.distance_matrix = self.get_single_distance_matrix(
                self.distance_metric_kwargs
            )

    def load_distance_matrix(self):
        with gzip.open(self.distance_matrix_path, "rb") as fh_in:
            distance_matrix = pickle.load(fh_in)
        # Check for NaNs
        if np.isnan(distance_matrix).any():
            raise ValueError(
                f"Distance matrix loaded from {self.distance_matrix_path} contains NaN values. "
                "This indicates corrupted data or calculation errors."
            )
        # Cast to specified dtype in memory if configured
        if self.distance_matrix_dtype is not None:
            distance_matrix = distance_matrix.astype(self.distance_matrix_dtype)
        self.distance_matrix = distance_matrix

    def post_process_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        if self.distance_matrix_post_processing is not None:
            distance_matrix = self.distance_matrix_post_processing(distance_matrix)
        return distance_matrix

    def _convert_numpy_types_to_python(self, params: dict) -> dict:
        """Convert numpy types to native Python types for compatibility.

        Some libraries (like HDBSCAN) have strict type checking that fails
        with numpy types even though they're functionally equivalent.

        Args:
            params: Dictionary of parameters potentially containing numpy types

        Returns:
            Dictionary with numpy types converted to Python types
        """
        params_converted = {}
        for key, val in params.items():
            if isinstance(val, np.integer):
                params_converted[key] = int(val)
            elif isinstance(val, np.floating):
                params_converted[key] = float(val)
            else:
                params_converted[key] = val
        return params_converted

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

            # Convert numpy types to native Python types for compatibility
            HP_params = [self._convert_numpy_types_to_python(p) for p in HP_params]

            if isinstance(alg, str):
                # alg_func = CLUSTERING_ALGORITHMS[alg]
                alg_name = alg
            else:
                # alg_func = alg
                alg_name = alg.__name__

            # Fit models in parallel and collect results
            # Use external worker function if provided (for script execution)
            # Otherwise use module-level worker (for interactive use)
            worker_func = (
                self.clustering_worker_func
                if self.clustering_worker_func is not None
                else _fit_clustering_worker
            )

            fitted_models_and_silhouettes = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                max_nbytes=None,
                temp_folder=None,
            )(
                delayed(worker_func)(
                    self.distance_matrix,
                    alg_name,  # Pass name instead of class for pickling
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

        # Create/update index using dedicated function
        self._save_clustering_results_index(all_results)

        # Set the last result as the default for backward compatibility
        if all_results:
            self.clustering_result_filename = all_results[-1]["filename"]

        return all_results

    def _save_clustering_results_index(self, all_results: List[dict]):
        """Save clustering results index to disk.

        Args:
            all_results: List of result dictionaries to save
        """
        index_path = os.path.join(self.cluster_dir, "clustering_results_index.pklz")
        with gzip.open(index_path, "wb") as f:
            pickle.dump(all_results, f)

        print(f"\nSaved {len(all_results)} clustering results to {self.cluster_dir}")
        print(f"Results index saved to: {index_path}")

    def create_clustering_results_index(self, overwrite: bool = True):
        """Create an index of all clustering results by scanning the cluster directory.

        Args:
            overwrite: If True, recreate index even if it already exists

        Returns:
            List of dicts containing filepath, filename, algorithm, params, and optionally silhouette_score
        """
        index_path = os.path.join(self.cluster_dir, "clustering_results_index.pklz")

        # Check if index exists and overwrite is False
        if os.path.exists(index_path) and not overwrite:
            print(
                f"Index already exists at {index_path}. Use overwrite=True to recreate."
            )
            return self.load_clustering_results_index()

        # Scan directory for result files
        all_results = []
        if not os.path.exists(self.cluster_dir):
            print(f"Cluster directory does not exist: {self.cluster_dir}")
            return all_results

        print(f"Scanning {self.cluster_dir} for clustering results...")
        for filename in os.listdir(self.cluster_dir):
            if (
                filename.endswith(".pklz")
                and filename != "clustering_results_index.pklz"
            ):
                filepath = os.path.join(self.cluster_dir, filename)

                try:
                    # Load result to extract metadata
                    with gzip.open(filepath, "rb") as f:
                        result = pickle.load(f)

                    result_info = {
                        "filepath": filepath,
                        "filename": filename,
                        "algorithm": result.get("algorithm", "unknown"),
                        "params": result.get("params", {}),
                    }

                    # Add silhouette score if available
                    if "silhouette_score" in result:
                        result_info["silhouette_score"] = result["silhouette_score"]

                    all_results.append(result_info)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        # Save index using dedicated function
        self._save_clustering_results_index(all_results)
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
        self,
        result_filenames: List[str] = [],
        n_best: int = None,
        relative_score_threshold: float = None,
        algorithm: str = None,
        **plot_kwargs,
    ):
        """Plot multiple clustering results.

        Args:
            result_filenames: List of specific result files to plot. If provided, ignores n_best and algorithm.
            n_best: Number of best results to plot based on silhouette score.
                If None, plots all results for the specified algorithm(s).
            algorithm: Filter results by algorithm name (e.g., 'agg', 'dbscan', 'optics').
            relative_score_threshold: If provided, only plots results with silhouette score within this fraction of the best score.
                If None, plots n_best from each algorithm separately.
            **plot_kwargs: Additional arguments passed to plotting function
        """
        if n_best is not None and relative_score_threshold is not None:
            raise ValueError("Cannot specify both n_best and relative_score_threshold.")

        # Load all results index
        if not result_filenames:
            all_results = self.load_clustering_results_index()

        if result_filenames:
            # Use provided filenames directly
            pass
        elif "silhouette_score" not in all_results[0]:
            raise ValueError("No silhouette scores found in results index.")
        else:
            if algorithm is not None:
                # Filter by specific algorithm
                filtered_results = [
                    res for res in all_results if res["algorithm"] == algorithm
                ]
                if not filtered_results:
                    raise ValueError(f"No results found for algorithm '{algorithm}'")

                # Sort by silhouette score
                sorted_results = sorted(
                    filtered_results,
                    key=lambda x: x.get("silhouette_score", -1),
                    reverse=True,
                )

                # Select n_best
                if n_best is not None:
                    sorted_results = sorted_results[:n_best]
                    print(f"Plotting top {len(sorted_results)} results for {algorithm}")
                else:
                    print(f"Plotting all {len(sorted_results)} results for {algorithm}")

                # Apply relative score threshold if specified
                if relative_score_threshold is not None:
                    best_score = sorted_results[0].get("silhouette_score", 0)
                    threshold_score = best_score * (1 - relative_score_threshold)
                    sorted_results = [
                        res
                        for res in sorted_results
                        if res.get("silhouette_score", 0) >= threshold_score
                    ]
                    print(
                        f"Applying relative score threshold: {relative_score_threshold:.2%} of best score ({best_score:.4f}), resulting in {len(sorted_results)} results to plot."
                    )

                result_filenames = [res["filename"] for res in sorted_results]
            else:
                # Group by algorithm and get n_best from each
                from itertools import groupby

                # Group results by algorithm
                all_results_sorted = sorted(all_results, key=lambda x: x["algorithm"])
                grouped = {
                    alg: list(group)
                    for alg, group in groupby(
                        all_results_sorted, key=lambda x: x["algorithm"]
                    )
                }

                result_filenames = []
                for alg_name, alg_results in grouped.items():
                    # Sort by silhouette score within each algorithm
                    sorted_alg_results = sorted(
                        alg_results,
                        key=lambda x: x.get("silhouette_score", -1),
                        reverse=True,
                    )

                    # Select n_best from this algorithm
                    if n_best is not None:
                        selected = sorted_alg_results[:n_best]
                        print(f"Plotting top {len(selected)} results for {alg_name}")
                    else:
                        selected = sorted_alg_results
                        print(f"Plotting all {len(selected)} results for {alg_name}")

                    result_filenames.extend([res["filename"] for res in selected])

        # Plot each result
        print(f"\nPreparing to plot {len(result_filenames)} clustering results...")
        for i, filename in enumerate(result_filenames, 1):
            print(f"\n[{i}/{len(result_filenames)}] Processing {filename}...")
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
