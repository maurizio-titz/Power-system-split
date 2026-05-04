#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import sys
from scipy.stats import uniform
from scipy.stats import loguniform
import numpy as np

sys.path.append("../")
sys.path.append("./")

from utils.data_handling import get_co2_levels
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from hdbscan import HDBSCAN

from loguru import logger


# Module-level worker function for parallel clustering (must be at top level for pickling)
def _fit_clustering_worker_script(
    distance_matrix, alg_name, params, fit_params, calc_silhouette=False
):
    """Worker function for parallel clustering execution.

    Defined at module level in the script to be picklable by joblib.
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
    import numpy as np

    alg_map = {
        "agg": AgglomerativeClustering,
        "dbscan": DBSCAN,
        "optics": OPTICS,
        "hdbscan": HDBSCAN,
    }
    alg_func = alg_map.get(alg_name, alg_name)

    # HDBSCAN requires float64 (double precision)
    if alg_name == "hdbscan" and distance_matrix.dtype != np.float64:
        distance_matrix = distance_matrix.astype(np.float64)

    model = alg_func(**params)
    model.fit(distance_matrix, **fit_params)

    silhouette_avg = None
    if calc_silhouette:
        from sklearn.metrics import silhouette_score

        cluster_labels = model.labels_
        if len(set(cluster_labels)) <= 1:
            print("Only one cluster found; silhouette score is undefined.")
            return model, -1
        silhouette_avg = silhouette_score(
            distance_matrix, cluster_labels, metric="precomputed"
        )
        print(f"Silhouette Score: {silhouette_avg:.4f}")

    return model, silhouette_avg


n_jobs_distance = 16
n_jobs_clustering = 16
distance_metric_kwargs = {
    "name": "composite",
    "metrics": [
        {
            "name": "hamming",
            "n_jobs": n_jobs_distance,
        },
        {
            "name": "cosine_distance",
            "preprocessing": {
                "method": "boundary_field",
                "target": "classes",  # Transform classes
                "alpha": 0.4,
                "steps": 3,
            },
            "n_jobs": n_jobs_distance,
        },
    ],
    "combiner": geometric_mean,
    "cache_components": True,
}

n_clusters_list = [16, 32, 64, 80]
clustering_params = {
    "agg": {
        "n_iter": 1e4,
        "n_jobs": n_jobs_clustering,
        "HPs": {
            "n_clusters": n_clusters_list,
            "linkage": [
                "average",
                "complete",
                # "single",
            ],  # "single" is not used as it leads to bad results
            "metric": ["precomputed"],
        },
        "calc_silhouette": True,
    },
}

# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster blackouts")
    parser.add_argument(
        "--testing", action="store_true", help="Run in testing mode", default=False
    )
    parser.add_argument(
        "--co2l", nargs="+", type=float, help="CO2 levels to cluster", default=None
    )
    parser.add_argument(
        "--threshold", type=float, help="Blackout size threshold", default=0.1
    )

    args = parser.parse_args()
    testing = args.testing
    threshold = args.threshold

    n_nodes = 600
    if testing:
        co2l_list = [0.6, 0.5]
        random_subsample_size = 0.001
    else:
        if args.co2l is not None:
            co2l_list = args.co2l
        else:
            co2l_list = list(get_co2_levels(n_nodes))
        random_subsample_size = None

    co2l_clustering = [co2l_list]
    # %%
    for co2l_iter in co2l_clustering:
        print(f"\n\n=== Clustering for CO2 levels: {co2l_iter} ===\n\n")
        # Initialize clustering object
        cl = Clustering(
            n_nodes,
            co2l_iter,
            indicator_type="rocof",
            transformation="blackout",
            blackout_size_threshold=threshold,
            distance_metric_kwargs=distance_metric_kwargs,
            clustering_params=clustering_params,
            distance_matrix_dtype=np.float16,
            random_subsample_size=random_subsample_size,
            clustering_worker_func=_fit_clustering_worker_script,  # Use script-level worker for pickling
        )
        
        cl.load_data()
        logger.info("Loaded data")
        cl.transform_vectors()
        logger.info("Transformed Vectors")
        cl.filter_data()
        logger.info("filtered data")
        cl.get_distance_matrix()
        logger.info("go dist matrix")
        cl.fit_clusters()
        logger.info("fitted clusters")
        cl.create_clustering_results_index()
        logger.info("created index")
        cl.plot_cluster_multiple(
            # relative_score_threshold=0.1,
            n_best=4,
            average_over_classes=False,
            algorithm="agg",
            sort_by="frequency",
        )
        cl.plot_cluster_multiple(
            n_best=4,
            # relative_score_threshold=0.1,
            average_over_classes=False,
            algorithm="agg",
            sort_by="accumulative_lost_load",
        )
