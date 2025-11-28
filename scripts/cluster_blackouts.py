# %%
# %load_ext autoreload
# %autoreload 2
from datetime import datetime
from functools import partial
import gzip
import os
import pickle
import sys

from scipy import sparse

sys.path.append("../")
sys.path.append("./")

from utils.data_handling import get_co2_levels
from utils.clustering.blackout_clustering_class import Clustering

# %%
n_nodes = 600
co2l_list = list(get_co2_levels(n_nodes))
# co2l_list = [0.6, 0.5]

from utils.clustering.distance_metrics import geometric_mean

distance_metric_kwargs = {
    "name": "composite",
    "metrics": [
        {
            "name": "ACC",
            "preprocessing": {
                "method": "low_weight_boundary_field",
                "target": "weights",  # Use as weights
                "tau": 2.5,
            },
            "n_jobs": 8,
        },
        {
            "name": "cosine_distance",
            "preprocessing": {
                "method": "boundary_field",
                "target": "classes",  # Transform classes
                "tau": 2.5,
            },
            "n_jobs": 8,
        },
    ],
    "combiner": geometric_mean,
    "cache_components": True,
}

n_clusters_list = [16, 32, 64, 80, 128, 150, 256, 300, 400, 512]
clustering_params = {
    # "optics": {
    #     "clustering_algorithm": OPTICS,
    #     "min_samples": loguniform(5, 100),  # sample min_samples on a log scale
    #     "cluster_method": ["xi"],
    #     "max_eps": loguniform(0.02, 0.2),
    #     "n_jobs": [1],
    #     "xi": loguniform(0.02, 0.2),  # sample xi on a log scale
    #     "metric": ["precomputed"],
    # },
    "agg": {
        "n_iter": 16,
        "n_jobs": 8,
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
    }
}
# %%
cl = Clustering(
    n_nodes,
    co2l_list,
    indicator_type="rocof",
    transformation="blackout",
    blackout_size_threshold=0.1,
    distance_metric_kwargs=distance_metric_kwargs,
    clustering_params=clustering_params,
    # random_subsample_size=0.001,
)
# %%

cl.load_data()
cl.transform_vectors()
cl.filter_data()
# %%
cl.get_distance_matrix()
# %%
# %%
cl.set_attribute("clustering_params", clustering_params)
# %%
cl.fit_clusters()
cl.prepare_visualize_clusters()
# %%prun -s cumulative -q -l 10 -T prun0
# We profile the cell, sort the report by "cumulative
# time", limit it to 10 lines, and save it to a file
# named "prun0".
cl.plot_cluster_multiple(
    n_best=3,
    average_over_classes=False,
)
# %%


# cl.run_all()


# #%%
# import numpy as np
# all_results = cl.load_clustering_results_index()

# print(f"\nFound {len(all_results)} clustering results:")

# # Load and analyze each result
# for result_info in all_results:
#     result = cl.load_clustering_result(result_info["filename"])

#     model = result["model"]
#     params = result["params"]
#     algorithm = result["algorithm"]

#     # Get cluster labels
#     labels = model.labels_
#     n_clusters = len(np.unique(labels))
#     n_noise = np.sum(labels == -1)

#     print(f"\n{algorithm} with params {params}:")
#     print(f"  N clusters: {n_clusters}")
#     print(f"  N noise points: {n_noise}")
#     print(f"  File: {result_info['filename']}")
