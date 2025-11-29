# %%
# %load_ext autoreload
# %autoreload 2
from datetime import datetime
from functools import partial
import gzip
import os
import pickle
import sys
from scipy.stats import uniform
from scipy.stats import loguniform
import numpy as np
from scipy import sparse

sys.path.append("../")
sys.path.append("./")

from utils.data_handling import get_co2_levels
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean


# Module-level worker function for parallel clustering (must be at top level for pickling)
def _fit_clustering_worker_script(
    distance_matrix, alg_name, params, fit_params, calc_silhouette=False
):
    """Worker function for parallel clustering execution.

    Defined at module level in the script to be picklable by joblib.
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS

    alg_map = {
        "agg": AgglomerativeClustering,
        "dbscan": DBSCAN,
        "optics": OPTICS,
    }
    alg_func = alg_map.get(alg_name, alg_name)

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


# %%
if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Cluster blackouts")
    parser.add_argument(
        "--testing", action="store_true", help="Run in testing mode", default=False
    )
    args = parser.parse_args()
    testing = args.testing

    n_nodes = 600
    if testing:
        co2l_list = [0.6, 0.5]
        random_subsample_size = (0.001,)
    else:
        co2l_list = list(get_co2_levels(n_nodes))
        random_subsample_size = (None,)

    n_jobs_distance = 32
    n_jobs_clustering = 4
    distance_metric_kwargs = {
        "name": "composite",
        "metrics": [
            {
                "name": "hamming_weighted",
                "preprocessing": {
                    "method": "low_weight_boundary_field",
                    "target": "weights",  # Use as weights
                    "alpha": 0.4,
                    "steps": 3,
                },
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

    n_clusters_list = np.arange(32, 1025, step=32).tolist()
    # n_clusters_list = [16, 32, 64, 80]
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
        "optics": {
            "n_iter": 64,
            "n_jobs": n_jobs_clustering,
            "HPs": {
                "min_samples": loguniform(
                    1e-6, 1e-2
                ),  # returns int via rvs() then cast
                "cluster_method": ["xi"],
                "max_eps": loguniform(0.02, 0.2),
                "n_jobs": [1],
                "xi": loguniform(0.02, 0.2),
                "metric": ["precomputed"],
            },
            "calc_silhouette": True,
            "HP_transforms": {
                "min_samples": lambda x: int(x)  # ensure integer conversion
            },
        },
    }
    # %%
    cl = Clustering(
        n_nodes,
        co2l_list,
        indicator_type="rocof",
        transformation="blackout",
        blackout_size_threshold=0.05,
        distance_metric_kwargs=distance_metric_kwargs,
        clustering_params=clustering_params,
        distance_matrix_dtype=np.float16,
        random_subsample_size=random_subsample_size,
        clustering_worker_func=_fit_clustering_worker_script,  # Use script-level worker for pickling
    )
    # %%

    cl.load_data()
    cl.transform_vectors()
    cl.filter_data()
    # %%
    # cl.set_attribute("distance_metric_kwargs", distance_metric_kwargs)
    cl.get_distance_matrix()
    # %%
    cl.set_attribute("clustering_params", clustering_params)
    cl.fit_clusters()
    cl.prepare_visualize_clusters()
    # %%prun -s cumulative -q -l 10 -T prun0
    # We profile the cell, sort the report by "cumulative
    # time", limit it to 10 lines, and save it to a file
    # named "prun0".
    cl.plot_cluster_multiple(
        n_best=12,
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
#     #     print(f"  File: {result_info['filename']}")
# #%%
# np.where(np.isnan(cl.distance_matrix[:100,:100]))
# #%%
# (np.isnan(cl.component_distance_matrices["ACC"][:100,:100])).sum()
# #%%
# (np.isnan(cl.component_distance_matrices["cosine_distance"][:100,:100])).sum()
# #%%
# def geometric_mean(data) -> np.ndarray:
#     product = data[0]
#     for mat in data[1:]:
#         product = np.multiply(product, mat)
#     return np.sqrt(product)

# m = geometric_mean(
#     [
#         cl.component_distance_matrices["ACC"][:100,:100],
#         cl.component_distance_matrices["cosine_distance"][:100,:100],
#     ]
# )
# # %%
# a = cl.component_distance_matrices["ACC"][:100,:100]
# b = cl.component_distance_matrices["cosine_distance"][:100,:100]
# weights = boundary_field(cl.unique_vecs, I=cl.I, L=cl.L, tau=2.5)
# #%%
# import numpy as np
# from scipy.sparse import csr_matrix, diags

# def boundary_field_rw_smoothing(labels, incidence_mat, A, alpha=0.5, steps=3):
#     """
#     Compute smoothed boundary fields using a row-stochastic random-walk operator.

#     Args:
#         labels (np.ndarray): shape (N_samples, N_nodes), node labels
#         incidence_mat (csr_matrix): incidence matrix (N_nodes, N_edges)
#         A (csr_matrix): adjacency matrix, shape (N_nodes, N_nodes)
#         alpha (float): smoothing strength per step, 0 < alpha < 1
#         steps (int): number of diffusion steps (3–6 good)

#     Returns:
#         b_s (np.ndarray): smoothed boundary fields, shape (N_samples, N_nodes)
#     """

#     # -----------------------------
#     # 1) Compute edge cuts
#     # -----------------------------
#     # diff = label difference per edge
#     diff = labels @ incidence_mat           # (N_samples, n_edges)
#     c = (diff != 0).astype(float)           # 1 if cut edge

#     # -----------------------------
#     # 2) Map cuts back to nodes
#     # -----------------------------
#     b = ((c @ incidence_mat.T) != 0).astype(float)  # boundary indicator at nodes
#     # b is in {0,1}

#     # -----------------------------
#     # 3) Build row-stochastic P = D^{-1} A
#     # -----------------------------
#     deg = np.array(A.sum(axis=1)).ravel()
#     deg_safe = np.maximum(deg, 1e-12)
#     Dinv = diags(1.0 / deg_safe)
#     P = Dinv @ A    # P is row-stochastic (rows sum to 1), no amplification possible

#     # -----------------------------
#     # 4) Random-walk smoothing
#     # -----------------------------
#     # Each sample is a row vector of length n_nodes.
#     # Use s_new = (1-alpha) s + alpha (P^T @ s^T)^T
#     # which in row form is: s_new = (1-alpha)*s + alpha*(s @ P)
#     s = b.copy()

#     for _ in range(steps):
#         s = (1 - alpha) * s + alpha * (s @ P)    # both terms preserve [0,1]

#     return s
# weigts_new = boundary_field_rw_smoothing(
#     cl.unique_vecs, cl.I, cl.A, alpha=0.5, steps=8
# )
# #%%
# c = np.multiply(a, b)
# np.isnan(c).sum()
# #%%
# d = np.power(c, 1/2)
# np.isnan(d).sum()
# #%%
# nan_idxs = np.where(np.isnan(d))
# nan_idxs = list(zip(nan_idxs[0], nan_idxs[1]))
# # %%
# for i, j in nan_idxs:
#     print(f"Index ({i}, {j}): ACC={a[i,j]}, Cosine={b[i,j]}, product= {c[i,j]}, Geometric Mean={d[i,j]}")

# #%%
# cl.set_grid_matrices()
# #%%

# # %%
# w = weights[:100,:100]
# # %%
# from utils.clustering.distance_metrics import ACC_weighted_pairwise
# acc_dist = ACC_weighted_pairwise(
#     cl.unique_vecs[:100],
#     cl.unique_vecs[:100],
#     weights[:100],
#     weights[:100],
# )
# #%%
# np.isnan(acc_dist).sum()

# #%% sample randomly from weights and plot histogram
# import matplotlib.pyplot as plt
# sample_size = 10000
# weight_samples = weights.flatten()
# weight_samples = weight_samples[~np.isnan(weight_samples)]
# if len(weight_samples) > sample_size:
#     weight_samples = np.random.choice(weight_samples, size=sample_size, replace=False)
# plt.hist(weight_samples, bins=50)
# plt.xlabel("Weight value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Boundary Weights")
# plt.show()

# #%%
# from scipy.sparse import diags
# L = cl.I.dot(cl.I.T)
# D = diags(L.diagonal())
# A = D - L
# deg = np.array(D.diagonal())
# Dinv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
# Lsym = diags(np.ones_like(deg)) - Dinv_sqrt @ A @ Dinv_sqrt

# #%%
# # Check adjacency correctness
# print("max offdiag L:", (L - diags(L.diagonal())).max())
# print("min offdiag L:", (L - diags(L.diagonal())).min())

# # Check A >= 0
# print("A min:", A.min())

# # Check eigenvalues of Lsym
# from scipy.sparse.linalg import eigsh
# eigvals = eigsh(Lsym, k=5, which="LM", return_eigenvectors=False)
# print("Largest eigenvalue:", eigvals.max())    # must be ≤ 2
