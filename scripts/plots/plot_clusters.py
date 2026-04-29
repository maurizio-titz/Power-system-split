#!usr/bin/env python
# -*- coding: utf-8 -*-
# %%
import numpy as np
import sys

# sys.path.append("../")
sys.path.append("./")
from utils.data_handling import get_co2_levels
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from hdbscan import HDBSCAN

from loguru import logger

from scripts.cluster_blackouts import (
    distance_metric_kwargs,
    clustering_params,
)

n_nodes = 600
co2l_iter = list(get_co2_levels(n_nodes))
logger.info(f"Cluster plots for {co2l_iter}")
cl = Clustering(
    n_nodes,
    co2l_iter,
    indicator_type="rocof",
    transformation="blackout",
    blackout_size_threshold=0.1,
    distance_metric_kwargs=distance_metric_kwargs,
    clustering_params=clustering_params,
    distance_matrix_dtype=np.float16,
)
# %%
logger.info("Loading data")
cl.load_data()
cl.transform_vectors()
cl.filter_data()
cl.create_clustering_results_index()
# %%
logger.info("Plotting multiple clusters ordered by occurence")
cl.plot_cluster_multiple(
    # relative_score_threshold=0.1,
    n_best=4,
    average_over_classes=False,
    algorithm="agg",
    sort_by="frequency",
)
# %%
logger.info("Plotting multiple clusters ordered by accum. lost load")
cl.plot_cluster_multiple(
    n_best=4,
    # relative_score_threshold=0.1,
    average_over_classes=False,
    algorithm="agg",
    sort_by="accumulative_lost_load",
)

# %%
