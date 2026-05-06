#!usr/bin/env python
# -*- coding: utf-8 -*-
# %%

import os
import numpy as np
import sys

import gzip
import pickle

# sys.path.append("../")
sys.path.append("./")
from utils.data_handling import get_co2_levels
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from hdbscan import HDBSCAN

# Logging
from loguru import logger
import time

from scripts.cluster_blackouts import (
    distance_metric_kwargs,
    clustering_params,
)

from utils.config import path_to_figures_sclopf

def plot_best_clustering_results(n_nodes: int = 600,
                                 n_best: int = 4,
                                 calc_again: bool = False,
                                 blackout_size_threshold: float = .1):
    """Plot the cluster centroids of the clustering results."""
    co2l_iter = list(get_co2_levels(n_nodes))
    logger.info(f"Cluster plots for {co2l_iter}")
    fpath_cluster_class = path_to_figures_sclopf + \
        f"/data_plot_cluster_class_n{n_nodes}_" + \
        f"thresBlackoutSize{blackout_size_threshold:.3f}.pklz"
    if not os.path.exists(fpath_cluster_class) or calc_again:
        logger.info("Building cluster class and saving later.")
        cl = Clustering(
            n_nodes,
            co2l_iter,
            indicator_type="rocof",
            transformation="blackout",
            blackout_size_threshold=blackout_size_threshold,
            distance_metric_kwargs=distance_metric_kwargs,
            clustering_params=clustering_params,
            distance_matrix_dtype=np.float16,
        )
        # %%
        logger.info(" ---> Loading data")
        cl.load_data()
        cl.transform_vectors()
        cl.filter_data()
        cl.create_clustering_results_index()
        
        with gzip.open(fpath_cluster_class, "wb") as fh_out:
            pickle.dump(cl, fh_out)
        logger.info(" ----> Saved cluster class!")
    # %%
    else:
        logger.info("Loading previously pickled cluster class")
        with gzip.open(fpath_cluster_class, "rb") as fh_in:
            cl = pickle.load(fh_in)
    t0 = time.time()
    logger.info("Plotting multiple clusters ordered by occurence")
    cl.plot_cluster_multiple(
        n_best=n_best,
        average_over_classes=False,
        algorithm="agg",
        sort_by="frequency",
        save_plot_data=True
    )
    # %%
    logger.info("Plotting multiple clusters ordered by accum. lost load")
    cl.plot_cluster_multiple(
        n_best=n_best,
        average_over_classes=False,
        algorithm="agg",
        sort_by="accumulative_lost_load",
    )
    
    # Only lines
    logger.info("Plot only lines sorted by occurence")
    cl.plot_cluster_multiple(n_best=n_best, average_over_classes=False,
                             algorithm="agg", 
                             sort_by="frequency",
                             use_only_lines=True)
    
    logger.info("Plot only lines sorted by accm. load")
    cl.plot_cluster_multiple(n_best=n_best, average_over_classes=False,
                             algorithm="agg", 
                             sort_by="frequency",
                             use_only_lines=True)
    
    duration_s = time.time() - t0
    logger.info(f"Finished plotting in {duration_s/60.:.2f} min")
    # %%

if __name__ == "__main__":
    plot_best_clustering_results(calc_again=True)
