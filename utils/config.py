#!/usr/bin/env python
import os

use_extensions = False  # legacy, do not use

# Read dataset selection from environment variable, default to "no_extensions"
DATASET = os.getenv("POWER_SYSTEM_DATASET")

# Base paths configuration for different datasets
root_path = "/srv/data/jlange/power-system-split/no_extensions/"
RESULTS_CONFIGS = {
    "no_load_inertia": "results_no_load_inertia/",
    "current": "results/",
}

# Get current dataset config
config = RESULTS_CONFIGS.get(DATASET)
if config is None:
    raise ValueError(
        f"Unknown dataset: {DATASET}. Available: {list(RESULTS_CONFIGS.keys())}"
    )


data_path = root_path + "/data/"
path_to_grid_data = root_path + "/grid_params/"
results_path = root_path + "/" + config
path_to_sclopf_data = root_path + "/data/European_networks_sclopf/"
path_to_lopf_data = root_path + "/data/European_networks_lopf/"

# sclopf paths
path_to_sclopf_results = results_path + "sclopf/"
# path_to_sclopf_results = results_path + "BACKU_sclopf_no_load_inertia/"
path_to_indicator_vectors_sclopf = path_to_sclopf_results + "indicator_vectors/"
path_to_clustering_results_sclopf = path_to_sclopf_results + "clustering/"
path_to_cascade_results_sclopf = path_to_sclopf_results + "cascade_results/"
path_to_evaluation_results_sclopf = path_to_sclopf_results + "evaluation_results/"
path_to_pypsa_network_sclopf = root_path + "/data/European_networks_sclopf/"
path_to_inertia_mitigation_results_sclopf = (
    path_to_sclopf_results + "syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_sclopf = (
    path_to_sclopf_results + "line_extension_mitigation/"
)
path_to_vis_results_sclopf = path_to_sclopf_results + "split_visualization/"
path_to_pre_outage_sclopf = path_to_sclopf_results + "pre_outage_data/"
path_to_figures_sclopf = results_path + "/figures/"

mattermost_url = None
