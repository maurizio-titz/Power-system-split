#!/usr/bin/env python

from utils.config_local import root_path

data_path = root_path + "/data/"
results_path = root_path + "/results/"
path_to_indicator_vectors = root_path + "/results/sclopf/indicator_vectors/"
path_to_clustering_results = root_path + "/results/sclopf/clustering/"
path_to_cascade_results = root_path + "/results/sclopf/cascade_results/"
path_to_evaluation_results = root_path + "/results/sclopf/evaluation_results/"
path_to_pypsa_network_sclopf = root_path + "/data/European_networks_sclopf/"
path_to_pypsa_network_lopf = root_path + "/data/European_networks_lopf/"
path_to_inertia_mitigation_results = (
    root_path + "/results/sclopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation = (
    root_path + "results/sclopf/line_extension_mitigation"
)
path_to_vis_results = results_path + "sclopf/split_visualization/"
path_to_pre_outage = results_path + "sclopf/pre_outage_data/"
path_to_figures = results_path + "/figures/sclopf/"


mattermost_url = None
