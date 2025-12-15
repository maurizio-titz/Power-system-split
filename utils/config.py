#!/usr/bin/env python

use_extensions = False  # set to False if no line extensions should be considered
if use_extensions:
    root_path = "/srv/data/jlange/power-system-split/w_extensions/"  # add the root path for your data and results here
    path_to_lopf_data = "/srv/data/jlange/PYPSA3/sclopf-iter/workflow/submodules/pypsa-eur/s_max_pu_0.7/results/networks/"
else:
    root_path = "/srv/data/jlange/power-system-split/no_extensions/"  # add the root path for your data and results here
    path_to_lopf_data = root_path + "/data/European_networks_lopf/"

data_path = root_path + "/data/"
path_to_grid_data = root_path + "/grid_params/"
results_path = root_path + "/results/"
path_to_sclopf_data = root_path + "/data/European_networks_sclopf/"

# sclopf paths
path_to_sclopf_results = results_path + "sclopf/"
path_to_indicator_vectors_sclopf = results_path + "sclopf/indicator_vectors/"
path_to_clustering_results_sclopf = results_path + "sclopf/clustering/"
path_to_cascade_results_sclopf = results_path + "sclopf/cascade_results/"
path_to_evaluation_results_sclopf = results_path + "sclopf/evaluation_results/"
path_to_pypsa_network_sclopf = root_path + "/data/European_networks_sclopf/"
path_to_inertia_mitigation_results_sclopf = (
    results_path + "sclopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_sclopf = (
    results_path + "sclopf/line_extension_mitigation/"
)
path_to_vis_results_sclopf = results_path + "sclopf/split_visualization/"
path_to_pre_outage_sclopf = results_path + "sclopf/pre_outage_data/"
path_to_figures_sclopf = results_path + "/figures/sclopf/"

mattermost_url = None
