#!/usr/bin/env python

from utils.config_local import root_path

data_path = root_path + "/data/"
results_path = root_path + "/results/"

# sclopf paths
path_to_indicator_vectors_sclopf = root_path + "/results/sclopf/indicator_vectors/"
path_to_clustering_results_sclopf = root_path + "/results/sclopf/clustering/"
path_to_cascade_results_sclopf = root_path + "/results/sclopf/cascade_results/"
path_to_evaluation_results_sclopf = root_path + "/results/sclopf/evaluation_results/"
path_to_pypsa_network_sclopf = root_path + "/data/European_networks_sclopf/"
path_to_inertia_mitigation_results_sclopf = (
    root_path + "/results/sclopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_sclopf = (
    root_path + "results/sclopf/line_extension_mitigation"
)
path_to_vis_results_sclopf = results_path + "sclopf/split_visualization/"
path_to_pre_outage_sclopf = results_path + "sclopf/pre_outage_data/"
path_to_figures_sclopf = results_path + "/figures/sclopf/"

# LOPF paths
path_to_indicator_vectors_lopf = root_path + "/results/lopf/indicator_vectors/"
path_to_clustering_results_lopf = root_path + "/results/lopf/clustering/"
path_to_cascade_results_lopf = root_path + "/results/lopf/cascade_results/"
path_to_evaluation_results_lopf = root_path + "/results/lopf/evaluation_results/"
path_to_pypsa_network_lopf = root_path + "/data/European_networks_lopf/"
path_to_inertia_mitigation_results_lopf = (
    root_path + "/results/lopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_lopf = (
    root_path + "results/lopf/line_extension_mitigation"
)
path_to_vis_results_lopf = results_path + "lopf/split_visualization/"
path_to_pre_outage_lopf = results_path + "lopf/pre_outage_data/"
path_to_figures_lopf = results_path + "/figures/lopf/"


mattermost_url = None


# /media/data/system_split//results/lopf/syn_inertia_mitigation/synthetic_inertia_placement_Co20.2_N800_deltarotE5000_rocofthres-1_lshare0_maxiter10000_random.pklz
# /media/data/system_split//results/lopf/syn_inertia_mitigation/synthetic_inertia_placement_Co20_N800_deltarotE5000_rocofthres-1_lshare0_maxiter10000_random.pklz
