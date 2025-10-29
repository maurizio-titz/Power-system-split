#!/usr/bin/env python

use_extensions = False  # set to False if no line extensions should be considered
if use_extensions:
    root_path = "./"  # add the root path for your data and results here
else:
    root_path = "./"  # add the root path for your data and results here

data_path = root_path + "/data/"
path_to_grid_data = root_path + "/grid_params/"
results_path = root_path + "/results/"
if use_extensions:
    results_path += "w_extensions/"
else:
    results_path += "no_extensions/"
path_to_sclopf_data = root_path + "/data/European_networks_sclopf/"
path_to_lopf_data = root_path + "/data/European_networks_lopf/"
# path_to_lopf_data = "/srv/data/jlange/PYPSA3/sclopf-iter/workflow/submodules/pypsa-eur/s_max_pu_0.7/results/networks/"
path_to_pypsa_network_lopf = path_to_lopf_data


# sclopf paths
path_to_sclopf_results = root_path + "/results/sclopf/"
path_to_indicator_vectors_sclopf = root_path + "/results/sclopf/indicator_vectors/"
path_to_clustering_results_sclopf = root_path + "/results/sclopf/clustering/"
path_to_cascade_results_sclopf = root_path + "/results/sclopf/cascade_results/"
path_to_evaluation_results_sclopf = root_path + "/results/sclopf/evaluation_results/"
path_to_pypsa_network_sclopf = root_path + "/data/European_networks_sclopf/"
path_to_inertia_mitigation_results_sclopf = (
    root_path + "/results/sclopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_sclopf = (
    root_path + "results/sclopf/line_extension_mitigation/"
)
path_to_vis_results_sclopf = results_path + "sclopf/split_visualization/"
path_to_pre_outage_sclopf = results_path + "sclopf/pre_outage_data/"
path_to_figures_sclopf = results_path + "/figures/sclopf/"

# LOPF paths
path_to_indicator_vectors_lopf = root_path + "/results/lopf/indicator_vectors/"
path_to_clustering_results_lopf = root_path + "/results/lopf/clustering/"
path_to_cascade_results_lopf = root_path + "/results/lopf/cascade_results/"
path_to_evaluation_results_lopf = root_path + "/results/lopf/evaluation_results/"
# path_to_pypsa_network_lopf = root_path + "/data/European_networks_lopf/"
path_to_inertia_mitigation_results_lopf = (
    root_path + "/results/lopf/syn_inertia_mitigation/"
)
path_to_line_extension_mitigation_lopf = (
    root_path + "results/lopf/line_extension_mitigation/"
)
path_to_vis_results_lopf = results_path + "lopf/split_visualization/"
path_to_pre_outage_lopf = results_path + "lopf/pre_outage_data/"
path_to_figures_lopf = results_path + "/figures/lopf/"


mattermost_url = None


# /media/data/system_split//results/lopf/syn_inertia_mitigation/synthetic_inertia_placement_Co20.2_N800_deltarotE5000_rocofthres-1_lshare0_maxiter10000_random.pklz
# /media/data/system_split//results/lopf/syn_inertia_mitigation/synthetic_inertia_placement_Co20_N800_deltarotE5000_rocofthres-1_lshare0_maxiter10000_random.pklz
