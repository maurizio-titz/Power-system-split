#!/usr/bin/env python
import os

from pathlib import Path

project_folder_path = str(Path(__file__).parent.parent)

# Base paths configuration for different datasets
root_path = project_folder_path

data_path = root_path + "/data"
path_to_grid_data = root_path + "/results/grid_params"
results_path = root_path + "/results"
path_to_sclopf_data = data_path + "/European_networks_sclopf"
path_to_lopf_data = data_path + "/European_networks_lopf"

# lopf paths
path_to_pypsa_network_lopf = data_path + "/European_networks_sclopf"
path_to_lopf_results = results_path + "/lopf"
path_to_cascade_results_lopf = path_to_lopf_results + "/cascade_results"


# sclopf paths
path_to_sclopf_results = results_path + "/sclopf"
path_to_indicator_vectors_sclopf = path_to_sclopf_results + "/indicator_vectors"
path_to_clustering_results_sclopf = path_to_sclopf_results + "/clustering"
path_to_cascade_results_sclopf = path_to_sclopf_results + "/cascade_results"
path_to_evaluation_results_sclopf = path_to_sclopf_results + "/evaluation_results"
path_to_pypsa_network_sclopf = data_path + "/European_networks_sclopf"


path_to_inertia_mitigation_results_sclopf = (
    path_to_sclopf_results + "/syn_inertia_mitigation"
)
path_to_line_extension_mitigation_sclopf = (
    path_to_sclopf_results + "/line_extension_mitigation"
)
path_to_vis_results_sclopf = path_to_sclopf_results + "/split_visualization"
path_to_pre_outage_sclopf = path_to_sclopf_results + "/pre_outage_data"
path_to_figures_sclopf = results_path + "/figures"
# Intermediate data that is created while running plot script for the first time
path_to_plot_data = path_to_figures_sclopf + "/plot_data"

# Allow plot output directory override via environment variable
_plot_output_dir = os.getenv("PLOT_OUTPUT_DIR", "").strip()
_plot_style = os.getenv("PLOT_STYLE", "").strip().lower()
if not _plot_output_dir and _plot_style in {"joule"}:
    _plot_output_dir = os.path.join(path_to_figures_sclopf, _plot_style)
if _plot_output_dir:
    path_to_figures_sclopf = _plot_output_dir

# Meteo analysis data + figures subfolder
path_to_WR_meteo_data = data_path + "/weather_data"
path_to_meteo_figures = path_to_figures_sclopf + "/meteo"

mattermost_url = None
