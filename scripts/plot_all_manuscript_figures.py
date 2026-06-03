#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set Joule style
import os
os.environ["PLOT_STYLE"] = "joule"
from utils.plot_style import *

import sys
import subprocess
import psutil

import gc
from matplotlib import pyplot as plt

from scripts.plots.plot_combined_generation_storage import create_combined_generation_storage_plot, create_generation_by_country_stacked_bar_plot
from scripts.plots.plot_combined_flow_split_statistics import create_combined_flow_and_split_statistics_plot
from scripts.plots.plot_clustering_analysis import create_clustering_analysis_plot_from_data, create_cluster_plot_only_lines_with_zoom
from scripts.plots.plot_mitigation import create_combined_mitigation_plot, blackout_size_histogram_after_mitigation
from scripts.plots.plot_line_failure_probs import create_total_line_failure_plot
from scripts.plots.plot_spatial_power_inhomogeneity import plot_spi_histograms_n_mean_spi_per_month
from scripts.plots.plot_additional_pypsa_figures import histogram_lineloading_accross_co2lvls, \
    graph_with_num_parallel, plot_load_dispatch, \
    plot_investments_capital_costs
from scripts.plots.plot_split_statistics import plot_blackout_size_histograms, plot_blackout_size_distributions_with_zoom_in, plot_component_number_vs_blackout_size, create_blackout_statistics_common_vs_different_corridor_plot
from scripts.plots.plot_inertia_by_country import create_inertia_by_country_all_co2_plot
from scripts.plots.plot_compare_lineMitigation_plannedExtension import plot_planned_line_extensions_single_co2, run_planned_extension_comparison

# Paths
from utils.config import path_to_plot_data, path_to_line_extension_mitigation_sclopf


def enough_memory(threshold_pct: float = 90.0) -> bool:
    return psutil.virtual_memory().percent < threshold_pct

def plot_manuscript_figures(
                     low_memory: bool = False):
    """Plot the figures for the paper, which mainly calls functions from 'scripts/plots/'

    Args:
        low_memory (bool, optional): If True, some figures are not plotted that require 
            more than 32GB of memory. Defaults to False.
    """
    
    ## Main fiugres
    # Fig. 1: Scenarios for decarbonisation of The European power system
    create_combined_generation_storage_plot(save_prefix="fig1")
    
    # Fig. 2: Evolution of risks during decarbonistaion
    if low_memory:
        logger.warning("Skipping 'Fig.2' + 'Fig.3' due to low memory!")
    else:
        create_combined_flow_and_split_statistics_plot(mean_distance=False, 
                                                   load_normalization=False, 
                                                   show_blackout_stats=True, 
                                                   secondary_proba_axis=True,
                                                   use_steps_load_loss=True,
                                                   use_equal_panels=True,
                                                   save_prefix="fig2")
    
    # Fig. 3: Characteristic geographic patterns of system split
    # This requires data that is created when 'plot_clusts.py' is run and the clustering 
    # data was create before (see cluster_blackouts.py).
    fpath_cluster_plot_data = os.path.join(path_to_plot_data, 
                                            "plot_data_agg_params_01bf408e.pklz")
    create_clustering_analysis_plot_from_data(fpath_cluster_plot_data, 
                                                sort_by="frequency",
                                                save_prefix="fig3")
    
    
    # Fig. 4: Conditional probability of transmission lines participating in casc. failures
    
    create_total_line_failure_plot(
        unified_colorbar=True,
        powernorm=False,
        scale_width=True,
        width_scale_sqrt=True,
        cmap="crameri:Batlow_r",
        save_prefix="fig4"
    )
    plt.close('all')
    gc.collect()
    # Fig. 5: Mitigation of split-induced blackouts via inertia and grid reinforcements
    create_combined_mitigation_plot(use_annualized_costs=True,
                                    co2_lvl_map=0.2,
                                    build_380kV_only=True,
                                    plot_intertia_cost=True,
                                    blackoutthreshold=.8,
                                    target="num_GSS",
                                    plot_rows="both", 
                                    organize_plots=False,
                                    save_prefix="fig5"
                                    )
        
    return


def plot_supplementary_figures(low_memory: bool = False):
    """Plot the supplemantry figures for the SI of the paper."""
    
    
    fpath_cluster_plot_data = os.path.join(path_to_plot_data, 
                                           "plot_data_agg_params_01bf408e.pklz")
    create_cluster_plot_only_lines_with_zoom(fpath_cluster_plot_data, 
                                             0,
                                             zoom_middle=(-0.07, 42.84), 
                                             zoom_radius_x=5.2,
                                             save_prefix="SI")
    create_clustering_analysis_plot_from_data(fpath_cluster_plot_data, use_only_lines=True,
                                              sort_by="frequency",
                                              save_prefix="SI")
    
    plt.close('all')
    gc.collect()
    
    # SPI vectors
    plot_spi_histograms_n_mean_spi_per_month(save_prefix="SI", 
                                             selected_co2_lvls=(.6,.4,.2, 0))
    
    # Grid map showing num parallel
    graph_with_num_parallel(save_prefix="SI")
    
    # Daily Profile Nuclear Generation
    ## please run the script 'scripts/plots/plot_nuclear_profiles.py' from bash
    subprocess.run([sys.executable, 
                    "scripts/plots/plot_nuclear_profiles.py"])
    
    # Co2 Scenarios vs Generation by country stacked
    create_generation_by_country_stacked_bar_plot(save_prefix="SI")
    
    # System Costs for different CO2 levels
    plot_investments_capital_costs(save_prefix="SI")
    plt.close('all')
    gc.collect()
    
    # System Energy Balance 0, 20, 60 % in seasons
    plot_load_dispatch(save_prefix="SI")
    if low_memory:
        logger.warning("Skipping figures showing histograms of blackout sizes" + 
                       " histograms.")
        
    else:
        ## Blackout sizes after mitigation
        blackout_size_histogram_after_mitigation(calc_inertia_again=False, 
                                                save_prefix="SI",
                                                n_bins=50)
        # Blackout Size Distribution (different versions) and for 3 with zoom in
        ## Blackout Size (different versions)
        plot_blackout_size_histograms(yscale_log=True, n_cols=2, save_prefix="SI")
        plot_blackout_size_histograms(yscale_log=True, n_cols=2, n_bins=50,
                                    fname_suffix="_lessbins", save_prefix="SI")
        
        plot_blackout_size_histograms(yscale_log=True, xscale_log=True, n_cols=2,
                                    xlims=(1e-2, 1e2), fname_suffix="_log", save_prefix="SI")
        plot_blackout_size_histograms(yscale_log=True, xscale_log=True, n_cols=2,
                                    xlims=(1e-2, 1e2), n_bins=50, 
                                    fname_suffix="_log_lessbins", save_prefix="SI")
    
        ## Zoom in
        plot_blackout_size_distributions_with_zoom_in(save_prefix="SI")
    
        # 2D Historgram of Number of Components and Share of load not served
        plot_component_number_vs_blackout_size(save_prefix="SI")

    # Fig.3 sorted by contribution to load not served
    create_clustering_analysis_plot_from_data(fpath_cluster_plot_data, 
                                            use_only_lines=False,
                                            sort_by="accumulative_lost_load",
                                            save_prefix="SI")
    plt.close('all')
    gc.collect()
    # Detailed analysis of cluster 11: Daily Profile, Generation histogram and largest comp
    ## just run script 'scripts/plots/plot_individual_cluster.py'
    if low_memory:
        logger.warning("Skipping plot of individual clusters analysis.")
    else:
        subprocess.run([sys.executable, 
                    "scripts/plots/plot_individual_cluster.py"])
    
    # Blackout Statistics same vs different corridor
    create_blackout_statistics_common_vs_different_corridor_plot(save_prefix="SI",
                                                                 show_number_and_normalized=True)
    
    # Total inertia placed per country to compare with ENTSO-E results
    create_inertia_by_country_all_co2_plot(
        co2_lvl_ref=0.6,
        co2_lvls=[0.4, 0.2, 0.0],
        target="num_GSS",
        blackoutthreshold=0.8,
        save_prefix="SI"
    )
    
    # TYNP line extensions vs 20%
    path_to_csv_planned_comp = os.path.join(
            path_to_line_extension_mitigation_sclopf,
            "planned_line_extensions_n600_target_num_GSS_annualized_380kVonly_blackoutthres0.8.csv",
        )
    if not os.path.exists(path_to_csv_planned_comp):
        logger.warning("Need to generate planned extension data. Might take around 25 minutes.")
        run_planned_extension_comparison(
            n_nodes=600,
            target="num_GSS",
            blackoutthreshold=0.8,
            build_380kV_only=True,
            annualized_costs=True,
        )
    plot_planned_line_extensions_single_co2(
        csv_path=path_to_csv_planned_comp,
        co2l=0.2,
        figure_name="planned_line_extensions_co2l0.2_blackoutthres0.8",
        red_lines_rank=[11],
        pad_deg=0.75,
    )
    plt.close('all')
    gc.collect()
    
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster blackouts")

    
    parser.add_argument(
        "--low_memory", action="store_true", help="Use if memory of machine below 32GB", default=False
    )
    
    parser.add_argument(
        "--no_SI", action="store_true", help="Use if memory of machine below 32GB", default=False
    )
    
    args = parser.parse_args()
    
    low_memory = args.low_memory
    also_supplementary_figs = not args.no_SI
    
    plot_manuscript_figures(low_memory=low_memory)
    if also_supplementary_figs:
        plot_supplementary_figures(low_memory=low_memory)