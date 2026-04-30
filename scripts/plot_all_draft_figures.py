#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set Joule style
import os
os.environ["PLOT_STYLE"] = "joules"
from utils.plot_style import *

from scripts.plots.plot_combined_generation_storage import create_combined_generation_storage_plot
from scripts.plots.plot_combined_flow_split_statistics import create_combined_flow_and_split_statistics_plot
from scripts.plots.plot_mitigation import create_combined_mitigation_plot

def plot_all_figures(plot_supplementary_figures: bool=False):
    
    ## Main fiugres
    # Fig. 1: Scenarios for decarbonisation of The European power system
    create_combined_generation_storage_plot()
    
    # Fig. 2: Evolution of risks during decarbonistaion
    create_combined_flow_and_split_statistics_plot(mean_distance=False, load_normalization=False, 
                                                   show_blackout_stats=True, secondary_proba_axis=True)
    
    # Fig. 3: Characteristic geographic patterns of system split
    
    
    # Fig. 4: Conditional probability of transmission lines participating in casc. failures
    create_combined_mitigation_plot(use_annualized_costs=True,
                                    co2_lvl_map=0.2,
                                    build_380kV_only=True,
                                    plot_intertia_cost=True,
                                    blackoutthreshold=.8,
                                    
                                    )
    
    # Fig. 5: Mitigation of split-induced blackouts via inertia and grid reinforcements
    
    
    if plot_supplementary_figures:
        pass
    
    return



if __name__ == "__main__":
    
    plot_all_figures()