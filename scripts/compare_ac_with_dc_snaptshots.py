#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Relax the approximation of DC approx by adding sampled voltages that vary by percent from nominal value, 
effectively changing transmission capacities, and regarding sinius in coupling. 
This compared the inital step after the triggering of a failure to see if that leads more often 
to a cascade and how often."""



def load_network(path_to_network):
    
    return


def setup_system_choosePVnPQ_distribtueQ(fraction_PV_nodes: float = 0.2, 
                                         power_factor_mean: float = .95,
                                         power_factor_fluct: float=.025):
    """Setup a system for running AC simulations by assigning PV and PQ nodes 
    and setting Q"""
    
    
    return


def solve_power_flow_dc_vs_ac():
    """Outer wrapper that solves the dc for a given system and dispatch, and 
    generates the AC solvable example and solves it."""
    
    return 

