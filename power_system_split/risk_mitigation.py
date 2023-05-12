"""
Mitigation of risks caused by system splits
"""


import numpy as np
import pandas as pd

from power_system_split.risk_analysis import val_at_risk, calc_rocof


def inertia_placement(comp_props, indicator_vectors, var_ref, m, q, number_of_simulations, node_list, epsilon):
    """Optimal placement of synthetic inertia through greedy search. 

    Args:
        comp_props (pandas.DataFrame): Split component properties for one Co2 level, with 
        'time_stamp', 'number_of_split', 'inertia_proxy' and 'load_imbalance' as columns. 
        indicator_vectors (ndarray): Indicator vectors of split components
        var_ref (float): Reference Value at risk that should be reached through inertia placement
        m (float): Amount of incremental inertia to place in each step
        q (float): Quantile for VaR calculation
        number_of_simulations (int): Number of cascade simulations for one CO2 level
        node_list (list): List of nodes in graph
        epsilon (float): Relative deviation from reference value that is allowed after optimization.

    Returns:
        tuple: component properties with synthetic inertia and information on greedy search steps.
    """    
    
    # Initialize values for inertia placement
    step=1
    comp_props_new = comp_props.copy()
    var_targ = val_at_risk(comp_props_new, number_of_simulations, q, method='abs')
    inertia_placement_info = pd.DataFrame(columns=['new_var', 'added_node_index'], dtype=np.float)
    inertia_placement_info.loc[0, 'added_node_index'] = np.nan
    inertia_placement_info.loc[0, 'new_var'] = var_targ
    
    
    # Continue placing inertia until the reference value is reached
    while var_targ > var_ref*(1+epsilon) :

        var_proposals = np.zeros(len(node_list))
        comp_props_proposal = comp_props_new.copy()
        
        
        # Try all potential nodes 
        for count, node in enumerate(node_list):


            # add new synthetic inertia
            new_inertia = add_syn_inertia(indicator_vectors, count, m)
            comp_props_proposal.loc[:,'inertia_proxy'] = comp_props_new.loc[:,'inertia_proxy'] + new_inertia
            comp_props_proposal.loc[:, 'rocof'] = calc_rocof(comp_props_proposal.load_imbalance, 
                                                                 comp_props_proposal.inertia_proxy)
            
            var_proposals[count] = val_at_risk(comp_props_proposal, number_of_simulations, q,
                                                   method='abs')

        
        # Select proposal and place inertia at best node
        count_opt = np.argmin(var_proposals)
        comp_props_new.loc[:,'inertia_proxy'] += add_syn_inertia(indicator_vectors,
                                                                 count_opt, m)
        var_targ = var_proposals[count_opt]
        
        print(step, ' New var:', var_targ, 'Target Var:', var_ref)
        
        # Save placement information    
        inertia_placement_info.loc[step, 'added_node_index'] = count_opt
        inertia_placement_info.loc[step, 'new_var'] = var_targ

        
        # Go to next placement step
        step+=1
    
    return comp_props_new, inertia_placement_info

def add_syn_inertia(indicator_vectors, node_count, m):
    """ Add inertia to split components that contain the node "node_count".

    Args:
        indicator_vectors (ndarray): Indicator vectors of split components
        node_count (int): Index of node to place inertia on
        m (float): Amount of inertia generation to place (in MW)

    Returns:
        ndarray: Inertia per split component
    """    
    # identify splits where this node is involved
    indices = np.where(indicator_vectors[:, node_count])[0]

    # assign additional_synthetic_inertia to all split components
    synthetic_inertia = np.zeros(indicator_vectors.shape[0])
    synthetic_inertia[indices] += m
    
    return  synthetic_inertia






