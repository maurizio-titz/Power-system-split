#!usr/bin/env python
# -*- coding: utf-8 -*-

"""This implemention follows Dirk's idea in the matlab script"""


import numpy as np
from numpy import dtype
from scipy import sparse
from scipy import linalg as sc_linalg
from scipy.sparse.csgraph import connected_components


def set_spmatrix_col_zero(II_in, B_d_in, del_idx):
    
    zero_spdiag = sparse.diags([1 if ii != del_idx else 0 for ii in range(II_in.shape[1])],
                                    dtype=II_in.dtype)
    II_in = II_in * zero_spdiag
    B_d_in[del_idx] = 0
    
    return

def remove_line_from_Bd(B_d_in, num_parallel_ls, line_limits_ls, del_idx, atol=1e-8):
    """Remove a line by changing the value in the sparse matrix B_d_in
    that collects all susceptences according to a heuristic that 
    keeps the effective nature of the links in mind.

    Args:
        B_d_in (sparse diagonal matrix): Matrix collecting the susceptances
        num_parallel_ls (list): number quantifying the effective number of parrallel lines
        line_limit_ls (list): List of line limits that will be modified.
        del_idx (idx of ): idx of edge in graph that will be modified due to overloaded power line
    """
    # Decide what to change num_parallel
    num_parallel = num_parallel_ls[del_idx]
    assert num_parallel >= 0, ('Line removal for num_parallel={0:3f} not correct.'+
                               ' Line was either already removed a wrong num_parallel was assigned. ')
    
    
    if 0 < num_parallel < .5:
        num_parallel_new = 0
    elif 0.5 <= num_parallel < (1 - atol):
        num_parallel_new = num_parallel/2.
    elif abs(num_parallel - 1) < atol:
        num_parallel_new = 0.
    else:
        num_parallel_new = num_parallel - 1.
        
    num_par_factor = (num_parallel_new /num_parallel)
    num_parallel_ls[del_idx] = num_parallel_new
    B_d_in[del_idx, del_idx] *= num_par_factor
    line_limits_ls[del_idx] *= num_par_factor
    
    return


def simulate_cascade(II_in, B_d_in,
                     P0, line_limits, num_parallel_in, failure_links):
    """Simulate a cascade with the given inital failure links:

    Args:
        II_in (sparse matrix): Incidence matrix
        B_d_in (_type_): Diagonal matrix with susecptance on diagonal.
        P0 (1d numpy array): power injections/extractions 
        line_limits (_type_): Limits of of power lines. 's_nom' in PyPSA
        num_parallel_in (list): List with 'num_parrallel' that gives a effective number
        for each edge the line quantifying different and also multiple lines between two nodes.
        failure_links (tuple): Collects the initial failure links

    Returns:
        failure_cascase, did_system_split: Links involved in the cascase, boolean if the system did split
    """
    II = II_in.copy()
    B_d = B_d_in.copy()
    num_parallel_ls = num_parallel_in.copy()
    
    # Solve the inital case
    # todo is this really the fastest to do the product???
    #LL = II.dot(num_parallel * B_d).dot(II.T)
    
    # Remove inital failures
    for del_idx in failure_links:
        remove_line_from_Bd(B_d, num_parallel_ls,
                            line_limits, del_idx)
        
    did_system_split = False
    still_going = True
    failure_cascade = [failure_links]
    
    while still_going:
        
        # Check if network is still connected
        LL_r = II.dot(B_d).dot(II.T)
        AA = - LL_r + sparse.diag(LL_r.diagonal())
        
        nr_components = connected_components(AA)[0]
        
        if nr_components > 1:
            did_system_split = True
            break
        
        # Solve the power flow Equation
        try:
            theta = sc_linalg.spsolve(LL_r, P0)
        except np.linalg.LinAlgError:
            LL_inv = np.linalg.pinv(LL_r.todense())
            theta = np.dot(LL_inv, P0)
            
        flows = B_d.dot((II.T).dot(theta))
        
        # Check line limits
        idxs_overloaded =  np.where(abs(flows) > line_limits)[0]
        # Stop if no new lines where overloaded
        if len(idxs_overloaded) == 0:
            # Cascade stopped
            still_going = False
        
        else:
            failure_cascade.append(idxs_overloaded)
            # Delete overloaded lines
            for idx_r in idxs_overloaded:
                remove_line_from_Bd(B_d, num_parallel_ls,
                                    line_limits, idx_r)

            
    return failure_cascade, did_system_split
    