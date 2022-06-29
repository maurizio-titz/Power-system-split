#!usr/bin/env python
# -*- coding: utf-8 -*-

"""This implemention follows Dirk's idea in the matlab script"""


import numpy as np
from numpy import dtype
from scipy import sparse
from scipy import linalg as sc_linalg
from scipy.sparse.csgraph import connected_components

def remove_line(II_in, B_d_in, del_idx):
    
    cap_now = 
    if 

def set_spmatrix_col_zero(II_in, B_d_in, del_idx):
    
    zero_spdiag = sparse.diags([1 if ii != del_idx else 0 for ii in range(II_in.shape[1])],
                                    dtype=II_in.dtype)
    II_in = II_in * zero_spdiag
    B_d_in[del_idx] = 0
    
    return

def simulate_cascade(AA_in, II_in, B_d,
                     P0, line_limits, num_parallel_in, failure_links):
    
    AA = AA_in.copy()
    II = II_in.copy()
    num_parallel = num_parallel_in.copy()
    
    # Solve the inital case
    # todo is this really the fastest to do the product???
    LL = II.dot(num_parallel * B_d).dot(II.T)
    
    for fail_r in failure_links:
        
        still_going = True
        
        while still_going:
            # Modify matrices accordingly that is II and B_d set to zero both ends
            # todo where to check if split occured?
            for del_idx in fail_r:
                set_spmatrix_col_zero(II, B_d, del_idx)
            
            # Check if network is still connected
            LL_r = II.dot(B_d).dot(II.T)
            AA = - LL_r + sparse.diag(LL_r.diagonal())
            
            nr_components = connected_components(AA)[0]
            
            if nr_components > 1:
                still_going = False
                break
            
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
                # todo save something
                still_going = False
            
            else:
                # Delete overloaded lines
                for idx_r in idxs_overloaded:
                    set_spmatrix_col_zero(II, B_d, idx_r)
                    
                    # todo Check if still connected
                
            LL_r = II.dot(B_d).dot(II.T)
            AA = - LL_r + sparse.diag(LL_r.diagonal())
            
            nr_components = connected_components(AA)[0]
            
    
    return
    