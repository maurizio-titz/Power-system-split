"""
Purely matrix-based simulation of cascading failures in power grids
"""


import numpy as np
from numpy import dtype
from scipy import sparse
from scipy import linalg as sc_linalg
from scipy.sparse.csgraph import connected_components
import itertools
import scipy


# def load_num_parallel_table():
#     """Load lookup table for changing number of parallel lines after
#     failure. 

#     Returns:
#         1d numpy array: lookup table
#     """
    
    
#     lookup_table = np.array([
#         0.28947368421052,					               0,
#         0.57894736842105,					0.28947368421052,
#         0.59210526315789,					               0,
#         0.86842105263157,					0.57894736842105,
#         0.88157894736842,					0.28947368421052,
#                        1,				                   0,
#         1.15789473684211,					0.86842105263157,
#         1.18421052631579,					0.59210526315789,
#         1.28947368421053,					0.28947368421052,
#         1.44736842105263,					1.15789473684211,
#         1.47368421052632,					0.88157894736842,
#         1.57894736842105,					0.57894736842105,
#         1.73684210526316,					1.44736842105263,
#         1.77631578947368,					1.18421052631579,
#         1.86842105263158,					0.86842105263157,
#                        2,					               1,
#         2.02631578947368,					1.73684210526316,
#         2.06578947368421,					1.47368421052632,
#         2.15789473684211,					1.15789473684211,
#         2.18421052631579,					1.18421052631579,
#         2.28947368421053,					1.28947368421053,
#         2.31578947368421,					2.02631578947368,
#         2.36842105263158,					1.77631578947368,
#         2.57894736842105,					1.57894736842105,
#         2.59210526315789,					1.59210526315789,
#         2.65789473684211,					2.06578947368421,
#         2.86842105263158,					1.86842105263158
#     ])
    
    
#     return lookup_table


def calc_num_parallel_after_failure(num_parallel):
    """Calculate the new effective number of circuits on a line 
    after removing one circuit. The new value depends on the line type
    and is indicated in a lookup table.  

    Args:
        num_parallel (float): Old value of effective number of circuits

    Returns:
        float: new value
    """        
    
    assert num_parallel >= 0, ('Line removal for num_parallel={0:3f} not correct.'+
                               ' Line was either already removed a wrong num_parallel was assigned. ')


    # First column: num_parallel before failure
    # Second column: num_parallel after failure
    lookup_table = np.array([
        [0.28947368, 0.        ],
        [0.57894737, 0.28947368],
        [0.59210526, 0.        ],
        [0.86842105, 0.57894737],
        [0.88157895, 0.28947368],
        [1.        , 0.        ],
        [1.15789474, 0.86842105],
        [1.18421053, 0.59210526],
        [1.28947368, 0.28947368],
        [1.44736842, 1.15789474],
        [1.47368421, 0.88157895],
        [1.57894737, 0.57894737],
        [1.59210526, 0.59210526],
        [1.73684211, 1.44736842],
        [1.77631579, 1.18421053],
        [1.86842105, 0.86842105],
        [2.        , 1.        ],
        [2.02631579, 1.73684211],
        [2.06578947, 1.47368421],
        [2.15789474, 1.15789474],
        [2.18421053, 1.18421053],
        [2.28947368, 1.28947368],
        [2.31578947, 2.02631579],
        [2.36842105, 1.77631579],
        [2.44736842, 1.44736842],
        [2.57894737, 1.57894737],
        [2.59210526, 1.59210526],
        [2.65789474, 2.06578947],
        [2.73684211, 1.73684211],
        [2.77631579, 1.77631579],
        [2.86842105, 1.86842105]
       ])


    if 0<num_parallel<3:
        num_parallel_case = np.argwhere(np.isclose(lookup_table[:,0], num_parallel))[0,0]
        num_parallel_new = lookup_table[num_parallel_case,1]
    elif num_parallel>=3:
        num_parallel_new = num_parallel - 1
    else:
        raise ValueError('num_parallel does not have a valid value!')

        
    return num_parallel_new

def calc_possible_double_line_failures(num_parallel_ls, bridge_idxs=None): 
    
    if bridge_idxs is None:
        edge_indices = range(len(num_parallel_ls))
    else:
        edge_indices = list(set(range(len(num_parallel_ls))) - set(bridge_idxs))
    
    # Add failures on two different links
    possible_failures = list(map(list, itertools.combinations(edge_indices,2)))
    
    # Add common mode failures (two circuits failing in one link)
    for i in edge_indices:
        num_parallel_one_fail = calc_num_parallel_after_failure(num_parallel_ls[i])
        
        # Only add common mode failure if more than one circuit is present
        # (i.e., first failure did not remove all circuits)
        if not np.isclose(num_parallel_one_fail,0):
            possible_failures += [[i,i]]
        
    return possible_failures


def remove_line_from_Bd(B_d_in, num_parallel_ls, line_limits_ls, del_idx, remove_all_circuits=False):
    """Remove a line by changing the susceptances, the number of parallel lines and 
    the line limits.

    Args:
        B_d_in (sparse diagonal matrix): Diagonal matrix with susceptances
        num_parallel_ls (list): number quantifying the effective number of parrallel circuits on a line
        line_limit_ls (list): List of line limits that will be modified.
        del_idx (idx of ): idx of edge in graph that will be modified due to overloaded power line
        remove_all_circuits (bool): If False, remove only one circuit from the line. 
        If True, remove the whole line with all circuits.  
    """

    # Select num_parallel of removed link
    num_parallel = num_parallel_ls[del_idx]
    assert num_parallel >= 0, ('Line removal for num_parallel={0:3f} not correct.'+
                               ' Line was either already removed a wrong num_parallel was assigned. ')
    
    # Calculate new num_parallel after removal
    if remove_all_circuits:
        num_parallel_new = 0 
    else:
        num_parallel_new = calc_num_parallel_after_failure(num_parallel)
        
    # Adapt network parameters accordingly
    num_par_factor = (num_parallel_new /num_parallel)
    num_parallel_ls[del_idx] = num_parallel_new
    B_d_in[del_idx, del_idx] *= num_par_factor
    line_limits_ls[del_idx] *= num_par_factor
    
    return

def solve_lpf(P, B_d, I, L=None):
    """Solve linear power flow.

    Args:
        P (1d numpy array): Power injections
        B_d (sparse matrix): Susceptance matrix
        I (sparse matrix): Incidence matrix
        L (sparse matrix): Laplacian matrix

    Returns:
        flows: Power flows
    """
    
    if L==None:
        L = I.dot(B_d).dot(I.T)
    
    try:
        theta = sparse.linalg.spsolve(L, P)
    except np.linalg.LinAlgError:
        L_inv = np.linalg.pinv(L.todense())
        theta = np.dot(L_inv, P)
        
    flows = B_d.dot((I.T).dot(theta))
    
    return flows

def simulate_cascade(II_in, B_d_in,
                     P0, line_limits_in, num_parallel_in, failure_links,
                     epsilon=0, max_cascade_length=np.inf):
    """Simulate a cascade with the given inital failure links:

    Args:
        II_in (sparse matrix): Incidence matrix
        B_d_in (_type_): Diagonal matrix with susecptance on diagonal.
        P0 (1d numpy array): power injections/extractions 
        line_limits (_type_): Limits of of power lines. 's_nom' in PyPSA
        num_parallel_in (list): List with 'num_parrallel' that gives a effective number
        for each edge the line quantifying different and also multiple lines between two nodes.
        failure_links (list): Collects the initial failure links
        epsilon (float): Margin above capacity that has to be exceeded for a link to fail. 
        The margin should be given as a share of the capacity (between 0 and 1).
        max_cascade_length (int): Maximum number of secondary failures to investigate.

    Returns:
        failure_cascase, did_system_split: Links involved in the cascase, boolean if the system did split
    """
    II = II_in.copy()
    B_d = B_d_in.copy()
    num_parallel_ls = num_parallel_in.copy()
    line_limits = line_limits_in.copy()
    
    # Solve the inital case
    # todo is this really the fastest to do the product???
    #LL = II.dot(num_parallel * B_d).dot(II.T)
    
    # Remove inital failures
    for del_idx in failure_links:
        remove_line_from_Bd(B_d, num_parallel_ls,
                            line_limits, del_idx)
        
    did_system_split = False
    still_going = True
    cascade_length = 0
    failure_cascade = failure_links.copy()
    
    while still_going:
        
        # Check if network is still connected
        LL_r = II.dot(B_d).dot(II.T)
        AA = - LL_r + sparse.diags(LL_r.diagonal())
  
        nr_components = connected_components(AA)[0]
        
        if nr_components > 1:
            did_system_split = True
            break
        
        flows = solve_lpf(P0, B_d, II, LL_r)
        
        # Check line limits
        idxs_overloaded =  np.where(abs(flows) > line_limits*(1+epsilon))[0]
        
        # Stop if no new lines where overloaded
        if len(idxs_overloaded) == 0:
            # Cascade stopped
            still_going = False
        
        else:
            failure_cascade += list(idxs_overloaded)
            # Delete overloaded lines
            for idx_r in idxs_overloaded:
                remove_line_from_Bd(B_d, num_parallel_ls,
                                    line_limits, idx_r,
                                    remove_all_circuits=True)
                
        # Stop if max length of cascade simulation is reached
        cascade_length+=1
        if cascade_length==max_cascade_length:
            break

            
    return failure_cascade, did_system_split
    
