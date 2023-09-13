"""
Purely matrix-based simulation of cascading failures in power grids
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import itertools


# First column: num_parallel before failure
# Second column: num_parallel after failure
LOOKUP_TABLE_NP = np.array([
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


def calc_num_parallel_after_failure(num_parallel):
    """Calculate the new effective number of circuits on a line 
    after removing one circuit. The new value depends on the line type
    and is indicated in a lookup table.  

    Args:
        num_parallel (float): Old value of effective number of circuits

    Returns:
        float: new value
    """        
    
    assert num_parallel >= 1e-8, ('Line removal for num_parallel=0 not correct.' +
                                  ' Line was either already removed a wrong num_parallel' +
                                  ' was assigned.')

    if 0 < num_parallel < 3:
        num_parallel_case = np.argwhere(np.isclose(LOOKUP_TABLE_NP[:, 0],
                                                   num_parallel))[0, 0]
        num_parallel_new = LOOKUP_TABLE_NP[num_parallel_case, 1]
        
    elif num_parallel >= 3:
        num_parallel_new = num_parallel - 1
        
    else:
        raise ValueError('num_parallel does not have a valid value!')

    return num_parallel_new


def calc_possible_double_line_failures(num_parallels, ignored_idxs=None): 
    """Determine the set of possible double line failures.

    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
        ignored_idxs (list-like, optional): Edges that should not fail. 

    Returns:
        list: List of lists, where each list contains two matrix indices of lines
    """


    if not ignored_idxs:
        edge_indices = range(len(num_parallels))
    else:
        edge_indices = list(set(range(len(num_parallels))) - set(ignored_idxs))
    
    # Add failures on two different lines
    possible_failures = list(map(list, itertools.combinations(edge_indices,2)))
    
    # Add common mode failures (two circuits failing in one line)
    for i in edge_indices:
        num_parallel_one_fail = calc_num_parallel_after_failure(num_parallels[i])
        
        # Only add common mode failure if more than one circuit is present
        # (i.e., first failure did not remove all circuits)
        if not np.isclose(num_parallel_one_fail,0):
            possible_failures += [[i,i]]
        
    return possible_failures


def calc_possible_single_line_failures(num_parallels, ignored_idxs=None):
    """Determine the set of possible single line failures.

    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
        ignored_idxs (list-like, optional): Edges that should not fail.

    Returns:
        list: List of lists, where each list contains one matrix index of a line
    """

    if not ignored_idxs:
        possible_failures = list(range(len(num_parallels)))
    else:
        possible_failures = list(set(range(len(num_parallels))) - set(ignored_idxs))

    return [[failure] for failure in possible_failures]


def remove_line_from_Bd(B_d_in, num_parallel_ls, line_limits_ls, del_idx,
                        remove_all_circuits=False):
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
    assert num_parallel >= 1e-8, ('Line removal for num_parallel=0 not correct.'+
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
    
    theta = np.zeros(L.shape[0])
    theta[1:] = sparse.linalg.spsolve(L[1:,1:], P[1:])
        
    flows = B_d.dot((I.T).dot(theta))
    
    return flows


def simulate_cascade(II_in, B_d_in,
                     P0, line_limits_in, num_parallel_in, failure_lines,
                     epsilon=1e-4, max_cascade_length=np.inf): 
    """Simulate a cascade with the given inital failure lines. 

    
    Args:
        II_in (sparse matrix): Incidence matrix
        B_d_in (sparse matrix): Diagonal matrix with susecptance on diagonal.
        P0 (1d numpy array): power injections/extractions 
        line_limits_in (1d numpy array): Limits of of power lines. 's_nom' in PyPSA
        num_parallel_in (1d numpy array): List with 'num_parrallel' that gives a effective number
        for each edge the line quantifying different and also multiple lines between two nodes.
        failure_lines (list): Collects the initial failure lines
        epsilon (float): Margin above capacity that has to be exceeded for a link to fail. 
        The margin should be given as a share of the capacity (between 0 and 1).
        max_cascade_length (int): Maximum number of secondary failures to investigate.

    Returns:
        failure_cascade, did_system_split: Lines involved in the cascade and boolean if the system did split.
        The cascade list only includes lines where all circuits have failed. 
    """
    II = II_in.copy()
    B_d = B_d_in.copy()
    num_parallel_ls = num_parallel_in.copy()
    line_limits = line_limits_in.copy()
    
    # Remove inital failures
    for del_idx in failure_lines:
        remove_line_from_Bd(B_d, num_parallel_ls,
                            line_limits, del_idx)
        
    did_system_split = False
    still_going = True
    cascade_length = 0

    # Only add initial failure that fully removed a line
    failure_cascade = list(np.argwhere(np.isclose(num_parallel_ls, 0))[:,0])
    
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
        idxs_overloaded =  np.where(abs(flows) > line_limits*(1 + epsilon))[0]
        
        # Stop if no new lines where overloaded
        if len(idxs_overloaded) == 0:
            # Cascade stopped
            still_going = False
        
        else:
            failure_cascade += list(idxs_overloaded)
            # Delete all circuits in a line if line is overloaded
            for idx_r in idxs_overloaded:
                remove_line_from_Bd(B_d, num_parallel_ls,
                                    line_limits, idx_r,
                                    remove_all_circuits=True)
                
        # Stop if max length of cascade simulation is reached
        cascade_length+=1
        if cascade_length==max_cascade_length:
            break

            
    return failure_cascade, did_system_split
    