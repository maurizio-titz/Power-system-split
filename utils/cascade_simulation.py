#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
Purely matrix-based simulation of cascading failures in power grids
"""

import itertools
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# First column: num_parallel before failure
# Second column: num_parallel after failure
## Not in original data and exists due to reduction from num_parallel > 3
## 0.59210526, 0.88157895, 1.18421053, 1.47368421, 1.59210526,
#  1.77631579, 2.06578947, 2.15789474, 2.18421053, 2.31578947, 2.36842105,
#  2.44736842, 2.57894737, 2.59210526, 2.73684211, 2.77631579
# 0.59210526 provides path to zero
LOOKUP_TABLE_NP = np.array(
    [
        [0.28947368, 0.0],
        [0.57894737, 0.28947368],
        [0.59210526, 0.0],
        [0.86842105, 0.57894737],
        [0.88157895, 0.28947368],
        [1.0, 0.0],
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
        [2.0, 1.0],
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
        [2.86842105, 1.86842105],
    ]
)
num_parallels_diffs_ = LOOKUP_TABLE_NP[:, 0] - LOOKUP_TABLE_NP[:, 1]
line_type_num_parallels = np.sort(np.unique(np.round(num_parallels_diffs_, 7)))

# The smallest cable seems to be 0.3351.. here instead of 0.2894.. as for sclopf
## Decision to reduce 2.34 by 1 and not be 0.3351...
LOOKUP_TABLE_NP_non_sclopf = np.array(
    [
        [0.33518006, 0.0],
        [0.67036011, 0.33518006],
        [0.67590027, 0.0],
        [1.0, 0.0],
        [1.00554017, 0.67036011],
        [1.01108033, 0.67590027],
        [1.33518006, 1.0],
        [1.34072022, 1.00554017],
        [1.34626039, 1.01108033],
        [1.67036011, 1.33518006],
        [1.67590028, 1.34072022],
        [1.68144040, 1.34626039],
        [2.0, 1.0],
        [2.00554017, 1.00554017],
        [2.01108033, 1.67590028],
        [2.01662050, 1.68144040],
        [2.33518006, 1.33518006],
        [2.34072022, 1.34072022],
        [2.34626039, 2.01108033],
        [2.67036011, 1.67036011],
        [2.67590028, 1.67590028],
        [2.68144044, 2.34626039],
    ]
)

# have to be added due to >3 being reduced by one. This are only the >2 ones
# add = [2.00554017, 2.01108033, 2.0166205, 2.33518006, 2.34072022, 2.34626039,
#       2.67036011, 2.67590028, 2.68144044] and 1.68144040, 0.67590027 is new smallest line
# not appearing in PyPSA network


def get_circuit_counts_in_all_lines(num_parallels, use_sclopf: bool = True):
    """Get the counts of different line types in all lines of the network.
    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a line and each column to a
            line type (defined by num_parallel reduction when one circuit is removed).
    """
    if use_sclopf:
        look_up_table = LOOKUP_TABLE_NP
    else:
        look_up_table = LOOKUP_TABLE_NP_non_sclopf

    num_p_paths_all = []
    for num_p in num_parallels:
        num_p_path = []
        while num_p > 0:
            num_p_path.append(num_p)
            if 0 < num_p < 3:
                num_parallel_case = np.argwhere(np.isclose(look_up_table[:, 0], num_p))[
                    0, 0
                ]
                num_p = look_up_table[num_parallel_case, 1]
            elif num_p >= 3:
                num_p -= 1
        num_p_paths_all.append(num_p_path + [0])

    # get the reduction of num_parallels at each step for each line
    line_type_counts = []
    for path in num_p_paths_all:
        path_diffs = [np.round(path[i] - path[i + 1], 7) for i in range(len(path) - 1)]
        # print(path_diffs)
        line_type_count = [
            sum(path_diffs == line_typ_num_par)
            for line_typ_num_par in line_type_num_parallels
        ]
        line_type_counts.append(line_type_count)
    line_type_counts = pd.DataFrame(
        line_type_counts,
        columns=[f"num_par_{num_par}" for num_par in line_type_num_parallels],
    )

    return line_type_counts


def get_partition(total_weight, weights: tuple = (0.29, 0.59, 1)):
    """get the partition of the total weight into the given weights.
    This function was graciously contributed by M. Titz"""
    import numpy as np
    from scipy.optimize import nnls

    # Convert to numpy array for easier manipulation
    weights_array = np.array(weights)

    # Try all possible combinations up to reasonable limits
    max_coeff = int(total_weight / min(weights)) + 1

    for c0 in range(max_coeff):
        for c1 in range(max_coeff - c0):
            for c2 in range(max_coeff - c0 - c1):
                candidate = np.array([c0, c1, c2])
                if np.isclose(
                    np.dot(candidate, weights_array), total_weight, atol=1e-6
                ):
                    return candidate
    raise ValueError("No valid partition found for the given total weight.")


def infer_circuit_counts_extension(
    num_parallels, num_parallels_extension, use_sclopf: bool = True
) -> pd.DataFrame:
    """Get the counts of different line types by which each line was extended.
    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
        num_parallels_extension (list-like): the per line num_parallel values by which each line was extended
    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a line and each column to a
            line type (defined by num_parallel reduction when one circuit is removed).
    """
    raise NotImplementedError("Function not yet implemented.")
    if use_sclopf:
        look_up_table = LOOKUP_TABLE_NP
    else:
        look_up_table = LOOKUP_TABLE_NP_non_sclopf


def remove_highest_volt_lvl_circuit(
    num_parallel: float,
    use_sclopf: bool = True,
):
    """Calculate the new effective number of circuits on a line
    after removing one circuit. The new value depends on the line type
    and is indicated in a lookup table.

    Args:
        num_parallel (float): Old value of effective number of circuits
        use_sclopf (bool): Decides which look up table to use, since sclopf and lopf PyPSA
        networks have different num_parallel value giving an effective line value.

    Returns:
        float: new value
    """
    if use_sclopf:
        look_up_table = LOOKUP_TABLE_NP
    else:
        look_up_table = LOOKUP_TABLE_NP_non_sclopf

    assert num_parallel >= 1e-8, (
        "Line removal for num_parallel=0 not correct."
        + " Line was either already removed a wrong num_parallel"
        + " was assigned."
    )

    if 0 < num_parallel < 3:
        # try:
        num_parallel_case = np.argwhere(np.isclose(look_up_table[:, 0], num_parallel))[
            0, 0
        ]
        num_parallel_new = look_up_table[num_parallel_case, 1]

        # except IndexError:
        #    raise LookupError(f"Num_parallel before failure '{num_parallel}' is not in Lookup table.")

    elif num_parallel >= 3:
        num_parallel_new = num_parallel - 1

    else:
        raise ValueError("num_parallel does not have a valid value!")

    return num_parallel_new


def calc_possible_double_line_failures(
    num_parallels,
    ignored_idxs=None,
    use_sclopf: bool = True,
    num_parallels_extension=None,
):
    """Determine the set of possible double line failures.

    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
        ignored_idxs (list-like, optional): Edges that should not fail.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network.
        num_parallels_extension (list-like, optional): the per line num_parallel values by which each line was extended

    Returns:
        list: List of dicts, where each dict contains two matrix indices of lines
    """
    circuit_counts_per_line = get_circuit_counts_in_all_lines(
        num_parallels, use_sclopf=use_sclopf
    )
    if num_parallels_extension is not None:
        raise NotImplementedError("Function not yet implemented.")

    if not ignored_idxs:
        edge_indices = range(len(num_parallels))
    else:
        edge_indices = list(set(range(len(num_parallels))) - set(ignored_idxs))

    possible_failures = []
    # Add failures on two different lines
    for idx_1 in edge_indices:
        for idx_2 in edge_indices:
            if idx_2 < idx_1:
                continue
            for line_type_1 in line_type_num_parallels:
                count_1 = circuit_counts_per_line.iloc[idx_1][f"num_par_{line_type_1}"]
                if count_1 > 0:
                    for line_type_2 in line_type_num_parallels:
                        if idx_1 == idx_2 and line_type_2 < line_type_1:
                            continue  # this prevents double counting same combinations

                        count_2 = circuit_counts_per_line.iloc[idx_2][
                            f"num_par_{line_type_2}"
                        ]
                        is_double_counting_case = (
                            idx_1 == idx_2 and line_type_1 == line_type_2
                        )
                        if is_double_counting_case:
                            count_2 -= 1  # avoid double counting same circuit

                        if count_2 > 0:
                            possible_failures.append(
                                {
                                    "failures": [
                                        (idx_1, line_type_1),
                                        (idx_2, line_type_2),
                                    ],
                                    "weight": (
                                        count_1 * count_2 / 2
                                        if is_double_counting_case
                                        else count_1 * count_2
                                    ),
                                }
                            )

    return possible_failures


def calc_possible_single_line_failures(num_parallels, ignored_idxs=None):
    """Determine the set of possible single line failures.

    Args:
        num_parallels (list-like): Effective number of parallel lines per edge.
        ignored_idxs (list-like, optional): Edges that should not fail.

    Returns:
        list: List of lists, where each list contains one matrix index of a line
    """
    circuit_counts_per_line = get_circuit_counts_in_all_lines(
        num_parallels, use_sclopf=True
    )

    if not ignored_idxs:
        edge_indices = range(len(num_parallels))
    else:
        edge_indices = list(set(range(len(num_parallels))) - set(ignored_idxs))

    possible_failures = []
    # Add failures on two different lines
    for idx_1 in edge_indices:
        for line_type_1 in line_type_num_parallels:
            count_1 = circuit_counts_per_line.iloc[idx_1][f"num_par_{line_type_1}"]
            if count_1 > 0:
                possible_failures.append(
                    {
                        "failures": [(idx_1, line_type_1)],
                        "weight": count_1,
                    }
                )

    return possible_failures


def remove_line_from_Bd(
    B_d_in,
    num_parallel_ls,
    line_limits_ls,
    del_idx,
    remove_all_circuits=False,
    use_sclopf: bool = True,
    num_parallel_reduction: Union[float, None] = None,
):
    """Remove a line by changing the susceptances, the number of parallel lines and
    the line limits.

    Args:
        B_d_in (sparse diagonal matrix): Diagonal matrix with susceptances
        num_parallel_ls (list): number quantifying the effective number of parrallel circuits on a line
        line_limit_ls (list): List of line limits that will be modified.
        del_idx (idx of ): idx of edge in graph that will be modified due to overloaded power line
        remove_all_circuits (bool): If False, remove only one circuit from the line.
        If True, remove the whole line with all circuits.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network.
    """

    # Select num_parallel of removed link
    num_parallel = num_parallel_ls[del_idx]
    assert num_parallel >= 1e-8, (
        "Line removal for num_parallel=0 not correct."
        + " Line was either already removed, a wrong num_parallel was assigned. "
    )
    assert (
        not remove_all_circuits or num_parallel_reduction is None
    ), "If remove_all_circuits is True, num_parallel_reduction has to be None."

    # Calculate new num_parallel after removal
    if remove_all_circuits:
        num_parallel_new = 0
    elif num_parallel_reduction is not None:  # if a specific value is given, use it
        num_parallel_new = num_parallel - num_parallel_reduction
    else:  # highest voltage level circuit is removed first by protective relays
        num_parallel_new = remove_highest_volt_lvl_circuit(
            num_parallel, use_sclopf=use_sclopf
        )
    if num_parallel_new < 1e-6:
        num_parallel_new = 0

    # Adapt network parameters accordingly
    num_par_factor = num_parallel_new / num_parallel
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

    if L == None:
        L = I.dot(B_d).dot(I.T)

    theta = np.zeros(L.shape[0])
    theta[1:] = sparse.linalg.spsolve(L[1:, 1:], P[1:])

    flows = B_d.dot((I.T).dot(theta))

    return flows


def simulate_cascade(
    II_in,
    B_d_in,
    P0,
    line_limits_in,
    num_parallel_in,
    failures: list,
    epsilon=1e-4,
    max_cascade_length=np.inf,
    use_sclopf: bool = True,
    initial_remove_all: bool = False,
):
    """Simulate a cascade with the given initial failure lines.


    Args:
        II_in (sparse matrix): Incidence matrix
        B_d_in (sparse matrix): Diagonal matrix with susceptance on diagonal.
        P0 (1d numpy array): power injections/extractions
        line_limits_in (1d numpy array): Limits of of power lines. 's_nom' in PyPSA
        num_parallel_in (1d numpy array): List with 'num_parallel' that gives a effective number
            for each edge the line quantifying different and also multiple lines between two nodes.
        failure_lines (list): Holds the tuples of the initial failure lines and their respective reductions in num_parallel. [(idx, line_type), ...]
        epsilon (float): Margin above capacity that has to be exceeded for a link to fail.
            he margin should be given as a share of the capacity (between 0 and 1).
        max_cascade_length (int): Maximum number of secondary failures to investigate.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network.

    Returns:
        failure_cascade, did_system_split: Lines involved in the cascade and boolean if the system did split.
            The cascade list only includes lines where all circuits have failed.
    """
    II = II_in.copy()
    B_d = B_d_in.copy()
    num_parallel_ls = num_parallel_in.copy()
    line_limits = line_limits_in.copy()

    if any(
        [
            not any(line_type_num_parallels - failure_tuple[1] < 1e-6)
            for failure_tuple in failures
        ]
    ):
        raise ValueError(
            "A num_parallel reduction value in failures is not valid."
            + f" Valid reductions are: {line_type_num_parallels}"
        )

    # Remove initial failures
    for del_idx, line_type in failures:
        remove_line_from_Bd(
            B_d,
            num_parallel_ls,
            line_limits,
            del_idx,
            num_parallel_reduction=line_type,
            use_sclopf=use_sclopf,
            remove_all_circuits=initial_remove_all,
        )

    did_system_split = False
    still_going = True
    cascade_length = 0

    # Only add initial failure that fully removed a line
    failure_cascade = list(np.argwhere(np.isclose(num_parallel_ls, 0))[:, 0])

    while still_going:

        # Check if network is still connected
        LL_r = II.dot(B_d).dot(II.T)
        AA = -LL_r + sparse.diags(LL_r.diagonal())

        nr_components = connected_components(AA)[0]

        if nr_components > 1:
            did_system_split = True
            break

        flows = solve_lpf(P0, B_d, II, LL_r)

        # Check line limits
        idxs_overloaded = np.where(abs(flows) > line_limits * (1 + epsilon))[0]

        # Stop if no new lines where overloaded
        if len(idxs_overloaded) == 0:
            # Cascade stopped
            still_going = False
        else:
            failure_cascade += list(idxs_overloaded)
            # Delete all circuits in a line if line is overloaded
            for idx_r in idxs_overloaded:
                remove_line_from_Bd(
                    B_d,
                    num_parallel_ls,
                    line_limits,
                    idx_r,
                    remove_all_circuits=True,
                    use_sclopf=use_sclopf,
                )

        # Stop if max length of cascade simulation is reached
        cascade_length += 1
        if cascade_length == max_cascade_length:
            break

    return failure_cascade, did_system_split
