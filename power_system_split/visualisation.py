#!usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains useful methods to analyse and
visualise cascade results"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx




def get_split_adjacencies_from_rocof_solutions(solution_dict,nx_graph):
    """obtain the adjacency matrices for each split in the
    rocof solution dictionary (see evaluate_split_observables).

    NOTE: This only takes into account splits that have been
    prefiltered according to the nodes criterion or the load criterion
    passed to the evaluation function"""

    adjacencies = []

    for key in solution_dict.keys():
        for number in solution_dict[key][::2]:

            split = splitting_cascades[key][number]
            F = nx_graph.copy()
            F.remove_edges_from(split)
            A = nx.adjacency_matrix(F).A
            adjacencies.append(A.astype('bool'))

    return adjacencies
