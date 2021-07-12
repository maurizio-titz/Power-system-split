#!/usr/bin/python3
# -*- coding: utf-8 -*

import networkx as nx
import numpy as np
import sys

sys.path.append('../power_system_split/')
import visualisation


def test_likelihood_no_solution_dict():

        G = nx.Graph()
        G.add_edge(1,2,
                   weight = 1,
                   orientation = (1,2),
                   line_index = [0],
                   s_nom = 1)
        G.add_edge(1,3,
                   weight = 1,
                   orientation = (1,3),
                   line_index = [1],
                   s_nom = 1)
        G.add_edge(3,2,
                   weight = 1,
                   orientation = (3,2),
                   line_index = [2],
                   s_nom = 1)

        splitting_cascades = {'0':[[(1,2),(1,3)],[(2,3)]],
                              '1':[[(2,3),(1,2)]]}

        number_of_snapshots = 2

        solution_dict = {}
        print(G.edges())
        primary_l, secondary_l = visualisation.calc_likelihood_failure(G,
                                                splitting_cascades,
                                                number_of_snapshots,
                                                solution_dict = solution_dict)

        assert(np.isclose(primary_l[(1,2)],0.5))
        assert(np.isclose(primary_l[(2,3)],1.0))
        assert(np.isclose(primary_l[(1,3)],0.0))

        assert(np.isclose(secondary_l[(2,3)],2/6.))
        assert(np.isclose(secondary_l[(1,2)],2/6.))
        assert(np.isclose(secondary_l[(1,3)],1/6.))

        return

def test_likelihood_with_solution_dict():

        G = nx.Graph()
        G.add_edge(1,2,
                   weight = 1,
                   orientation = (1,2),
                   line_index = [0],
                   s_nom = 1)
        G.add_edge(1,3,
                   weight = 1,
                   orientation = (1,3),
                   line_index = [1],
                   s_nom = 1)
        G.add_edge(3,2,
                   weight = 1,
                   orientation = (3,2),
                   line_index = [2],
                   s_nom = 1)

        splitting_cascades = {'0':[[(1,2),(1,3)],[(2,3)]],
                              '1':[[(2,3),(1,2)]]}

        number_of_snapshots = 2

        solution_dict = {'0':[0,'test',1,'test']}

        primary_l, secondary_l = visualisation.calc_likelihood_failure(G,
                                                splitting_cascades,
                                                number_of_snapshots,
                                                solution_dict = solution_dict)

        assert(np.isclose(primary_l[(1,2)],0.5))
        assert(np.isclose(primary_l[(2,3)],0.5))
        assert(np.isclose(primary_l[(1,3)],0.0))

        assert(np.isclose(secondary_l[(2,3)],1/6.))
        assert(np.isclose(secondary_l[(1,2)],1/6.))
        assert(np.isclose(secondary_l[(1,3)],1/6.))


        return
