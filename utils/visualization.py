""" 
Different functions to visualize system splits and their properties
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import RadiusNeighborsClassifier
from tqdm import tqdm

from utils import data_handling


def calc_likelihood_failure(nx_graph,
                            splitting_cascades,
                            number_of_simulations,
                            generator_snapshot_weightings,
                            ):
    """Calculate the likelihood that a) a given edge causes a split
    if it fails (primary likelihood) and b) the likelihood that a
    given edge fails at some point during a cascade (secondary likelihood). 

    Args:
        nx_graph (networkx graph): Graph of power system 
        splitting_cascades (dict): Nested dictionary with cascade results
        number_of_simulations (int): Number of initial failures times the number of time steps.
        generator_snapshot_weightings (pd.DataFrame): Number of hours for this snapshot.

    Returns:
        likelihood_primary, likelihood_secondary (dict, dict): Likelihood of primary and subsequent secondary
        failures induced by the link that is defined in the key.
    """

    likelihood_primary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    likelihood_secondary = {(u, v): 0.0 for u, v in nx_graph.edges()}
    
    #nr_edges = nx_graph.number_of_edges()
    #total_nr_splits = sum(len(vv) for vv in splitting_cascades.values())

    for timestamp, splits in tqdm(splitting_cascades.items()):
        weight = generator_snapshot_weightings[timestamp]

        
        for init_failure, cascade in splits.items():
            
            cascade_edges = data_handling.matrix_indices_to_nx_edges(cascade, nx_graph)
            
            for failed_edge in cascade_edges:
                
                likelihood_secondary[failed_edge] += weight/number_of_simulations

            init_edge1, init_edge2 = data_handling.matrix_indices_to_nx_edges(init_failure, nx_graph)
            likelihood_primary[init_edge1] += weight/number_of_simulations
            likelihood_primary[init_edge2] += weight/number_of_simulations
        
    return likelihood_primary, likelihood_secondary


def cluster_indicator_vectors_agglomerative(indicator_vectors, n_cluster=None, min_cluster_distance=0.1,
                                            cluster_distance_type='single'):
    """Cluster indicator vectors of graph components into similar groups with agglomerative clustering.

    The distance between vectors is quantified by the hamming distance,
    i.e. the relative number of nodes that are not in the same component.

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        n_cluster (int, optional): Number of cluster. If this is not None, min_cluster_distance has to be None.
        Defaults to None.
        min_cluster_distance (float, optional): The minimum distance between two clusters. Below this distance, two
        clusters will be merged. Defaults to 0.1.
        cluster_distance_type (str, optional): Method to compute the distance between two cluster.
        Available: "ward","average","single" and "maximum". Defaults to 'average'.

    Returns:
        tuple: number of cluster and array of cluster labels for each indicator vector
    """

    if n_cluster != None and min_cluster_distance != None:
        raise ValueError(
            'If n_cluster is not None, min_cluster_distance has to be None')

    agg_cluster = AgglomerativeClustering(affinity='hamming',
                                          n_clusters=n_cluster,
                                          distance_threshold=min_cluster_distance,
                                          linkage=cluster_distance_type,
                                          compute_distances=False)
    agg_cluster.fit(indicator_vectors)

    return agg_cluster.n_clusters_, agg_cluster.labels_


def cluster_indicator_vectors_combined(indicator_vectors, min_cluster_distance=0.1, subset_ratio=0.01,
                                       n_jobs=20):
    """Cluster indicator vectors of graph components into similar groups with Agglomerative clustering and 
    Nearest-Neighbor classification combined.

    The distance between vectors is quantified by the hamming distance,
    i.e. the relative number of nodes that are not in the same component (number between 0 and 1).
    
    First, a subset of indicator vectors is clustered with Agg. clustering. The rest is then assigned to the cluster
    with Nearest-Neighbor classification. We use a fixed radius to determine nearest neighbors to account
    for a minimum distance that should seperate the clusters. 

    Args:
        indicator_vectors (ndarray): Indicators of split components with shape n_vectors x n_nodes
        min_cluster_distance (float): The minimum hamming distance between two clusters (number between 0 and 1).
        Below this distance, two clusters will be merged during Agglomerative clustering. 
        subset_ratio (float):  Ratio of indicator vectors used in Agglomerative clustering. 
        n_jobs (int): Number of jobs.

    Returns:
        tuple: number of cluster and array of cluster labels for each indicator vector
    """

    # Select subset of indicator vectors
    indices_all_components = np.arange(indicator_vectors.shape[0])
    np.random.shuffle(indices_all_components)
    subset_indices = indices_all_components[:int(subset_ratio*indicator_vectors.shape[0])]
    remaining_indices = indices_all_components[int(subset_ratio*indicator_vectors.shape[0]):]
    
    print('\n',subset_indices.size, 'components will be used during agglomerative clustering.')
    print(remaining_indices.size, 'components will be assigned via NN classification.')

    # Cluster subset of vectors
    print('Agglomerative clustering of subset...')
    n_cluster, subset_c_labels = cluster_indicator_vectors_agglomerative(indicator_vectors[subset_indices], 
                                                                         min_cluster_distance = min_cluster_distance,
                                                                         cluster_distance_type = 'average')
    print('There are ',n_cluster, 'unique split cluster.')
    
    # Assign cluster labels to remaining vectors
    print('NN classification of remaining instances...')
    rnc = RadiusNeighborsClassifier(radius= min_cluster_distance, metric='hamming', outlier_label=-1, n_jobs=n_jobs)
    rnc.fit(indicator_vectors[subset_indices], subset_c_labels)
    remaining_c_labels = rnc.predict(indicator_vectors[remaining_indices])
    
    all_c_labels = np.zeros(indicator_vectors.shape[0])
    all_c_labels[subset_indices] = subset_c_labels
    all_c_labels[remaining_indices] = remaining_c_labels


    return n_cluster, all_c_labels


#TODO: Adapt and correct inertia placement functions below.
 
# def inertia_placement(comp_props, indicator_vectors, var_ref, m, q, number_of_simulations, node_list, epsilon):
#     """Optimal placement of synthetic inertia through greedy search. 

#     Args:
#         comp_props (pandas.DataFrame): Split component properties for one Co2 level, with 
#         'time_stamp', 'number_of_split', 'inertia_proxy' and 'load_imbalance' as columns. 
#         indicator_vectors (ndarray): Indicator vectors of split components
#         var_ref (float): Reference Value at risk that should be reached through inertia placement
#         m (float): Amount of incremental inertia to place in each step
#         q (float): Quantile for VaR calculation
#         number_of_simulations (int): Number of cascade simulations for one CO2 level
#         node_list (list): List of nodes in graph
#         epsilon (float): Relative deviation from reference value that is allowed after optimization.

#     Returns:
#         tuple: component properties with synthetic inertia and information on greedy search steps.
#     """    
    
#     # Initialize values for inertia placement
#     step=1
#     comp_props_new = comp_props.copy()
#     var_targ = val_at_risk(comp_props_new, number_of_simulations, q, method='abs')
#     inertia_placement_info = pd.DataFrame(columns=['new_var', 'added_node_index'], dtype=np.float)
#     inertia_placement_info.loc[0, 'added_node_index'] = np.nan
#     inertia_placement_info.loc[0, 'new_var'] = var_targ
    
    
#     # Continue placing inertia until the reference value is reached
#     while var_targ > var_ref*(1+epsilon) :

#         var_proposals = np.zeros(len(node_list))
#         comp_props_proposal = comp_props_new.copy()
        
        
#         # Try all potential nodes 
#         for count, node in enumerate(node_list):


#             # add new synthetic inertia
#             new_inertia = add_syn_inertia(indicator_vectors, count, m)
#             comp_props_proposal.loc[:,'inertia_proxy'] = comp_props_new.loc[:,'inertia_proxy'] + new_inertia
#             comp_props_proposal.loc[:, 'rocof'] = calc_rocof(comp_props_proposal.load_imbalance, 
#                                                                  comp_props_proposal.inertia_proxy)
            
#             var_proposals[count] = val_at_risk(comp_props_proposal, number_of_simulations, q,
#                                                    method='abs')

        
#         # Select proposal and place inertia at best node
#         count_opt = np.argmin(var_proposals)
#         comp_props_new.loc[:,'inertia_proxy'] += add_syn_inertia(indicator_vectors,
#                                                                  count_opt, m)
#         var_targ = var_proposals[count_opt]
        
#         print(step, ' New var:', var_targ, 'Target Var:', var_ref)
        
#         # Save placement information    
#         inertia_placement_info.loc[step, 'added_node_index'] = count_opt
#         inertia_placement_info.loc[step, 'new_var'] = var_targ

        
#         # Go to next placement step
#         step+=1
    
#     return comp_props_new, inertia_placement_info

# def add_syn_inertia(indicator_vectors, node_count, m):
#     """ Add inertia to split components that contain the node "node_count".

#     Args:
#         indicator_vectors (ndarray): Indicator vectors of split components
#         node_count (int): Index of node to place inertia on
#         m (float): Amount of inertia generation to place (in MW)

#     Returns:
#         ndarray: Inertia per split component
#     """    
#     # identify splits where this node is involved
#     indices = np.where(indicator_vectors[:, node_count])[0]

#     # assign additional_synthetic_inertia to all split components
#     synthetic_inertia = np.zeros(indicator_vectors.shape[0])
#     synthetic_inertia[indices] += m
    
#     return  synthetic_inertia

# def val_at_risk(data, n_total, q, target='rocof', method='abs'):
#     """ Calculate value at risk of a target for one Co2 level.

#     Args:
#         data (pandas.DataFrame): Split component properties for one Co2 level, with 
#         'time_stamp', 'number_of_split', <target> as columns. 
#         n_total (int): Total number of data points for which the quantile is calculated. If <data> does not
#         supply all data points, they will be extended by zeros. The quantile is then evaluated on the extended
#         data array.
#         q (float): Quantile, between 0 and 1. 
#         target (str, optional): Specify column name where target is stored. Defaults to 'rocof'. 
#         method (str, optional): Method to quantify impact of one split component. Defaults to 'abs'.

#     Returns:
#         [float]: value at risk
#     """    

        
#     if method=='min':
#         r = -data.loc[:,target].clip(upper=0).copy()
#     elif method=='max':
#         r = data.loc[:,target].clip(lower=0).copy()
#     elif method=='abs':
#         r = data.loc[:,target].abs().copy()
#     else:
#         raise('Method {} not implemented'.format(method))
        
#     group_keys = [data.time_stamp, data.number_of_split]
#     r = r.groupby(by=group_keys).max().values
    
#     n_r = r.shape[0]
#     n_missing = n_total - n_r

#     if n_missing > 0:
#         r = np.append(r, np.zeros(n_missing))
    
#     # Use interpolation other than 'linear', as that would cause problems with np.inf in the data
#     return np.quantile(r,q, interpolation='higher')


# def calc_rocof(load_imbalance, inertia_proxy):
#     return 50*load_imbalance /  (inertia_proxy*2*6)






