"""
Evaluation of inertia and power imbalance in subgraphs of PyPSA network 
"""
import numpy as np
import pandas as pd

def get_load_imbalance_subgraph(subgraph,
                                generators,
                                current_generation,
                                storages,
                                current_storage,
                                loads,
                                current_load,
                                HVDC_transport):
    """Calculate load imbalance due to subgraph"""
    gens   = generators[generators["bus"].isin(list(subgraph.nodes()))]
    loads  = loads[loads["bus"].isin(list(subgraph.nodes()))]
    stores = storages[storages["bus"].isin(list(subgraph.nodes()))]


    load_imbalance = current_generation.loc[list(gens.index)].sum() + current_storage.loc[list(stores.index)].sum()-\
                        current_load.loc[list(loads.index)].sum()
    load_imbalance -= HVDC_transport[HVDC_transport.index.isin(list(subgraph.nodes()))].sum()
    return load_imbalance

def get_load_subgraph(subgraph,
                    generators,
                    current_generation,
                    storages,
                    current_storage,
                    loads,
                    current_load,
                    HVDC_transport):
    """Calculate load in subgraph (ingnoring HVDC and storage consumption)"""
    loads  = loads[loads["bus"].isin(list(subgraph.nodes()))]
    #stores = storages[storages["bus"].isin(list(subgraph.nodes()))]
    #HVDC =...
    
    subgraph_load = current_load.loc[list(loads.index)].sum()
    
    return subgraph_load


def check_load_criterion(subgraphs,
                         generators,
                         current_generation,
                         storages,
                         current_storage,
                         loads,
                         current_load,
                         HVDC_transport):
    """ Check load/generation criterion which means
    that none of the subgraph accounts for 90 % of the load
    or generation at the current timestamp"""

    load_criterion = True

    threshold = 0.9

    overall_generation = current_generation.sum() + current_storage.sum()
    overall_load       = current_load.sum()

    subgraph_contributions = np.zeros((len(subgraphs),2))

    for count,subgraph in enumerate(subgraphs):
        gens                = generators[generators["bus"].isin(list(subgraph.nodes()))]
        loads               = loads[loads["bus"].isin(list(subgraph.nodes()))]
        stores              = storages[storages["bus"].isin(list(subgraph.nodes()))]
        subgraph_generation = current_generation.loc[list(gens.index)].sum() + current_storage.loc[list(stores.index)].sum()
        subgraph_generation -= HVDC_transport[HVDC_transport.index.isin(list(subgraph.nodes()))].sum()
        subgraph_load       = current_load.loc[list(loads.index)].sum()

        subgraph_contributions[count] = np.array([subgraph_generation/overall_generation,subgraph_load/overall_load])

    if np.any(subgraph_contributions>threshold):
        load_criterion = False
    return load_criterion




def get_inertia_gen_subgraph(subgraph,generators,current_generation,storages,current_storage,use_pnom,inertiaplants = None, inertia_storages = None):
    """get inertia generation for a subgraph"""
    if not inertiaplants:
        inertiaplants = ['CCGT','OCGT','coal','nuclear','oil','ror','lignite','biomass']
    if not inertia_storages:
        inertia_storages = ['PHS']

    # threshold below which a generator or storage is not counted as being participating
    participation_threshold = 0.05

    gens                     = generators[generators["bus"].isin(list(subgraph.nodes()))]
    existing_inertia_sources = list(set(inertiaplants)-(set(inertiaplants)-set(list(gens.carrier))))
    current_generators       = current_generation.loc[list(gens.index)]

    stores                    = storages[storages["bus"].isin(list(subgraph.nodes()))]
    existing_inertia_storages = list(set(inertia_storages)-(set(inertia_storages)-set(list(stores.carrier))))
    current_storages          = current_storage.loc[list(stores.index)]

    if use_pnom:
        participating_gens = current_generation.loc[list(gens.index)]>participation_threshold*gens["p_nom"]
        reduced_gens       = gens[participating_gens]
        existing_inertia_sources = list(set(existing_inertia_sources)-(set(existing_inertia_sources)-set(reduced_gens.carrier)))
        nominal_power            = reduced_gens["p_nom"]*reduced_gens["p_max_pu"]
        inertia_generation       = (nominal_power).groupby(reduced_gens.carrier).sum().loc[existing_inertia_sources].sum()

        participating_storages    = current_storages.loc[list(stores.index)]>participation_threshold*stores["p_nom"]
        reduced_stores            = stores[participating_storages]
        existing_inertia_storages = list(set(existing_inertia_storages)-(set(existing_inertia_storages)-set(reduced_stores.carrier)))
        inertia_generation += (reduced_stores["p_nom"]).groupby(reduced_stores.carrier).sum().loc[existing_inertia_storages].sum()

    else:
        inertia_generation = current_generators.groupby(gens.carrier).sum().loc[existing_inertia_sources].sum()
        inertia_generation += current_storages.groupby(stores.carrier).sum().loc[existing_inertia_storages].sum()
    return inertia_generation

#TODO: rename this function to better fit "network/subgraph evaluation"
def evaluate_split_observables(subgraphs,
                               pypsa_network,
                               timestamp,
                               use_pnom = True,
                               inertiaplants = None,
                               inertia_storages = None,
                               snet_index = None,
                               criterion = 'nodes'):
    """
    ARGUMENTS:

    split: list of edges split to be used with networkx graph created from pypsa network

    pypsa_network: pypsa network object containing solution for timestamps

    timestamp: a pandas timestamp that is contained in the snapshots of the
               pypsa network

    use_pnom (optional): boolean, whether or not to use the nominal power of a generator to
              to estimate its inertia
              defaults to True

    inertiaplants (optional): list of strings that describes the carriers that are assumed
                   to contribute to the system inertia

                   defaults to None which uses the list of inertia plants provided
                   in the function "get_inertia_gen_subgraph"

    inertia_storages (optional): list of strings that describes the carriers that are assumed
                   to contribute to the system inertia

                   defaults to None which uses the list of inertia storages provided
                   in the function "get_inertia_gen_subgraph"

    snet_index (optional): integer, gives the index of the subnet of the pypsa network to use
                 defaults to None

    criterion (optional): string, that describes which criterion to use to determine which
               subgraphs are evaluated.
               defaults to "nodes"

               If "nodes" is chosen, only subgraphs with
               at least 10 nodes are evaluated.

               If "load" is chosen, the split is only evaluated if none of the split
               components contains 90 % of the total load or generation. In this case,
               every split component is evaluated independent of the component size
               
               If "all" is chosen, all subgraphs are evaluated.

    RETURNS:

    results_dict: dictionary of results with keywords 'inertia_proxy' and
                  'load_imbalance'.
                  NOTE: TO GET THE ROCOF FROM LOAD IMBALANCE AND INERTIA PROXY
                  YOU NEED TO MULTIPLY INERTIA PROXY BY THE INERTIA CONSTANT
                  (SEE BELOW) AND TAKE INTO ACCOUNT THE SYSTEM FREQUENCY

                  Each keyword contains a list with length corresponding to the number
                  of split components as evaluated based on the criterion given.
                  To reproduce these split components call
                  "edge_list_to_subgraph(split,Graph, criterion = criterion)" with
                  the split and the networkx Graph representing the subnetwork under
                  consideration.

                  'inertia_proxy' is the inertia estimate based on the sum of nominal powers
                  of all plants and storages that contribute with at least 5 % of their
                  nominal power at the given point in time and are listed in the inertia
                  plants list.
                  NOTE: THIS ESTIMATE DOES NOT INCORPORATE THE INERTIA CONSTANT.
                  TO GET THE ACTUAL INERTIA, MULTIPLY EITHER BY A GLOBAL CONSTANT
                  (e.g. H = 6s^{-1}) OR DEFINE INDDIVIDUAL INERTIA CONSTANTS IN
                  THE FUNCTION "get_inertia_gen_subgraph"

                  'load_imbalance' is the load imbalance for each subgraph in units
                  of MW, i.e. its generation surplus or the missing generation.
                  It assumes that HVDC transport stays the same as before the split,
                  since the split is assumed to occur instantenously and thus leaves
                  HVDC transport unchanged.

    """

    assert isinstance(timestamp,pd.Timestamp)

    # Rescale load shedding since units for load shedding are different
    # than for generation, storage and load
    # (Note: In this project load shedding is not implement)
    load_shedding_indices = pypsa_network.generators[pypsa_network.generators.carrier.isin(['load'])].index

    current_generation = pypsa_network.generators_t.p.loc[timestamp].copy()
    current_generation[load_shedding_indices] /= 1e3
    current_storage    = pypsa_network.storage_units_t.p.loc[timestamp]
    current_load       = pypsa_network.loads_t.p.loc[timestamp]
    HVDC_transport     = pypsa_network.links_t.p0.loc[timestamp].groupby(pypsa_network.links["bus0"]).sum()
    HVDC_transport     = HVDC_transport.add(pypsa_network.links_t.p1.loc[timestamp].groupby(pypsa_network.links["bus1"]).sum(),
                                            fill_value = 0)

    assert np.abs(current_generation.sum()+current_storage.sum()-current_load.sum())<5e-2

    results_dict = {'inertia_proxy': [],'load_imbalance': [], 'load':[]} #, 'available_flexible_generation': [] }

    evaluate_results = False

    #TODO: separate edge_list_to_subgraph and checking the importance of a component!

    if (criterion == 'nodes') or (criterion =='all'):
        if len(subgraphs) >= 2:
            evaluate_results = True
    elif criterion == 'load':
        evaluate_results = check_load_criterion(subgraphs,
                                                pypsa_network.generators,
                                                current_generation,
                                                pypsa_network.storage_units,
                                                current_storage,
                                                pypsa_network.loads,
                                                current_load,
                                                HVDC_transport)

    if evaluate_results:
        for subgraph in subgraphs:
            inertia_generation = get_inertia_gen_subgraph(subgraph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                          use_pnom,
                                                          inertiaplants = inertiaplants,
                                                          inertia_storages = inertia_storages)

            load_imbalance = get_load_imbalance_subgraph(subgraph,
                                                          pypsa_network.generators,
                                                          current_generation,
                                                          pypsa_network.storage_units,
                                                          current_storage,
                                                          pypsa_network.loads,
                                                          current_load,
                                                          HVDC_transport)

            subgraph_load = get_load_subgraph(subgraph,
                                            pypsa_network.generators,
                                            current_generation,
                                            pypsa_network.storage_units,
                                            current_storage,
                                            pypsa_network.loads,
                                            current_load,
                                            HVDC_transport)

            results_dict['inertia_proxy'].append(inertia_generation)
            results_dict['load_imbalance'].append(load_imbalance)
            results_dict['load'].append(subgraph_load) 
            #results_dict['available_flexible_generation'].append(available_flexible_generation)

    return results_dict




