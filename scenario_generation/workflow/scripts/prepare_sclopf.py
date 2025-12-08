import pypsa
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def split_outage_lines(n, lookup):
    """
    Separate one parallel line in each corridor for which we consider the outage.
    Special handling for 220kV lines lifted to 380kV.
    """

    def create_nonoutage_lines(n, lookup):
        lookup.num_parallel_before = lookup.num_parallel_before.apply(lambda b: round(b,10))

        lines_nonout = n.lines.copy()
        no_split = lookup.query("num_parallel_after == 0.").num_parallel_before.values
        lines_nonout.num_parallel = lines_nonout.num_parallel.apply(lambda b: round(b, 10))
        lines_nonout = lines_nonout.query("num_parallel not in @no_split")
        lines_nonout.loc[:, "num_parallel"] = lines_nonout.num_parallel.apply(
            lambda np: (lookup.query("num_parallel_before == @np").num_parallel_after.values[0]
                        if np in lookup.num_parallel_before.values else np-1)
        )
        s_nom_vals = (
            ((lines_nonout.s_nom * lines_nonout.num_parallel)
             / n.lines.loc[lines_nonout.index].num_parallel)
        )
        lines_nonout.loc[s_nom_vals.index, "s_nom"] = s_nom_vals
        return lines_nonout

    lines_nonout = create_nonoutage_lines(n, lookup)

    n.lines.num_parallel = n.lines.num_parallel.apply(lambda b: round(b,10))
    n.lines.num_parallel = n.lines.num_parallel.apply(
        lambda np: (lookup.query("num_parallel_before == @np").num_parallel_failing.values[0]
                    if np in lookup.num_parallel_before.values else 1.)
    )
    n.lines.loc[lines_nonout.index, "s_nom"] = (
        n.lines.loc[lines_nonout.index, "s_nom"] * n.lines.loc[lines_nonout.index, "num_parallel"] /
        (lines_nonout.num_parallel+n.lines.loc[lines_nonout.index, "num_parallel"])
    )
    n.lines.index = [f"{i}_outage" for i in n.lines.index]
    n.lines = pd.concat([n.lines, lines_nonout])

    # prior results of pfs are now worthless.
    for key in n.lines_t.keys():
        n.lines_t[key] = pd.DataFrame(index=n.snapshots)

    n.calculate_dependent_values()

    return n

def fix_capacities(n, overdim_extendables):

    goi = n.generators.query("p_nom_extendable == True").index
    n.generators.loc[goi, "p_nom"] = overdim_extendables*n.generators.loc[goi, "p_nom_opt"]
    n.generators.loc[goi, "p_nom_extendable"] = False

    suoi = n.storage_units.query("p_nom_extendable == True").index
    n.storage_units.loc[suoi, "p_nom"] = overdim_extendables*n.storage_units.loc[suoi, "p_nom_opt"]
    n.storage_units.loc[suoi, "p_nom_extendable"] = False
    n.storage_units.marginal_cost *= -1

    soi = n.stores.query("e_nom_extendable == True").index
    n.stores.loc[soi, "e_nom"] = overdim_extendables*n.stores.loc[soi, "e_nom_opt"]
    n.stores.loc[soi, "e_nom_extendable"] = False

    lkoi = n.links.query("p_nom_extendable == True").index
    n.links.loc[lkoi, "p_nom"] = n.links.loc[lkoi, "p_nom_opt"]
    n.links.loc[lkoi, "p_nom_extendable"] = False

    loi = n.lines.query("s_nom_extendable == True").index
    n.lines.loc[loi, "s_nom"] = n.lines.loc[loi, "s_nom_opt"]
    n.lines.loc[loi, "s_nom_extendable"] = False

    n.lines = n.lines.loc[n.lines.num_parallel != 0.0]

if __name__ == "__main__":

    # config = snakemake.config
    config = snakemake.config
    group_size = int(config["groupsize"])
    temp_resolution = int(config["clustering"]["temporal"]["resolution_elec"].split("H")[0])
    load_shedding = snakemake.config["load_shedding"]
    artificial_load = snakemake.config["artificial_load"]

    n = pypsa.Network(snakemake.input.network)


    ### if loadshedding is activated, add loadshedding possibility to each bus ###
    if load_shedding:
        print("Loadshedding included")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            p_nom=1e9,  # kW
        )
    
    
    ### if artificial load is activated, add art_load possibility to each bus ###
    if artificial_load:
        print("Artificial Load included")
        n.add("Carrier", "art_load", color="#38761d", nice_name="artificial load")
        buses_i = n.buses.index
        n.madd(
            "Generator",
            buses_i,
            " art_load",
            bus=buses_i,
            carrier="art_load",
            sign=-1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            p_nom=1e9,  # kW
        )


    fix_capacities(n, config["overdim_extendables"])

    lookup = pd.read_csv(snakemake.input.lookup, header=[1]).iloc[:-1]
    lookup.num_parallel_before = lookup.num_parallel_before.astype(float)
    lookup.num_parallel_failing = lookup.num_parallel_failing.astype(float)
    lookup.num_parallel_after = lookup.num_parallel_after.astype(float)
    lookup = lookup[["num_parallel_before", "num_parallel_failing", "num_parallel_after"]]
    n = split_outage_lines(n, lookup)
    n.determine_network_topology()

    #otherwise, this will throw an error in network_lopf with extra functionality for sclopf
    to_delete = []
    for i, sub in n.sub_networks.iterrows():
        buses = sub.obj.buses()
        if len(buses) == 1:
            to_delete += list(buses.index)
    logger.warning(
        f"Currently, islands cannot be treated in the sc-lopf. Thus removing {to_delete}")

    n.mremove("Bus", to_delete)
    n.mremove("Load", n.loads.query("bus in @to_delete").index)
    n.mremove("Generator", n.generators.query("bus in @to_delete").index)
    n.mremove("Link", n.links.query("bus0 in @to_delete or bus1 in @to_delete").index)
    n.mremove("StorageUnit", n.storage_units.query("bus in @to_delete").index)
    n.mremove("Store", n.stores.query("bus in @to_delete").index)

    n.determine_network_topology()
    n.calculate_dependent_values()
 ### if loadshedding is activated, add loadshedding possibility to each bus ###
    if load_shedding:
        print("Loadshedding included")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            p_nom=1e9,  # kW
        )
    
    
    ### if artificial load is activated, add art_load possibility to each bus ###
    if artificial_load:
        print("Artificial Load included")
        n.add("Carrier", "art_load", color="#38761d", nice_name="artificial load")
        buses_i = n.buses.index
        n.madd(
            "Generator",
            buses_i,
            " art_load",
            bus=buses_i,
            carrier="art_load",
            sign=-1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            p_nom=1e9,  # kW
        )
        
        
    if config["force_storage"] == 'all':
        logger.info("Forcing state of charge for storage units according to lopf result.")
        n.storage_units.cyclic_state_of_charge = False
        
        n.storage_units_t.state_of_charge_set = n.storage_units_t.state_of_charge
            
    elif config["force_storage"] == 'boundaries':
        #### fixate start and endpoints ####
        # docs in  https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html:
        # If in the time series n.storage_units_t.state_of_charge_set there are values which are not NaNs, 
        # then it will be assumed that these are fixed state of charges desired for that time
        # and these will be added as extra constraints. 
        logger.info("Fixing the boundaries of each window to the storage levels of the lopf result")

        n.storage_units_t.state_of_charge_set = n.storage_units_t.state_of_charge

        i_values = [n for n in range(int(np.ceil(8760 / temp_resolution / group_size)))] # ToDo: change
        for i in i_values:
            n.storage_units_t.set = n.storage_units_t.state_of_charge
            
            snapshots = n.snapshots[group_size * i : group_size * i + group_size]
            last = snapshots[-1-1]
            first = snapshots[0+ 1]
            n.storage_units_t.state_of_charge_set.loc[first :last ] = np.NaN
            


    elif config["force_storage"] == 'upper_limit':
        logger.info("Setting upper bound for storage behavior according to lopf result.")
        n.storage_units.cyclic_state_of_charge = False
        
        discharge = n.storage_units_t.p.divide(n.storage_units.p_nom).clip(lower=0.)
        charge = n.storage_units_t.p.divide(n.storage_units.p_nom).clip(upper=0.)

        # set ratio of power that the storages have to suffice 
        p_charge = 1 
        p_discharge = 1
        
 
        n.storage_units_t.p_max_pu = (p_charge * charge + p_discharge * discharge)
        
   
        # add up shadow prices for improved merit order
        avg_cost = (
            n.storage_units_t.mu_lower.mean() +
            n.storage_units_t.mu_upper.abs().mean()
        ) / 2
        n.storage_units.marginal_cost = avg_cost
        
        
    elif config["force_storage"] == 'Martha':
        logger.info("Setting upper and lower bound for storage behavior according to lopf result.")
        n.storage_units.cyclic_state_of_charge = False
        
        discharge = n.storage_units_t.p.divide(n.storage_units.p_nom).clip(lower=0.)
        charge = n.storage_units_t.p.divide(n.storage_units.p_nom).clip(upper=0.)

        n.storage_units_t.p_max_pu = discharge
        n.storage_units_t.p_min_pu = charge

        # add up shadow prices for improved merit order
        avg_cost = (
            n.storage_units_t.mu_lower.mean() +
            n.storage_units_t.mu_upper.abs().mean()
        ) / 2
        n.storage_units.marginal_cost = avg_cost

    n.export_to_netcdf(snakemake.output[0])

