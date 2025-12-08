import pypsa
import numpy as np
import pandas as pd
import logging
from importlib.metadata import version

#############
from Johannes.utils import data_handling
from Johannes.check_n1_stab import check_n1_stab
from pyomo.util.infeasible import log_infeasible_constraints 
from linopy.common import print_single_constraint

#############

logger = logging.getLogger(__name__)

def get_branch_outages(n):
    """Creates list of considered outages"""
    branch_outages = []
    for l in n.lines.index:
        if len(l.split("_")) == 2:
            branch_outages = np.append(l, branch_outages)
    return branch_outages

def get_emissions(n, snapshots):

    gen = (
        n.generators_t.p.loc[snapshots]
        .multiply(n.snapshot_weightings.loc[snapshots].objective,axis=0)
        .divide(n.generators.efficiency,axis=1).fillna(0)
        .multiply(n.generators.carrier.map(n.carriers.co2_emissions))
        .fillna(0).sum().sum()
    )

    stog = (
        n.storage_units_t.p.loc[snapshots]
        .multiply(n.snapshot_weightings.loc[snapshots].objective,axis=0)
        .divide(n.storage_units.efficiency_dispatch,axis=1).fillna(0)
        .multiply(n.storage_units.carrier.map(n.carriers.co2_emissions))
        .fillna(0).sum().sum()
    )

    return gen+stog

def test_contingency(network, line_outages):
    now = network.snapshots[0]

    # set dispatch
    network.generators_t.p_set = network.generators_t.p_set.reindex(
        columns=network.generators.index
    )
    network.generators_t.p_set.loc[now] = network.generators_t.p.loc[now]

    network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(
        columns=network.storage_units.index
    )
    network.storage_units_t.p_set.loc[now] = network.storage_units_t.p.loc[now]

    network.links_t.p_set = network.links_t.p_set.reindex(
        columns=network.links.index
    )
    network.links_t.p_set.loc[now] = network.links_t.p0.loc[now]

    p0 = network.lpf_contingency(now, branch_outages=line_outages)

    return p0.loc["Line"].divide(network.lines.s_nom,axis=0).abs().max().max()


if __name__ == "__main__":

    i = int(snakemake.wildcards.i)

    solver = snakemake.config["solving"]
    group_size = snakemake.config["groupsize"]
    load_shedding = snakemake.config["load_shedding"]
    network_sclopf = snakemake.config["network_sclopf"]
    
    ####
    ####
    
    
    prep = pypsa.Network(snakemake.input.prepared)
    prep.lines.s_max_pu = 1. # overwrite default value of 0.7 to 1 for sclopf calculation

    snapshots = prep.snapshots[group_size * i : group_size * i + group_size]
    maxiter = len(prep.snapshots) / group_size
    if i > maxiter:
        raise ValueError("Can iterate only {maxiter} times. This would be iteration {i}.")
    logger.info(f"preparing sclopf for {len(snapshots)} snapshots...")

    # get co2 emissions after lopf
    emissions_lopf_i = get_emissions(prep, snapshots)

    # continue with sclopf for time window i
    n = prep.copy(snapshots=snapshots)

    branch_outages = get_branch_outages(n)

    kwargs = {
        "pyomo": False,
        "branch_outages": branch_outages,
        "solver_name": solver["solver"]["name"],
        "solver_options": solver["solver_options"],
        "formulation": "kirchhoff"
    }


    if i == 0:
        n.storage_units.state_of_charge_initial = (
            prep.storage_units_t.state_of_charge.loc[snapshots[0]]
        )
    if i > 0:
        prev = pypsa.Network(snakemake.input.previous)
        n.storage_units.state_of_charge_initial = (
            prev.storage_units_t.state_of_charge.iloc[-1]
        )
    del prep

    logger.info(
        "Removing global constraint for CO2 emissions and adding a local constraint "
        f"for the selected window of {group_size} snapshots."
    )
    # replace the upper CO2 limit from LOPF by equality
    # constraint for rolling window. But it doesn't work for some reason...
    n.remove("GlobalConstraint", "CO2Limit")
    n.add(
        "GlobalConstraint",
        "CO2Limit_upper",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=emissions_lopf_i,
    )

    if network_sclopf:
        pypsa_version = version('pypsa')
        assert pypsa_version <= '0.28.0', "network_sclopf is only supported for pypsa v0.28.0 or earlier."
        logger.info("Using network_sclopf")
        from pypsa.contingency import network_sclopf
        network_sclopf(n, snapshots=snapshots, **kwargs)
        
        
    else:
        status, condition = n.optimize.optimize_security_constrained(
            snapshots,
            branch_outages = pd.Index(branch_outages),
            solver_name = solver["solver"]["name"],
            solver_options = solver["solver_options"]
        )
        
        
        logger.info(f"SCLOPF status: {status} with condition {condition}.")
        
        if status != "ok":
            m = n.model

            # m.print_infeasibilities()
            print("")
            labels = m.compute_infeasibilities()
            res = [print_single_constraint(m, label) for label in labels]
            print("\n---------------------------------------- \nInfeasible Constraints:\n---------------------------------------- \n")
            logger.info("\n".join(res))
            
            
    to_remove = [k for k in n.lines_t.keys() if "mu_contingency" in k]
    for k in to_remove:
        n.lines_t.pop(k)

    print("\n---------------------------------------- \nCO2 Check:\n---------------------------------------- \n")
    # get co2 emissions after sclopf
    emissions_sclopf_i = get_emissions(n, snapshots)
    perc = 100 * emissions_sclopf_i / (emissions_lopf_i)
    # if (
    #     np.isclose(emissions_sclopf_i, 0, atol=10) and
    #     np.isclose(emissions_lopf_i, 0, atol=10)
    # ): 
    #     perc = 100
    rtol = 0.005
    if perc <= 100 * (1+rtol): 
        logger.info(
            f"Co2-Test successful. Emissions are {perc}% of LOPF window."
        )
    else:
        raise AssertionError(
            f"Co2-Test not succesful. Emissions are {perc}% of LOPF window."
        )

    # export network before contingency test because n.copy faces recursion error ??
    n.export_to_netcdf(snakemake.output[0])



