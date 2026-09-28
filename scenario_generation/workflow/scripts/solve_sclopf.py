import os
import pypsa
import numpy as np
import pandas as pd
import logging
from importlib.metadata import version

from pyomo.util.infeasible import log_infeasible_constraints 
from linopy.common import format_single_constraint

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
    if "snakemake" not in globals():
        # run this file directly (e.g. `python3 scripts/solve_sclopf.py`) to
        # iterate on solve/constraint logic for one window without going
        # through the full Snakemake DAG - only needs config.sclopf.yaml (this
        # rule doesn't read config.yaml), the wildcards below, and the
        # `prepared`/`previous` inputs to already exist on disk.
        import sys

        sys.path.insert(0, os.path.abspath("submodules/pypsa-eur"))
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sclopf_weather",
            configfiles=["configs/config.sclopf.yaml"],
            submodule_dir="submodules/pypsa-eur",  # root_dir here is already workflow/
            opts="Co2L0.6",
            weather_year="2012",
            i="0",
        )

    from scripts._helpers import configure_logging

    configure_logging(snakemake)

    i = int(snakemake.wildcards.i)

    solver = snakemake.config["solving"]
    group_size = snakemake.config["groupsize"]
    load_shedding = snakemake.config["load_shedding"]
    
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
    # replace the upper CO2 limit from LOPF by an equality constraint for the
    # rolling window.
    # pypsa-eur names this constraint "CO2Limit-<upper|lower>", not "CO2Limit" anymore
    co2_limit_constraints = n.global_constraints.index[
        n.global_constraints.index.str.startswith("CO2Limit")
    ]
    n.remove("GlobalConstraint", co2_limit_constraints)
    n.add(
        "GlobalConstraint",
        "CO2Limit_upper",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=emissions_lopf_i,
    )
    n.add(
        "GlobalConstraint",
        "CO2Limit_lower",
        carrier_attribute="co2_emissions",
        sense=">=",
        constant=emissions_lopf_i,
    )
        
    # else:
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
        res = [format_single_constraint(m, label) for label in labels]
        print("\n---------------------------------------- \nInfeasible Constraints:\n---------------------------------------- \n")
        logger.info("\n".join(res))
        raise RuntimeError(
            f"SCLOPF for window {i} did not solve to optimality (status={status}, "
            f"condition={condition}). See the infeasible constraints logged above."
        )


    to_remove = [k for k in n.lines_t.keys() if "mu_contingency" in k]
    for k in to_remove:
        n.lines_t.pop(k)

    print("\n---------------------------------------- \nCO2 Check:\n---------------------------------------- \n")
    # get co2 emissions after sclopf
    emissions_sclopf_i = get_emissions(n, snapshots)

    if emissions_lopf_i == 0:

        if emissions_sclopf_i != 0:
            raise AssertionError(
            f"Co2-Test not succesful. Emissions are {perc}% of LOPF window."
        )

        else: 
            logger.info(
                f"Co2-Test successful. Emissions are 0 as in LOPF window."
            )

    else: 

        perc = 100 * emissions_sclopf_i / (emissions_lopf_i)
        
        rtol = 0.005
        if perc <= 100 * (1+rtol): 
            logger.info(
                f"Co2-Test successful. Emissions are {perc}% of LOPF window."
            )
        else:
            raise AssertionError(
                f"Co2-Test not succesful. Emissions are {perc}% of LOPF window."
            )

    # export network
    n.export_to_netcdf(snakemake.output[0])



