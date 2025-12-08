import pypsa
import numpy as np
import pandas as pd
import logging
from importlib.metadata import version
from pypsa import Network
from collections.abc import Sequence



#############
from Johannes.utils import data_handling
from Johannes.check_n1_stab import check_n1_stab
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


def optimize_security_constrained(
    n: Network,
    snapshots: Sequence | None = None,
    branch_outages: Sequence | pd.Index | pd.MultiIndex | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict = {},
    constrain_soc: bool = False,
    **kwargs,
) -> tuple[str, str]:
    """
    Computes Security-Constrained Linear Optimal Power Flow (SCLOPF).

    This ensures that no branch is overloaded even given the branch outages.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like, optional
        Set of snapshots to consider in the optimization. The default is None.
    branch_outages : list-like/pandas.Index/pandas.MultiIndex, optional
        Subset of passive branches to consider as possible outages. If a list
        or a pandas.Index is passed, it is assumed to identify lines. If a
        multiindex is passed, its first level has to contain the component names,
        the second the assets. The default None results in all passive branches
        to be considered.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

    Returns
    -------
    None
    """
    all_passive_branches = n.passive_branches().index

    if branch_outages is None:
        branch_outages = all_passive_branches
    elif isinstance(branch_outages, (list, pd.Index)):
        branch_outages = pd.MultiIndex.from_product([("Line",), branch_outages])

        if diff := set(branch_outages) - set(all_passive_branches):
            raise ValueError(
                f"The following passive branches are not in the network: {diff}"
            )

    if not len(all_passive_branches):
        return n.optimize(
            snapshots,  # type: ignore
            multi_investment_periods=multi_investment_periods,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )
    
    if constrain_soc == True:
        print('Writing soc constraints for Sclopf')
        
        from pyomo.environ import Constraint


        # Loop über Speicher und Snapshots
        for i, storage in n.storage_units.iterrows():
            for t in snapshots:
                # Constraint: Neuer SoC ≥ Alter SoC
                soc_var = n.storage_units_t.state_of_charge[i, t]
                old_soc_value = n.storage_units_t.state_of_charge.at[t, i]
                m.add_component(f"enforce_soc_{i}_{t}", Constraint(expr=soc_var >= old_soc_value))


    for sub_network in n.sub_networks.obj:
        branches_i = sub_network.branches_i()
        outages = branches_i.intersection(branch_outages)

        if outages.empty:
            continue

        sub_network.calculate_BODF()
        BODF = pd.DataFrame(sub_network.BODF, index=branches_i, columns=branches_i)[
            outages
        ]

        for c_outage, c_affected in product(outages.unique(0), branches_i.unique(0)):
            c_outage_ = c_outage + "-outage"
            c_outages = outages.get_loc_level(c_outage)[1]
            flow_outage = m.variables[c_outage + "-s"].loc[:, c_outages]
            flow_outage = flow_outage.rename({c_outage: c_outage_})

            bodf = BODF.loc[c_affected, c_outage]
            bodf = xr.DataArray(bodf, dims=[c_affected, c_outage_])
            additional_flow = flow_outage * bodf
            for bound, kind in product(("lower", "upper"), ("fix", "ext")):
                coord = c_affected + "-" + kind
                constraint = coord + "-s-" + bound
                if constraint not in m.constraints:
                    continue
                rename = {c_affected: coord}
                added_flow = additional_flow.rename(rename)
                con = m.constraints[constraint]  # use this as a template
                # idx now contains fixed/extendable for the sub-network
                idx = con.lhs.indexes[coord].intersection(added_flow.indexes[coord])
                sel = {coord: idx}
                lhs = con.lhs.sel(sel) + added_flow.sel(sel)
                name = constraint + f"-security-for-{c_outage_}-in-{sub_network}"
                m.add_constraints(lhs, con.sign.sel(sel), con.rhs.sel(sel), name=name)
        # output = 
                n.optimize.solve_model(**kwargs)
        
    return n


if __name__ == "__main__":

    i = int(snakemake.wildcards.i)

    solver = snakemake.config["solving"]
    group_size = snakemake.config["groupsize"]
    load_shedding = snakemake.config["load_shedding"]
    network_sclopf = snakemake.config["network_sclopf"]
    
    ####
    co2_relaxation = snakemake.config["co2_relaxation"]
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


    #### moved to preparation for loadshedding quantification ####
    # if load_shedding:
    #     n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
    #     buses_i = n.buses.index
    #     n.madd(
    #         "Generator",
    #         buses_i,
    #         " load",
    #         bus=buses_i,
    #         carrier="load",
    #         sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
    #         marginal_cost=1e2,  # Eur/kWh
    #         p_nom=1e9,  # kW
    #     )

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
        constant=co2_relaxation*emissions_lopf_i,
    )
    logger.info(
        "Added a CO2 overdimension of "
        f"{co2_relaxation} times."
    )

    if network_sclopf:
        pypsa_version = version('pypsa')
        assert pypsa_version <= '0.28.0', "network_sclopf is only supported for pypsa v0.28.0 or earlier."
        logger.info("Using network_sclopf")
        from pypsa.contingency import network_sclopf
        network_sclopf(n, snapshots=snapshots, **kwargs)
    else:
        status, condition = optimize_security_constrained(
            n,
            snapshots,
            branch_outages = pd.Index(branch_outages),
            solver_name = solver["solver"]["name"],
            solver_options = solver["solver_options"],
            constrain_soc=True
        )
        logger.info(f"SCLOPF status: {status} with condition {condition}.")
        if status != "ok":
            n.model.print_infeasibilities()

    to_remove = [k for k in n.lines_t.keys() if "mu_contingency" in k]
    for k in to_remove:
        n.lines_t.pop(k)

    # get co2 emissions after sclopf
    emissions_sclopf_i = get_emissions(n, snapshots)
    perc = 100 * emissions_sclopf_i / (emissions_lopf_i)
    # if (
    #     np.isclose(emissions_sclopf_i, 0, atol=10) and
    #     np.isclose(emissions_lopf_i, 0, atol=10)
    # ): 
    #     perc = 100
    rtol = 0.005
    if perc <= 100 * co2_relaxation * (1+rtol): #np.isclose(perc, 100*co2_relaxation, rtol=0.05):
        logger.info(
            f"Co2-Test successful. Emissions are {perc}% of LOPF window."
        )
    else:
        raise AssertionError(
            f"Co2-Test not succesful. Emissions are {perc}% of LOPF window."
        )

    # export network before contingency test because n.copy faces recursion error ??
    n.export_to_netcdf(snakemake.output[0])

    # test contingency
    # branch_outages = [("Line", i) for i in branch_outages]
    # maxloading = test_contingency(n, branch_outages)
    # if np.isclose(maxloading, 1, rtol=0.05):
    #     logger.info(
    #         "Contingency test succesful. Network is N-1 secure with a "
    #         f"maximal line loading of {100*maxloading}%."
    #     )
    # else:
    #     raise AssertionError(
    #         "Contingency test not succesful,  maximal line loading is "
    #         f" {100*maxloading}%. Network might not be N-1 secure."
    #     )
    #############
    check_n1_stab(path_to_pypsa_network = snakemake.output[0], use_sclopf=True)


    #############
