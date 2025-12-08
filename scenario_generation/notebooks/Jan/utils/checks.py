import numpy as np
import pypsa

def get_emissions(n):

    gen = (
        n.generators_t.p
        .multiply(n.snapshot_weightings.objective,axis=0) # multiply by no. of hours (weight)
        .divide(n.generators.efficiency,axis=1).fillna(0) # efficiency and generation might be different (e.g. in the case of CCGT)
        .multiply(n.generators.carrier.map(n.carriers.co2_emissions)) # multiply by specific emissions
        .fillna(0).sum().sum()
    )

    stog = (
        n.storage_units_t.p
        .multiply(n.snapshot_weightings.objective,axis=0)
        .divide(n.storage_units.efficiency_dispatch,axis=1).fillna(0)
        .multiply(n.storage_units.carrier.map(n.carriers.co2_emissions))
        .fillna(0).sum().sum()
    )

    return gen+stog