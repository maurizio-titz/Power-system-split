"""
Operation-only LOPF for the weather-year sensitivity pipeline.

Takes the network produced by swap_weather_year.py (2013-investment-optimized
capacities, but wind/solar/hydro time series swapped to a different weather
year) and re-solves dispatch under the new weather year with capacities
frozen at their 2013 LOPF values - no further capacity expansion. This gives
a self-consistent generators_t.p/storage_units_t.p/state_of_charge trajectory
for the new weather year, which prepare_sclopf_weather.py then feeds into the
SC-LOPF redispatch stage (its force_storage='all' pins state_of_charge from
whatever network it's given - it must be one actually solved under the same
weather year's inflow, not the stale 2013 trajectory carried over from
solved_{HORIZON}.nc).
"""

import pypsa

from prepare_sclopf import fix_capacities

if __name__ == "__main__":
    if "snakemake" not in globals():
        # run standalone (e.g. `python3 scripts/solve_operation_weather.py`) to
        # iterate without going through the full Snakemake DAG
        import os
        import sys

        sys.path.insert(0, os.path.abspath("submodules/pypsa-eur"))
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operation_weather",
            configfiles=["configs/config.sclopf.yaml", "configs/config.yaml"],
            submodule_dir="submodules/pypsa-eur",
            opts="Co2L0.6",
            weather_year="2012",
        )

    from scripts._helpers import configure_logging

    configure_logging(snakemake)

    solver = snakemake.config["solving"]

    n = pypsa.Network(snakemake.input.network)

    # Freeze capacities at exactly their 2013 investment-LOPF optimum (no
    # overdimensioning, no further extendability) - only operations respond
    # to the new weather year here.
    fix_capacities(n, overdim_extendables=1.0, freeze_extendable=True)

    status, condition = n.optimize(
        solver_name=solver["solver"]["name"],
        solver_options=solver["solver_options"],
    )

    if status != "ok":
        raise RuntimeError(
            f"Operation LOPF under the swapped weather year did not solve to "
            f"optimality (status={status}, condition={condition})."
        )

    # fix_capacities() (via its outage-line-splitting bookkeeping, unused
    # here but harmless to compute) leaves a 'partition' column on n.lines
    # holding a per-line list, which netCDF/xarray cannot serialise.
    n.lines = n.lines.drop(columns=["partition"])
    n.export_to_netcdf(snakemake.output[0])
