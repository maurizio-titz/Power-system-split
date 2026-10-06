"""
Swap the weather-dependent time series (wind/solar p_max_pu, hydro/ror
inflow) of an already-solved LOPF network for a different weather year,
while leaving capacities, topology, loads, CO2 constraints, and everything
else untouched.

Wind/solar are point-sampled at each generator's bus (x, y) coordinate from
the new cutout, rather than area-weighted over region polygons, because the
old base network's original clustering regions no longer exist (see
/home/jlange/.claude/plans/clever-imagining-walrus.md). Hydro reuses
pypsa-eur's own build_hydro_profile.py output (per-country) redistributed
with the same per-plant dist_key formula as add_electricity.py's
attach_hydro().
"""

import atlite
import pandas as pd
import pypsa
import xarray as xr

WIND_CARRIERS = ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
SOLAR_CARRIERS = ["solar", "solar-hsat"]


def normed(s: pd.Series) -> pd.Series:
    return s / s.sum()


def point_sample(da: xr.DataArray, buses: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-cell time series at each bus's (x, y), one column per bus."""
    # buses.index carries pypsa's own index name (e.g. "name"), and passing a
    # *named* pd.Index straight into coords={} makes xarray bind the coordinate
    # to a dimension derived from that name instead of the "bus" key given here
    # -> CoordinateValidationError. Use .values (plain ndarray) to strip it.
    bus_names = buses.index.values
    sampled = da.sel(
        x=xr.DataArray(buses["x"].values, dims="bus", coords={"bus": bus_names}),
        y=xr.DataArray(buses["y"].values, dims="bus", coords={"bus": bus_names}),
        method="nearest",
    )
    return sampled.transpose("time", "bus").to_pandas()


def remap_to_base_year(df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    Relabel a time series (in the new weather year's calendar) onto the base
    network's snapshot calendar (e.g. new weather year 2012 -> base year
    2013), so it aligns with n.snapshots for reindexing. Drops Feb 29 first
    so a leap-year source never collides with a non-leap-year base (or
    vice versa).
    """
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df.set_axis(
        pd.DatetimeIndex([t.replace(year=base_year) for t in df.index]), axis=0
    )


if __name__ == "__main__":
    n = pypsa.Network(snakemake.input.network)
    cutout = atlite.Cutout(snakemake.input.cutout)
    renewable_cfg = snakemake.params.renewable
    base_year = n.snapshots[0].year

    # --- wind: point-sample the new cutout at each generator's bus ---
    for carrier in WIND_CARRIERS:
        gens = n.generators[n.generators.carrier == carrier]
        if gens.empty:
            continue
        cfg = renewable_cfg[carrier]
        resource = cfg["resource"]
        cf = cutout.wind(
            turbine=resource["turbine"],
            smooth=resource.get("smooth", False),
            add_cutout_windspeed=resource.get("add_cutout_windspeed", True),
            capacity_factor_timeseries=True,
        )
        series = point_sample(cf, n.buses.loc[gens["bus"]])
        series.columns = gens.index
        series = remap_to_base_year(series, base_year)
        series *= cfg.get("correction_factor", 1.0)
        clip = cfg.get("clip_p_max_pu")
        if clip:
            series = series.where(series >= clip, 0)
        n.generators_t.p_max_pu[gens.index] = series.reindex(n.snapshots)

    # --- solar: same, via cutout.pv() ---
    for carrier in SOLAR_CARRIERS:
        gens = n.generators[n.generators.carrier == carrier]
        if gens.empty:
            continue
        cfg = renewable_cfg[carrier]
        resource = cfg["resource"]
        cf = cutout.pv(
            panel=resource["panel"],
            orientation=resource["orientation"],
            tracking=resource.get("tracking"),
            capacity_factor_timeseries=True,
        )
        series = point_sample(cf, n.buses.loc[gens["bus"]])
        series.columns = gens.index
        series = remap_to_base_year(series, base_year)
        series *= cfg.get("correction_factor", 1.0)
        clip = cfg.get("clip_p_max_pu")
        if clip:
            series = series.where(series >= clip, 0)
        n.generators_t.p_max_pu[gens.index] = series.reindex(n.snapshots)

    # --- hydro & ror: redistribute the new per-country inflow profile,
    # mirroring add_electricity.py's attach_hydro() dist_key formula ---
    ror = n.generators[n.generators.carrier == "ror"]
    hydro = n.storage_units[n.storage_units.carrier == "hydro"]
    inflow_idx = ror.index.union(hydro.index)

    if not inflow_idx.empty:
        p_nom = pd.concat([ror["p_nom"], hydro["p_nom"]]).loc[inflow_idx]
        bus = pd.concat([ror["bus"], hydro["bus"]]).loc[inflow_idx]
        country = bus.map(n.buses["country"])
        dist_key = p_nom.groupby(country).transform(normed)

        with xr.open_dataarray(snakemake.input.hydro_profile) as inflow:
            inflow_countries = pd.Index(country[inflow_idx])
            missing_c = inflow_countries.unique().difference(inflow.indexes["countries"])
            assert missing_c.empty, (
                f"'{snakemake.input.hydro_profile}' is missing inflow time-series "
                f"for at least one country: {', '.join(missing_c)}"
            )
            inflow_t = (
                inflow.sel(countries=inflow_countries)
                .rename({"countries": "name"})
                .assign_coords(name=inflow_idx)
                .transpose("time", "name")
                .to_pandas()
                .multiply(dist_key, axis=1)
            )
        inflow_t = remap_to_base_year(inflow_t, base_year).reindex(n.snapshots)

        if not ror.empty:
            n.generators_t.p_max_pu[ror.index] = (
                inflow_t[ror.index].divide(ror["p_nom"], axis=1).clip(upper=1.0)
            )
        if not hydro.empty:
            n.storage_units_t.inflow[hydro.index] = inflow_t[hydro.index]

    n.export_to_netcdf(snakemake.output[0])
