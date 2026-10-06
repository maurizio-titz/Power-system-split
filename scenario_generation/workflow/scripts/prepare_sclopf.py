import pypsa
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

# A circuit's contribution to num_parallel (relative to one 380kV 4-bundle circuit)
# is (its own voltage * its type's i_nom) / (380kV * the 380kV 4-bundle type's i_nom).
# This table covers both base_network choices:
#
# - electricity.base_network: entsoegridkit only ever assigns a type by *exact*
#   voltage match (base_network.py's _set_electrical_parameters_lines_eg), so only
#   the standard classes from electricity.voltages / lines.types occur: 220/300/
#   330/380/400/500/750kV (330, 400 and 500 reuse the 300kV/380kV types at a
#   different voltage, per config.default.yaml's lines.types).
# - electricity.base_network: osm has real-world, non-standard OSM-tagged voltages
#   (e.g. 225/236/254/275/420kV) that get *nearest-match* type assignment instead
#   (base_network.py's _get_linetype_by_voltage) - verified against this project's
#   actual data/entsoegridkit-independent OSM base network. Each is listed at its
#   real voltage with the type it gets nearest-matched to, so its weight comes out
#   different from the same type's "clean" voltage.
#
# One entry is not a voltage class at all: "HTLS 4-bundle 380.0" is a
# High-Temperature-Low-Sag reconductoring type used by some NEP/TYNDP transmission
# projects (data/transmission_projects/nep/new_lines.csv), independent of which
# base_network is chosen. Every such project line has a clean num_parallel=2.0 in
# the raw data, but simplify_network.py's simplify_network_to_380()
# (scripts/simplify_network.py:78-81) forces every line onto the standard
# "Al/St 240/40 4-bundle 380.0" type while *preserving s_nom*:
#   n.lines["type"] = linetype_380
#   n.lines["i_nom"] = n.line_types.i_nom[linetype_380]
#   n.lines["num_parallel"] = n.lines.eval("s_nom / (sqrt(3) * v_nom * i_nom)")
# HTLS carries much more current per circuit (i_nom~4.00kA) than the standard type
# (i_nom~2.58kA), so preserving capacity under the standard type's lower i_nom
# inflates num_parallel by that same ratio (~1.550387597 per original circuit) -
# a real, physical rescaling, not a data error, but no longer a clean circuit count
# under the standard-type basis. Listed here as (reference_voltage, actual_type) so
# its weight is computed the same way as a real voltage class.
_VOLTAGE_TYPE = [
    # standard classes (electricity.voltages / lines.types, both base_networks)
    (220.0, "Al/St 240/40 2-bundle 220.0"),
    (300.0, "Al/St 240/40 3-bundle 300.0"),
    (330.0, "Al/St 240/40 3-bundle 300.0"),
    (380.0, "Al/St 240/40 4-bundle 380.0"),
    (400.0, "Al/St 240/40 4-bundle 380.0"),
    (500.0, "Al/St 240/40 4-bundle 380.0"),
    (750.0, "Al/St 560/50 4-bundle 750.0"),
    # real, non-standard OSM voltages, nearest-matched to a standard type (osm only)
    (225.0, "Al/St 240/40 2-bundle 220.0"),
    (236.0, "Al/St 240/40 2-bundle 220.0"),
    (254.0, "Al/St 240/40 2-bundle 220.0"),
    (275.0, "Al/St 240/40 3-bundle 300.0"),
    (420.0, "Al/St 240/40 4-bundle 380.0"),
    # transmission-project conductor type, independent of base_network (see above)
    (380.0, "HTLS 4-bundle 380.0"),
]
_REFERENCE_VOLTAGE = 380.0
_REFERENCE_TYPE = "Al/St 240/40 4-bundle 380.0"


def _num_parallel_weights():
    """num_parallel contribution of one circuit of each (voltage, type) pair in
    _VOLTAGE_TYPE, relative to one 380kV 4-bundle circuit."""
    line_types = pypsa.Network().line_types
    ref_i_nom = line_types.loc[_REFERENCE_TYPE, "i_nom"]
    return {
        (v_nom, type_name): (v_nom * line_types.loc[type_name, "i_nom"]) / (_REFERENCE_VOLTAGE * ref_i_nom)
        for v_nom, type_name in _VOLTAGE_TYPE
    }


# ascending, so index 0 is the lowest voltage/weight and index -1 the highest
TYPE_WEIGHTS = sorted(set(_num_parallel_weights().values()))

_SCALE = 1_000_000
_TOL = 10  # +-10 units at 1e-6 scale. pypsa's line_types table itself stores i_nom
           # rounded to 2 decimals, and _build_reachability rounds each weight to
           # the nearest 1e-6 before summing - both errors accumulate with circuit
           # count (observed up to ~7e-6 for 4 HTLS circuits summed by
           # cluster_network). Wide enough to absorb that, while still rejecting
           # genuinely different values (which differ by orders of magnitude more,
           # e.g. ~6600e-6 for the closest non-match seen in practice).


def _build_reachability(weights, max_target):
    """Boolean array: reachable[s] is True iff s (in units of 1/_SCALE) can be written
    as a non-negative integer combination of `weights` (also scaled by _SCALE)."""
    w_int = [int(round(w * _SCALE)) for w in weights]
    size = int(round(max_target * _SCALE)) + _TOL + 1
    reachable = np.zeros(size, dtype=bool)
    reachable[0] = True
    for w in w_int:
        for s in range(w, size):
            if reachable[s - w]:
                reachable[s] = True
    return reachable, w_int


def get_partition(total_weight, reachable, w_int):
    """Decompose total_weight into non-negative integer counts of `w_int` (the
    per-voltage weights, scaled by _SCALE), searching within a small tolerance for
    floating-point noise. Generalises the old np_lookup.csv table (which only covered
    220/300/380kV circuits) to every voltage in _VOLTAGE_TYPE."""
    target = int(round(total_weight * _SCALE))
    for t in range(max(0, target - _TOL), min(len(reachable) - 1, target + _TOL) + 1):
        if not reachable[t]:
            continue
        counts = [0] * len(w_int)
        s = t
        while s > 0:
            for idx, w in enumerate(w_int):
                if w <= s and reachable[s - w]:
                    counts[idx] += 1
                    s -= w
                    break
            else:
                raise RuntimeError("inconsistent reachability table")
        return np.array(counts)
    raise ValueError(f"No valid partition found for the given total weight {total_weight}.")


def _assert_capacity_conserved(original, split_lines, atol=1e-6):
    """Sanity check: splitting a corridor into outage/non-outage variants must only
    rearrange its s_nom and num_parallel, never create or destroy capacity."""
    base_name = split_lines.index.to_series().str.replace(r"_outagetype\d+$", "", regex=True)
    totals = split_lines.groupby(base_name)[["s_nom", "num_parallel"]].sum()
    totals = totals.reindex(original.index)
    for col in ["s_nom", "num_parallel"]:
        diff = (totals[col] - original[col]).abs()
        bad = diff[diff > atol]
        if len(bad):
            raise AssertionError(
                f"split_outage_lines changed total {col} for lines {list(bad.index)}: "
                f"before={original.loc[bad.index, col].to_dict()}, "
                f"after={totals.loc[bad.index, col].to_dict()}"
            )


def split_outage_lines(n):
    """
    Separate all parallel lines in each corridor for which we consider the outage.
    """
    original = n.lines[["s_nom", "num_parallel"]].copy()

    def nonzero_mask(partitions, type_idx):
        return partitions.apply(lambda row: [int(j == type_idx and row[j] > 0) for j in range(len(row))])

    def remove_linetype(partitions, type_idx):
        """Remove one line type from all partitions safely."""
        outage_partitions = nonzero_mask(partitions, type_idx)
        partitions_arr = np.vstack(partitions)
        outage_arr = np.vstack(outage_partitions)
        return pd.Series((partitions_arr - outage_arr).tolist(), index=partitions.index)

    def create_nonoutage_lines(n):
        lines_nonout = n.lines.copy()
        partitions = n.lines['partition']

        nonout_partitions = partitions
        # iteratively remove each line type starting from the highest capacity,
        # neglecting outages that would lead to all-zero partitions
        for type_idx in reversed(range(len(TYPE_WEIGHTS))):
            nonout_partitions = remove_linetype(nonout_partitions, type_idx)

        lines_nonout['num_parallel'] = nonout_partitions.apply(lambda p: (np.array(p) * TYPE_WEIGHTS).sum())
        lines_nonout['s_nom'] = lines_nonout['s_nom'] * (lines_nonout['num_parallel'] / n.lines['num_parallel'])

        arr_part = np.vstack(partitions.values)
        arr_nonout = np.vstack(nonout_partitions.values)
        outage_arr = arr_part - arr_nonout
        outage_line_partitions = pd.Series(list(outage_arr), index=n.lines.index)

        return lines_nonout, outage_line_partitions

    lines_nonout, outage_partitions = create_nonoutage_lines(n)

    def create_outage_lines(n, outage_partitions):
        lines_out = pd.DataFrame()
        # loop through each linetype to create fictitious outage lines
        for type_idx in reversed(range(len(TYPE_WEIGHTS))):
            lines_out_type = n.lines.copy()

            # get indices where the outage partition has non-zero entries for this type
            outage_indices = outage_partitions.apply(lambda row: row[type_idx] > 0)

            lines_out_type = lines_out_type[outage_indices].copy()
            orig_num_parallel = lines_out_type['num_parallel']
            lines_out_type['num_parallel'] = TYPE_WEIGHTS[type_idx]
            lines_out_type['s_nom'] = (lines_out_type['s_nom'] * TYPE_WEIGHTS[type_idx] /
                                        orig_num_parallel)

            # partition is 0 except for the current type, there is a 1
            partition = [0] * len(TYPE_WEIGHTS)
            partition[type_idx] = 1
            lines_out_type['partition'] = [partition] * len(lines_out_type)
            lines_out_type.index = pd.Index(
                [f"{i}_outagetype{type_idx}" for i in lines_out_type.index],
                name=lines_out_type.index.name,
            )

            lines_out = pd.concat([lines_out, lines_out_type])
        return lines_out

    lines_out = create_outage_lines(n, outage_partitions)
    n.lines = pd.concat([lines_out, lines_nonout])
    _assert_capacity_conserved(original, n.lines)

    # prior results of pfs are now worthless.
    for key in n.lines_t.keys():
        n.lines_t[key] = pd.DataFrame(index=n.snapshots)

    n.calculate_dependent_values()

    return n


def fix_capacities(n, overdim_extendables, freeze_extendable=True):
    """Set p_nom/e_nom of previously-extendable components to their optimal
    value (times overdim_extendables). If freeze_extendable is True (default,
    matches the pipeline's normal behavior), these components also become
    non-extendable, so SC-LOPF is a pure fixed-capacity redispatch. If False,
    the optimal value instead becomes a floor (p_nom_min/e_nom_min) and the
    component stays extendable, so solve_sclopf's per-window LP can still add
    capacity on top of it - used by the weather-year sensitivity variant to
    let the brownfield fleet grow under a different weather year's profiles.
    """

    goi = n.generators.query("p_nom_extendable == True").index
    n.generators.loc[goi, "p_nom"] = overdim_extendables*n.generators.loc[goi, "p_nom_opt"]
    if freeze_extendable:
        n.generators.loc[goi, "p_nom_extendable"] = False
    else:
        # overdim_extendables can push the floor above a generator's fixed
        # potential ceiling (p_nom_max) for units already built out to (or
        # near) that ceiling in the original solve - clip so p_nom_min never
        # exceeds p_nom_max, which would otherwise make the LP infeasible.
        n.generators.loc[goi, "p_nom"] = n.generators.loc[goi, "p_nom"].clip(
            upper=n.generators.loc[goi, "p_nom_max"]
        )
        n.generators.loc[goi, "p_nom_min"] = n.generators.loc[goi, "p_nom"]

    suoi = n.storage_units.query("p_nom_extendable == True").index
    n.storage_units.loc[suoi, "p_nom"] = overdim_extendables*n.storage_units.loc[suoi, "p_nom_opt"]
    if freeze_extendable:
        n.storage_units.loc[suoi, "p_nom_extendable"] = False
    else:
        n.storage_units.loc[suoi, "p_nom"] = n.storage_units.loc[suoi, "p_nom"].clip(
            upper=n.storage_units.loc[suoi, "p_nom_max"]
        )
        n.storage_units.loc[suoi, "p_nom_min"] = n.storage_units.loc[suoi, "p_nom"]
    n.storage_units.marginal_cost *= -1

    soi = n.stores.query("e_nom_extendable == True").index
    n.stores.loc[soi, "e_nom"] = overdim_extendables*n.stores.loc[soi, "e_nom_opt"]
    if freeze_extendable:
        n.stores.loc[soi, "e_nom_extendable"] = False
    else:
        n.stores.loc[soi, "e_nom"] = n.stores.loc[soi, "e_nom"].clip(
            upper=n.stores.loc[soi, "e_nom_max"]
        )
        n.stores.loc[soi, "e_nom_min"] = n.stores.loc[soi, "e_nom"]

    lkoi = n.links.query("p_nom_extendable == True").index
    n.links.loc[lkoi, "p_nom"] = n.links.loc[lkoi, "p_nom_opt"]
    n.links.loc[lkoi, "p_nom_extendable"] = False

    loi = n.lines.query("s_nom_extendable == True").index
    n.lines.loc[loi, "s_nom"] = n.lines.loc[loi, "s_nom_opt"]
    n.lines.loc[loi, "s_nom_extendable"] = False

    n.lines = n.lines.loc[n.lines.num_parallel != 0.0]

    # decompose each line's num_parallel into counts of the per-voltage weights in
    # TYPE_WEIGHTS, so split_outage_lines knows which voltage classes a corridor is
    # made of and can split off the appropriate outage/non-outage variants.
    reachable, w_int = _build_reachability(TYPE_WEIGHTS, n.lines.num_parallel.max())
    partitions = n.lines['num_parallel'].apply(lambda v: get_partition(v, reachable, w_int))
    corr_num_parallel = partitions.apply(lambda p: (p * TYPE_WEIGHTS).sum())
    n.lines['num_parallel_uncorrected'] = n.lines['num_parallel']
    n.lines['num_parallel'] = corr_num_parallel
    n.lines['correction_num_parallel'] = n.lines['num_parallel'] - n.lines['num_parallel_uncorrected']
    n.lines['partition'] = partitions

if __name__ == "__main__":
    if "snakemake" not in globals():
        # run standalone (e.g. `python3 scripts/prepare_sclopf.py`) to iterate
        # without going through the full Snakemake DAG
        import os
        import sys

        sys.path.insert(0, os.path.abspath("submodules/pypsa-eur"))
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_sclopf_weather",
            configfiles=["configs/config.sclopf.yaml", "configs/config.yaml"],
            submodule_dir="submodules/pypsa-eur",
            opts="Co2L0.6",
            weather_year="2012",
        )

    from scripts._helpers import configure_logging

    configure_logging(snakemake)

    # config = snakemake.config
    config = snakemake.config
    group_size = int(config["groupsize"])
    temp_resolution = int("".join(filter(str.isdigit, config["clustering"]["temporal"]["averaging"])))
    load_shedding = snakemake.config["load_shedding"]
    artificial_load = snakemake.config["artificial_load"]

    n = pypsa.Network(snakemake.input.network)

    fix_capacities(
        n,
        config["overdim_extendables"],
        freeze_extendable=config.get("freeze_extendable_capacities", True),
    )

    n = split_outage_lines(n)
    n.determine_network_topology()

    #otherwise, this will throw an error in network_lopf with extra functionality for sclopf
    to_delete = []
    for i, sub in n.sub_networks.iterrows():
        buses = sub.obj.buses()
        if len(buses) == 1:
            to_delete += list(buses.index)
    logger.warning(
        f"Currently, islands cannot be treated in the sc-lopf. Thus removing {to_delete}")

    n.remove("Bus", to_delete)
    n.remove("Load", n.loads.query("bus in @to_delete").index)
    n.remove("Generator", n.generators.query("bus in @to_delete").index)
    n.remove("Link", n.links.query("bus0 in @to_delete or bus1 in @to_delete").index)
    n.remove("StorageUnit", n.storage_units.query("bus in @to_delete").index)
    n.remove("Store", n.stores.query("bus in @to_delete").index)

    n.determine_network_topology()
    n.calculate_dependent_values()


 ### if loadshedding is activated, add loadshedding possibility to each bus ###
    if load_shedding:
        print("Loadshedding included")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        n.add(
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
        n.add(
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
        logger.info("Fixing the boundaries of each window to the storage levels of the lopf result")

        n.storage_units_t.state_of_charge_set = n.storage_units_t.state_of_charge

        i_values = [n for n in range(int(np.ceil(8760 / temp_resolution / group_size)))]
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

    # 'partition' holds a per-line array (circuit counts per TYPE_WEIGHTS entry),
    # which netCDF/xarray cannot serialise as a line attribute; only needed
    # internally by fix_capacities/split_outage_lines above.
    n.lines = n.lines.drop(columns=["partition"])
    n.export_to_netcdf(snakemake.output[0])
