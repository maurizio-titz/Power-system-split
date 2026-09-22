#!/usr/bin/env python
"""
Report CO2 emissions (as % of the 1990 baseline) for the solved LOPF networks
and, where available, their reassembled SCLOPF counterparts.

Standalone script, not a Snakemake script: run it directly, e.g.

    cd workflow
    ../../submodules/pypsa-eur/.pixi/envs/default/bin/python3 scripts/report_co2_emissions.py

Run with -h for options (e.g. to point at a different config or scenario subset).
"""

import argparse
import os
import sys
from pathlib import Path

import pypsa
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = WORKFLOW_DIR.parent.parent
sys.path.append(str(PROJECT_DIR))

from utils.config import path_to_sclopf_data  # noqa: E402

GT_TO_TONNES = 1e9


def load_config(workflow_dir: Path) -> dict:
    config = {}
    for fn in ["configs/config.sclopf.yaml", "configs/config.yaml"]:
        with open(workflow_dir / fn) as f:
            config.update(yaml.safe_load(f) or {})
    return config


def get_emissions(n: pypsa.Network) -> float:
    """Total CO2 emissions [tonnes] from generator and storage dispatch."""
    gen = (
        n.generators_t.p.multiply(n.snapshot_weightings.objective, axis=0)
        .divide(n.generators.efficiency, axis=1)
        .fillna(0)
        .multiply(n.generators.carrier.map(n.carriers.co2_emissions))
        .fillna(0)
        .sum()
        .sum()
    )
    stog = (
        n.storage_units_t.p.multiply(n.snapshot_weightings.objective, axis=0)
        .divide(n.storage_units.efficiency_dispatch, axis=1)
        .fillna(0)
        .multiply(n.storage_units.carrier.map(n.carriers.co2_emissions))
        .fillna(0)
        .sum()
        .sum()
    )
    return gen + stog


def get_baseline_1990(n: pypsa.Network, fraction: float) -> float | None:
    """
    Back out the 1990 baseline [tonnes] that pypsa-eur used to scale co2_budget,
    from the network's own CO2Limit global constraint:
        constant [tonnes] = fraction * baseline_1990 [Gt] * GT_TO_TONNES * nyears
    """
    if not fraction:
        return None
    nyears = n.snapshot_weightings.objective.sum() / 8760.0
    for name in n.global_constraints.index:
        if name.startswith("CO2Limit"):
            constant = n.global_constraints.at[name, "constant"]
            return constant / (fraction * nyears)
    return None


def co2_run_name(opts: str) -> str:
    """'Co2L0.6' -> 'co2-0.6'"""
    return "co2-" + opts.replace("Co2L", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="CO2 scenario fractions to report, e.g. 0.0 0.05 0.1 ... "
        "(default: config.sclopf.yaml's Co2-scenarios)",
    )
    args = parser.parse_args()

    config = load_config(WORKFLOW_DIR)
    scenarios = args.scenarios or config["Co2-scenarios"]

    horizons = config["planning_horizons"]
    horizon = str(horizons[0] if isinstance(horizons, list) else horizons)
    nclusters = str(config["clustering"]["cluster_network"]["n_clusters"])

    lopf_dir = WORKFLOW_DIR / "submodules/pypsa-eur/results"
    sclopf_dir = Path(path_to_sclopf_data)

    rows = []
    shared_baseline_t = None

    for s in scenarios:
        fraction = float(s)
        lopf_fn = lopf_dir / co2_run_name(f"Co2L{s}") / "networks" / f"solved_{horizon}.nc"
        sclopf_fn = sclopf_dir / f"sclopf-elec_s_{nclusters}_ec_lv1.0_Co2L{s}.nc"

        lopf_pct = sclopf_pct = None

        if lopf_fn.exists():
            n_lopf = pypsa.Network(str(lopf_fn))
            lopf_emissions_t = get_emissions(n_lopf)
            if shared_baseline_t is None and fraction:
                shared_baseline_t = get_baseline_1990(n_lopf, fraction)
            if shared_baseline_t:
                lopf_pct = 100 * lopf_emissions_t / shared_baseline_t

        if sclopf_fn.exists():
            n_sclopf = pypsa.Network(str(sclopf_fn))
            sclopf_emissions_t = get_emissions(n_sclopf)
            if shared_baseline_t:
                sclopf_pct = 100 * sclopf_emissions_t / shared_baseline_t

        rows.append((s, lopf_fn.exists(), lopf_pct, sclopf_fn.exists(), sclopf_pct))

    if shared_baseline_t is None:
        print("Could not determine the 1990 baseline (no scenario with fraction > 0 found).")
        return

    print(f"1990 baseline: {shared_baseline_t / 1e6:.2f} Mt CO2\n")
    header = f"{'scenario':>10} | {'LOPF % of 1990':>15} | {'SCLOPF % of 1990':>17}"
    print(header)
    print("-" * len(header))
    for s, lopf_exists, lopf_pct, sclopf_exists, sclopf_pct in rows:
        lopf_str = f"{lopf_pct:.2f}%" if lopf_pct is not None else ("missing" if not lopf_exists else "n/a")
        sclopf_str = f"{sclopf_pct:.2f}%" if sclopf_pct is not None else ("missing" if not sclopf_exists else "n/a")
        print(f"{s:>10} | {lopf_str:>15} | {sclopf_str:>17}")


if __name__ == "__main__":
    main()
