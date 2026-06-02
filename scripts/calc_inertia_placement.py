#!usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate optimal inertia placement to reduce lost load due to RoCoF. Start with large step size delta_rot_ls and decrease it by a factor of 2 in each iteration. Throws an error the placement of inertia can not mitigate any loss of load (i.e. delta_rot_ls is too small push the RoCoF over -1 for any split component)."""

import sys

sys.path.append("./")

import argparse
from utils.data_handling import get_co2_levels
from utils.mitigation_visualization import plot_map_inertia_placement_final
from utils.synthetic_inertia_placement import (
    run_different_parameters_for_co2lvl,
    run_specific_co2lvl_n_size,
)
from utils.config import path_to_sclopf_data
import datetime
import os


def main(single_co2=None):
    """Main entry. If single_co2 is provided, run only that CO2 level, otherwise iterate over default list."""

    n_nodes = 600
    use_sclopf = True

    if single_co2 is None:
        co2l_list = get_co2_levels(n_nodes)
    else:
        co2l_list = [single_co2]

    # resolve_strategies = [
    #     "random",
    #     "concentrate",
    #     "hindsight",
    #     "hindsight_concentrate",
    # ]
    resolve_strategy = (
        "random"  # performance is very similar, random gives the most intuitive results
    )
    delta_rot_dict = {
        0.0: 20000,
        0.05: 10000,
        0.1: 3000,
        0.2: 2000,
        0.3: 2000,
        0.4: 2000,
        0.5: 2000,
        0.6: 1000,
    }
    # delta_rot = 1000
    max_iter = 1000
    for co2_lvl in co2l_list:
        print(f"Running for co2_lvl: {co2_lvl}")
        delta_rot = delta_rot_dict.get(co2_lvl, 1000)
        for blackout_size_threshold in [0, 0.8]:
            run_specific_co2lvl_n_size(
                co2_lvl,
                nn_nodes=n_nodes,
                delta_rot_energy=delta_rot,
                show_progress=True,
                save_it=True,
                max_iter=max_iter,
                resolve_equality_method=resolve_strategy,
                revert_ch_rotE_fac=False,
                use_sclopf=use_sclopf,
                blackout_size_threshold=blackout_size_threshold,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate optimal inertia placement. Optionally run for a single CO2 level."
    )
    parser.add_argument(
        "--co2",
        type=float,
        default=None,
        help="Single CO2 level to run (e.g. 0.6). If omitted, script will iterate over default levels.",
    )
    args = parser.parse_args()
    main(single_co2=args.co2)
