"""calculate optimal inertia placement to reduce lost load due to RoCoF. Start with large step size delta_rot_ls and decrease it by a factor of 2 in each iteration. Throws an error the placement of inertia can not mitigate any loss of load (i.e. delta_rot_ls is too small push the RoCoF over -1 for any split component)."""

import sys

sys.path.append("./")


from utils.config import get_co2_levels
from utils.plot_mitigation_strategies import plot_map_inertia_placement_final
from utils.synthetic_inertia_placement import (
    run_different_parameters_for_co2lvl,
    run_specific_co2lvl_n_size,
)

if __name__ == "__main__":
    from utils.config import path_to_sclopf_data
    import datetime
    import os

    n_nodes = 600
    use_sclopf = True

    co2l_list = get_co2_levels(n_nodes)
    # resolve_strategies = [
    #     "random",
    #     "concentrate",
    #     "hindsight",
    #     "hindsight_concentrate",
    # ]
    resolve_strategy = (
        "random"  # performance is very similar, random gives the most intuitive results
    )
    resolve_strategies = [resolve_strategy]
    delta_rot_ls = [1000, 500, 200, 100]
    for co2_lvl in co2l_list:
        print(f"Running for co2_lvl: {co2_lvl}")
        run_different_parameters_for_co2lvl(
            co2_lvl,
            n_nodes,
            use_sclopf,
            delta_rot_ls,
            max_iter=10000,
            nr_processes=1,
            revert_chrotE_fac=False,
            resolve_equality_method_ls=resolve_strategies,
        )
    for co2_lvl in co2l_list:
        for delta_rot in delta_rot_ls:
            try:
                plot_map_inertia_placement_final(
                    co2_lvl,
                    nn=n_nodes,
                    max_iter=10000,
                    max_node_size=800,
                    edge_width=0.2,
                    delta_Erot=delta_rot,
                    resolve_strategy=resolve_strategy,
                    save_fig=True,
                    use_sclopf=use_sclopf,
                )
            except ZeroDivisionError as e:
                pass
