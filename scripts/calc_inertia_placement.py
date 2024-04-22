"""calculate optimal inertia placement to reduce lost load due to RoCoF. Start with large step size delta_rot_ls and decrease it by a factor of 2 in each iteration. Throws an error the placement of inertia can not mitigate any loss of load (i.e. delta_rot_ls is too small push the RoCoF over -1 for any split component).
    """

import sys

sys.path.append("./")


from utils.plot_mitigation_strategies import plot_map_inertia_placement_final
from utils.synthetic_inertia_placement import (
    run_different_parameters_for_co2lvl,
    run_specific_co2lvl_n_size,
)

if __name__ == "__main__":
    n_nodes = 400
    # resolve_strategies = [
    #     "random",
    #     # "concentrate",
    #     # "hindsight",
    #     # "hindsight_concentrate",
    # ]
    resolve_strategy = "random"
    # delta_rot_ls = [5000, 2500, 1000, 500, 100]
    # delta_rot_ls = [5000]
    delta_rot = 5000
    for co2_lvl in [0.0]:
        # for co2_lvl in [0.0, 0.2, 0.3, 0.4, 0.5]:
        print(f"Running for co2_lvl: {co2_lvl}")
        # run_different_parameters_for_co2lvl(
        #     co2_lvl,
        #     n_nodes,
        #     delta_rot_ls,
        #     max_iter=10000,
        #     nr_processes=1,
        #     revert_chrotE_fac=False,
        #     resolve_equality_method_ls=resolve_strategies,
        # )
        # run_specific_co2lvl_n_size(
        #     co2_lvl,
        #     nn_nodes=n_nodes,
        #     delta_rot_energy=delta_rot,
        #     show_progress=False,
        #     save_it=True,
        #     max_iter=10000,
        #     resolve_equality_method=resolve_strategy,
        #     revert_ch_rotE_fac=False,
        # )
        plot_map_inertia_placement_final(
            co2_lvl,
            nn=400,
            max_iter=10000,
            max_node_size=800,
            edge_width=0.2,
            delta_Erot=delta_rot,
            resolve_strategy=resolve_strategy,
            save_fig=True,
        )
