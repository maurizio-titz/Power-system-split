from utils.indicator_utils import run_different_parameters_for_co2lvl
"""calculate optimal inertia placement to reduce lost load due to RoCoF. Start with large step size delta_rot_ls and decrease it by a factor of 2 in each iteration. Throws an error the placement of inertia can not mitigate any loss of load (i.e. delta_rot_ls is too small push the RoCoF over -1 for any split component).
    """
n_nodes = 400
delta_rot_ls = [5000/(2**i) for i in range(9)]
for co2_lvl in [0.0, 0.1]:
    run_different_parameters_for_co2lvl(
        co2_lvl,
        n_nodes,
        delta_rot_ls,
        max_iter=10000,
        nr_processes = 5,
        revert_chrotE_fac = False,
        resolve_equality_method_ls = ["random", "concentrate",
                                            "hindsight", "hindsight_concentrate"],
    )