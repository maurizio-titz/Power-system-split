import sys


sys.path.append("./")

import os

import cartopy.geodesic as gd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from utils.data_handling import get_co2_levels
from utils import data_handling, subgraph_evaluation
from utils.config import (
    path_to_pre_outage_sclopf,
    path_to_pypsa_network_sclopf,
    path_to_sclopf_data,
    use_extensions,
)
from scripts.evaluate_cascade import load_inertia_constant

n_nodes = 600
co2l_list = get_co2_levels(n_nodes)
use_sclopf = True

print(f"n_nodes: {n_nodes}")
print(f"use_sclopf: {use_sclopf}")
print(f"co2l_list: {co2l_list}")

path_to_pre_outage = path_to_pre_outage_sclopf
path_to_pypsa_network = path_to_pypsa_network_sclopf
# Setup paths
os.makedirs(path_to_pre_outage, exist_ok=True)
# Select a particular subnetwork for calculations (if the pypsa network has different ones).
# For our data set, "0" indicates the Continental European AC grid.
snet_index = 0

# Get number of time steps and graph
network = data_handling.load_pypsa_network(0.0, n_nodes, use_sclopf=use_sclopf)
nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)
n_time_steps = network.snapshots.shape[0]

### Extract pre-outage inertia data ##### not used anymore

inertia_time_series = np.zeros((len(co2l_list), n_time_steps))
nodal_inertia_min_max = np.zeros((len(co2l_list), 2, nx_graph.number_of_nodes()))

for i, co2l in enumerate(co2l_list):
    print("Co2 level %.2f" % co2l)

    network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf=use_sclopf)
    nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)

    for t_count, timestamp in enumerate(network.snapshots):
        obs = subgraph_evaluation.evaluate_observables_for_subgraphs(
            [nx_graph], network, timestamp, snet=0
        )
        load_inertia = float(obs[0, 2]) * load_inertia_constant
        assert obs.shape[0] == 1, "Expected one subgraph"
        inertia_time_series[i, t_count] = obs[0, 0] + load_inertia

    # Select timestamps with min and max total inertia
    t_largest = network.snapshots[np.argmax(inertia_time_series[i])]
    t_smallest = network.snapshots[np.argmin(inertia_time_series[i])]

    for t_count, timestamp in enumerate([t_smallest, t_largest]):
        for nodecount, node in enumerate(nx_graph.nodes()):
            subgraph = nx.Graph()
            subgraph.add_node(node)
            obs = subgraph_evaluation.evaluate_observables_for_subgraphs(
                [subgraph], network, timestamp, snet=0
            )
            load_inertia = float(obs[0, 2]) * load_inertia_constant

            nodal_inertia_min_max[i, t_count, nodecount] = obs[0, 0] + load_inertia

inertia_time_series_file_path = (
    path_to_pre_outage + f"inertia_time_series_all_co2ls_{n_nodes}.npy"
)
np.save(
    inertia_time_series_file_path,
    inertia_time_series,
)
print(f"Saved inertia time series to {inertia_time_series_file_path}")
nodal_inertia_min_max_file_path = (
    path_to_pre_outage + f"min_max_nodal_inertia_generation_all_co2ls_{n_nodes}.npy"
)
np.save(
    nodal_inertia_min_max_file_path,
    nodal_inertia_min_max,
)
print(f"Saved min/max nodal inertia generation to {nodal_inertia_min_max_file_path}")


# #### Calculate dipole vectors #### not used in the publication, but can be used for further analysis of the spatial power inhomogeneity

# pos = nx.get_node_attributes(nx_graph, "pos")
# dipole_vector = np.zeros((len(co2l_list), 2, len(network.snapshots)))
# mean_consumption_vector = np.zeros((len(co2l_list), nx_graph.number_of_nodes()))
# weighted_mean_consumption_vector = np.zeros(
#     (len(co2l_list), nx_graph.number_of_nodes())
# )
# graph_net_mismatch = np.zeros((len(co2l_list), len(network.snapshots)))

# for i, co2l in enumerate(co2l_list):

#     print("Co2 level %.2f" % co2l)

#     network = data_handling.load_pypsa_network(co2l, n_nodes, use_sclopf=use_sclopf)
#     nx_graph = data_handling.build_networkx_graph(network, snet_index=snet_index)

#     ### NOTE: Here, the mean position is subtracted from the coordinates
#     position_vector = np.array([pos[n] for n in nx_graph.nodes()])
#     position_vector[:, 1] -= np.mean(position_vector[:, 1])
#     position_vector[:, 0] -= np.mean(position_vector[:, 0])

#     nodal_balance = (
#         network.generators_t.p.mul(network.generators.sign)
#         .T.groupby(network.generators["bus"])
#         .sum()
#     )
#     nodal_balance = nodal_balance.add(
#         network.storage_units_t.p.T.groupby(network.storage_units["bus"]).sum(),
#         fill_value=0,
#     ).add(-network.loads_t.p.T.groupby(network.loads["bus"]).sum(), fill_value=0)

#     ## add hvdc link subtraction/addition
#     nodal_balance = nodal_balance.add(
#         -network.links_t.p0.T.groupby(network.links["bus0"]).sum(), fill_value=0
#     )
#     nodal_balance = nodal_balance.add(
#         -network.links_t.p1.T.groupby(network.links["bus1"]).sum(), fill_value=0
#     )

#     nodal_balance = -nodal_balance

#     for nodecount, node in enumerate(nx_graph.nodes()):
#         mean_consumption_vector[i, nodecount] = nodal_balance.mean(axis=1).loc[node]
#         weighted_mean_consumption_vector[i, nodecount] = (
#             nodal_balance.mul(network.snapshot_weightings.generators, axis="columns")
#             .mean(axis=1)
#             .loc[node]
#         )
#         graph_net_mismatch[i] += nodal_balance.loc[node]
#         dipole_vector[i] += np.outer(
#             position_vector[nodecount], nodal_balance.loc[node].to_numpy()
#         )
#         if np.any(np.isnan(dipole_vector)):
#             raise ValueError("SPI coordinates are not valid!")

# np.save(
#     path_to_pre_outage + f"dipole_vector_time_series_all_co2ls_{n_nodes}.npy",
#     dipole_vector,
# )
# np.save(
#     path_to_pre_outage + f"mean_nodal_consumption_all_co2ls_{n_nodes}.npy",
#     mean_consumption_vector,
# )
# np.save(
#     path_to_pre_outage + f"weighted_mean_nodal_consumption_all_co2ls_{n_nodes}.npy",
#     weighted_mean_consumption_vector,
# )
# np.save(
#     path_to_pre_outage
#     + f"graph_net_power_mismatch_time_series_all_co2ls_{n_nodes}.npy",
#     graph_net_mismatch,
# )

# #### Calculate spatial power inhomogeneity ####
# # (To calculate the spatial power inhomogeneity (spi) we rescale the spi vector
# # by the below scale factor, since it has the units of power and positions in long and lat
# # We then evaluate the geodesic distance between the mean position (given in logitude and latitude)
# # and the end point of the rescaled spi vector)

# print("\nCalculate spatial power inhomogeneity...")
# vec_norm = np.zeros((len(co2l_list), len(network.snapshots)))
# positions = np.array([pos[n] for n in nx_graph.nodes()])
# mean_pos = np.array([np.mean(positions[:, 0]), np.mean(positions[:, 1])])
# scale_factor = 1e6

# k = gd.Geodesic()

# for i, co2l in enumerate(co2l_list):
#     print("Co2 level %.2f" % co2l)

#     for j in range(len(network.snapshots)):
#         shapely_pos = Point(dipole_vector[i, :, j] / scale_factor + mean_pos)
#         distance = k.inverse(shapely_pos.coords, mean_pos)[0, 0] / 1000
#         vec_norm[i, j] = distance
#         if np.any(np.isnan(vec_norm)):
#             raise ValueError("SPI coordinates are not valid!")
# np.save(path_to_pre_outage + f"spi_time_series_all_co2ls_{n_nodes}.npy", vec_norm)


#### get actual co2 emission lvls ####
# %%
import sys
import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np

from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_pre_outage_sclopf,
    path_to_sclopf_results,
)

n_nodes = 600

# # Load network graph and node positions
# # (it is equal for all CO2 levels)
# snet_index = 0
# network = data_handling.load_pypsa_network_from_path(path_to_pypsa_network_sclopf +
#                         f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc", True)
# nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
# pos = nx.get_node_attributes(nx_graph, 'pos')
# I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)
# n_lines = nx_graph.number_of_edges()

# # Determine number of simulations,
# # i.e., total number of initial failures over the simulated period of one year
# bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph),
#                                                            nx_graph)
# n_2_failures = cascade_simulation.calc_possible_double_line_failures(num_parallels,
#                                                                      ignored_idxs=bridge_idxs)
# num_failures = len(n_2_failures)
# num_failures_weighted = network.snapshot_weightings.objective.sum() * num_failures
# n_snapshots = network.snapshots.shape[0]

co2ls = get_co2_levels(n_nodes)


# %%
networks = {
    co2l: data_handling.load_pypsa_network(
        n_nodes=600, co2lvl=co2l, use_sclopf=True, lopt=use_extensions
    )
    for co2l in co2ls
}

# %%
# get actual emission levels
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import os

import sys

# root_path = "../"#'/srv/data/jlange/PYPSA3/sclopf-iter' # This defaults to './', which should be the repository path

# sys.path.append(root_path)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


relax = 1.0  # Relaxation factor for the CO2 constraint
p_heurist = 0.7  # Heuristic for the maximum line loading
NCLUSTERS = 600
groupsize = 60
TEMP_RESOLUTION = 3  # H
load_shedding = True
n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))


# Co2_scenarios = ['0.0','0.05', '0.1','0.2', '0.3', '0.4', '0.5', '0.6'] #, '0.2'


# networks_path = f"{root_path}/workflow/co2_rel_{relax}/s_max_pu_{p_heurist}/postnetworks/"


# root_path = "/srv/data/jlange/PYPSA3/sclopf-iter"  # This defaults to './', which should be the repository path
# networks_path = (
#     f"{root_path}/workflow/co2_rel_{relax}/s_max_pu_{p_heurist}/postnetworks/"
# )


def get_emissions(n):

    gen = (
        n.generators_t.p.multiply(
            n.snapshot_weightings.objective, axis=0
        )  # multiply by no. of hours (weight)
        .divide(n.generators.efficiency, axis=1)
        .fillna(
            0
        )  # efficiency and generation might be different (e.g. in the case of CCGT)
        .multiply(
            n.generators.carrier.map(n.carriers.co2_emissions)
        )  # multiply by specific emissions
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


# os.makedirs('pics/lopf', exist_ok=True)
Co2_scenarios = co2ls
# Define the carriers
renewable_carriers = [
    "solar",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "hydro",
]
conventional_carriers = [
    "nuclear",
    "oil",
    "OCGT",
    "CCGT",
    "coal",
    "lignite",
    "geothermal",
    "biomass",
]

# Combine them into a single list for the DataFrame columns
all_carriers = renewable_carriers + conventional_carriers  # network.gener.index

generation_by_carrier_and_co2l = pd.DataFrame(
    None, index=all_carriers, columns=Co2_scenarios
)

CO2_shadowprices = pd.Series(index=Co2_scenarios)
CO2_values = pd.Series(index=Co2_scenarios)
CO2_global = pd.Series(index=Co2_scenarios)
# emissions = pd.Dataframe(None,index = [0], columns = Co2_scenarios)

for Co2l in Co2_scenarios:

    network = networks[Co2l]

    shadow_CO2 = network.global_constraints["mu"]["CO2Limit"]
    CO2_global[Co2l] = network.global_constraints["constant"]["CO2Limit"]

    CO2_shadowprices[Co2l] = shadow_CO2
    CO2_values[Co2l] = get_emissions(network)

    print(f"{Co2l} CO2 shadow price: {shadow_CO2}")

    df = network.generators_t.p.sum(axis=0)

    # remove carrier id
    df.index = df.index.str[6:]
    # aggregate same carrier types
    df = df.groupby(df.index).sum()

    for carrier in df.index:

        # print(f"{carrier} {df[carrier]}")
        generation_by_carrier_and_co2l.loc[carrier, Co2l] = df[carrier]
        # generation_by_carrier_and_co2l.loc[carrier, "co2_emissions"] = network.carriers.co2_emissions[carrier]#df[carrier]

    # plot energy balance with inbuilt energy_balance method

    # number of different carriers
    num_bars = len(
        network.statistics.energy_balance().loc[:, :, "AC"].groupby("carrier").sum()
    )

    # colors
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, num_bars))

    # plot
    fig, ax_loss_lvl = plt.subplots(figsize=(10, 6), layout="constrained")
    # tmp.rename({"-":"Load"},inplace=True)
    # data in plot
    # energy_balance_data =
    network.statistics.energy_balance().loc[:, :, "AC"].groupby("carrier").sum().rename(
        {"-": "Load"}
    ).to_frame().T.plot.bar(
        stacked=True, ax=ax_loss_lvl, color=colors, title=f"Energy Balance {Co2l}"
    )

    ax_loss_lvl.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
    # fig.savefig(f"pics/lopf/Energymix_lopf_{Co2l}-{NCLUSTERS}.pdf")
    plt.close()


# generation_fossil = generation_by_carrier_and_co2l.loc[["coal", "lignite", "CCGT", "OCGT"]].sum(axis=0)
# fossil_percentage = generation_fossil.div(generation_by_carrier_and_co2l.sum(axis=0) - generation_fossil)*100

# emission_by_carrier_by_co2l = generation_by_carrier_and_co2l.mul(network.carriers.co2_emissions, axis="index")
# emission_by_carrier_by_co2l.dropna(axis=0, thresh=1, inplace=True)
emission_by_co2l = CO2_values  # emission_by_carrier_by_co2l.sum(axis=0)
emission_by_co2l_perc = emission_by_co2l / emission_by_co2l[0.5] * 50

x_values = np.array(emission_by_co2l_perc.index, dtype=float) * 100

# plot red line
plt.plot([60, 0], [60, 0], color="r", alpha=0.5)

plt.scatter(x_values, emission_by_co2l_perc.values)

plt.gca().invert_xaxis()
plt.grid()
plt.title("emission level normalized to 50%")
plt.xlabel("intended emission lvl[%]")
plt.ylabel("calculated emission lvl[%]")

plt.figure()
plt.scatter(x_values, emission_by_co2l.values, label="Data")
plt.scatter(x_values, CO2_global.values, label="Global constraints", marker="x")

plt.legend()
plt.gca().invert_xaxis()
plt.grid()
plt.xlabel("intended emission lvl[%]")
plt.ylabel("calculated emissions")
# plt.savefig(f"pics/lopf/emissions.pdf")


for cx, Co2l in enumerate(Co2_scenarios):
    print(
        f"{Co2l}: Actual CO2lvl: {emission_by_co2l.values[cx]/CO2_global.values[cx]*float(Co2l)*100}"
    )


actual_co2ls = {
    Co2l: emission_by_co2l.values[cx] / (CO2_global.values[cx] + 10**-10) * float(Co2l)
    for cx, Co2l in enumerate(Co2_scenarios)
}

actual_co2ls = pd.DataFrame.from_dict(
    actual_co2ls, orient="index", columns=["actual_co2_levels"]
)
actual_co2ls.sort_index(inplace=True)
actual_co2ls.index.name = "co2_level"
actual_co2ls.to_csv(path_to_sclopf_results + "actual_co2_levels.csv")
