"""this module analyses the blackout clusters. Especially, it investigates which nodes have an overfrequency blackout and what carrier was dominant before the blackout."""
#%%
import gzip
import pickle
import numpy as np
import sys

import pickle
import pandas as pd

sys.path.append("..")  # To import from parent directory

from utils import data_handling
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from utils.data_handling import get_co2_levels
from utils import config

%load_ext autoreload
%autoreload 2
#%%

n_nodes = 600
co2l_list = list(get_co2_levels(n_nodes))

n_jobs_distance = 32
n_jobs_clustering = 2
distance_metric_kwargs = {
    "name": "composite",
    "metrics": [
        {
            "name": "hamming",
            # "preprocessing": {
            #     "method": "low_weight_boundary_field",
            #     "target": "weights",  # Use as weights
            #     "alpha": 0.4,
            #     "steps": 3,
            # },
            "n_jobs": n_jobs_distance,
        },
        {
            "name": "cosine_distance",
            "preprocessing": {
                "method": "boundary_field",
                "target": "classes",  # Transform classes
                "alpha": 0.4,
                "steps": 3,
            },
            "n_jobs": n_jobs_distance,
        },
    ],
    "combiner": geometric_mean,
    "cache_components": True,
}

n_clusters_list = np.arange(32, 1025, step=32).tolist()
# n_clusters_list = [16, 32, 64, 80]
clustering_params = {
    "agg": {
        "n_iter": 1e4,
        "n_jobs": n_jobs_clustering,
        "HPs": {
            "n_clusters": n_clusters_list,
            "linkage": [
                "average",
                "complete",
                # "single",
            ],  # "single" is not used as it leads to bad results
            "metric": ["precomputed"],
        },
        "calc_silhouette": True,
    },
    # "hdbscan": {
    #     "n_iter": 128,
    #     "n_jobs": n_jobs_clustering,
    #     "HPs": {
    #         "min_cluster_size": np.arange(5, 1000),
    #         "min_samples": np.arange(10, 1000),
    #         "cluster_selection_epsilon": loguniform(0.005, 0.1),
    #         "metric": ["precomputed"],
    #         "cluster_selection_method": ["eom", "leaf"],
    #         "core_dist_n_jobs": [32 // n_jobs_clustering],
    #     },
    #     "calc_silhouette": True,
    # },
}
# %%
# Initialize clustering object
cl = Clustering(
    n_nodes,
    co2l_list,
    indicator_type="rocof",
    transformation="blackout",
    blackout_size_threshold=0.1,
    distance_metric_kwargs=distance_metric_kwargs,
    clustering_params=clustering_params,
    distance_matrix_dtype=np.float16,
    # random_subsample_size=random_subsample_size,
    # clustering_worker_func=_fit_clustering_worker_script,  # Use script-level worker for pickling
)
# %%

# load clustering results
res = cl.load_clustering_results_index()
# %%
res = pd.DataFrame(res).sort_values(by=["silhouette_score"], ascending=False).reset_index(drop=True)

with gzip.open(res.loc[0, "filepath"].replace(".pklz", "_centroid_res.pklz"), "rb") as f:
    cluster_res = pickle.load(f)
# %%
cluster_number = 10
cluster_idx = cluster_res["centroids_df"].sort_values(by="n_samples", ascending=False).iloc[cluster_number,:].name
# %%
centroid = cluster_res["centroids"][cluster_idx]
#%% # plot centroid
import matplotlib.pyplot as plt
import networkx as nx
colors_classes=["blue", "lightgray", "red"]
f, ax = plt.subplots(figsize=(8, 8))
nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
pos = nx.get_node_attributes(nx_graph, "pos")
for i_node, node in enumerate(nx_graph.nodes()):
    ax.pie(
        centroid[i_node, :],
        colors=colors_classes,
        radius=0.5,
        center=(pos[node][0], pos[node][1]),
        # zorder=2,
    )
    pos_arr = np.array(list(pos.values()))
    ax.set_xlim(pos_arr[:, 0].min() - 1, pos_arr[:, 0].max() + 1)
    ax.set_ylim(pos_arr[:, 1].min() - 1, pos_arr[:, 1].max() + 1)
#%%
edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax,
            edge_color="black",
            width=1,
            # edge_cmap=cmap,
        )
edges.set_zorder(0)
#%%
f
# %%
node_idxs_red = np.where(centroid[:,2]>0.5)
# %%
nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
# %%
nodes_red = np.array(nx_graph.nodes)[node_idxs_red]
# %%
cluster_mask = cluster_res["group_masks"][cluster_idx]
# %%
split_properties = pd.read_hdf(
            f"{cl.data_dir}/data_filtered.h5",
            key="split_props",
        )
props_clust = split_properties.loc[cluster_mask]
#%%
co2l = 0.0
fpath_component_in = (
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
    )
component_df = pd.read_hdf(fpath_component_in, key="df")
#%%
props_clust_lvl = props_clust[props_clust.index.get_level_values("co2l") == co2l]
idx_clust_lvl = props_clust_lvl.index
#%%
component_df["co2l"] = co2l
component_df = component_df.set_index(["co2l", "time_stamp", "split_number"])
#%%
component_df_cluster = component_df.loc[idx_clust_lvl]
component_df_cluster.reset_index(inplace=True, drop=False)
#%%
# Keep only the row with largest load_share for each index group
idx = component_df_cluster.groupby(by=["co2l", "time_stamp", "split_number"])['load_share'].idxmax()
largest_component_df = component_df_cluster.loc[idx]
# %%
largest_component_df.load_share.hist(bins=30, log=True)
largest_component_df.power_imbalance.hist(bins=30, log=True)
largest_component_df.rot_energy.hist(bins=30, log=True)
largest_component_df.rocof.hist(bins=30, log=True)
#%%
snaps_clust = props_clust_lvl.index.get_level_values("time_stamp").value_counts().sort_index().index
idxs_snaps_clust = [snapshot_to_idx[snap] for snap in snaps_clust]
snap_counts_clust = props_clust_lvl.index.get_level_values("time_stamp").value_counts().sort_index().values
#%%
intertia_cluster = inertia_time[idxs_snaps_clust]
#%%
(component_df[component_df.split_number_snapshot==1].groupby(component_df[component_df.split_number_snapshot==1].index.get_level_values("time_stamp")).rot_energy.sum()/1000).hist(bins=30, log=True)
#%%
largest_component_sub = largest_component_df.loc[np.random.choice(largest_component_df.index, size=10000, replace=False)]
plt.scatter(largest_component_sub.power_imbalance/1000, largest_component_sub.rot_energy/1000)
plt.xlabel("Power Imbalance [GW]")
plt.ylabel("Rotational Energy [GWs]")
# plt.colorbar(label="Counts")
inertia_rocof_border = - 50 * largest_component_sub.power_imbalance/ 2
plt.plot(largest_component_sub.power_imbalance/1000, inertia_rocof_border/1000, label="RoCoF Border", linewidth=2, color="red")
# %%
co2l_cluster = props_clust.index.get_level_values("co2l").unique().values
#%%
networks = {co2l: data_handling.load_pypsa_network(co2lvl=co2l, n_nodes=600) for co2l in co2l_cluster}
# %%
co2l = co2l_cluster[0]
n = networks[co2l]
snapshots = props_clust.xs(co2l, level="co2l").index.get_level_values("time_stamp").unique().values
#%%
node_cols_gen = [col for col in n.generators_t["p"].columns if any(str(node) in col for node in nodes_red)]
node_cols_store = n.storage_units.index[n.storage_units.bus.isin(nodes_red)].tolist()
#%%
gen_nodes = n.generators_t["p"].loc[snapshots, node_cols_gen]
#%%
store_nodes = n.storage_units_t["p"].loc[snapshots, node_cols_store]

#%%

gen_by_carrier = gen_nodes.groupby(gen_nodes.columns.str.split().str[-1], axis=1).sum().astype(int)
# remove columns with less than 
gen_by_carrier = gen_by_carrier.loc[:,gen_by_carrier.max(axis=0) > gen_by_carrier.sum(axis=1).mean() * 0.01]
#%%
load = n.loads_t["p"].loc[snapshots, n.loads.index[n.loads.bus.isin(nodes_red)]].sum(axis=1)
#%%
storage_by_carrier = store_nodes.groupby(n.storage_units.loc[store_nodes.columns].carrier, axis=1).sum().astype(int)
storage_by_carrier = storage_by_carrier.loc[:,storage_by_carrier.abs().max(axis=0) > storage_by_carrier.sum(axis=1).mean() * 0.05]
storage_by_carrier
#%%
power_imbalance = gen_by_carrier.sum(axis=1) + storage_by_carrier.sum(axis=1) - load
#%%
inertia_time = (
    np.load(
        config.path_to_pre_outage_sclopf + f"inertia_time_series_all_co2ls_{n_nodes}.npy"
    )
    / 1000
)[-1,:]
all_snapshots = n.loads_t["p"].index
inertia_time_cluster = inertia_time[np.isin(all_snapshots, snapshots)]

rocofs = (
                        50 * power_imbalance/1000 / ((inertia_time_cluster + 1e-8) * 2)
                    ) # /1000 because power imbalance and inertia in GW
#%%
snapshot_to_idx = {snap: idx for idx, snap in enumerate(all_snapshots)}
#%%
# plot histograms of generation and storage power
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,5, figsize=(16,5))
for col in gen_by_carrier.columns:
    axs[0].hist(gen_by_carrier[col].values/1000, bins=30, alpha=0.5, label=col)
for col in storage_by_carrier.columns:
    axs[0].hist(storage_by_carrier[col].values/1000, bins=30, alpha=0.5, label=col)
axs[0].set_title("Generation power overfreq")
axs[0].legend()
axs[0].set_ylim(1, axs[0].get_ylim()[1])
axs[0].set_xlabel("Power [GW]")

axs[1].hist(load.values/1000, bins=30, alpha=0.7, color="orange")
axs[1].set_title("Load overfreq")
axs[1].set_xlabel("Power [GW]")

axs[2].hist(power_imbalance.values/1000, bins=30, alpha=0.7, color="green")
axs[2].set_title("Power imbalance overfreq")
axs[2].set_xlabel("Power imbalance [GW]")

axs[3].hist(inertia_time_cluster, bins=30, alpha=0.7, color="red")
axs[3].set_title("Total Inertia")
axs[3].set_xlabel("Inertia [s]")

axs[4].hist(rocofs, bins=30, alpha=0.7, color="purple")
axs[4].set_title("RoCoF distribution")
axs[4].set_xlabel("RoCoF [Hz/s]")
#%%
# Plot histograms of snapshots - daily and yearly profiles
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Convert snapshots to pandas datetime if needed
snapshots_dt = pd.to_datetime(snapshots)

# Daily profile - histogram of hours
hours = snapshots_dt.hour
axs[0].hist(hours, bins=24, range=(0, 24), edgecolor='black', alpha=0.7)
axs[0].set_xlabel('Hour of Day')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Daily Profile: Blackout Distribution by Hour')
axs[0].set_xticks(range(0, 24, 2))
axs[0].grid(axis='y', alpha=0.3)

# Yearly profile - histogram of day of year
day_of_year = snapshots_dt.dayofyear
axs[1].hist(day_of_year, bins=52, range=(1, 366), edgecolor='black', alpha=0.7)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Yearly Profile: Blackout Distribution Throughout the Year')
axs[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
# %%

# get installed generation capacity in the nodes of the cluster
gen_capacity = n.generators.loc[n.generators.bus.isin(nodes_red)].groupby(n.generators.loc[n.generators.bus.isin(nodes_red)].carrier).p_nom.sum().astype(int)
gen_capacity = gen_capacity[[carrier for carrier in gen_capacity.index if not "load" in carrier.lower()]]
gen_capacity[gen_capacity>gen_capacity.max()*0.01]
gen_capacity.loc["solar"]/1000

#%% get inertia distribution