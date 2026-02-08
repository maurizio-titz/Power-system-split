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
from utils import plot_style
%load_ext autoreload
%autoreload 2

cluster_number = 10
save_dir = f"{config.path_to_figures_sclopf}./analysis_cluster_{cluster_number}/"
#%%
with gzip.open("/srv/data/jlange/power-system-split/no_extensions/results/sclopf/clustering/rocof_blackout_Co2L0.6_0.5_0.4_0.3_0.2_0.1_0.05_0.0_n600_lls0.1/clustering_results/clustering_results_index.pklz", "rb") as f:
# with gzip.open("/srv/data/jlange/power-system-split/no_extensions/results_no_load_inertia/sclopf/clustering/rocof_blackout_Co2L0.6_0.5_0.4_0.3_0.2_0.1_0.05_0.0_n600_lls0.05/clustering_results/clustering_results_index.pklz", "rb") as f:
    df = pickle.load(f)
df = pd.DataFrame(df)
df.sort_values(by="silhouette_score", ascending=False)
#%%
fpath = "/srv/data/jlange/power-system-split/no_extensions/results/sclopf/clustering/rocof_blackout_Co2L0.6_0.5_0.4_0.3_0.2_0.1_0.05_0.0_n600_lls0.1/clustering_results/agg_params_68e9bdc7.pklz"
with gzip.open(fpath, "rb") as f:
    res = pickle.load(f)
#%%
m = res["model"]
m.labels_.shape
#%% load clustering results
n_nodes = 600
co2l_list = list(get_co2_levels(n_nodes))

n_jobs_distance = 32
n_jobs_clustering = 2
distance_metric_kwargs = {
    "name": "composite",
    "metrics": [
        {
            "name": "hamming",
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
}
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
)

# load clustering results
res = cl.load_clustering_results_index()
res = pd.DataFrame(res).sort_values(by=["silhouette_score"], ascending=False).reset_index(drop=True)

with gzip.open(res.loc[0, "filepath"].replace(".pklz", "_centroid_res.pklz"), "rb") as f:
    cluster_res = pickle.load(f)
# %% select cluster to analyse
cluster_idx = cluster_res["centroids_df"].sort_values(by="n_samples", ascending=False).iloc[cluster_number,:].name
centroid = cluster_res["centroids"][cluster_idx]
edge_centroids = cluster_res["edge_centroids"][cluster_idx]
failed_edges_prob = edge_centroids.copy()
# plot centroid
plot_cluster = True
edge_log_scale = False
if edge_log_scale:
    vmin_edge = 1e-3
    vmax_edge = 1.0
else:
    vmin_edge = 0.0
    vmax_edge = 1.0
vmin_edge_orig = vmin_edge
vmax_edge_orig = vmax_edge
import matplotlib as mpl
from utils.clustering_visualisation import truncate_colormap
edge_cmap = mpl.cm.get_cmap("inferno_r")
edge_cmap = truncate_colormap(edge_cmap, 0.1, 0.95, 1000)
edge_cmap.set_under("gainsboro", 1.0)
if plot_cluster:
    import matplotlib.pyplot as plt
    import networkx as nx
    plot_order = np.argsort(centroid[:, 2])  # Sort by the value of the third class
    colors_classes=["blue", "lightgray", "red"]
    f, ax = plt.subplots(figsize=(8, 8))
    nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
    pos = nx.get_node_attributes(nx_graph, "pos")
    nodes = np.array(list(nx_graph.nodes()))
    for i_node in plot_order:
        node = nodes[i_node]
        ax.pie(
            centroid[i_node, :],
            colors=colors_classes,
            radius=0.2,
            center=(pos[node][0], pos[node][1]),
            # zorder=2,
        );
        pos_arr = np.array(list(pos.values()))
        ax.set_xlim(pos_arr[:, 0].min() - 1, pos_arr[:, 0].max() + 1)
        ax.set_ylim(pos_arr[:, 1].min() - 1, pos_arr[:, 1].max() + 1)
    # edges = nx.draw_networkx_edges(
    #             nx_graph,
    #             pos=pos,
    #             ax=ax,
    #             edge_color="black",
    #             width=1,
    #             # edge_cmap=cmap,
    #         )
    if edge_log_scale:
        failed_edges_prob = np.array(
            [np.log10(x) if x > 1e-12 else -np.inf for x in failed_edges_prob]
        )
        vmin_edge = np.log10(vmin_edge)
        vmax_edge = np.log10(vmax_edge)

    edges = nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax,
        # edge_color="black",
        width=6,
        edge_cmap=edge_cmap,
        edge_color=failed_edges_prob,
        edge_vmin=vmin_edge,
        edge_vmax=vmax_edge,
    )
    edges.set_zorder(0)
    import matplotlib.colors as mplcolors
    if edge_log_scale:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.LogNorm(vmin=vmin_edge_orig, vmax=vmax_edge_orig)
        )
    else:
        sm_edge = plt.cm.ScalarMappable(
            cmap=edge_cmap, norm=mplcolors.Normalize(vmin=vmin_edge_orig, vmax=vmax_edge_orig)
        )
    cb_edge = fig.colorbar(sm_edge, ax=ax)
    # plt.colorbar(sm_edges, ax=ax, label="Edge failure probability")
    
#%%
from utils.clustering_visualisation import plot_centroid_with_failures
f, ax = plt.subplots(figsize=(8, 8))
plot_order = np.argsort(centroid[:, 2])  # Sort by the value of the third class
nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
pos = nx.get_node_attributes(nx_graph, "pos")
plot_centroid_with_failures(
    nx_graph,
    pos,
    cmap,
    edge_cmap,
    vmax,
    vmin,
    vmin_edge,
    vmax_edge,
    centroid,
    failed_edges_prob,
    ax,
)
# %%
node_idxs_red = np.where(centroid[:,2]>0.5)
nx_graph = data_handling.load_networkx_graph(snet_index=0, co2lvl=0.0)
nodes_red = np.array(nx_graph.nodes)[node_idxs_red]
cluster_mask = cluster_res["group_masks"][cluster_idx]
split_properties = pd.read_hdf(
            f"{cl.data_dir}/data_filtered.h5",
            key="split_props",
        )
props_clust = split_properties.loc[cluster_mask]

# restrict to single co2 level for analysis
co2l = 0.0
fpath_component_in = (
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
    )
component_df = pd.read_hdf(fpath_component_in, key="df")
props_clust_lvl = props_clust[props_clust.index.get_level_values("co2l") == co2l]
idx_clust_lvl = props_clust_lvl.index
component_df["co2l"] = co2l
component_df = component_df.set_index(["co2l", "time_stamp", "split_number"])
component_df_cluster = component_df.loc[idx_clust_lvl]
component_df_cluster.reset_index(inplace=True, drop=False)
#%%
# Keep only the row with largest load_share for each index group
idx = component_df_cluster.groupby(by=["co2l", "time_stamp", "split_number"])['load_share'].idxmax()
largest_component_df = component_df_cluster.loc[idx]
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

largest_component_df.load_share.hist(bins=30, log=True, ax=axs[0,0])
(largest_component_df.power_imbalance/1000).hist(bins=30, log=True, ax=axs[0,1])
(largest_component_df.rot_energy/1000).hist(bins=30, log=True, ax=axs[1,0])
largest_component_df.rocof.hist(bins=30, log=True, ax=axs[1,1])
axs[0,0].set_title("Load Share Distribution")
axs[0,1].set_title("Power Imbalance Distribution [GW]")
axs[1,0].set_title("Rotational Energy Distribution [GWs]")
axs[1,1].set_title("RoCoF Distribution [Hz/s]")
#%%
# co2l_cluster = props_clust.index.get_level_values("co2l").unique().values
# networks = {co2l: data_handling.load_pypsa_network(co2lvl=co2l, n_nodes=600) for co2l in co2l_cluster}
# co2l = co2l_cluster[0]
# n = networks[co2l]

co2l = 0.0
n = data_handling.load_pypsa_network(co2lvl=co2l, n_nodes=600)
snapshots = props_clust.xs(co2l, level="co2l").index.get_level_values("time_stamp").unique().values

#%%
all_snapshots = pd.to_datetime(n.loads_t["p"].index)
snapshot_to_idx = {snap: idx for idx, snap in enumerate(all_snapshots)}
snaps_clust = props_clust_lvl.index.get_level_values("time_stamp").value_counts().sort_index().index
idxs_snaps_clust = [snapshot_to_idx[snap] for snap in snaps_clust]
snap_counts_clust = props_clust_lvl.index.get_level_values("time_stamp").value_counts().sort_index().values
#%%
# intertia_cluster = inertia_time[idxs_snaps_clust]
#%%
(component_df[component_df.split_number_snapshot==1].groupby(component_df[component_df.split_number_snapshot==1].index.get_level_values("time_stamp")).rot_energy.sum()/1000).hist(bins=30, log=True)
#%%
from operator import itemgetter
# largest_component_sub = largest_component_df.loc[np.random.choice(largest_component_df.index, size=10000, replace=False)]
largest_component_sub = largest_component_df
inertia_time = (
    np.load(
        config.path_to_pre_outage_sclopf + f"inertia_time_series_all_co2ls_{n_nodes}.npy"
    )
    / 1000
)[-1,:]
largest_component_sub["time_stamp"] = pd.to_datetime(largest_component_sub["time_stamp"])
time_stamps = largest_component_sub["time_stamp"].tolist()
total_inertia_sub = np.array(inertia_time)[np.array(itemgetter(*time_stamps)(snapshot_to_idx))]
inertia_share_largest_component = largest_component_sub.rot_energy / (total_inertia_sub + 1e-8) / 1000
plt.scatter(largest_component_sub.power_imbalance/1000, largest_component_sub.rot_energy/1000, c=largest_component_sub.load_share, cmap="viridis", alpha=1, s=10)
plt.colorbar(label="Load Share of Component", orientation="vertical")
cbar_lims = (inertia_share_largest_component.min(), 1)
plt.clim(cbar_lims)
plt.xlabel("Power Imbalance [GW]")
plt.ylabel("Rotational Energy [GWs]")
xlims = plt.xlim()
ylims = plt.ylim()
# plt.colorbar(label="Counts")
x = np.array([0, largest_component_sub.power_imbalance.min()*2])
inertia_rocof_border = - 50 * np.array(x)/ 2
plt.plot(x/1000, inertia_rocof_border/1000, label="RoCoF Border", linewidth=0.5, color="red")
# fill stable area
plt.fill_between(
    x / 1000,
    inertia_rocof_border / 1000,
    ylims[1],
    color="lightgrey",
    alpha=1,
    zorder=0,
)
plt.fill_between(
    x / 1000,
    inertia_rocof_border / 1000,
    ylims[1],
    color="lightgrey",
    alpha=1,
    zorder=0,
)
plt.text(
    x=largest_component_sub.power_imbalance.max()/1000*1.1,
    y=largest_component_sub.rot_energy.max()/1000*0.95,
    s="Stable",
    horizontalalignment="right",
    verticalalignment="top",
    fontsize=14,
)
# fill rocof violation area
plt.fill_between(
    x / 1000,
    ylims[0],
    inertia_rocof_border / 1000,
    color="red",
    alpha=0.1,
    zorder=0,
)
plt.text(
    x=largest_component_sub.power_imbalance.min()/1000*0.9,
    y=largest_component_sub.rot_energy.max()/1000*0.95,
    s="RoCoF Violation",
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=14,
)
plt.xlim(xlims)
plt.ylim(ylims)
plt.title("Largest Component")
f_name = f"power_imbalance_vs_rot_energy_co2l{co2l}"
plot_style.save_figure(plt.gcf(), save_dir, f_name)
#%%
ax, f = plt.subplots(figsize=(6,6))
plt.scatter(largest_component_sub.load_share, inertia_share_largest_component, alpha=0.5, s=10)
plt.xlabel("Load Share of Component")
plt.ylabel("Inertia Share of Component")
plt.title("Largest Component")
xlims = plt.xlim()
ylims = plt.ylim()
plt.plot([0,1], [0,1], color="red", linestyle="-", linewidth=0.5)
plt.xlim(xlims)
plt.ylim(ylims)
f_name = f"load_share_vs_inertia_share_co2l{co2l}"
plot_style.save_figure(plt.gcf(), save_dir, f_name)


#%% analyse overfrequency nodes

node_cols_gen = [col for col in n.generators_t["p"].columns if any(str(node) in col for node in nodes_red)]
node_cols_store = n.storage_units.index[n.storage_units.bus.isin(nodes_red)].tolist()
gen_nodes = n.generators_t["p"].loc[snapshots, node_cols_gen]
store_nodes = n.storage_units_t["p"].loc[snapshots, node_cols_store]
gen_by_carrier = gen_nodes.groupby(gen_nodes.columns.str.split().str[-1], axis=1).sum().astype(int)
# remove columns with less than 1% average contribution
# gen_by_carrier = gen_by_carrier.loc[:,gen_by_carrier.max(axis=0) > gen_by_carrier.sum(axis=1).mean() * 0.01]
load = n.loads_t["p"].loc[snapshots, n.loads.index[n.loads.bus.isin(nodes_red)]].sum(axis=1)
storage_by_carrier = store_nodes.groupby(n.storage_units.loc[store_nodes.columns].carrier, axis=1).sum().astype(int)
# storage_by_carrier = storage_by_carrier.loc[:,storage_by_carrier.abs().max(axis=0) > storage_by_carrier.sum(axis=1).mean() * 0.05]
# storage_by_carrier
power_imbalance = gen_by_carrier.sum(axis=1) - storage_by_carrier.sum(axis=1) - load

inertia_time_cluster = inertia_time[np.isin(all_snapshots, snapshots)]

rocofs = (
                        50 * power_imbalance/1000 / ((inertia_time_cluster + 1e-8) * 2)
                    ) # /1000 because power imbalance and inertia in GW
#%%
relative_contribution = gen_by_carrier.div(gen_by_carrier.sum(axis=1), axis=0)
relative_contribution.describe()
#%%
# since solar contributes more than 95% on average we will not plot by carrier but only total generation and storage
#%%
# plot histograms of generation and storage power
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,4, figsize=(15,5))
# gen_store_vals = np.concatenate(
#     [gen_by_carrier.values.flatten(), storage_by_carrier.values.flatten()]
# ) / 1000
# shared_bins = np.histogram_bin_edges(gen_store_vals, bins=30)
# for col in gen_by_carrier.columns:
#     axs[0].hist(gen_by_carrier[col].values/1000, bins=shared_bins, alpha=0.5, label=col)
axs[2].hist(gen_by_carrier.sum(axis=1).values/1000, alpha=0.7, bins=30, label="Generation")
axs[2].set_ylabel("Counts")

# for col in storage_by_carrier.columns:
#     axs[2].hist(storage_by_carrier[col].values/1000, bins=shared_bins, alpha=0.5, label=col)
# axs[2].set_title("Generation")
# axs[2].legend()
axs[2].set_ylim(1, axs[2].get_ylim()[1])
axs[2].set_xlabel("Generation [GW]")

axs[0].hist(load.values/1000, bins=30, alpha=0.7)
# axs[0].set_title("Load")
# axs[0].set_xlabel("Load + Storage [GW]")
axs[0].set_xlabel("Load [GW]")
axs[0].set_ylabel("Counts")

axs[1].hist(storage_by_carrier.sum(axis=1).values/1000, bins=30, alpha=0.7)
# axs[1].set_title("Load")
axs[1].set_xlabel("Storage [GW]")
axs[1].set_ylabel("Counts")


axs[3].hist(power_imbalance.values/1000, bins=30, alpha=0.7)
# axs[3].set_title("Power imbalance")
axs[3].set_xlabel("Power imbalance [GW]")
axs[3].set_ylabel("Counts")

plt.tight_layout()
# axs[3].hist(inertia_time_cluster, bins=30, alpha=0.7, color="red")
# axs[3].set_title("Total Inertia")
# axs[3].set_xlabel("Inertia [s]")

# axs[4].hist(rocofs, bins=30, alpha=0.7, color="purple")
# axs[4].set_title("RoCoF distribution")
# axs[4].set_xlabel("RoCoF [Hz/s]")
f_name = f"gen_load_storage_imbalance_hist_co2l{co2l}"
plot_style.save_figure(fig, save_dir, f_name)
#%%
# Plot histograms of snapshots - daily and yearly profiles
fig, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[0.5, 1])

# Convert snapshots to pandas datetime if needed
snapshots_dt = pd.to_datetime(snaps_clust)
possible_hours= pd.to_datetime(all_snapshots).hour.unique()

# Daily profile - bar plot centered on ticks
hours = snapshots_dt.hour
hours_counts = hours.value_counts().sort_index()
axs[0].bar(hours_counts.index, hours_counts.values, width=0.9, edgecolor='black', alpha=0.7)
axs[0].set_xlabel('Hour of Day')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Daily Profile')
axs[0].set_xticks(possible_hours)
axs[0].set_xlim(possible_hours.min() - 0.5, possible_hours.max() + 0.5)
axs[0].grid(axis='y', alpha=0.3)

# Yearly profile - histogram of day of year
day_of_year = snapshots_dt.dayofyear
axs[1].hist(day_of_year, bins=len(snapshots_dt.dayofyear.unique()), range=(1, 366), edgecolor='black', alpha=0.7)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Yearly Profile')
# axs[1].set_xticks(np.sort(day_of_year.unique()))
axs[1].grid(axis='y', alpha=0.3)

# add suptitle
plt.suptitle(f"Temporal Distribution of Blackouts", fontsize=16, y=1.1)

plt.tight_layout()
f_name = f"blackout_daily_yearly_profile_co2l{co2l}"
plot_style.save_figure(plt.gcf(), save_dir, f_name)
# %%

# get installed generation capacity in the nodes of the cluster
gen_capacity = n.generators.loc[n.generators.bus.isin(nodes_red)].groupby(n.generators.loc[n.generators.bus.isin(nodes_red)].carrier).p_nom.sum().astype(int)
gen_capacity = gen_capacity[[carrier for carrier in gen_capacity.index if not "load" in carrier.lower()]]
gen_capacity[gen_capacity>gen_capacity.max()*0.01]
gen_capacity.loc["solar"]/1000

#%% get inertia distribution