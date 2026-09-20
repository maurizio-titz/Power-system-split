#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""this module analyses the blackout clusters. 
Especially, it investigates which nodes have an overfrequency blackout 
and what carrier was dominant before the blackout."""

import gzip
import pickle
import numpy as np
import sys
import os

import pickle
import pandas as pd

sys.path.append("..")  # To import from parent directory
sys.path.append("./")  # To import from parent directory

from utils import data_handling
from utils.clustering.blackout_clustering_class import Clustering
from utils.clustering.distance_metrics import geometric_mean
from utils.data_handling import get_co2_levels
from utils import config
from utils import plot_style

import matplotlib.pyplot as plt
import networkx as nx
from loguru import logger

from utils import subgraph_evaluation

plot_style.setup_matplotlib_style()

# Paths
from utils.config import path_to_sclopf_results

MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# a node counts as over-frequency if it ends up in an island with a RoCoF above
# ROCOF_THRESHOLD (the threshold of the "blackout" transformation, see
# utils.clustering.data_handling.transform_indicator_vectors) in more than
# OVER_FREQUENCY_SHARE of the cluster's splits
ROCOF_THRESHOLD = 1.0  # Hz/s
OVER_FREQUENCY_SHARE = 0.5


def plot_cluster_summary(
    gen_by_carrier,
    storage_by_carrier,
    load,
    power_imbalance,
    snaps_clust,
    all_snapshots,
    co2l,
    save_dir,
    cluster_number=None,
    f_name=None,
):
    """Combine generation by carrier, the temporal blackout profiles and the
    load/storage/generation/imbalance histograms into a single figure.

    Layout: the upper row holds two equally sized plots (the daily and yearly
    blackout profiles on the left, generation by carrier on the right), the lower
    row holds the four load/storage/generation/imbalance histograms. Each of the
    three parts carries its own heading.

    Args:
        gen_by_carrier: DataFrame of generation per carrier [MW], snapshots x carriers
        storage_by_carrier: DataFrame of storage dispatch per carrier [MW]
        load: Series of total load [MW] per snapshot
        power_imbalance: Series of generation + storage - load - HVDC export [MW] per
            snapshot
        snaps_clust: snapshots of the cluster (used for the temporal profiles)
        all_snapshots: all snapshots of the network (used for the hour ticks)
        co2l: CO2 level, only used for the file name
        save_dir: directory the figure is written to
        cluster_number: cluster the data belongs to, only used for the file name
        f_name: file name without extension (default: cluster_summary_cluster
            {cluster_number}_co2l{co2l})
    """
    # axis labels and titles are larger than the default, as the panels are small
    label_fs = plot_style.AXIS_LABEL_FONTSIZE + 4
    tick_fs = plot_style.TICK_LABEL_FONTSIZE + 2
    heading_fs = plot_style.TITLE_FONTSIZE + 6
    # the panel titles of the profile part are set in the same size as the labels
    title_fs = label_fs

    # subfigures instead of a plain gridspec, so each of the three parts can
    # carry its own heading and the upper two are separated more clearly
    fig = plt.figure(figsize=(15, 10.5), layout="constrained")
    # the bottom row carries no heading any more, so the rows are put right next to each
    # other; the padding of the constrained layout keeps them apart, which is why
    # negative values do not close the gap any further
    sfig_top, sfig_bottom = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.0)
    # the profile part gets more of the upper row than the carrier histogram
    sfig_profile, sfig_carrier = sfig_top.subfigures(
        1, 2, width_ratios=[1.2, 0.8], wspace=0.12
    )

    ax_daily, ax_yearly = sfig_profile.subplots(1, 2, width_ratios=[0.5, 1])
    ax_carrier = sfig_carrier.subplots()
    axs_bottom = list(sfig_bottom.subplots(1, 4))

    sfig_profile.suptitle("Temporal Distribution of Blackouts", fontsize=heading_fs)
    # the node selection is not named here, it belongs into the caption of the figure
    head_carrier = sfig_carrier.suptitle("Generation by Carrier", fontsize=heading_fs)
    # the bottom row carries no heading, its panels are labelled well enough by their
    # x labels

    # --- generation by carrier ---------------------------------------------
    bins = np.arange(0, gen_by_carrier.sum(axis=1).max() / 1000 * 1.2, step=0.5)
    for col in gen_by_carrier.columns:
        sign_vals = gen_by_carrier[col].values[gen_by_carrier[col].values >= 500]
        if len(sign_vals) == 0:
            continue
        ax_carrier.hist(
            gen_by_carrier[col].values / 1000,
            alpha=0.5,
            bins=bins,
            edgecolor="black",
            label=col.replace("_", r"\_"),
            weights=np.ones(len(gen_by_carrier[col])) / len(gen_by_carrier[col]),
        )
    ax_carrier.set_yscale("log")
    ax_carrier.legend(fontsize=plot_style.LEGEND_FONTSIZE + 3)
    ax_carrier.set_xlabel("Generation [GW]", fontsize=label_fs)
    ax_carrier.set_ylabel("Relative Frequency", fontsize=label_fs)

    # --- daily and yearly blackout profile ----------------------------------
    snapshots_dt = pd.to_datetime(snaps_clust)
    possible_hours = list(pd.to_datetime(all_snapshots).hour.unique()) + [24]

    hours_counts = snapshots_dt.hour.value_counts().sort_index()
    hours_rel = hours_counts.values / hours_counts.values.sum()
    ax_daily.bar(
        hours_counts.index + 1.5, hours_rel, width=3, edgecolor="black", alpha=0.7
    )
    ax_daily.set_title("Daily Profile", fontsize=title_fs)
    ax_daily.set_xlabel("Hour of Day", fontsize=label_fs)
    # only the left panel of the profile part carries the y label
    ax_daily.set_ylabel("Relative Frequency", fontsize=label_fs)
    ax_daily.set_xticks(possible_hours)
    ax_daily.set_xticklabels(possible_hours, rotation=90)
    ax_daily.set_xlim(0, 24)
    ax_daily.grid(axis="y", alpha=0.3)

    day_of_year = snapshots_dt.dayofyear
    ax_yearly.hist(
        day_of_year,
        bins=len(snapshots_dt.dayofyear.unique()),
        range=(1, 366),
        edgecolor="black",
        alpha=0.7,
        weights=np.ones(len(day_of_year)) / len(day_of_year),
    )
    ax_yearly.set_title("Yearly Profile", fontsize=title_fs)
    ax_yearly.set_xlabel("Month", fontsize=label_fs)
    ax_yearly.set_xticks(MONTH_STARTS)
    ax_yearly.set_xticklabels(MONTH_LABELS, rotation=90)
    ax_yearly.grid(axis="y", alpha=0.3)
    ax_yearly.set_xlim(1, 366)

    # --- load, storage, generation and power imbalance ----------------------
    panels = [
        (load.values / 1000, "Load [GW]"),
        # the sign convention of PyPSA is kept, so the label states it explicitly
        (storage_by_carrier.sum(axis=1).values / 1000, "Storage Dispatch [GW]"),
        (gen_by_carrier.sum(axis=1).values / 1000, "Generation [GW]"),
        (power_imbalance.values / 1000, "Power Imbalance [GW]"),
    ]
    for ax, (vals, xlabel) in zip(axs_bottom, panels):
        ax.hist(
            vals,
            bins=30,
            alpha=0.7,
            edgecolor="black",
            weights=np.ones(len(vals)) / len(vals),
        )
        ax.set_xlabel(xlabel, fontsize=label_fs)
    # only the left panel of the bottom row carries the y label
    axs_bottom[0].set_ylabel("Relative Frequency", fontsize=label_fs)

    # the panel labels follow the reading order of the figure
    for i, ax in enumerate([ax_daily, ax_yearly, ax_carrier] + axs_bottom):
        # the labels of the bottom row are lifted a bit, otherwise they touch the
        # uppermost y tick label of their panel
        plot_style.add_panel_label(
            ax, i, y_offset=0.05 if ax not in axs_bottom else 0.085
        )
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)

    # a suptitle is centred on its subfigure, but the y label and the tick labels only
    # take space on the left, so the panel itself sits right of the subfigure centre.
    # the heading is centred on the panel instead, once the layout has settled
    fig.canvas.draw()
    head_carrier.set_x(
        sfig_carrier.transSubfigure.inverted().transform(
            (ax_carrier.get_window_extent().intervalx.mean(), 0)
        )[0]
    )

    if f_name is None:
        cluster_part = "" if cluster_number is None else f"_cluster{cluster_number}"
        f_name = f"cluster_summary{cluster_part}_co2l{co2l}"
    # the subdirectory is set explicitly, otherwise it would be derived from the
    # file name and carry the cluster number as well
    plot_style.save_figure(
        fig, f_name, os.path.join(save_dir, "cluster_summary"), organize_plots=False
    )


def plot_largest_component_two_panel(
    largest_component,
    inertia_share,
    co2l,
    save_dir,
    f_name=None,
):
    """Plot the load/inertia share and the power imbalance/rotational energy of the
    largest component of each split next to each other.

    Layout: the left panel holds the load share against the inertia share together with
    the 1:1 line, the right panel holds the power imbalance against the rotational
    energy, coloured by the load share and shaded by the RoCoF border.

    Args:
        largest_component: DataFrame of the largest component of each split, holding
            load_share, power_imbalance [MW] and rot_energy [MWs]
        inertia_share: Series of the share of the system inertia the largest component
            carries, aligned with largest_component
        co2l: CO2 level, only used for the file name
        save_dir: directory the figure is written to
        f_name: file name without extension (default:
            largest_component_two_panel_co2l{co2l})
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1 (left): Load share vs inertia share
    axs[0].scatter(
        largest_component.load_share,
        inertia_share,
        alpha=0.5,
        s=10,
    )
    axs[0].set_xlabel("Load Share of Component")
    axs[0].set_ylabel("Inertia Share of Component")
    xlims = axs[0].get_xlim()
    ylims = axs[0].get_ylim()
    axs[0].plot([0, 1], [0, 1], color="red", linestyle="-", linewidth=0.5)
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)

    # Plot 2 (right): Power imbalance vs rotational energy
    sc = axs[1].scatter(
        largest_component.power_imbalance / 1000,
        largest_component.rot_energy / 1000,
        c=largest_component.load_share,
        cmap="viridis",
        alpha=1,
        s=10,
    )
    fig.colorbar(sc, ax=axs[1], label="Load Share of Component", orientation="vertical")
    cbar_lims = (inertia_share.min(), 1)
    sc.set_clim(cbar_lims)
    axs[1].set_xlabel("Power Imbalance [GW]")
    axs[1].set_ylabel("Rotational Energy [GWs]")
    xlims = axs[1].get_xlim()
    ylims = axs[1].get_ylim()
    x = np.array([0, largest_component.power_imbalance.min() * 2])
    inertia_rocof_border = -50 * np.array(x) / 2
    axs[1].plot(
        x / 1000,
        inertia_rocof_border / 1000,
        label="RoCoF Border",
        linewidth=0.5,
        color="red",
    )
    axs[1].fill_between(
        x / 1000,
        inertia_rocof_border / 1000,
        ylims[1],
        color="lightgrey",
        alpha=1,
        zorder=0,
    )
    axs[1].text(
        x=largest_component.power_imbalance.max() / 1000 * 1.1,
        y=largest_component.rot_energy.max() / 1000 * 0.95,
        s="Stable",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=14,
    )
    axs[1].fill_between(
        x / 1000,
        ylims[0],
        inertia_rocof_border / 1000,
        color="red",
        alpha=0.1,
        zorder=0,
    )
    axs[1].text(
        x=largest_component.power_imbalance.min() / 1000 * 0.9,
        y=largest_component.rot_energy.max() / 1000 * 0.95,
        s="RoCoF Violation",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=14,
    )
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)

    for i, ax in enumerate(axs):
        plot_style.add_panel_label(ax, i)

    plt.tight_layout()

    if f_name is None:
        f_name = f"largest_component_two_panel_co2l{co2l}"
    plot_style.save_figure(fig, f_name, save_dir)


if __name__ == "__main__":
    for cluster_number in [11, 12]:
        # sort_by = "n_samples"
        sort_by = "weighted_lost_load"

        save_dir = os.path.join(
            config.path_to_figures_sclopf, 
            f"analysis_cluster_{cluster_number}"
        )
   
        fpath = path_to_sclopf_results + \
            "/clustering/rocof_blackout_Co2L0.6_0.5_0.4_0.3_0.2_0.1_0.05_0.0_n600_lls0.1/clustering_results/agg_params_01bf408e.pklz"
        with gzip.open(fpath, "rb") as f:
            res = pickle.load(f)

        with gzip.open(fpath.replace(".pklz", "_centroid_res.pklz"), "rb") as f:
            cluster_res = pickle.load(f)

        # load clustering results
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
    
        cluster_idx = (
            cluster_res["centroids_df"]
            .sort_values(by=sort_by, ascending=False)
            .iloc[cluster_number - 1, :]
            .name
        )
        centroid = cluster_res["centroids"][cluster_idx]
        edge_centroids = cluster_res["edge_centroids"][cluster_idx]
        failed_edges_prob = edge_centroids.copy()
        # plot centroid
        plot_cluster = False
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

        edge_cmap = plt.get_cmap("inferno_r")
        edge_cmap = truncate_colormap(edge_cmap, 0.1, 0.95, 1000)
        edge_cmap.set_under("gainsboro", 1.0)
        if plot_cluster:
            plot_order = np.argsort(centroid[:, 2])  # Sort by the value of the third class
            colors_classes = ["blue", "lightgray", "red"]
            f, ax = plt.subplots(figsize=(8, 8))
            nx_graph = data_handling.load_networkx_graph(snet_index='0', 
                                                         co2lvl=0.0)
            pos = nx.get_node_attributes(nx_graph, "pos")
            nodes = np.array(list(nx_graph.nodes()))
            for i_node in plot_order:
                node = nodes[i_node]
                ax.pie(
                    centroid[i_node, :],
                    colors=colors_classes,
                    radius=0.2,
                    center=(pos[node][0], pos[node][1]),
                )
                pos_arr = np.array(list(pos.values()))
                ax.set_xlim(pos_arr[:, 0].min() - 1, pos_arr[:, 0].max() + 1)
                ax.set_ylim(pos_arr[:, 1].min() - 1, pos_arr[:, 1].max() + 1)
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
                    cmap=edge_cmap,
                    norm=mplcolors.LogNorm(vmin=vmin_edge_orig, vmax=vmax_edge_orig),
                )
            else:
                sm_edge = plt.cm.ScalarMappable(
                    cmap=edge_cmap,
                    norm=mplcolors.Normalize(vmin=vmin_edge_orig, vmax=vmax_edge_orig),
                )
            cb_edge = f.colorbar(sm_edge, ax=ax)
            # plt.colorbar(sm_edges, ax=ax, label="Edge failure probability")

        # the third class of the centroid holds the weighted share of the cluster's splits
        # in which the node ended up in an island with a RoCoF above ROCOF_THRESHOLD
        node_idxs_red = np.where(centroid[:, 2] > OVER_FREQUENCY_SHARE)
        nx_graph = data_handling.load_networkx_graph(snet_index='0',
                                                     co2lvl=0.0)
        nodes_red = np.array(nx_graph.nodes)[node_idxs_red]
        logger.info(
            f"cluster {cluster_number}: {len(nodes_red)} of {centroid.shape[0]} nodes of the "
            f"AC subnetwork have a RoCoF > {ROCOF_THRESHOLD:g} Hz/s in more than "
            f"{OVER_FREQUENCY_SHARE:.0%} of the cluster's splits "
            f"({', '.join(nodes_red)}; shares "
            f"{np.round(centroid[node_idxs_red, 2], 3).ravel().tolist()})"
        )
        cluster_mask = cluster_res["group_masks"][cluster_idx]
        split_properties = pd.read_hdf(
            f"{cl.data_dir}/data_filtered.h5",
            key="split_props",
        )
        props_clust = split_properties.loc[cluster_mask]

        # restrict to single co2 level for analysis
        co2l = 0.0
        fpath_component_in_all = config.path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
        component_df = pd.read_hdf(fpath_component_in_all, key="df")
        component_df = component_df[component_df.co2l == co2l]
        component_df.reset_index(inplace=True, drop=True)
    
        props_clust_lvl = props_clust[props_clust.index.get_level_values("co2l") == co2l]
        idx_clust_lvl = props_clust_lvl.index
        component_df = component_df.set_index(["co2l", "time_stamp", "split_number"])
        component_df_cluster = component_df.loc[idx_clust_lvl]
        component_df_cluster.reset_index(inplace=True, drop=False)

        # Keep only the row with largest load_share for each index group
        idx = component_df_cluster.groupby(by=["co2l", "time_stamp", "split_number"])[
            "load_share"
        ].idxmax()
        largest_component_df = component_df_cluster.loc[idx]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        largest_component_df.load_share.hist(
            bins=30,
            log=True,
            ax=axs[0, 0],
            weights=np.ones(len(largest_component_df)) / len(largest_component_df),
        )
        (largest_component_df.power_imbalance / 1000).hist(
            bins=30,
            log=True,
            ax=axs[0, 1],
            weights=np.ones(len(largest_component_df)) / len(largest_component_df),
        )
        (largest_component_df.rot_energy / 1000).hist(
            bins=30,
            log=True,
            ax=axs[1, 0],
            weights=np.ones(len(largest_component_df)) / len(largest_component_df),
        )
        largest_component_df.rocof.hist(
            bins=30,
            log=True,
            ax=axs[1, 1],
            weights=np.ones(len(largest_component_df)) / len(largest_component_df),
        )
        axs[0, 0].set_title("Load Share Distribution")
        axs[0, 1].set_title("Power Imbalance Distribution [GW]")
        axs[1, 0].set_title("Rotational Energy Distribution [GWs]")
        axs[1, 1].set_title("RoCoF Distribution [Hz/s]")
        for ax in axs.ravel():
            ax.set_ylabel("Relative Frequency")

        co2l = 0.0
        n = data_handling.load_pypsa_network(co2lvl=co2l, n_nodes=600)
        snapshots = (
            props_clust.xs(co2l, level="co2l")
            .index.get_level_values("time_stamp")
            .unique()
            .values
        )
        logger.info(
            f"cluster {cluster_number}: {len(snapshots)} snapshots at co2l={co2l}"
        )

        all_snapshots = pd.to_datetime(n.loads_t["p"].index)
        snapshot_to_idx = {snap: idx for idx, snap in enumerate(all_snapshots)}
        snaps_clust = (
            props_clust_lvl.index.get_level_values("time_stamp")
            .value_counts()
            .sort_index()
            .index
        )
        idxs_snaps_clust = [snapshot_to_idx[snap] for snap in snaps_clust]
        snap_counts_clust = (
            props_clust_lvl.index.get_level_values("time_stamp")
            .value_counts()
            .sort_index()
            .values
        )

        rot_energy_series = (
            component_df[component_df.split_number_snapshot == 1]
            .groupby(
                component_df[
                    component_df.split_number_snapshot == 1
                ].index.get_level_values("time_stamp")
            )
            .rot_energy.sum()
            / 1000
        )

        from operator import itemgetter

        largest_component_sub = largest_component_df
        inertia_time = (
            np.load(
                config.path_to_pre_outage_sclopf
                + f"/inertia_time_series_all_co2ls_{n_nodes}.npy"
            )
            / 1000
        )[-1, :]
        largest_component_sub["time_stamp"] = pd.to_datetime(
            largest_component_sub["time_stamp"]
        )
        time_stamps = largest_component_sub["time_stamp"].tolist()
        total_inertia_sub = np.array(inertia_time)[
            np.array(itemgetter(*time_stamps)(snapshot_to_idx))
        ]
        inertia_share_largest_component = (
            largest_component_sub.rot_energy / (total_inertia_sub + 1e-8) / 1000
        )

        plot_largest_component_two_panel(
            largest_component=largest_component_sub,
            inertia_share=inertia_share_largest_component,
            co2l=co2l,
            save_dir=save_dir,
        )

        # analyse overfrequency nodes

        # the generators are picked by their bus attribute, not by a substring of the
        # column name: a bus name such as "DE1 1" is a prefix of "DE1 19", so a substring
        # match would pull in the generators of the other nodes as well
        node_cols_gen = n.generators.index[n.generators.bus.isin(nodes_red)].tolist()
        node_cols_store = n.storage_units.index[
            n.storage_units.bus.isin(nodes_red)
        ].tolist()
        gen_nodes = n.generators_t["p"].loc[snapshots, node_cols_gen]
        store_nodes = n.storage_units_t["p"].loc[snapshots, node_cols_store]
        gen_by_carrier = (
            gen_nodes.groupby(gen_nodes.columns.str.split().str[-1], axis=1)
            .sum()
            .astype(int)
        )
        # remove columns with less than 1% average contribution
        # gen_by_carrier = gen_by_carrier.loc[:,gen_by_carrier.max(axis=0) > gen_by_carrier.sum(axis=1).mean() * 0.01]
        load = (
            n.loads_t["p"]
            .loc[snapshots, n.loads.index[n.loads.bus.isin(nodes_red)]]
            .sum(axis=1)
        )
        storage_by_carrier = (
            store_nodes.groupby(n.storage_units.loc[store_nodes.columns].carrier, axis=1)
            .sum()
            .astype(int)
        )
        # storage_by_carrier = storage_by_carrier.loc[:,storage_by_carrier.abs().max(axis=0) > storage_by_carrier.sum(axis=1).mean() * 0.05]
        # storage_by_carrier
        # net power leaving the nodes through HVDC links; links without a time series carry
        # no power and are filled with zero
        links_red = n.links[n.links.bus0.isin(nodes_red) | n.links.bus1.isin(nodes_red)]
        p0_red = n.links_t["p0"].reindex(columns=links_red.index, fill_value=0.0).loc[
            snapshots
        ]
        p1_red = n.links_t["p1"].reindex(columns=links_red.index, fill_value=0.0).loc[
            snapshots
        ]
        hvdc_export = (
            p0_red.loc[:, links_red.bus0.isin(nodes_red)].sum(axis=1)
            + p1_red.loc[:, links_red.bus1.isin(nodes_red)].sum(axis=1)
        )
        if links_red.empty:
            logger.info(f"cluster {cluster_number}: no HVDC links connect to these nodes")
        else:
            logger.info(
                f"cluster {cluster_number}: {len(links_red)} HVDC links connect to these "
                f"nodes ({', '.join(links_red.index)}), max net export "
                f"{hvdc_export.abs().max() / 1000:.2f} GW"
            )

        # storage_units_t.p is discharge-positive and hvdc_export is the power leaving the
        # nodes, following subgraph_evaluation.calc_power_imbalance_subgraph
        power_imbalance = (
            gen_by_carrier.sum(axis=1)
            + storage_by_carrier.sum(axis=1)
            - load
            - hvdc_export
        )

        inertia_time_cluster = inertia_time[np.isin(all_snapshots, snapshots)]

        rocofs = (
            50 * power_imbalance / 1000 / ((inertia_time_cluster + 1e-8) * 2)
        )  # /1000 because power imbalance and inertia in GW

        relative_contribution = gen_by_carrier.div(gen_by_carrier.sum(axis=1), axis=0)
        relative_contribution.describe()

        # since solar contributes more than 95% on average we will not plot by carrier but only total generation and storage

        # plot histograms of generation and storage power
        bins = np.arange(0, gen_by_carrier.sum(axis=1).max() / 1000 * 1.2, step=0.5)

        fig, ax = plt.subplots()
        for col in gen_by_carrier.columns:
            sign_vals = gen_by_carrier[col].values[gen_by_carrier[col].values >= 500]
            if len(sign_vals) == 0:
                continue
            ax.hist(
                gen_by_carrier[col].values / 1000,
                alpha=0.5,
                bins=bins,
                label=col,
                weights=np.ones(len(gen_by_carrier[col])) / len(gen_by_carrier[col]),
            )
        plt.yscale("log")
        ax.legend()
        ax.set_xlabel("Generation [GW]")
        ax.set_ylabel("Relative Frequency")
        plt.title("Generation by Carrier")
        f_name = f"generation_by_carrier_hist_co2l{co2l}"
        plot_style.save_figure(fig, f_name, save_dir)

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        # gen_store_vals = np.concatenate(
        #     [gen_by_carrier.values.flatten(), storage_by_carrier.values.flatten()]
        # ) / 1000
        # shared_bins = np.histogram_bin_edges(gen_store_vals, bins=30)
        # for col in gen_by_carrier.columns:
        #     axs[0].hist(gen_by_carrier[col].values/1000, bins=shared_bins, alpha=0.5, label=col)
        gen_total_vals = gen_by_carrier.sum(axis=1).values / 1000
        axs[2].hist(
            gen_total_vals,
            alpha=0.7,
            bins=30,
            label="Generation",
            weights=np.ones(len(gen_total_vals)) / len(gen_total_vals),
        )
        axs[2].set_ylabel("Relative Frequency")

        # for col in storage_by_carrier.columns:
        #     axs[2].hist(storage_by_carrier[col].values/1000, bins=shared_bins, alpha=0.5, label=col)
        # axs[2].set_title("Generation")
        # axs[2].legend()
        # axs[2].set_ylim(1, axs[2].get_ylim()[1])
        axs[2].set_xlabel("Generation [GW]")

        load_vals = load.values / 1000
        axs[0].hist(
            load_vals,
            bins=30,
            alpha=0.7,
            weights=np.ones(len(load_vals)) / len(load_vals),
        )
        # axs[0].set_title("Load")
        # axs[0].set_xlabel("Load + Storage [GW]")
        axs[0].set_xlabel("Load [GW]")
        axs[0].set_ylabel("Relative Frequency")

        storage_vals = storage_by_carrier.sum(axis=1).values / 1000
        axs[1].hist(
            storage_vals,
            bins=30,
            alpha=0.7,
            weights=np.ones(len(storage_vals)) / len(storage_vals),
        )
        # axs[1].set_title("Load")
        axs[1].set_xlabel("Storage [GW]")
        axs[1].set_ylabel("Relative Frequency")

        imbalance_vals = power_imbalance.values / 1000
        axs[3].hist(
            imbalance_vals,
            bins=30,
            alpha=0.7,
            weights=np.ones(len(imbalance_vals)) / len(imbalance_vals),
        )
        # axs[3].set_title("Power imbalance")
        axs[3].set_xlabel("Power imbalance [GW]")
        axs[3].set_ylabel("Relative Frequency")

        plt.tight_layout()
        # axs[3].hist(inertia_time_cluster, bins=30, alpha=0.7, color="red")
        # axs[3].set_title("Total Inertia")
        # axs[3].set_xlabel("Inertia [s]")

        # axs[4].hist(rocofs, bins=30, alpha=0.7, color="purple")
        # axs[4].set_title("RoCoF distribution")
        # axs[4].set_xlabel("RoCoF [Hz/s]")
        f_name = f"gen_load_storage_imbalance_hist_co2l{co2l}"
        plot_style.save_figure(fig, f_name, save_dir)
        # 
        # Plot histograms of snapshots - daily and yearly profiles
        fig, axs = plt.subplots(1, 2, figsize=(6, 5), width_ratios=[0.5, 1])

        # Convert snapshots to pandas datetime if needed
        snapshots_dt = pd.to_datetime(snaps_clust)
        possible_hours = list(pd.to_datetime(all_snapshots).hour.unique()) + [24]

        # Daily profile - bar plot centered on ticks
        hours = snapshots_dt.hour
        hours_counts = hours.value_counts().sort_index()
        hours_rel = hours_counts.values / hours_counts.values.sum()
        axs[0].bar(
            hours_counts.index + 1.5, hours_rel, width=3, edgecolor="black", alpha=0.7
        )
        axs[0].set_xlabel("Hour of Day")
        axs[0].set_ylabel("Relative Frequency")
        axs[0].set_title("Daily Profile")
        axs[0].set_xticks(possible_hours)
        # rotate x-axis labels
        axs[0].set_xticklabels(possible_hours, rotation=90)
        axs[0].set_xlim(0, 24)
        axs[0].grid(axis="y", alpha=0.3)

        # Yearly profile - histogram of day of year
        day_of_year = snapshots_dt.dayofyear
        axs[1].hist(
            day_of_year,
            bins=len(snapshots_dt.dayofyear.unique()),
            range=(1, 366),
            edgecolor="black",
            alpha=0.7,
            weights=np.ones(len(day_of_year)) / len(day_of_year),
        )
        # axs[1].set_xlabel('Day of Year')
        axs[1].set_ylabel("Relative Frequency")
        axs[1].set_title("Yearly Profile")
        # Set month ticks
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        axs[1].set_xticks(month_starts)
        # rotate x-axis labels
        axs[1].set_xticklabels(month_labels, rotation=90)
        axs[1].grid(axis="y", alpha=0.3)
        axs[1].set_xlim(1, 366)

        # add suptitle
        plt.suptitle(f"Temporal Distribution of Blackouts", fontsize=16, y=0.93)

        plt.tight_layout()
    
        f_name = f"blackout_daily_yearly_profile_co2l{co2l}"
        plot_style.save_figure(plt.gcf(), f_name, save_dir)

        # combined figure of the three plots above
        plot_cluster_summary(
            gen_by_carrier=gen_by_carrier,
            storage_by_carrier=storage_by_carrier,
            load=load,
            power_imbalance=power_imbalance,
            snaps_clust=snaps_clust,
            all_snapshots=all_snapshots,
            co2l=co2l,
            save_dir=save_dir,
            cluster_number=cluster_number,
        )

        # get installed generation capacity in the nodes of the cluster
        gen_capacity = (
            n.generators.loc[n.generators.bus.isin(nodes_red)]
            .groupby(n.generators.loc[n.generators.bus.isin(nodes_red)].carrier)
            .p_nom.sum()
            .astype(int)
        )
        gen_capacity = gen_capacity[
            [carrier for carrier in gen_capacity.index if not "load" in carrier.lower()]
        ]
        gen_capacity = gen_capacity[gen_capacity > gen_capacity.max() * 0.01]
        gen_capacity


        # inertia contribution by snapshot (full network)
    

        participation_threshold = 0.05
        load_inertia_constant = 0.4

        inertia_by_snapshot_rows = []
        snapshot_index = pd.to_datetime(snapshots)

        for snap in snapshots:
            current_generation = n.generators_t.p.loc[snap].mul(n.generators.sign).copy()
            current_storage = n.storage_units_t.p.loc[snap]
            current_load = n.loads_t.p.loc[snap]

            inertia_by_type = subgraph_evaluation.get_inertia_by_type_subgraph(
                None,
                n.generators,
                current_generation,
                n.storage_units,
                current_storage,
                n.loads,
                current_load,
                participation_threshold=participation_threshold,
                load_inertia_constant=load_inertia_constant,
            )
            inertia_by_snapshot_rows.append(inertia_by_type)

        inertia_by_snapshot_df = pd.DataFrame(
            inertia_by_snapshot_rows, index=snapshot_index
        ).fillna(0.0)
        inertia_by_snapshot_df["total"] = inertia_by_snapshot_df.sum(axis=1)
        # 
        inertia_by_snapshot_relative = inertia_by_snapshot_df.div(
            inertia_by_snapshot_df.total, axis=0
        )
        no_inertia_cols = inertia_by_snapshot_relative.columns[
            inertia_by_snapshot_relative.sum(axis=0) == 0
        ]
        inertia_by_snapshot_relative = inertia_by_snapshot_relative.drop(
            columns=no_inertia_cols
        )
        inertia_by_snapshot_relative.describe()
        # 
        inertia_rel_clust = inertia_by_snapshot_relative.loc[snapshots_dt]
        inertia_rel_clust.describe()
        # 
        # (inertia_by_snapshot_relative.H2 * 100).hist()
        # plt.title("Relative inertia contribution of H2 storage")
        # plt.xlabel("Relative inertia contribution [\%]")
        # plt.ylabel("Frequency")
        # plot_style.save_figure(plt.gcf(), "h2_inertia_contribution_hist", save_dir)
        # 
        # (inertia_by_snapshot_relative.H2 * 100).hist()
        # plt.title(f"Relative inertia contribution of H2 storage cluster {cluster_number}")
        # plt.xlabel("Relative inertia contribution [\%]")
        # plt.ylabel("Frequency")
        # plot_style.save_figure(
        #     plt.gcf(),
        #     f"h2_inertia_contribution_hist_cluster{cluster_number}",
        #     save_dir,
        # )
