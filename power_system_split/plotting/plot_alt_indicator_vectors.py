#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot details on the indicators values"""

import os

import pandas as pd
import numpy as np

import copy

import gzip
import pickle

from glob import glob
from tqdm import tqdm

import cartopy.crs as cartopy_crs
import networkx as nx

from utils.data_handling import (
    load_pypsa_network_from_path,
    build_networkx_graph,
    matrix_indices_to_nx_edges,
    nx_edges_to_matrix_indices,
    get_matrices_from_nx_graph,
)
from utils.cascade_simulation import calc_possible_double_line_failures

import matplotlib

matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 24,
        "axes.labelsize": 24,
        "axes.titlesize": 28,
        "figure.titlesize": 28,
    }
)
matplotlib.rcParams["text.usetex"] = True
from matplotlib import pyplot as plt

out_data_path = "results/sclopf/split_visualization/"
if not os.path.exists(out_data_path):
    os.mkdir(out_data_path)


def draw_map(
    graph: nx.Graph, xlims: tuple = (-11, 29), ylims: tuple = (36, 59), save_prefix=None
):

    ax = plt.axes(projection=cartopy_crs.PlateCarree())
    ax.coastlines()
    fig = plt.gcf()

    pos_nx = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_nodes(graph, pos_nx, node_size=20, ax=ax)
    nx.draw_networkx_edges(graph, pos_nx, width=2.0, ax=ax)

    # Aesthetics
    ax.set_xlim(left=min(xlims), right=max(xlims))
    ax.set_ylim(bottom=min(ylims), top=max(ylims))

    ## Remove background color and frame
    ax.axis("off")

    if save_prefix is None:
        plt.show()
    else:
        fig_path = "plots/" + save_prefix
        fig.savefig(fig_path, bbox_inches="tight", transparent=True)

        fig.clear()
        plt.close(fig)

    return


def calc_likelihood_failure_alt_indicator(
    co2_lvl: float, n_nodes: int, save_it: bool = True, verbose: bool = False
):
    """Find the likelihood of a link failing by using the indicators vectors"""

    # Load PyPSA and generate networkx graph
    pypsa_net = load_pypsa_network_from_path(
        co2_lvl, n_nodes, "data/European_networks_sclopf/"
    )
    nx_graph = build_networkx_graph(pypsa_net, snet_index=0)  # snet_idx is Europe

    # Find number of total experiments
    _, _, num_paralles, _ = get_matrices_from_nx_graph(nx_graph)
    bridge_idxs = nx_edges_to_matrix_indices(nx.bridges(nx_graph), nx_graph)
    n_2_failures = calc_possible_double_line_failures(
        num_paralles, ignored_idxs=bridge_idxs
    )
    snapshot_weights = pypsa_net.snapshot_weightings.generators
    number_total_experiments = int(len(n_2_failures) * snapshot_weights.sum())
    if verbose:
        print(f"Total Nr. Experiments: {number_total_experiments}")

    # Load edge split indicator vector
    fpath_indicator_vector = (
        "results/sclopf/indicator_vectors_rocof_lshare_edges/"
        + "indicator_vector_failed_edges_Co2L"
        + "f{co2_lvl}_n{n_nodes}.pklz"
    )
    with gzip.open(fpath_indicator_vector) as fh_indi_in:
        [edge_names_ls, edge_index_ls, index_tuple_splits, failed_edges_arr] = (
            pickle.load(fh_indi_in)
        )

    failed_edges_arr = np.array(failed_edges_arr, dtype=float)

    time_stamps = pd.to_datetime([xx[0] for xx in index_tuple_splits])
    unique_times = np.unique(time_stamps)

    for uni_time_r in unique_times:
        snapshot_weight_r = snapshot_weights[uni_time_r]
        idxs_time = np.where(time_stamps == uni_time_r)[0]
        failed_edges_arr[idxs_time] *= snapshot_weight_r

    probabilities = failed_edges_arr.sum(axis=0)
    probabilities /= number_total_experiments
    # idx_nx_edges = nx_edges_to_matrix_indices(edge_names_ls, nx_graph)

    # Output dictionaries with likelihood as values and links as keys
    # likelihood_primary = {(uu, vv): 0.0 for uu, vv in nx_graph.edges()}
    likelihood_secondary = {(uu, vv): 0.0 for uu, vv in nx_graph.edges()}

    for idx, edge_name_r in enumerate(edge_names_ls):
        likelihood_secondary[edge_name_r] = probabilities[idx]

    if save_it:
        fpath_out = (
            out_data_path
            + f"/indicator_vector_edge_likelihood_co2l{co2_lvl}_n{n_nodes}.pklz"
        )
        with gzip.open(fpath_out, "wb") as fh_out:
            pickle.dump(likelihood_secondary, fh_out)

    return likelihood_secondary


def calc_all_secondary_likelihoods(n_nodes=400, verbose: bool = True):

    glob_pypsa_search_str = (
        "data/European_networks_sclopf/" + "sclopf-elec_s_{0}*.nc".format(n_nodes)
    )
    pypsa_file_ls = glob(glob_pypsa_search_str)
    co2_lvl_ls = [float(xx.split("Co2L")[-1].split("-")[0]) for xx in pypsa_file_ls]
    pbar = tqdm(co2_lvl_ls, disable=not verbose)
    for co2_r in pbar:
        pbar.set_description(f"Processing CO2_lvl {co2_r:.2f}")
        if co2_r > 0.0:
            calc_likelihood_failure_alt_indicator(co2_r, n_nodes)

    return


def plot_probability_secondary_edge_failures(
    co2_lvl_list: tuple = (0.8, 0.6, 0.4, 0.1),
    vlims: tuple = (-5, -3),
    save_fig: bool = False,
    plot_diff_to_orig: bool = True,
):
    """Plot the likelihood of link triggering an initial failure."""

    # proj = cartopy_crs.PlateCarree()
    if plot_diff_to_orig:

        fig, [ax, ax_old, ax_diff] = plt.subplots(
            3, len(co2_lvl_list), sharex="all", sharey="all", figsize=(16, 6)
        )
    #                       subplot_kw={'projection': proj},
    #                       gridspec_kw={})

    else:
        fig, ax = plt.subplots(
            1, len(co2_lvl_list), sharex="all", sharey="all", figsize=(32, 8)
        )

    cmap = copy.copy(matplotlib.colormaps["viridis_r"])
    cmap.set_under("gainsboro", 1.0)

    cmap_diff = copy.copy(matplotlib.colormaps["inferno_r"])
    cmap_diff.set_under("gainsboro", 1.0)
    cmap_diff.set_over("red", 1.0)

    pypsa_net = load_pypsa_network_from_path(0.1, 400, "data/European_networks_sclopf/")
    nx_graph = build_networkx_graph(pypsa_net, snet_index=0)
    node_pos = nx.get_node_attributes(nx_graph, "pos")

    # Load old secondary edge_failures
    with open(
        "results/sclopf/split_visualization/"
        + "edge_likelihoods_secondary_all_"
        + "co2ls_n400.pickle",
        "rb",
    ) as fh_old_in:
        old_likelihoods_secondary = pickle.load(fh_old_in)

    for idx_co2, co2_lvl_r in enumerate(co2_lvl_list):
        ax_r = ax[idx_co2]
        ax_r_old = ax_old[idx_co2]
        ax_r_diff = ax_diff[idx_co2]

        # load probability
        fpath_out = (
            out_data_path
            + f"/indicator_vector_edge_likelihood_co2l{co2_lvl_r}_n400.pklz"
        )
        with gzip.open(fpath_out, "rb") as fh_in:
            like_secondary = pickle.load(fh_in)

        # Draw networkx
        color_edges = [
            (
                np.log10(like_secondary[edge_r])
                if like_secondary[edge_r] > 1e-12
                else -np.inf
            )
            for edge_r in nx_graph.edges()
        ]

        nx.draw_networkx_edges(
            nx_graph,
            node_pos,
            width=2,
            edge_cmap=cmap,
            edge_vmin=min(vlims),
            edge_vmax=max(vlims),
            edge_color=color_edges,
            ax=ax_r,
        )

        # ax_r.set_title(f"$CO_2$--level {co2_lvl_r*100}\%", size=24)

        ## Old results
        old_like_sec_co2_r = old_likelihoods_secondary[np.round(co2_lvl_r, decimals=1)]
        old_color_edges = [
            (
                np.log10(old_like_sec_co2_r[edge_r])
                if old_like_sec_co2_r[edge_r] > 1e-12
                else -np.inf
            )
            for edge_r in nx_graph.edges()
        ]
        nx.draw_networkx_edges(
            nx_graph,
            node_pos,
            width=2,
            edge_cmap=cmap,
            edge_vmin=min(vlims),
            edge_vmax=max(vlims),
            edge_color=old_color_edges,
            ax=ax_r_old,
        )

        diff_color_edges = [
            np.log10(abs(old_like_sec_co2_r[edge_r] - like_secondary[edge_r]))
            # if np.abs(old_like_sec_co2_r[edge_r] - like_secondary[edge_r]) > 1e-12 else -np.inf
            for edge_r in nx_graph.edges()
        ]
        nx.draw_networkx_edges(
            nx_graph,
            node_pos,
            width=2,
            edge_cmap=cmap_diff,
            edge_color=diff_color_edges,
            ax=ax_r_diff,
            edge_vmin=-6,
            edge_vmax=-2,
        )

    # Aesthetics
    for ax_r in np.array([ax, ax_old, ax_diff]).flatten():
        ax_r.axis("off")

    offset_label = -0.8
    ypos_label = 0.5
    ax[0].text(offset_label, ypos_label, "Indicator Vector", transform=ax[0].transAxes)
    ax_old[0].text(
        offset_label, ypos_label, "Previous Results", transform=ax_old[0].transAxes
    )
    ax_diff[0].text(
        offset_label, ypos_label, "Difference", transform=ax_diff[0].transAxes
    )

    fig.subplots_adjust(left=0.15, right=0.95)

    ## Co2 labels
    for idx_r, ax_r in enumerate(ax_diff):
        ax_r.text(
            0.5,
            -0.2,
            f"{co2_lvl_list[idx_r]*100} \%",
            transform=ax_r.transAxes,
            horizontalalignment="center",
        )

    if save_fig:
        fig_path = "alt_indicator_vec_prob_secondar_failure.png"

        fig.savefig(fig_path, bbox_inches="tight")
        fig.clear()

        plt.close(fig)

    else:
        plt.show()

    return diff_color_edges


def plot_histogram_number_failed_links(co2_lvl: float, save_it: bool = True):
    """Plot the histogram of number failed links."""

    # Load data

    # Find histogram

    # Plot it

    return


def plot_rocof_failure_map():

    # Load data

    return
