#!/usr/bin/env python3
"""
Combined figure: Flow & Inertia (top row) + Split Statistics (bottom row).
"""

import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.append("./")

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_pre_outage_sclopf,
    path_to_vis_results_sclopf,
)
from utils import data_handling, cascade_simulation
from utils.plot_style import (
    setup_matplotlib_style,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure,
)


def create_combined_flow_and_split_statistics_plot(
    mean_distance: bool = False,
    load_normalization: bool = False,
    show_blackout_stats: bool = True,
    secondary_proba_axis: bool = True,
    use_steps_load_loss: bool = False,
    n_nodes: int = 600,
    use_equal_panels: bool = False,
    save_prefix: str | None = None):
    """Create combined figure with flow/inertia (top) and 
    split statistics (bottom)."""

    # Setup
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)

    # Load network and data (flow/inertia)
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index='0')

    co2ls = get_co2_levels(n_nodes)
    networks = {
        co2l: data_handling.load_pypsa_network(
            n_nodes=n_nodes, co2lvl=co2l, use_sclopf=True
        )
        for co2l in co2ls
    }

    inertia_time = (
        np.load(
            path_to_pre_outage_sclopf + f"/inertia_time_series_all_co2ls_{n_nodes}.npy"
        )
        / 1000
    )

    # Load split statistics data
    _, _, num_parallels, _ = data_handling.get_matrices_from_nx_graph(nx_graph)
    bridge_idxs = data_handling.nx_edges_to_matrix_indices(
        nx.bridges(nx_graph), nx_graph
    )
    n_2_failures = cascade_simulation.calc_possible_double_line_failures(
        num_parallels, ignored_idxs=bridge_idxs
    )

    component_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"/component_properties_all_n{n_nodes}.h5"
    )
    component_props.time_stamp = pd.to_datetime(component_props.time_stamp)

    split_props = pd.read_hdf(
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5", index_col=0
    )
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
        float
    )

    if "neglidgeable_blackout" in split_props.columns:
        split_props.rename(
            columns={"neglidgeable_blackout": "negligible"}, inplace=True
        )

    component_props["total_weighting"] = (
        component_props["snapshot_weighting"] * component_props["trigger_weighting"]
    )

    if load_normalization:
        component_props_filtered = component_props[component_props.load_share > 0.1]
    else:
        component_props_filtered = component_props[component_props.load_share > 0.1]

    # Setup matplotlib styling
    setup_matplotlib_style()

    # Figure layout: two stacked rows (top: flow/inertia, bottom: split stats)
    
    
    if use_equal_panels:
        if not show_blackout_stats:
            raise NotImplementedError("Not implemented for not 'show_blackout_stats'" + 
                                      " and 'equal_panels'")
        fig, [[ax_flow, ax_inertia, ax_empty],
              [ax1_imbalance, ax2_inertia, ax3_num_splits]] = plt.subplots(2, 3, figsize=(12, 9))
        height_ratios_main = [1, 1]
    else:
        fig = plt.figure(figsize=(12, 9))
        height_ratios_main = [1, 1.05]
        
    gs_outer = GridSpec(2, 1, figure=fig, hspace=0.3, 
                        height_ratios=height_ratios_main)

    # --- Top row: Flow & Inertia ---
    if use_equal_panels:
        gs_top_main = GridSpecFromSubplotSpec(1, 3, 
                                          subplot_spec=gs_outer[0], 
                                          wspace=0.2)
    else:
        gs_top_main = GridSpecFromSubplotSpec(1, 2, 
                                          subplot_spec=gs_outer[0], 
                                          wspace=0.23)
        
        ax_flow = fig.add_subplot(gs_top_main[0])
        ax_inertia = fig.add_subplot(gs_top_main[1])

    cmap = plt.get_cmap("cividis")
    unit_factor = 1e6
    selected_co2ls_spi = sorted(np.array([0.6, 0.2, 0.0]))

    flows_lvls = []
    for co2l in selected_co2ls_spi:
        n = networks[co2l]
        flow_distance = (n.lines_t.p0.abs().mul(network.lines.length)).sum(axis=1)
        if mean_distance:
            total_transmissed_power = (
                n.generators_t.p.mul(network.generators.sign).sum(axis=1)
            ) + n.storage_units_t.p[n.storage_units_t.p > 0].sum(axis=1)
            flow_distance = flow_distance / total_transmissed_power
        else:
            flow_distance = flow_distance / unit_factor
        flows_lvls.append(flow_distance)
    flows_lvls = np.array(flows_lvls)

    bins = np.linspace(flows_lvls.min(), flows_lvls.max(), 40)
    for co2l in selected_co2ls_spi:
        density, bins_ = np.histogram(
            flows_lvls[selected_co2ls_spi.index(co2l), :],
            bins=bins,
            weights=network.snapshot_weightings.generators
            / network.snapshot_weightings.generators.sum(),
        )
        density_masked = density.copy()
        density_masked[density_masked == 0] = np.nan

        ax_flow.stairs(
            density_masked,
            bins,
            label=r"{}\%".format(get_actual_co2_level(co2l, percent=True)),
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
        )

    ax_flow.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    if mean_distance:
        ax_flow.set_xlabel(
            r"Mean transmission distance [km]", fontsize=AXIS_LABEL_FONTSIZE
        )
    else:
        ax_flow.set_xlabel(
            r"Total power flow distance [TW$\cdot$km]", fontsize=AXIS_LABEL_FONTSIZE
        )
    ax_flow.set_ylabel(r"Frequency", fontsize=AXIS_LABEL_FONTSIZE)
    ax_flow.grid(True)

    bins = np.linspace(0, inertia_time.max(), 40)
    for co2l in selected_co2ls_spi:
        ind = np.where(np.round(co2ls, 2) == co2l)[0][0]
        density, bins_ = np.histogram(
            inertia_time[ind, :],
            bins=bins,
            weights=network.snapshot_weightings.generators
            / network.snapshot_weightings.generators.sum(),
        )
        density_masked = density.copy()
        density_masked[density_masked == 0] = np.nan

        ax_inertia.stairs(
            density_masked,
            bins,
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
        )

    ax_inertia.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_inertia.set_xlabel("Rotational energy [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_inertia.grid(True)

    add_panel_label(ax_flow, 0, y_offset=0.08)
    add_panel_label(ax_inertia, 1, y_offset=0.08)

    # --- Bottom row: Split Statistics ---
    if not use_equal_panels:
        if show_blackout_stats:
            n_cols = 4
            width_ratios = [1, 1, 1, 1]
            wspace = 0.40 if secondary_proba_axis else 0.4
        else:
            n_cols = 2
            width_ratios = [1, 1]
            wspace = 0.3
        
    
    if not use_equal_panels:
        gs_bottom = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_outer[1], hspace=0.55, height_ratios=[1, 0.15]
        )
        gs_bottom_main = GridSpecFromSubplotSpec(
            1,
            n_cols,
            subplot_spec=gs_bottom[0],
            width_ratios=width_ratios,
            hspace=0,
            wspace=wspace,
        )

        if show_blackout_stats:
            ax3_num_splits = fig.add_subplot(gs_bottom_main[2])
            ax3_num_splits_legend = fig.add_subplot(gs_bottom_main[3])
        else:
            ax3_num_splits = None
            ax3_num_splits_legend = None

    # Panel c: loss of load share distribution
    if show_blackout_stats:
        cmap = plt.get_cmap("inferno_r")

        bins = np.linspace(0, 100, 6).astype(int)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        data = []
        for co2l in co2ls[::-1]:
            vals = (
                split_props[split_props.co2l == co2l].lost_load_share_blackout.values
                * 100
            )
            counts, _ = np.histogram(vals, bins=bins)
            data.append(counts)

        data = np.stack(data)
        data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

        plt.sca(ax3_num_splits)
        markers = ["o", "s", "D", "^", "v", "."]
        for i, bin_center in enumerate(bin_centers):
            color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
            counts = data.loc[:, bin_center]
            if use_steps_load_loss:
                plt.step(
                get_actual_co2_level(co2ls[::-1], percent=True),
                counts,
                where="mid",
                label=rf"{bins[i]}-{bins[i+1]}\%",
                alpha=0.8,
                color=color,
                marker=markers[i % len(markers)],
                markersize=5,)
            else:
                plt.plot(
                    get_actual_co2_level(co2ls[::-1], percent=True),
                    counts,
                    label=rf"{bins[i]}-{bins[i+1]}\%",
                    alpha=0.8,
                    color=color,
                    marker=markers[i % len(markers)],
                    markersize=5,
                )

        ax3_num_splits.set_xlabel(
            r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax3_num_splits.set_ylabel(
            "Number of System Splits", fontsize=AXIS_LABEL_FONTSIZE
        )
        x_ticks_major = [0, 20, 40, 60]
        x_ticks_minor = [10, 30, 50]
        ax3_num_splits.set_xticks(x_ticks_major)
        ax3_num_splits.set_xticks(x_ticks_minor, minor=True)
        ax3_num_splits.grid(False)
        ax3_num_splits.tick_params(
            axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE
        )
        ax3_num_splits.set_yscale("log")
        ax3_num_splits.invert_xaxis()

        if secondary_proba_axis:
            ax3_num_splits_secondary = ax3_num_splits.twinx()
            for i, bin_center in enumerate(bin_centers):
                color = cmap(i / (len(bin_centers) + 1) + (1 / (len(bin_centers) + 1)))
                counts = data.loc[:, bin_center]
                plt.plot(
                    get_actual_co2_level(co2ls[::-1], percent=True),
                    counts
                    / len(n_2_failures)
                    / network.snapshot_weightings.generators.sum(),
                    label=rf"{bins[i]}-{bins[i+1]}\%",
                    alpha=0.0,
                )
            ax3_num_splits_secondary.set_ylabel(
                "Fraction of total contingencies", fontsize=AXIS_LABEL_FONTSIZE
            )
            ax3_num_splits_secondary.set_yscale("log")
            ax3_num_splits_secondary.tick_params(
                axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE
            )

        h, l = ax3_num_splits.get_legend_handles_labels()
        if not use_equal_panels:
            
            if secondary_proba_axis:
                ax3_num_splits_legend.legend(
                    h,
                    l,
                    title="Share of load\n not served",
                    loc="center left",
                    bbox_to_anchor=(0.1, 0.5),
                    ncols=1,
                    columnspacing=0.5,
                )
            else:
                ax3_num_splits_legend.legend(
                    h,
                    l,
                    title="Share of load\n not served",
                    loc="center",
                    ncols=1,
                    columnspacing=0.5,
                )
            ax3_num_splits_legend.axis("off")            
            

    # Panel b: Inertia histograms
    cmap = plt.get_cmap("cividis")
    if not use_equal_panels:
        ax2_inertia = fig.add_subplot(gs_bottom_main[1])
        ax2_inertia_legend = fig.add_subplot(gs_bottom[1, :])

    rot_energy = component_props_filtered.rot_energy / 1000
    if load_normalization:
        rot_energy = component_props_filtered.rot_energy / component_props_filtered.load

    selected_co2ls = np.array([0.0, 0.2, 0.6])
    max_vals = [
        rot_energy[component_props_filtered.co2l == co2l].max()
        for co2l in selected_co2ls
    ]
    
    min_vals = [
        rot_energy[component_props_filtered.co2l == co2l].min()
        for co2l in selected_co2ls
    ]
    
    bins = np.linspace(min(min_vals), max(max_vals), 15)

    for co2l in selected_co2ls:
        ax2_inertia.hist(
            rot_energy[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.total_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            label=rf"{get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)} \%",
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax2_inertia.set_yscale("log")
    if load_normalization:
        ax2_inertia.set_xlabel(
            "Norm. rotational energy [s]", 
            fontsize=AXIS_LABEL_FONTSIZE
        )
    else:
        ax2_inertia.set_xlabel("Rotational energy [GWs]", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_inertia.set_ylabel("Number of split components", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_inertia.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    # Panel a: power imbalance histograms
    if not use_equal_panels:
        ax1_imbalance = fig.add_subplot(gs_bottom_main[0])
        
    if load_normalization:
        power_imbalance = (
            component_props_filtered.power_imbalance / component_props_filtered.load
        )
    else:
        power_imbalance = component_props_filtered.power_imbalance / 1000
    bins = np.linspace(power_imbalance.min(), power_imbalance.max(), 15)

    for co2l in selected_co2ls:
        ax1_imbalance.hist(
            power_imbalance[component_props_filtered.co2l == co2l],
            weights=component_props_filtered.total_weighting[
                component_props_filtered.co2l == co2l
            ],
            bins=bins,
            histtype="step",
            linewidth=3,
            label=rf"{get_actual_co2_level(co2l, n_nodes=n_nodes, percent=True)} \%",
            color=cmap(np.where(co2ls == co2l)[0][0] / (len(co2ls) - 1)),
            alpha=0.8,
            density=False,
        )

    ax1_imbalance.set_yscale("log")
    ax1_imbalance.set_ylabel("Number of split components", fontsize=AXIS_LABEL_FONTSIZE)
    if load_normalization:
        ax1_imbalance.set_xlabel(
            "Norm. power imbalance [1]", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax1_imbalance.set_xticks(
            np.arange(
                np.ceil(power_imbalance.min()),
                np.floor(power_imbalance.max()) + 1,
                step=1,
            )
        )
    else:
        ax1_imbalance.set_xlabel("Power imbalance [GW]", 
                                 fontsize=AXIS_LABEL_FONTSIZE)

    ax1_imbalance.tick_params(axis="both", which="both", 
                              labelsize=TICK_LABEL_FONTSIZE)

    
    h, l = ax2_inertia.get_legend_handles_labels()
    handles = [Line2D([], [], color="none")] + h[::-1]
    labels = [r"CO$_2$ level [\% of 1990]"] + l[::-1]
    if not use_equal_panels:
        
        ax2_inertia_legend.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(-0.08, 0.45),
            ncols=4,
            columnspacing=1,
            handletextpad=0.5,
        )
        ax2_inertia_legend.axis("off")
        

    axes_to_label = [ax1_imbalance, ax2_inertia]
    if show_blackout_stats and ax3_num_splits is not None:
        axes_to_label.append(ax3_num_splits)

    for idx, ax_loss_lvl in enumerate(axes_to_label, start=2):
        add_panel_label(ax_loss_lvl, idx, y_offset=0.1, 
                        x_offset=-.125)

    plt.tight_layout()
    
    # Legend in the case for equal panel size
    if use_equal_panels:
        ax_empty.axis('off')
        
        pos_ax_inertia = ax_inertia.get_position()
        
        leg_co2 = ax_inertia.legend(h, l,
                   title="CO$_2$ level \n[\\% of 1990]",
                   loc="upper left",
                   bbox_to_anchor=(1, 1.05),
                   fontsize=18, title_fontsize=20)
        leg_title = leg_co2.get_title()
        leg_title.set_horizontalalignment('center')
        
        pos_ax3 = ax3_num_splits.get_position()
        h, l = ax3_num_splits.get_legend_handles_labels()
        leg_sp = ax3_num_splits.legend(h, l, title="Share of load\n not served",
                loc="lower right",
                bbox_to_anchor=(1.05, 1.),
                fontsize=20, title_fontsize=18)
        title_sp = leg_sp.get_title()
        title_sp.set_horizontalalignment('center')
        
    # Align x and ylabel
    fig.align_ylabels([ax1_imbalance, ax_flow])
    fig.align_ylabels([ax2_inertia, ax_inertia])
    
    if show_blackout_stats:
        fig.align_xlabels([ax1_imbalance, ax2_inertia, ax3_num_splits])
    else:
        fig.align_xlabels([ax1_imbalance, ax2_inertia])

    save_name = "combined_flow_inertia_split_statistics"
    if mean_distance:
        save_name += "_mean_distance"
        
    if load_normalization:
        save_name += "_normalized"
        
    if show_blackout_stats:
        save_name += "_with_blackout_stats"
        
    if secondary_proba_axis:
        save_name += "_with_secondary_proba_axis"
        
    if use_steps_load_loss:
        save_name += "_steps_lloss"
        
    if use_equal_panels:
        save_name += "_equal_panels"

    if save_prefix is not None:
        save_name = save_prefix + "_" + save_name
    
    save_figure(fig, save_name, save_path)

if __name__ == "__main__":
    create_combined_flow_and_split_statistics_plot(
        mean_distance=False,
        load_normalization=False,
        show_blackout_stats=True,
        secondary_proba_axis=True,
    )
