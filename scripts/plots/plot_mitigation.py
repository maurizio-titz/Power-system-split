#!/usr/bin/env python3
"""
Line Extension Mitigation Plot
Shows line extension costs and mitigation effects for reducing system splits.
"""

import gzip
import sys
import pickle
import copy
import os
import warnings

import glob

warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm
from loguru import logger

import networkx as nx
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


sys.path.append("./")

from utils.data_handling import get_actual_co2_level, get_co2_levels
from utils.config import (
    path_to_pypsa_network_sclopf,
    path_to_figures_sclopf,
    path_to_vis_results_sclopf,
    path_to_line_extension_mitigation_sclopf,
    path_to_pre_outage_sclopf,
    path_to_inertia_mitigation_results_sclopf,
    path_to_evaluation_results_sclopf,
    path_to_plot_data
)
from utils import data_handling, cascade_simulation
from utils.cascade_simulation import LOOKUP_TABLE_NP
from utils.mitigation_visualization import (
    plot_map_inertia_placement_final,
    calc_inertia_placement_ref_loss,
    color1,
    color2,
    plot_map_inertia_placement_new,
)
from utils.plot_style import (
    setup_matplotlib_style,
    TITLE_FONTSIZE,
    SUBTITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PANEL_LABEL_FONTSIZE,
    add_panel_label,
    save_figure,
)

# Backwards compatibility
AXIS_LABELSIZE = AXIS_LABEL_FONTSIZE
TICK_LABELSIZE = TICK_LABEL_FONTSIZE
SUBLABEL_FONTSIZE = PANEL_LABEL_FONTSIZE

# Plot data path
if not os.path.exists(path_to_plot_data):
    os.makedirs(path_to_plot_data, exist_ok=True)

annualized_cost_per_MWs_max_DE = 888.5  # € per MWs/a of synthetic intertia as per https://www.netztransparenz.de/de-de/Systemdienstleistungen/Frequenzhaltung/Marktgest%C3%BCtzte-Beschaffung-von-Momentanreserve
# annualized_cost_per_MVAs_GB = (
#     5080 * 1.1672
# )  # € per MWs/a of synthetic intertia as per https://www.neso.energy/news/neso-awards-first-contracts-under-mid-term-y-1-stability-market?utm_source=chatgpt.com However, this is not directly comparable to inertia, so we don't use it.
# # mean exchange rate in 2025 from https://www.exchangerates.org.uk/GBP-EUR-spot-exchange-rates-history-2025.html
# # 25.4 millionn Pound for 5 GVAs: 25.4 / 5 * 1e6 / 1e3 = 5080 £ per MWs/a
annualized_cost_per_MWs_max = annualized_cost_per_MWs_max_DE
annualized_cost_per_GWs_max = annualized_cost_per_MWs_max * 1e3  # € per GWs/a


def calc_annualized_cost_per_km(
    capex_per_km, discount_rate=0.05, lifetime=40, fom_fraction=0.0
):
    """
    Calculate annualized transmission cost per km.

    Parameters
    ----------
    capex_per_km : float
        Investment cost [€/km]
    discount_rate : float
        Discount rate (e.g. 0.05 for 5%)
    lifetime : int
        Asset lifetime in years
    fom_fraction : float, optional
        Fixed O&M cost as fraction of CAPEX per year
        (e.g. 0.02 for 2%), default = 0

    Returns
    -------
    float
        Annualized cost [€/km/year]
    """
    r = discount_rate
    n = lifetime

    # Capital Recovery Factor
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)

    annualized_capital = capex_per_km * crf
    annual_fom = capex_per_km * fom_fraction

    return annualized_capital + annual_fom


capital_expenditure_per_km = (
    4.5 * 1e6
)  # € per km for 380kV line, source: https://www.netzentwicklungsplan.de/sites/default/files/2023-03/230321_NEP_Kostenschaetzung_NEP2037_2045_V2023_1.Entwurf.pdf


def calculate_line_extension(
    split_properties,
    nx_graph,
    co2ls,
    ref_network,
    n_nodes=600,
    build_380kV_only=False,
    target="num_GSS",
    blackoutthreshold=0.0,
):

    from utils.cascade_simulation import LOOKUP_TABLE_NP

    if isinstance(co2ls, float):
        co2ls = [co2ls]
    print("Calculating line extension mitigation and saving results...")
    table_rounded = np.round(LOOKUP_TABLE_NP, 5)
    num_par_paths = {}
    for tup in table_rounded:
        current_tup = tup
        num_par_paths[tup[0]] = [current_tup[0] - current_tup[1]]
        while current_tup[1] != 0:
            current_tup = table_rounded[
                np.where((table_rounded[:, 0] == current_tup[1]))[0][0]
            ]
            num_par_paths[tup[0]].append(current_tup[0] - current_tup[1])
    max_num_par = {k: max(v) for k, v in num_par_paths.items()}

    os.makedirs(path_to_line_extension_mitigation_sclopf, exist_ok=True)

    # extension_cost_per_MWkm = 1100  # € / MVA / km ##  source: Tom Brown updated fig of https://ariadneprojekt.de/publikation/report-szenarien-zur-klimaneutralitat-2045-2/ Fig 7.2
    # annualized_extension_cost_per_MWkm = (
    #     0.058 * extension_cost_per_MWkm
    # )  # € / MVA / km / a
    annualized_cost_per_km = calc_annualized_cost_per_km(capital_expenditure_per_km)

    n_lines = nx_graph.number_of_edges()

    for annualized_costs in [True, False]:

        for co2l in co2ls:
            if co2l == 0.6:
                continue
            f_name = create_gridExt_mitigation_filename(
                n_nodes,
                build_380kV_only,
                target,
                blackoutthreshold,
                annualized_costs,
                co2l,
            )
            split_props_lvl = split_properties[split_properties.co2l == co2l].copy()
            split_props_lvl.lost_load_share_blackout = (
                split_props_lvl.lost_load_share_blackout.astype(float)
            )
            if blackoutthreshold is not None and blackoutthreshold > 0.0:
                remaining_splits = split_props_lvl[
                    split_props_lvl.lost_load_share_blackout > blackoutthreshold
                ]
            else:
                remaining_splits = split_props_lvl
            remaining_splits = remaining_splits.loc[
                :,
                [
                    "lost_load_share_blackout",
                    "init_failure_0",
                    "init_failure_1",
                    "total_weighting",
                ],
            ]
            costs_ = ref_network.lines.loc[
                :,
                [
                    "bus0",
                    "bus1",
                    "capital_cost",
                    "s_nom",
                    "num_parallel",
                    "length",
                ],
            ].copy()
            s_380kV = (
                (
                    np.sqrt(3)
                    * ref_network.lines["type"].map(ref_network.line_types.i_nom)
                    * ref_network.lines.bus0.map(ref_network.buses.v_nom)
                )
                .unique()
                .astype(float)
            )
            # if only building 380kV lines, set s_380kV accordingly
            if build_380kV_only:
                costs_["num_par_ext"] = 1
            else:
                # take the highest voltage line possible for extension, so that failures on the corridor are save
                costs_["num_par_ext"] = costs_.num_parallel.apply(
                    lambda x: max_num_par[np.round(x, 5)] if x in max_num_par else 1
                )
            if annualized_costs:
                costs_["extension_cost"] = annualized_cost_per_km * costs_.length
            else:
                costs_["extension_cost"] = capital_expenditure_per_km * costs_.length

            loss_with_mitigation = [
                (
                    split_props_lvl.lost_load_share_blackout
                    * split_props_lvl.total_weighting
                ).sum()
            ]
            reinforced_lines = []
            cost = []
            num_blackouts_with_mitigation = [remaining_splits.total_weighting.sum()]

            while remaining_splits.shape[0] > 0:
                trigger0 = (
                    remaining_splits.loc[
                        :,
                        [
                            "lost_load_share_blackout",
                            "init_failure_0",
                            "total_weighting",
                        ],
                    ]
                    .groupby("init_failure_0")
                    .sum()
                )
                trigger1 = (
                    remaining_splits.loc[
                        :,
                        [
                            "lost_load_share_blackout",
                            "init_failure_1",
                            "total_weighting",
                        ],
                    ]
                    .groupby("init_failure_1")
                    .sum()
                )
                trigger1.rename_axis("trigger", inplace=True)
                trigger0.rename_axis("trigger", inplace=True)
                mitigated_loss_by_trigger = pd.Series(
                    index=list(range(n_lines)), data=0
                )
                mitigated_loss_by_trigger.rename_axis("trigger", inplace=True)
                mitigated_loss_by_trigger = mitigated_loss_by_trigger.add(
                    trigger0.lost_load_share_blackout * trigger0.total_weighting,
                    fill_value=0,
                )
                mitigated_loss_by_trigger = mitigated_loss_by_trigger.add(
                    trigger1.lost_load_share_blackout * trigger1.total_weighting,
                    fill_value=0,
                )
                mitigated_loss_per_dollar = mitigated_loss_by_trigger / costs_[
                    "extension_cost"
                ].reset_index(drop=True)

                mitigated_splits_by_trigger = pd.Series(
                    index=list(range(n_lines)), data=0
                )
                mitigated_splits_by_trigger.rename_axis("trigger", inplace=True)
                mitigated_splits_by_trigger = mitigated_splits_by_trigger.add(
                    trigger0.total_weighting, fill_value=0
                )
                mitigated_splits_by_trigger = mitigated_splits_by_trigger.add(
                    trigger1.total_weighting, fill_value=0
                )
                mitigated_splits_per_dollar = mitigated_splits_by_trigger / costs_[
                    "extension_cost"
                ].reset_index(drop=True)

                if target == "num_GSS":
                    trigger = mitigated_splits_per_dollar.idxmax()
                else:
                    trigger = mitigated_loss_per_dollar.idxmax()

                cost.append(costs_["extension_cost"][int(trigger)])
                remaining_splits = remaining_splits[
                    (remaining_splits.init_failure_1 != trigger)
                    & (remaining_splits.init_failure_0 != trigger)
                ]
                loss_with_mitigation.append(
                    (
                        remaining_splits.lost_load_share_blackout
                        * remaining_splits.total_weighting
                    ).sum()
                )
                reinforced_lines.append(trigger)
                num_blackouts_with_mitigation.append(
                    remaining_splits.total_weighting.sum()
                )

            with open(os.path.join(path_to_line_extension_mitigation_sclopf, f_name), "wb") as f:
                pickle.dump(
                    (
                        reinforced_lines,
                        loss_with_mitigation,
                        num_blackouts_with_mitigation,
                        cost,
                    ),
                    f,
                )

            # num_lines_to_reach_ref_loss = np.where(
            #     loss_with_mitigation < lost_load_reference_lvl
            # )[0][0]
            # num_lines_to_reach_double_ref_loss = np.where(
            #     loss_with_mitigation < 2 * lost_load_reference_lvl
            # )[0][0]
            # cost_to_reach_ref_loss[co2l] = sum(cost[:num_lines_to_reach_ref_loss])
            # cost_to_reach_double_ref_loss[co2l] = sum(
            #     cost[:num_lines_to_reach_double_ref_loss]
            # )
    if len(co2ls) == 1:
        return (
            reinforced_lines,
            loss_with_mitigation,
            num_blackouts_with_mitigation,
            cost,
        )
    else:
        return None


def create_gridExt_mitigation_filename(
    n_nodes, build_380kV_only, target, blackoutthreshold, annualized_costs, co2l
):
    f_name = "heuristic_costMin"
    if target == "num_GSS":
        f_name += "_GSS_mitigation"
    else:
        f_name += "_loss_mitigation"
    if annualized_costs:
        f_name = f_name + f"_annualized_Co2L{co2l}_n{n_nodes}.pkl"
    else:
        f_name = f_name + f"_Co2L{co2l}_n{n_nodes}.pkl"
    if build_380kV_only:
        f_name = f_name.replace(".pkl", "_380kVonly.pkl")
    if blackoutthreshold is not None and blackoutthreshold > 0.0:
        f_name = f_name.replace(".pkl", f"_blackoutthres{blackoutthreshold}.pkl")

    return f_name


def create_combined_mitigation_plot(
    use_annualized_costs=True,
    co2_lvl_map=0.1,
    build_380kV_only=False,
    plot_intertia_cost=False,
    blackoutthreshold=0.0,
    target: str = "num_GSS",
    plot_rows: str = "both",
    n_nodes: int = 600,
    organize_plots: bool = True,
    save_prefix: str | None = None
):
    """Create combined mitigation plot with inertia on top and line extension below.

    Parameters
    ----------
    plot_rows : str, optional
        Which rows to plot: "both" (default), "inertia", or "line_extension"
    """

    # Setup
    save_path = path_to_figures_sclopf
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(path_to_line_extension_mitigation_sclopf, exist_ok=True)

    # Load network
    network = data_handling.load_pypsa_network_from_path(
        path_to_pypsa_network_sclopf
        + f"/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.6-2920SEG.nc",
        True,
    )
    nx_graph = data_handling.build_networkx_graph(network, snet_index='0')
    I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(
        nx_graph
    )
    n_lines = nx_graph.number_of_edges()

    # Get CO2 levels and split properties
    co2ls = get_co2_levels(n_nodes)
    split_properties = pd.read_hdf(
        path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5", index_col=0
    )

    # Reference levels and setup
    co2l_ref = 0.6
    ref_loss_factors = [1]  # Only use ref_loss_factor=1

    # Validate plot_rows parameter
    if plot_rows not in ["both", "inertia", "line_extension"]:
        raise ValueError(
            f"plot_rows must be 'both', 'inertia', or 'line_extension', got '{plot_rows}'"
        )

    # Setup matplotlib with consistent styling
    setup_matplotlib_style()

    # === INERTIA MITIGATION DATA PREPARATION ===
    if plot_rows in ["both", "inertia"]:
        inertia_time = (
            np.load(
                path_to_pre_outage_sclopf
                + f"/inertia_time_series_all_co2ls_{n_nodes}.npy"
            )
            / 1000
        )
        ind = np.where(np.round(co2ls, 2) == co2l_ref)[0][0]
        median_inertia_ref = np.median(inertia_time[ind, :])

        # try loading results
        fname = f"/inertia_placement_results_refLoss_allCo2lvls_n{n_nodes}_{target}_blackoutthres{blackoutthreshold}.pkl"
        try:
            with gzip.open(
                path_to_inertia_mitigation_results_sclopf + fname,
                "rb",
            ) as f:
                (
                    res_tuples_by_lvl,
                    inertia_needed_by_ref_loss,
                    mitigation_curves_by_ref_loss,
                    steps_to_reach_target_by_ref_loss,
                    delta_Erot_by_lvl,
                ) = pickle.load(f)
        except FileNotFoundError:
            # Calculate inertia needed by level
            inertia_needed_by_ref_loss = {}
            mitigation_curves_by_ref_loss = {}
            steps_to_reach_target_by_ref_loss = {}
            for ref_loss_factor in ref_loss_factors:
                (
                    res_tuples_by_lvl,
                    mitigation_curve,
                    inertia_at_ref_loss_by_lvl,
                    steps_to_reach_target_by_lvl,
                    delta_Erot_by_lvl,
                ) = calc_inertia_placement_ref_loss(
                    n_nodes=600,
                    delta_Erot=1000,
                    co2_lvl_ref=co2l_ref,
                    split_properties=split_properties,
                    ref_loss_factor=ref_loss_factor,
                    co2_lvls=co2ls,
                    blackoutthreshold=blackoutthreshold,
                    target=target,
                )

                inertia_needed_by_ref_loss[ref_loss_factor] = inertia_at_ref_loss_by_lvl
                mitigation_curves_by_ref_loss[ref_loss_factor] = mitigation_curve
                steps_to_reach_target_by_ref_loss[ref_loss_factor] = (
                    steps_to_reach_target_by_lvl
                )
            # save results
            with gzip.open(
                path_to_inertia_mitigation_results_sclopf + fname,
                "wb",
            ) as f:
                pickle.dump(
                    (
                        res_tuples_by_lvl,
                        inertia_needed_by_ref_loss,
                        mitigation_curves_by_ref_loss,
                        steps_to_reach_target_by_ref_loss,
                        delta_Erot_by_lvl,
                    ),
                    f,
                )

    # === LINE EXTENSION MITIGATION DATA PREPARATION ===
    if plot_rows in ["both", "line_extension"]:
        # Calculate parallel line extensions lookup
        table_rounded = np.round(LOOKUP_TABLE_NP, 5)
        num_par_paths = {}
        for tup in table_rounded:
            current_tup = tup
            num_par_paths[tup[0]] = [current_tup[0] - current_tup[1]]
            while current_tup[1] != 0:
                current_tup = table_rounded[
                    np.where((table_rounded[:, 0] == current_tup[1]))[0][0]
                ]
                num_par_paths[tup[0]].append(current_tup[0] - current_tup[1])
        max_num_par = {k: max(v) for k, v in num_par_paths.items()}

    split_properties_reference = split_properties[
        split_properties.co2l == co2l_ref
    ].copy()
    if target == "total_loss":
        target_value = split_properties_reference.lost_load_share_blackout.sum()
    if target == "num_GSS":
        target_value = (
            split_properties_reference["total_weighting"]
            * (split_properties_reference.lost_load_share_blackout > blackoutthreshold)
        ).sum()
    elif target == "lost_load_GSS":
        target_value = (
            split_properties_reference["total_weighting"]
            * split_properties_reference.lost_load_share_blackout
        ).sum()

    # Calculate costs for line extensions
    if plot_rows in ["both", "line_extension"]:
        cost_to_reach_ref_loss = {}
        lines_to_reach_ref_loss = {}  # Store number of lines needed

        for co2l in co2ls:
            if co2l == 0.6:
                continue

            f_name = create_gridExt_mitigation_filename(
                n_nodes,
                build_380kV_only,
                target,
                blackoutthreshold,
                use_annualized_costs,
                co2l,
            )

            try:
                reinforced_lines, post_mitigation_value, num_blackouts, cost = (
                    pickle.load(
                        open(os.path.join(path_to_line_extension_mitigation_sclopf, f_name), "rb")
                    )
                )
            except FileNotFoundError as e:
                reinforced_lines, post_mitigation_value, num_blackouts, cost = (
                    calculate_line_extension(
                        split_properties,
                        nx_graph,
                        co2l,
                        network,
                        n_nodes=600,
                        build_380kV_only=build_380kV_only,
                        blackoutthreshold=blackoutthreshold,
                        target=target,
                    )
                )

            # Find number of lines needed
            if target == "num_GSS":
                num_lines_to_reach_ref_loss = np.where(
                    np.array(num_blackouts) < target_value
                )[0][0]
            else:
                num_lines_to_reach_ref_loss = np.where(
                    np.array(post_mitigation_value) < target_value
                )[0][0]

            cost_to_reach_ref_loss[co2l] = sum(cost[:num_lines_to_reach_ref_loss])
            lines_to_reach_ref_loss[co2l] = num_lines_to_reach_ref_loss

    # === CREATE COMBINED FIGURE ===
    # Determine figure size and layout based on plot_rows
    if plot_rows == "both":
        figsize = (16, 9)
        n_rows = 2
        height_ratios = [1, 1]
    else:
        figsize = (16, 4.5)
        n_rows = 1
        height_ratios = [1]

    f = plt.figure(figsize=figsize)

    # Main grid
    gs_main = GridSpec(n_rows, 1, figure=f, height_ratios=height_ratios, hspace=0.5)

    co2_ref_percent = get_actual_co2_level(co2l_ref, percent=True)
    co2_lvl_map_percent = get_actual_co2_level(co2_lvl_map, percent=True)

    if target == "num_GSS":
        target_symbol = "N^{\\text{GSS}}"
    else:
        target_symbol = "R"
    if target == "total_loss":
        loss_axis_label = rf"Expected loss of load [${target_symbol}_{{{int(round(co2_lvl_map_percent))}\%}}/{target_symbol}_{{{int(round(co2_ref_percent))}\%}}$]"
    elif target == "num_GSS":
        loss_axis_label = rf"Globally Severe Splits [${target_symbol}_{{{int(round(co2_lvl_map_percent))}\%}}/{target_symbol}_{{{int(round(co2_ref_percent))}\%}}$]"
    elif target == "lost_load_GSS":
        loss_axis_label = rf"Expected lost load in GSS [${target_symbol}_{{{int(round(co2_lvl_map_percent))}\%}}/{target_symbol}_{{{int(round(co2_ref_percent))}\%}}$]"

    # === CREATE SUBPLOTS BASED ON plot_rows ===
    if plot_rows in ["both", "inertia"]:
        # Inertia mitigation section
        subplot_idx = 0 if plot_rows == "both" else 0
        gs_inertia = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=gs_main[subplot_idx],
            wspace=0.1,
            width_ratios=[1.5, 2.2, 1],
        )
        ax_inertia_loss = f.add_subplot(gs_inertia[0])  # Loss reduction curve
        ax_inertia_map = f.add_subplot(gs_inertia[1])  # Map
        ax_inertia_all = f.add_subplot(gs_inertia[2])  # All CO2 levels

    if plot_rows in ["both", "line_extension"]:
        # Line extension section
        subplot_idx = 1 if plot_rows == "both" else 0
        gs_line = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=gs_main[subplot_idx],
            wspace=0.1,
            width_ratios=[1.5, 2.2, 1],
        )
        ax_line_loss = f.add_subplot(gs_line[0])  # Loss reduction curve
        ax_line_map = f.add_subplot(gs_line[1])  # Map
        ax_line_all = f.add_subplot(gs_line[2])  # All CO2 levels

    # === PLOT INERTIA MITIGATION ===
    if plot_rows in ["both", "inertia"]:
        unit = "GWs"
        unit_factor = 1e-3
        # Use consistent colors across both mitigation types
        color_reference_loss = color1  # Same color for both loss reduction curves
        color_all_levels = color2  # Same color for both all-levels curves
        color_loss_curve = "black"

        # Inertia: All CO2 levels plot
        for i, ref_loss_factor in enumerate(ref_loss_factors):
            inertia_at_ref_loss_by_lvl = inertia_needed_by_ref_loss[ref_loss_factor]
            label = f"$R_{{\\textrm{{ref}}}}$"  # Simplified label for ref_loss_factor=1

            ax_inertia_all.plot(
                np.array(
                    get_actual_co2_level(
                        list(inertia_at_ref_loss_by_lvl.keys()), percent=True
                    )
                ),
                np.array(list(inertia_at_ref_loss_by_lvl.values())) * unit_factor,
                label=label,
                color=color_reference_loss,
                linewidth=2,
            )
            ax_inertia_all.plot(
                np.array(
                    get_actual_co2_level(
                        list(inertia_at_ref_loss_by_lvl.keys()), percent=True
                    )
                ),
                median_inertia_ref * np.ones(len(inertia_at_ref_loss_by_lvl)),
                linestyle="--",
                color="black",
            )
            # add description of median line as text
            ax_inertia_all.text(
                np.array(
                    get_actual_co2_level(
                        list(inertia_at_ref_loss_by_lvl.keys()), percent=True
                    )
                )[0]
                / 2,
                median_inertia_ref,
                f"Median inertia at {get_actual_co2_level(co2l_ref, percent=True)}\\%",
                color="black",
                ha="center",
                va="bottom",
                rotation=0,
                zorder=np.inf,
                fontsize=TICK_LABELSIZE,
            )

            if plot_intertia_cost:
                # add secondary axis for cost of inertia
                ax_inertia_all_cost = ax_inertia_all.twinx()
                ax_inertia_all_cost.set_ylabel(
                    f"Annualized cost [billion €]", fontsize=AXIS_LABELSIZE
                )
                # scale inertia to cost
                rescale_factor = annualized_cost_per_GWs_max / 1e6
                ticks_cost = ax_inertia_all.get_yticks() * rescale_factor
                ax_inertia_all_cost.set_yticks(
                    ticks_cost,
                    labels=[int(tick) / 1e3 for tick in ax_inertia_all.get_yticks()],
                )
                ax_inertia_all_cost.tick_params(
                    axis="y",
                    which="major",
                    labelsize=TICK_LABELSIZE,
                )
                ax_inertia_all_cost.set_ylim(ax_inertia_all.get_ylim())
                ax_inertia_all_cost.grid(False)

        ax_inertia_all.set_ylim(top=median_inertia_ref * 1.1)
        ax_inertia_all.invert_xaxis()
        ax_inertia_all.set_ylabel(f"Inertia placed [{unit}]", fontsize=AXIS_LABELSIZE)
        ax_inertia_all.set_xlabel("CO$_2$ level [\\% of 1990]", fontsize=AXIS_LABELSIZE)
        ax_inertia_all.set_title("Additional inertia needed", fontsize=TITLE_FONTSIZE)
        ax_inertia_all.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax_inertia_all.grid(True, alpha=0.3)

        # Inertia: Map and loss reduction
        for i, ref_loss_factor in enumerate(ref_loss_factors):
            plot_curve = i == 0
            plot_map_inertia_placement_new(
                target_value=target_value,
                co2_lvl=co2_lvl_map,
                idx_reached_target=steps_to_reach_target_by_ref_loss[ref_loss_factor][
                    co2_lvl_map
                ],
                res_tuple=res_tuples_by_lvl[co2_lvl_map],
                mitigation_curve=mitigation_curves_by_ref_loss[ref_loss_factor][
                    co2_lvl_map
                ],
                axes=(ax_inertia_map, ax_inertia_loss),
                nn=600,
                max_iter=10000,
                max_node_size=100,
                edge_width=0.2,
                delta_Erot=delta_Erot_by_lvl[co2_lvl_map],
                resolve_strategy="random",
                show_step_number=False,
                plot_split_number=False,
                save_fig=False,
                co2_lvl_ref=co2l_ref,
                ref_loss_factor=ref_loss_factor,
                unit=unit,
                color=color_reference_loss,
                plot_curve=plot_curve,
                line_color=color_loss_curve,
                annualized_cost_per_GWs_max=annualized_cost_per_GWs_max,
            )

        # Set consistent styling for inertia plots
        title = f"Additional inertia mitigation \n CO$_2$ level={get_actual_co2_level(co2_lvl_map, percent=True)}\\%"
        ax_inertia_loss.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
        )
        ax_inertia_loss.tick_params(
            axis="both", which="major", labelsize=TICK_LABELSIZE
        )
        ax_inertia_loss.set_xlabel("Inertia placed [GWs]", fontsize=AXIS_LABELSIZE)

        ax_inertia_loss.set_ylabel(loss_axis_label, fontsize=AXIS_LABELSIZE)
        ax_inertia_loss.grid(True, alpha=0.3)

        ax_inertia_map.set_aspect("equal")
        # Set map title with appropriate symbol based on target
        ax_inertia_map.set_title(
            rf"Additional inertia to reach ${target_symbol}_{{{int(round(co2_ref_percent))}\%}}$"
            + "\n"
            + rf"CO$_2$ level={get_actual_co2_level(co2_lvl_map, percent=True)}\%",
            fontsize=TITLE_FONTSIZE,
        )

    # === PLOT LINE EXTENSION MITIGATION ===
    if plot_rows in ["both", "line_extension"]:
        # Use consistent colors (define if not already defined in inertia section)
        if plot_rows == "line_extension":
            color_reference_loss = color1
            color_all_levels = color2
            color_loss_curve = "black"

        # Load specific data for the map level
        f_name = create_gridExt_mitigation_filename(
            n_nodes,
            build_380kV_only,
            target,
            blackoutthreshold,
            use_annualized_costs,
            co2_lvl_map,
        )

        reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(
            open(os.path.join(path_to_line_extension_mitigation_sclopf, f_name), "rb")
        )
        if target == "num_GSS":
            post_mitigation_value = num_blackouts
        elif target == "total_loss" or target == "lost_load_GSS":
            post_mitigation_value = loss_with_mitigation

        num_lines_to_reach_ref_loss = np.where(
            np.array(post_mitigation_value) < target_value
        )[0][0]
        cost = [0] + cost  # Add 0 cost for the initial state with no lines reinforced
        cost_to_reach_ref_loss_lvl = sum(cost[: num_lines_to_reach_ref_loss + 1])

        # Line extension: Loss reduction curve
        ax_line_loss.plot(
            np.arange(len(post_mitigation_value)),
            np.array(post_mitigation_value) / target_value,
            color=color_loss_curve,
            linewidth=2,
            label="Loss reduction",
        )

        right_xlim = np.where(np.array(post_mitigation_value) / target_value < 0.5)[0][
            0
        ]

        # Add secondary y-axis for cost
        if use_annualized_costs:
            rescale_factor = 1 / 0.025
        else:
            rescale_factor = 1 / 1
        cost_rescaled = np.cumsum(cost) * rescale_factor / 1e9
        # Add second y axis for costs
        ax2_line_cost = ax_line_loss.twinx()
        ax2_line_cost.plot(
            np.arange(len(cost)),
            cost_rescaled,
            color="black",
            # color=color_loss_curve,
            linestyle="dotted",
            linewidth=2,
            label="Annualized cost" if use_annualized_costs else "Cost",
        )
        if use_annualized_costs:
            cost_label = "Annualized cost [billion €]"
        else:
            cost_label = "Cost [billion €]"
        ax2_line_cost.set_ylabel(cost_label, fontsize=AXIS_LABELSIZE)
        # ax2_line_cost.set_ylim((0, 1.1 * sum(cost[:right_xlim]) / 1e9))
        ax2_line_cost.tick_params(axis="y", which="major", labelsize=TICK_LABELSIZE)

        ax_line_loss.set_xlim(0, right_xlim)
        ax_line_loss.set_ylim(bottom=0)
        ax_line_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ticks_primary = ax_line_loss.get_yticks()
        ax2_line_cost.set_yticks(ticks_primary)[:-1]
        ticks_labels_secondary = ticks_primary / rescale_factor
        if all(ticks_labels_secondary % 1 == 0):
            ticks_labels_secondary = ticks_labels_secondary.astype(int)
        ax2_line_cost.set_yticklabels(ticks_labels_secondary)

        # Adjust y-limits if cost exceeds primary y-axis limits
        primary_lims = ax_line_loss.get_ylim()
        if cost_rescaled[num_lines_to_reach_ref_loss + 1] > primary_lims[1]:
            primary_lims = (
                primary_lims[0],
                cost_rescaled[num_lines_to_reach_ref_loss] * 1.1,
            )
            ax_line_loss.set_ylim(primary_lims)
        ax2_line_cost.set_ylim(primary_lims)

        # Add reference line for line number
        ax_line_loss.plot(
            [0, num_lines_to_reach_ref_loss],
            [1, 1],
            "--",
            c=color_reference_loss,
        )
        ax_line_loss.plot(
            [num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss],
            [0, 1],
            "--",
            c=color_reference_loss,
        )
        ax_line_loss.text(
            num_lines_to_reach_ref_loss + right_xlim / 200 * 5,
            1.1,
            f"{num_lines_to_reach_ref_loss} lines",
            verticalalignment="bottom",
            horizontalalignment="left",
            zorder=np.inf,
            fontsize=TICK_LABELSIZE,
            c=color_reference_loss,
        )

        # Add reference line for cost
        ax2_line_cost.plot(
            [num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss],
            [0, cost_to_reach_ref_loss_lvl * rescale_factor / 1e9],
            "--",
            c=color_reference_loss,
        )
        ax2_line_cost.text(
            num_lines_to_reach_ref_loss - right_xlim / 200 * 2.5,
            cost_to_reach_ref_loss_lvl * rescale_factor / 1e9 + primary_lims[1] / 50,
            f"{cost_to_reach_ref_loss_lvl/ 1e9:.2f} bn. €",
            verticalalignment="center",
            horizontalalignment="right",
            zorder=np.inf,
            fontsize=TICK_LABELSIZE,
            c=color_reference_loss,
        )

        ax_line_loss.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax_line_loss.set_xlabel("Number of reinforced lines", fontsize=AXIS_LABELSIZE)
        ax_line_loss.set_ylabel(loss_axis_label, fontsize=AXIS_LABELSIZE)
        title = f"Grid extension mitigation \n CO$_2$ level={get_actual_co2_level(co2_lvl_map, percent=True)}\\%"
        ax_line_loss.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
        )
        ax_line_loss.grid(True, alpha=0.3)

        # Create shared legend for both axes
        lines1, labels1 = ax_line_loss.get_legend_handles_labels()
        lines2, labels2 = ax2_line_cost.get_legend_handles_labels()
        ax_line_loss.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=LEGEND_FONTSIZE,
        )

        # Line extension: Map
        pos = nx.get_node_attributes(nx_graph, "pos")
        reinforced_lines_selected = reinforced_lines[:num_lines_to_reach_ref_loss]
        lines_not_extended = np.setdiff1d(np.arange(n_lines), reinforced_lines_selected)
        mitigated_loss = np.array(
            [
                post_mitigation_value[i] - post_mitigation_value[i + 1]
                for i in range(len(post_mitigation_value) - 1)
            ]
        )
        mitigated_loss = np.concatenate(
            (mitigated_loss, -np.ones(n_lines - len(mitigated_loss)))
        )
        mitigated_loss_selected = -np.ones(n_lines)

        reinforced_lines_selected_int = np.array(reinforced_lines_selected).astype(int)
        mitigated_loss_selected[reinforced_lines_selected_int] = (
            mitigated_loss[:num_lines_to_reach_ref_loss] / target_value
        )

        width = 2 * (mitigated_loss_selected > 0).astype(int) + 1
        cmap = copy.copy(plt.cm.get_cmap("plasma_r"))
        cmap.set_under("gainsboro", 1.0)

        nx.draw_networkx_nodes(
            nx_graph, pos=pos, ax=ax_line_map, node_color="black", node_size=0
        )

        # Draw all lines first (background), then reinforced lines on top
        nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax_line_map,
            edge_color="gainsboro",
            width=1,
        )

        edges_list = list(nx_graph.edges())
        reinforced_edgelist = [
            edges_list[i] for i in reinforced_lines_selected_int if i < len(edges_list)
        ]
        reinforced_colors = mitigated_loss_selected[reinforced_lines_selected_int]
        reinforced_widths = width[reinforced_lines_selected_int]

        edges = nx.draw_networkx_edges(
            nx_graph,
            pos=pos,
            ax=ax_line_map,
            edgelist=reinforced_edgelist,
            edge_color=reinforced_colors,
            width=reinforced_widths,
            edge_cmap=cmap,
            edge_vmin=0,
        )

        # Add colorbar for line extension map
        bbox = ax_line_map.get_position()
        ax_line_map_legend = f.add_axes(
            (bbox.x0 + bbox.width * 0.1, bbox.y0 - 0.01, bbox.width * 0.8, 0.03)
        )
        y_label_line = f"$\\Delta {target_symbol}_{{\\ell}}/{target_symbol}_{{{int(round(co2_ref_percent))}\\%}}$"
        cbar_line = plt.colorbar(
            edges,
            ax=ax_line_map,
            cax=ax_line_map_legend,
            label=y_label_line,
            shrink=0.5,
            orientation="horizontal",
        )
        cbar_line.ax.tick_params(labelsize=TICK_LABELSIZE)
        cbar_line.ax.set_xlabel(y_label_line, fontsize=AXIS_LABELSIZE)

        ax_line_map.axis("off")
        ax_line_map.set_aspect("equal")
        # Set map title with appropriate symbol based on target
        ax_line_map.set_title(
            rf"Grid extension to reach ${target_symbol}_{{{int(round(co2_ref_percent))}\%}}$"
            + "\n"
            + rf"CO$_2$ level={get_actual_co2_level(co2_lvl_map, percent=True)}\%",
            fontsize=TITLE_FONTSIZE,
        )

        # Line extension: All CO2 levels plot
        if cost_to_reach_ref_loss:
            cost_df = pd.DataFrame.from_dict(
                cost_to_reach_ref_loss, orient="index", columns=["cost_ref_loss"]
            )
            lines_df = pd.DataFrame.from_dict(
                lines_to_reach_ref_loss, orient="index", columns=["lines_needed"]
            )

            actual_co2_ref = get_actual_co2_level(co2l_ref, percent=True)

            # Plot number of lines as primary axis
            ax_line_all.plot(
                get_actual_co2_level(lines_df.index, percent=True),
                lines_df.lines_needed,
                label="Num. lines",
                color=color_reference_loss,
                linewidth=2,
            )
            ax_line_all.invert_xaxis()
            ax_line_all.set_ylim(bottom=0)

            # Add secondary y-axis for cost
            if use_annualized_costs:
                rescale_factor = 1 / 0.05
            else:
                rescale_factor = 20 / 2
            cost_rescaled = (
                cost_df.cost_ref_loss * rescale_factor / 1e9
            )  # Convert to billion €

            ax_line_all_sec = ax_line_all.twinx()
            ax_line_all_sec.plot(
                get_actual_co2_level(cost_df.index, percent=True),
                cost_rescaled,
                label="Annualized cost" if use_annualized_costs else "Cost",
                linestyle="dotted",
                color=color_reference_loss,
                linewidth=2,
                marker="o",
                markersize=4,
            )

            ax_line_all.set_xlabel(
                "CO$_2$ level [\\% of 1990]", fontsize=AXIS_LABELSIZE
            )
            ax_line_all.set_ylabel(
                "Number of reinforced lines", fontsize=AXIS_LABELSIZE
            )
            if use_annualized_costs:
                cost_all_label = "Annualized cost [billion €]"
            else:
                cost_all_label = "Cost [billion €]"
            ax_line_all_sec.set_ylabel(cost_all_label, fontsize=AXIS_LABELSIZE)
            ax_line_all.invert_xaxis()
            ax_line_all_sec.invert_xaxis()
            ax_line_all.set_title("Grid extension needed", fontsize=TITLE_FONTSIZE)
            ax_line_all.tick_params(
                axis="both", which="major", labelsize=TICK_LABELSIZE
            )
            ax_line_all_sec.tick_params(
                axis="y", which="major", labelsize=TICK_LABELSIZE
            )
            ax_line_all.grid(True, alpha=0.3)

            ticks_primary = ax_line_all.get_yticks()
            ax_line_all_sec.set_yticks(ticks_primary)[
                -1
            ]  # Exclude last tick because it changes ylim
            ticks_labels_secondary = ticks_primary / rescale_factor
            if all(ticks_labels_secondary % 1 == 0):
                ticks_labels_secondary = ticks_labels_secondary.astype(int)
            ax_line_all_sec.set_yticklabels(ticks_labels_secondary)
            # ax_line_all_sec.set_yticklabels(
            #     [f"{tick:.0f}" for tick in ticks_secondary]
            # )
            primary_lims = ax_line_all.get_ylim()
            ax_line_all_sec.set_ylim(primary_lims)

            # Add combined legend
            lines1, labels1 = ax_line_all.get_legend_handles_labels()
            lines2, labels2 = ax_line_all_sec.get_legend_handles_labels()
            ax_line_all.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper left",
                fontsize=LEGEND_FONTSIZE,
            )

    # === ADD SUBPLOT LABELS ===
    if plot_rows == "both":
        axes_all = [
            ax_inertia_loss,
            ax_inertia_map,
            ax_inertia_all,
            ax_line_loss,
            ax_line_map,
            ax_line_all,
        ]
    elif plot_rows == "inertia":
        axes_all = [
            ax_inertia_loss,
            ax_inertia_map,
            ax_inertia_all,
        ]
    else:  # line_extension
        axes_all = [
            ax_line_loss,
            ax_line_map,
            ax_line_all,
        ]

    # Align panel labels to a consistent height per row
    if plot_rows == "both":
        axes_rows = [
            [ax_inertia_loss, ax_inertia_map, ax_inertia_all],
            [ax_line_loss, ax_line_map, ax_line_all],
        ]
    else:
        axes_rows = [axes_all]

    label_index = 0
    default_y_offset = 0.13
    for row_axes in axes_rows:
        row_positions = [ax.get_position() for ax in row_axes]
        row_max_y = max(pos.y1 for pos in row_positions)
        row_max_h = max(pos.height for pos in row_positions)
        y_target = row_max_y + default_y_offset * row_max_h

        for ax in row_axes:
            pos = ax.get_position()
            y_offset = (y_target - pos.y1) / pos.height
            add_panel_label(ax, label_index, x_offset=-0.10, y_offset=y_offset)
            label_index += 1

    # === APPLY CONSISTENT STYLING TO ALL AXES ===
    # Apply consistent font styling to all text elements
    styling_axes = []
    if plot_rows in ["both", "inertia"]:
        styling_axes.extend([ax_inertia_loss, ax_inertia_all])
    if plot_rows in ["both", "line_extension"]:
        styling_axes.extend([ax_line_loss, ax_line_all])

    for ax in styling_axes:

        # Apply consistent styling to map titles
        map_axes = []
        if plot_rows in ["both", "inertia"]:
            map_axes.append(ax_inertia_map)
        if plot_rows in ["both", "line_extension"]:
            map_axes.append(ax_line_map)

        for ax in map_axes:
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)

    plt.tight_layout()

    if plot_rows == "both":
        f_name = "combined"
    elif plot_rows == "inertia":
        f_name = "inertia"
    else:  # line_extension
        f_name = "lineExtension"
    f_name += f"_mitigation_plot"
    f_name += f"_{co2_lvl_map}"
    if use_annualized_costs:
        f_name = f_name + "_annualized"
    if build_380kV_only:
        f_name = f_name + "_380kVonly"
    if plot_intertia_cost:
        f_name = f_name + "_inertiaCost"
    if blackoutthreshold is not None and blackoutthreshold > 0.0:
        f_name = f_name + f"_blackoutThres{blackoutthreshold}"
    f_name = f_name + f"_{target}"
    if plot_rows != "both":
        f_name = f_name + f"_{plot_rows}"
        
    if save_prefix is not None:
        f_name = save_prefix + "_" + f_name

    save_figure(f, f_name, save_path, organize_plots=organize_plots)


def blackout_size_histogram_after_mitigation(
    n_nodes: int = 600,
    co2_lvls_tup: tuple[float, ...] | None = (0.0, .2),
    budget_invest_bn_tup: tuple[float, ...] = (2, .3),
    n_bins: int = 51,
    xlims: tuple[float, float] = (0, 100),
    ylog_scale: bool = True,
    calc_inertia_again: bool = False,
    linewidth: float = 1.5,
    alpha: float = .8,
    save_prefix: str | None = None):
    """Plot the size of blackouts after mitigation and compare it with before.

    Args:
        n_nodes (int, optional): Number of nodes of the PyPSA network. Defaults to 600.
        co2_lvls_tup (tuple[float, ...] | None, optional): CO2 levels that are supposed 
            to be shown. Defaults to (0.0, .2).
        budget_invest_bn_tup (tuple[float, ...], optional): Tuple with different budgets in 
            bn euro that can be used for mitigation. Defaults to (2, .3).
        n_bins (int, optional): Number of bins for the blackout sizes. Defaults to 101.
        xlims (tuple[float, float], optional): x-limits that is being plotted. 
            Defaults to (0, 100).
        ylog_scale (bool, optional): Is the y axis in log_scale? Defaults to True.
        calc_inertia_again (bool, optional): If 'True', the calculation 
            of the split properites dataframe for a given budget is repeated. 
            Defaults to False.
        linewidth (float, optional): Line width to the plot. Defaults to 1.5.
        alpha (float, optional): Color alpha of the lines. Defaults to .8.
        save_prefix (str): If not None, this str is put before the filename.
    """
    
    # Load data
    ## Original
    split_props = pd.read_hdf(path_to_vis_results_sclopf + 
                              f"/split_properties_all_n{n_nodes}.h5", index_col=0)
    split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(float)
    
    split_props["total_weighting"] = (
        split_props["snapshot_weighting"] * split_props["trigger_weighting"]
    )
    
    # Plot it
    setup_matplotlib_style()
    color_ls = ['#1b9e77',
                '#d95f02',
                '#7570b3',
                '#e7298a']
    linestyle_ls = ["--", "-.", ":"]
    
    if len(budget_invest_bn_tup) > 3:
        raise ValueError("Only a maximum of 3 different budget values is reasonable!")
    
    co2_lvls_available = np.array(get_co2_levels(n_nodes))
    
    if co2_lvls_tup is None:
        co2_lvls_tup = co2_lvls_available
    
    # Pick the co2lvls from the list and check if everything in in the list
    co2_lvls_picked_ls = list()
    for idx, co2l_r in enumerate(co2_lvls_tup):
        diff_list = abs(co2_lvls_available - co2l_r)
        idx_co2_list = np.argmin(diff_list)
        
        if diff_list[idx_co2_list] > 1e-6 or np.count_nonzero(diff_list < 1e-6) != 1:
            raise ValueError(f"The co2 value {co2l_r} is not in the available co2 levels!")
        co2_lvls_picked_ls.append(co2_lvls_available[idx_co2_list])
    
    # Sort provided lists/tuples
    co2_lvls_picked_ls = sorted(co2_lvls_picked_ls)[::-1]
    budget_invest_bn_tup = sorted(budget_invest_bn_tup)
    
    n_rows = len(co2_lvls_picked_ls)
    
    fig, ax_arr = plt.subplots(n_rows, 2, figsize=(8, 2*n_rows))
    ax_ls_line = ax_arr[:, 1].flatten()
    ax_ls_inertia = ax_arr[:, 0].flatten()
    
    bins = np.linspace(min(xlims), max(xlims), n_bins)
    
    ## Orignal Plots
    all_counts = list()
    pbar = tqdm(enumerate(co2_lvls_picked_ls), 
                            total=len(co2_lvls_picked_ls))
    for idx, co2l_r in pbar:
        pbar.set_description(f"Co2={co2l_r:.2f}")
        pbar.update()
        co2_mask = split_props.co2l == co2l_r
        split_props_co2 = split_props[co2_mask]
        vals =(
            split_props_co2.lost_load_share_blackout.values * 100.
        )
        
        ax_ls_line[idx].hist(vals, bins=bins, 
                            weights=split_props_co2.total_weighting,
                            alpha=alpha, 
                            histtype="step",
                            log=ylog_scale,
                            label=rf"CO$_2$: {get_actual_co2_level(co2l_r, n_nodes=n_nodes, percent=True)}\%" ,
                            zorder=-2,
                            linewidth=linewidth,
                            color=color_ls[0])
        ax_ls_inertia[idx].hist(vals, bins=bins, 
                            weights=split_props_co2.total_weighting,
                            alpha=alpha, 
                            histtype="step",
                            log=ylog_scale,
                            label=rf"CO$_2$: {get_actual_co2_level(co2l_r, n_nodes=n_nodes, percent=True)}\%",
                            zorder=-2,
                            linewidth=linewidth,
                            color=color_ls[0])
        
        # Line mitigation
        fname_line_mitigation = os.path.join(path_to_line_extension_mitigation_sclopf, 
                                             f"heuristic_costMin_GSS_mitigation_annualized_Co2L{co2l_r}_n{n_nodes}_380kVonly_blackoutthres0.8.pkl")
        with open(fname_line_mitigation, 'rb') as fh_line_in:
            reinforced_lines, loss_w_mitigation, num_blackouts_with_mitigation, \
                cost =  pickle.load(fh_line_in)
                
        cumu_cost = np.cumsum(cost)
        
        for idx_budget, budget_invest_bn_r in enumerate(budget_invest_bn_tup):
            idxs_under_budget = np.where(cumu_cost <= (budget_invest_bn_r*1e9))

            lines_reinforced_under_budget = np.array(reinforced_lines)[idxs_under_budget]
            
            if len(lines_reinforced_under_budget) == len(reinforced_lines):
                logger.warning("Chosen line extenstion budget is over the value that was calculated")
            
            mask_sprops_lines = (split_props_co2.init_failure_0.isin(lines_reinforced_under_budget)) |\
                (split_props_co2.init_failure_1.isin(lines_reinforced_under_budget))
            
            split_props_co2_lines = split_props_co2[~mask_sprops_lines]
            
            vals_lines = split_props_co2_lines.lost_load_share_blackout.values * 100.
            
            ax_ls_line[idx].hist(vals_lines, bins=bins,
                                weights=split_props_co2_lines.total_weighting,
                                alpha=alpha,
                                histtype="step",
                                log=ylog_scale,
                                zorder=0,
                                linestyle=linestyle_ls[idx_budget],
                                label=f"{budget_invest_bn_r}",
                                linewidth=linewidth,
                                color=color_ls[1+idx_budget])
        
        # Additional inertia mitigation
        search_pattern = os.path.join(path_to_inertia_mitigation_results_sclopf, 
                                                f"synthetic_inertia_placement_Co2{co2l_r:g}_N{n_nodes}_deltarotE*_rocofthres1_lshare0_maxiter1000_random_blackoutthres0.8.pklz")
        fname_inertia_ls = glob.glob(search_pattern)
        if len(fname_inertia_ls) != 1:
            print(search_pattern)
            raise IOError(f"Either not file found or multiple found. List has lenght {len(fname_inertia_ls)}")
        else:
            fname_inertia = fname_inertia_ls[0]
        
        inertia_step_delta_rotE_MWs = float(fname_inertia.split("deltarotE")[-1].split("_rocofthres")[0])
        
        with gzip.open(fname_inertia, 'rb') as fh_inertia_in:
            _, _, inertia_placed_loss_mitigated_ls, comp_mitigated_step, _, _ = pickle.load(fh_inertia_in)
        
        fname_comp_property_df = os.path.join(path_to_evaluation_results_sclopf, 
                                              f"component_properties_Co2L{co2l_r}_n{n_nodes}.h5")
        
        comp_property_df = pd.read_hdf(fname_comp_property_df, key="df")
        # Get how much inertia is placed before reaching budges. Get number of steps
        comp_mitigated_step = comp_mitigated_step.sort_values()
        comp_mitigated_step = comp_mitigated_step[comp_mitigated_step != -1]
        
        inertia_unit_fac = 1e-3
        inertia_placed_df = pd.DataFrame(inertia_placed_loss_mitigated_ls,
                                         columns=["idx_step", "idx_node", "delta_Erot_fac", 
                                                  "max_change", 
                                                  "components_beyond_threshold"])
        inertia_placed_df["delta_Erot_GWs"] = inertia_placed_df.loc[:, "delta_Erot_fac"] \
            * inertia_step_delta_rotE_MWs * inertia_unit_fac
        inertia_placed_df["cum_Erot_GWs"] = inertia_placed_df["delta_Erot_GWs"].cumsum()
        
        scale_Erot_to_bn = annualized_cost_per_GWs_max / 1e9
        
        inertia_placed_df["cost_bn"] = inertia_placed_df["delta_Erot_GWs"] * scale_Erot_to_bn 
        inertia_placed_df["cum_cost_bn"] = inertia_placed_df["cost_bn"].cumsum()
        
        for idx_budget_i, budget_inertia in enumerate(budget_invest_bn_tup):
            comp_property_r = comp_property_df.copy()
            fname_inertia_sprops = f"mitigated_split_properties_Co2L{co2l_r}_n{n_nodes}_budget{budget_inertia:.4f}.h5"
            fpath_inertia_mitigated_sprops = os.path.join(path_to_plot_data, 
                                                          fname_inertia_sprops)
            
            if not os.path.exists(fpath_inertia_mitigated_sprops) or calc_inertia_again:
                logger.info(f"Finding sprops for inertia mitigation w. budget {budget_inertia}")
                # Find out components that have been mitigated with the current budget
                inertia_mask_under_budget = inertia_placed_df.cum_cost_bn <= budget_inertia
            
                inertia_budget_row = inertia_placed_df[inertia_mask_under_budget].iloc[-1]
                
                inertia_last_step = inertia_budget_row.idx_step
                
                # TODO should this effectively reordering take place of should it just be cut outs?
                inertia_comps_miti_index = comp_mitigated_step[comp_mitigated_step < inertia_last_step].index
                
                # Set the 'blackout_load_loss_share' to zero where it was mitigated               
                comp_property_r.loc[inertia_comps_miti_index, "blackout_load_loss_share"] = 0
                logger.info(f"---> {len(inertia_comps_miti_index)} components were mitigated")
                inertia_split_groups = comp_property_r.groupby(["time_stamp", "split_number"])

                inertia_split_props_r = pd.DataFrame(index=inertia_split_groups.groups.keys())
                inertia_split_props_r.index = inertia_split_props_r.index.rename(
                    ["time_stamp", "split_number_snapshot"]
                )
                
                inertia_split_props_r["lost_load_share_blackout"] = (
                    inertia_split_groups.blackout_load_loss_share.sum().astype(float)
                )
                trigger_weighting = inertia_split_groups.trigger_weighting.unique().astype(int)
                inertia_split_props_r["trigger_weighting"] = trigger_weighting
                
                inertia_split_props_r["load"] = inertia_split_groups.load.sum().astype(float)
                
                snapshot_weightings = data_handling.load_pypsa_network(
                    co2lvl=co2l_r, n_nodes=n_nodes, use_sclopf=True
                    ).snapshot_weightings
                inertia_split_props_r["snapshot_weighting"] = snapshot_weightings.generators.loc[
                    inertia_split_props_r.index.get_level_values("time_stamp")
                    ].values.astype(int)
                inertia_split_props_r["total_weighting"] = (
                    inertia_split_props_r["snapshot_weighting"] * \
                        inertia_split_props_r["trigger_weighting"]
                )
                
                inertia_split_props_r.to_hdf(fpath_inertia_mitigated_sprops, key="df",
                                            complib='zlib', complevel=9)
                logger.info("Finished sprops calc.")
            else:
                inertia_split_props_r = pd.read_hdf(fpath_inertia_mitigated_sprops, key="df")
            
            # Plot it
            vals_inertia = inertia_split_props_r.lost_load_share_blackout * 100.
            
            #color_i_r = f"C{idx_budget_i + 1}"
            ax_ls_inertia[idx].hist(vals_inertia, bins=bins,
                                    weights=inertia_split_props_r.total_weighting,
                                    alpha=alpha,
                                    histtype="step",
                                    log=ylog_scale,
                                    zorder=0,
                                    linestyle=linestyle_ls[idx_budget_i],
                                    label=f"{budget_inertia:.2f}",
                                    linewidth=linewidth,
                                    color=color_ls[1+idx_budget_i])
            
        
    # Aesthetics
    [xx.sharex(ax_ls_line[0]) for xx in ax_ls_line[1:]]
    [xx.sharex(ax_ls_inertia[0]) for xx in ax_ls_inertia[1:]]
    
    [xx.sharey(ax_ls_line[0]) for xx in ax_ls_line[1:]]
    [xx.sharey(ax_ls_inertia[0]) for xx in ax_ls_inertia[1:]]
    
    for idx_r, ax_r in enumerate(ax_arr[:, 0]):
        ax_r.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        ax_r.text(
            -0.3, 0.5,
            f"CO$_2$ level $= {round(get_actual_co2_level(co2_lvls_picked_ls[idx_r], n_nodes=n_nodes)*100)}$\\%",
            transform=ax_r.transAxes,
            fontsize=AXIS_LABEL_FONTSIZE,
            ha="center", va="center",
            rotation=90,
        )
    
    for ax_r in [ax_ls_line[0],ax_ls_inertia[0]]:
        ax_r.set_xlim(left=min(bins), right=max(bins))
        
    for ax_r in ax_arr[:-1, :].flatten():
        ax_r.tick_params(labelbottom=False)
        
    for ax_r in ax_arr[:, 1:].flatten():
        ax_r.tick_params(labelleft=False)
    
    for ax_r in ax_arr[-1, :].flatten():
        ax_r.set_xlabel(
                r"Share of load not served [\%]", fontsize=AXIS_LABEL_FONTSIZE
            )
    
    xlabel = ax_arr[-1, 0].xaxis.get_label()
    xlabel_bbox = xlabel.get_window_extent(fig.canvas.get_renderer())
    xlabel_fig_pos = xlabel_bbox.transformed(fig.transFigure.inverted())
    
    y_pos = xlabel_fig_pos.y0 - 0.0  # 5% below xlabel
    
    labels_fig_legend = [f"{0:.2f}"] + [f"{xx:.2f}" for xx in budget_invest_bn_tup]
    
    leg = fig.legend(labels_fig_legend,
               loc="upper center",
               ncols=3,
               title="Mitigation Budget [bn €]",
               bbox_to_anchor=(0.5, y_pos ),
               fontsize=LEGEND_FONTSIZE,
               )
    
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_linewidth'):
            handle.set_linewidth(2.5)
    
    ax_ls_inertia[0].text(.5, 1.25, "Additional Inertia",
                          transform=ax_ls_inertia[0].transAxes,
                          ha="center", fontsize=TITLE_FONTSIZE)
    ax_ls_line[0].text(.5, 1.25, "Line Reinforcement",
                       transform=ax_ls_line[0].transAxes,
                       ha="center", fontsize=TITLE_FONTSIZE)
    
    for idx_r, ax_r in enumerate(ax_arr.flatten()):
        add_panel_label(ax_r, idx_r, x_offset=-.15, y_offset=.075)
    
    # Save it
    fname_fig = f"blackout_sizes_after_mitigation_n{n_nodes}"
    
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return

if __name__ == "__main__":
    params = [
        {"blackoutthreshold": 0.8, "target": "num_GSS"},
        # {"blackoutthreshold": None, "target": "total_loss"},
    ]
    for param in params:
        blackoutthreshold = param["blackoutthreshold"]
        target = param["target"]
        for plot_rows_option in ["line_extension", "both", "inertia"]:
            print(
                "Mitigation params:",
                f"blackoutthreshold={blackoutthreshold}",
                f"target={target}",
                f"plot_rows={plot_rows_option}",
            )
            create_combined_mitigation_plot(
                use_annualized_costs=True,
                co2_lvl_map=0.2,
                build_380kV_only=True,
                plot_intertia_cost=True,
                blackoutthreshold=blackoutthreshold,
                plot_rows=plot_rows_option,  # Options: "both", "inertia", "line_extension"
                target=target,
            )
