#!/usr/bin/env python

"""Plot figure that give more insights into the scenarios
generated with PyPSA."""

import os 

import gzip
import pickle

from utils.data_handling import \
    (load_pypsa_network, build_networkx_graph, 
     get_co2_levels, get_actual_co2_level)

from utils.plot_style import *
setup_matplotlib_style()

from utils.config import path_to_figures_sclopf

from utils import calculate_line_loadings

import networkx as nx

from tqdm import tqdm
from loguru import logger

from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
import pandas as pd
import matplotlib.pyplot as plt
from utils.plot_style import FOUR_PANEL_SIZE,LEGEND_FONTSIZE,TICK_LABEL_FONTSIZE,setup_matplotlib_style
from matplotlib import ticker


def histogram_lineloading_accross_co2lvls(n_nodes: int = 600, calc_again: bool = False,
                                          condition_on_split: bool = False,
                                          n_bins: int = 100, xscale_log: bool = False,
                                          bin_range: tuple[float, float] = (0.5, 1.)):
    """Draw histograms of line loadings for different co2 levels
    """
    
    condition_str = "_givenSplit" if condition_on_split else ""
    
    path_to_results = path_to_figures_sclopf + \
        f"/plot_data/data_line_loading_n{n_nodes}{condition_str}.pklz"
    
    available_co2_lvls = get_co2_levels(n_nodes=n_nodes)
    
    if not os.path.exists(path_to_results) or calc_again:
        # Find line loading for all co2_lvls
        
        flow_n_lineloading_dict: dict[float, tuple[dict, dict]] = dict()
        
        for co2l_r in tqdm(available_co2_lvls, desc="Co2lvls"):
            flow_snapshot_dict, line_loading_dict = calculate_line_loadings.calculate_line_loadings_for_all_snapshots(co2l_r, n_nodes=n_nodes,
                                                                              condition_on_split=False)
        
            flow_n_lineloading_dict[co2l_r] = flow_snapshot_dict, line_loading_dict
        
        with gzip.open(path_to_results, "wb") as fh_out:
            pickle.dump(flow_n_lineloading_dict, fh_out)
        
    else:
        with gzip.open(path_to_results, "rb") as fh_in:
            flow_n_lineloading_dict = pickle.load(fh_in)
        
        if not(set(flow_n_lineloading_dict) == set(available_co2_lvls)):
            raise ValueError("Dictionary should contain all available co2 levels. Please check if all PyPSA Sclopf Scenarios" + 
                             " are in the right place and calcualte line loading again!")
    
    # Plot it
    fig, ax = plt.subplots()
    
    cmap_co2 = plt.get_cmap("cividis").copy()
    
    if xscale_log:
        line_loading_bins = np.logspace(np.log10(min(bin_range)), np.log10(bin_range), n_bins, endpoint=True)
    else:
        line_loading_bins = np.linspace(min(bin_range), max(bin_range), n_bins, endpoint=True)
    
    for key_r, [flow_dict_r, ll_dict_r] in tqdm(flow_n_lineloading_dict.items(), desc="Plotting hists..."):
        
        color_r = cmap_co2(np.where(available_co2_lvls == key_r)[0][0] / (len(co2ls) - 1))
        
        all_line_loadings = [item for sublist in ll_dict_r.values() for item in sublist]
        ax.hist(all_line_loadings, bins=line_loading_bins,
                histtype='step', density=True,
                color=color_r, 
                linewidth=LINE_WIDTH, label=f"{key_r * 100}\\%")
    
    # Aesthetics
    if xscale_log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Line Loading $\\ell_{\\text{load}}$")
    ax.set_ylabel("$P(\\ell_{\\text{load}})$")
    
    ax.set_xlim(min(bin_range), max(bin_range))
    
    ax.legend(loc="lower left",
              title="CO$_2$ level [\\% of 1990]",
              fontsize=14, title_fontsize=14)
    
    fname_fig = f"line_loading_histogram_n{n_nodes}"
    if xscale_log:
        fname_fig += "_xlog"
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return


def graph_with_num_parallel(co2lvl: float = .6, n_nodes: int = 600, 
                            snet_index: str = '0',
                            edge_width: float = 2.,
                            log_scale: bool = False,
                            vlims_edges: tuple[float, float] | None = (1e-1, 1e1)):
    """Plot the graph of the CE network with num_parallel on the acis"""
    
    net = load_pypsa_network(co2lvl=co2lvl, n_nodes=n_nodes)
    
    nx_graph = build_networkx_graph(net, snet_index=snet_index)
    pos_graph = nx.get_node_attributes(nx_graph, "pos")
    num_parallel_vals = np.array([xx for _, _, xx in nx_graph.edges(data="num_parallel")])
    
    if any(num_parallel_vals < -1e-8):
        raise ValueError("Num Parallel values should be >= 0")
    if log_scale:
        cval_edges: list[float] = [np.log10(xx) if xx > 1e-12 else -np.inf 
                      for xx in num_parallel_vals ]
    else:
        cval_edges: list[float] = list(num_parallel_vals)
    
    # Plot it
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    """nodes = nx.draw_networkx_nodes(nx_graph, pos=pos_graph, 
                                   ax=ax, node_color="k",
                                   node_size=0)
    nodes.set_zorder(-1)"""
    
    
    cmap_edges = plt.get_cmap("viridis").copy()
    
    edges = nx.draw_networkx_edges(nx_graph, pos=pos_graph, 
                                   edge_color=cval_edges,
                                   edge_cmap=cmap_edges,
                                   width=edge_width)
    
    edges.set_zorder(0)
    
    if vlims_edges is None:
        vlims_edges = (min(num_parallel_vals), max(num_parallel_vals))
        
    
    ## Colorbar
    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes([ax_pos.x1-.075, ax_pos.y0, 
                            .025, ax_pos.height])
    if log_scale:
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, 
                                        norm=mplcolors.LogNorm(vmin=min(vlims_edges), 
                                                               vmax=max(vlims_edges)))
    else:
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, 
                                        norm=mplcolors.Normalize(vmin=min(vlims_edges), 
                                                                 vmax=max(vlims_edges)))
    
    cb_edges = fig.colorbar(sm_edge, cax=cbar_ax)
    cb_edges.set_label("$\\ell_{\\text{parallel}}$", size=AXIS_LABEL_FONTSIZE*1.5)
    
    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    
    fname_fig = f"graph_num_parallel_co2l{co2lvl}_n{n_nodes}"
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return



def plot_investments_capital_costs(n_nodes,agg_solar:bool = True, agg_gas:bool = True):


    '''Plot the capital and operational costs for different co2 levels.
    This function loads the PyPSA networks for different CO2 levels, calculates the operational and capital costs for generators and storage units, and plots the costs as stacked bar charts for each CO2 level. The resulting figure shows how the cost composition changes with CO2 levels.
    args:
    n_nodes (int): The number of nodes in the PyPSA networks to load and analyze.
    agg_solar (bool): Whether to aggregate solar and solar-hsat costs into a single "solar" category.
    agg_gas (bool): Whether to aggregate CCGT and OCGT costs into a single "gas" category.

    '''
    co2ls = get_co2_levels(n_nodes=n_nodes)
    networks = {
        c:   load_pypsa_network(co2lvl=c, n_nodes=n_nodes)
        for c in co2ls
    }

    n_ref = networks[co2ls[0]]

    colors_gens = n_ref.carriers.color
    nice_names = n_ref.carriers.nice_name
    short_nice_names = {"battery": "Battery", "hydro": "Hydro", "PHS": "Pumped hydro"}
    for name, nicename in short_nice_names.items():
        nice_names[name] = nicename


    opex = {}
    capex = {}
    storage_capex = {}

    for co2, n in networks.items():

        # Operational costs
        gen_dispatch = n.generators_t.p.sum()

        gen_opex = (
            gen_dispatch
            * n.generators.marginal_cost
        ) #€/MWh

        gen_opex = gen_opex.groupby(
            n.generators.carrier
        ).sum()

        opex[co2] = gen_opex
       # Generator investment costs

        gen_capex = (
            n.generators.p_nom_opt
            * n.generators.capital_cost
        )

        gen_capex = gen_capex.groupby(
            n.generators.carrier
        ).sum()
       # Storage investment costs

        if len(n.storage_units) > 0:
            sto_cap = (
                n.storage_units.p_nom_opt
                * n.storage_units.capital_cost
            )

            sto_cap = sto_cap.groupby(
                n.storage_units.carrier
            ).sum()
        else:
            sto_cap = pd.Series(dtype=float)

        capex[co2] = gen_capex
        storage_capex[co2] = sto_cap


    # CONVERT TO DATAFRAMES
    opex_df = pd.DataFrame(opex).fillna(0).T
    capex_df = pd.DataFrame(capex).fillna(0).T
    storage_df = pd.DataFrame(storage_capex).fillna(0).T


    opex_df = opex_df.drop(columns=["geothermal", "art_load","load"])
    capex_df = capex_df.drop(columns=["geothermal", "art_load","load"])

    if agg_solar == True:
        opex_df_solar = (opex_df['CCGT'] + opex_df['OCGT']).copy()
        opex_df = opex_df.drop(columns=["CCGT", "OCGT"])
        opex_df['gas'] = opex_df_solar
        capex_df_solar = (capex_df['solar'] + capex_df['solar-hsat']).copy()
        capex_df = capex_df.drop(columns=["solar", "solar-hsat"])
        capex_df['solar'] = capex_df_solar

    if agg_gas == True:
        opex_df_solar = (opex_df['solar'] + opex_df['solar-hsat']).copy()
        opex_df = opex_df.drop(columns=["solar", "solar-hsat"])
        opex_df['solar'] = opex_df_solar
        capex_df_gas = (capex_df['CCGT'] + capex_df['OCGT']).copy()
        capex_df = capex_df.drop(columns=["CCGT", "OCGT"])
        capex_df['gas'] = capex_df_gas
        colors_gens['gas'] =   "#cc0099"
        nice_names['gas'] = 'Gas'

    colors_gens['H2'] = "green"
    colors_gens['battery'] = "tab:olive"

    opex_df_offwind = (opex_df['offwind-dc'] + opex_df['offwind-ac']+ opex_df['offwind-float']).copy()
    opex_df = opex_df.drop(columns=["offwind-dc", "offwind-ac",'offwind-float'])
    opex_df['offwind'] = opex_df_offwind

    capex_df_offwind = (capex_df['offwind-dc'] + capex_df['offwind-ac'] + capex_df['offwind-float']).copy()
    capex_df = capex_df.drop(columns=["offwind-dc", "offwind-ac",'offwind-float'])
    capex_df['offwind'] = capex_df_offwind

    capex_df /= 1e9
    opex_df /= 1e9
    storage_df /= 1e9
    colors_gens['offwind'] =  colors_gens['offwind-dc']
    nice_names['offwind'] = 'Offshore Wind'

    # PLOT
    fname_fig = "supl_costs_per_co2"
    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 12),
        sharex=True
    )

    # ----------------------------
    # OPEX
    # ----------------------------

    opex_df.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=colors_gens[opex_df.columns]

    )

    axes[0].set_ylabel(r"Cost [bn €/a]", fontsize=AXIS_LABEL_FONTSIZE)

    # CAPEX
    capex_df.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=colors_gens[capex_df.columns],
        legend=False

    )

    axes[1].set_ylabel(r"Cost [bn €/a]", fontsize=AXIS_LABEL_FONTSIZE)

    # STORAGE CAPEX
    storage_df.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        color=colors_gens[storage_df.columns],
    )

    axes[2].set_ylabel(r"Cost [€/a]", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)


    handles, labels = axes[0].get_legend_handles_labels()

    new_labels = [nice_names.get(l, l).capitalize() for l in labels.copy()]

    axes[0].legend(handles, new_labels,
                    loc="upper left",
                    ncol=2,
                    fontsize=LEGEND_FONTSIZE,
    )

    handles, labels = axes[2].get_legend_handles_labels()

    new_labels = [nice_names.get(l, l).capitalize() for l in labels.copy()]

    axes[2].legend(handles, new_labels, 
                    loc="upper right",
                    ncol=2,
                    fontsize=LEGEND_FONTSIZE,
    )


    co2_order = np.array(sorted(co2ls))
    x_labels = co2_order.copy()#[f"{get_actual_co2_level(c, percent=True):g}%" for c in co2_order]
    x_labels[-1] = 0.58
    axes[2].set_xticklabels(x_labels, rotation=0)

    for idx, ax in enumerate(axes):
        add_panel_label(ax, idx, x_offset=-0.-0.03)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    plt.show()

    save_figure(fig, fname_fig, path_to_figures_sclopf)

    return




setup_matplotlib_style()

def plot_load_dispatch(n_nodes,agg_solar:bool = True, agg_gas:bool = True, agg_offwind:bool = True,drop_carriers:bool = True):

    '''Plot the average load and dispatch profiles for different seasons and co2 levels.

    This function loads the PyPSA networks for different CO2 levels, aggregates the generator and storage dispatch profiles by carrier, and plots the average profiles for each season (winter, spring, summer, autumn). The resulting figure shows how the supply and load profiles change with CO2 levels and seasons.

    args:
        n_nodes (int): The number of nodes in the PyPSA networks to load and analyze.
        agg_solar (bool): Whether to aggregate solar and solar-hsat profiles into a single "solar" category.
        agg_gas (bool): Whether to aggregate CCGT and OCGT profiles into a single "gas" category.
        agg_offwind (bool): Whether to aggregate offwind-dc, offwind-ac, and offwind-float profiles into a single "offwind" category.
        drop_carriers (bool): Whether to drop certain carriers (geothermal, art_load, load) from the profiles before plotting.
    '''
    # Load networks for all CO2 levels
    co2ls = get_co2_levels(n_nodes=n_nodes)
    networks = {
        c:   load_pypsa_network(co2lvl=c, n_nodes=n_nodes)
        for c in co2ls
    }

    n_ref = networks[co2ls[0]]

    nice_names = n_ref.carriers.nice_name
    colors_gens = n_ref.carriers.set_index("nice_name").color

    colors_gens["Offshore Wind"] = n_ref.carriers.color.get("offwind-dc", "#1f77b4")
    colors_gens['Hydrogen Storage'] = "green"
    colors_gens['Battery Storage'] = "tab:olive"
    nice_names['gas'] = 'Gas'
    nice_names['offwind'] = 'Offshore Wind'
    colors_gens['Gas'] =   "#cc0099"


    idx = n_ref.snapshots

    winter = idx.month.isin([12, 1, 2])
    spring = idx.month.isin([3, 4, 5])
    summer = idx.month.isin([6, 7, 8])
    autumn = idx.month.isin([9, 10, 11])



    for co2, n in networks.items():
        fig, axes = plt.subplots(2,2,sharex=True, sharey=True, figsize=FOUR_PANEL_SIZE)
        for ax, season,title in zip(axes.flatten(), [winter, spring, summer, autumn],["Winter", "Spring", "Summer", "Autumn"]):
            gen = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum()
            sto = n.storage_units_t.p.groupby(n.storage_units.carrier, axis=1).sum()
            supply = sto.add(gen, fill_value=0)

            load = n.loads_t.p.sum(axis=1)

            def _agg_carriers(df, agg_solar, agg_gas, agg_offwind,drop_carriers):
                df = df.copy()
                if drop_carriers:
                    drop_cols = ["geothermal", "art_load","load"]
                    df = df.drop(columns=drop_cols)
                if agg_gas:
                    cols = [c for c in ["CCGT", "OCGT"] if c in df.columns]
                    if cols:
                        df["gas"] = df[cols].sum(axis=1)
                        df = df.drop(columns=cols)
                if agg_solar:
                    cols = [c for c in ["solar", "solar-hsat"] if c in df.columns]
                    if len(cols) > 1:
                        df["solar"] = df[cols].sum(axis=1)
                        df = df.drop(columns=[c for c in cols if c != "solar"])
                if agg_offwind:
                    cols = [c for c in ["offwind-ac", "offwind-dc", "offwind-float"]
                            if c in df.columns]
                    if cols:
                        df["offwind"] = df[cols].sum(axis=1)
                        df = df.drop(columns=cols)
                return df

            supply = _agg_carriers(supply, agg_solar=True, agg_gas=True, agg_offwind=True,drop_carriers=True)
            supply_season = supply.loc[season]
            load_season = load.loc[season]



            hour = supply_season.index.hour
            avg_profile = supply_season.groupby(hour).mean()
            load = n_ref.loads_t.p.sum(axis=1)
            load_season = load.loc[season]
            avg_load = load_season.groupby(load_season.index.hour).mean()

            def stackedplot(ax, df, colors, x_col=None, alpha=0.8):
                if x_col is not None:
                    x = df[x_col].values
                    data_cols = [c for c in df.columns if c != x_col]
                else:
                    x = df.index.values
                    data_cols = list(df.columns)
                # split into pos and neg values for stacking
                pos_bottoms = np.zeros(len(x))
                neg_bottoms = np.zeros(len(x))

                #  plot each column separately
                for i, col in enumerate(data_cols):
                    vals = df[col].values
                    pos_vals = np.clip(vals, 0, None)
                    neg_vals = np.clip(vals, None, 0)

                    ax.bar(x, pos_vals, bottom=pos_bottoms, label=col,  color=colors[i], alpha=alpha)
                    ax.bar(x, neg_vals, bottom=neg_bottoms, color=colors[i], alpha=alpha)

                    pos_bottoms += pos_vals
                    neg_bottoms += neg_vals

                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            # plot supply and load
            stackedplot(ax, avg_profile/1e3, colors=[colors_gens.get(c, "#333333") for c in nice_names[avg_profile.columns]], x_col=None, alpha=0.8)
            ax.plot(avg_load.index, avg_load/1e3, color="black", linewidth=2,label="Load")

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("GW")
            ax.set_title(f"{title}")

        for idx, ax in enumerate(axes.flatten()):
            add_panel_label(ax, idx, x_offset=-0.-0.03)
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

        # add legend
        legend_ax = fig.add_axes([0.1, -0.05, 0.8, 0.12])
        legend_ax.axis("off")
        handles, labels = axes[0,0].get_legend_handles_labels()
        labels = [nice_names.get(l, l) for l in labels]
        labels = [name.capitalize() for name in labels]
        legend_ax.legend(handles, labels, ncol=4, loc="center", frameon=True,fontsize = LEGEND_FONTSIZE)

        save_figure(fig, f"load_dispatch_{n_nodes}_{co2}.pdf", path_to_figures_sclopf)

    return
