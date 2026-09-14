#!/usr/bin/env python

"""Plot figure that give more insights into the scenarios
generated with PyPSA."""

import os 

import gzip
import pickle

from utils.data_handling import \
    (load_pypsa_network, build_networkx_graph,
     get_co2_levels, get_actual_co2_level, warn_if_out_of_bin_range)

from utils.plot_style import *
setup_matplotlib_style()

from utils.config import path_to_figures_sclopf, path_to_plot_data

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


def histogram_lineloading_accross_co2lvls(n_nodes: int = 600, 
                                          calc_again: bool = False,
                                          condition_on_split: bool = False,
                                          n_bins: int = 100, 
                                          xscale_log: bool = False,
                                          bin_range: tuple[float, float] = (0., 1.),
                                          x_lim: tuple[float, float] = (.5, 1.),
                                          save_prefix: str | None = None):
    """Draw histograms of line loadings for different co2 levels
    """
    
    condition_str = "_givenSplit" if condition_on_split else ""
    
    path_to_results = os.path.join(path_to_plot_data , 
                                   f"data_line_loading_n{n_nodes}{condition_str}.pklz")
    
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
        # The bin range is fixed, not derived from the data. Line loading is |flow| divided
        # by the line limit from a re-solved load flow, so it is not guaranteed to stay
        # within bin_range the way a load share is bounded by 1 — clipping would misreport a
        # genuine overload as exactly fully loaded. Warn instead, so a dropped tail shows up
        # in the log rather than silently thinning the histogram.
        warn_if_out_of_bin_range(
            all_line_loadings, line_loading_bins,
            context=f"histogram_lineloading_accross_co2lvls (Co2L={key_r:.2f})",
        )
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
    
    ax.set_xlim(min(x_lim), max(x_lim))
    
    ax.legend(loc="lower left",
              title="CO$_2$ level [\\% of 1990]",
              fontsize=14, title_fontsize=14)
    
    fname_fig = f"line_loading_histogram_n{n_nodes}"
    if xscale_log:
        fname_fig += "_xlog"
        
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return


def graph_with_num_parallel(co2lvl: float = .6, n_nodes: int = 600, 
                            snet_index: str = '0',
                            edge_width: float = 2.,
                            log_scale: bool = False,
                            vlims_edges: tuple[float, float] | None = (1e-1, 1e1),
                            save_prefix: str | None = None):
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
    
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig
    
    save_figure(fig, fname_fig, path_to_figures_sclopf)
    
    return



def plot_investments_capital_costs(n_nodes: int = 600,
                                   agg_solar: bool = True, 
                                   agg_gas: bool = True,
                                   save_prefix: str | None = None,
                                   co2_lvls_selected: tuple[float, ...] | None = (0.0, 
                                                                             0.05, 
                                                                             0.2, 
                                                                             0.4,
                                                                             0.6)):
    """Plot the capital and operational costs for different co2 levels.
    This function loads the PyPSA networks for different CO2 levels, 
    calculates the operational and capital costs for generators and storage units, and plots the costs as stacked bar charts for each CO2 level. The resulting figure shows how the cost composition changes with CO2 levels.
    
    args:
    n_nodes (int): The number of nodes in the PyPSA networks to load and analyze.
    agg_solar (bool): Whether to aggregate solar and solar-hsat costs into a single "solar" category.
    agg_gas (bool): Whether to aggregate CCGT and OCGT costs into a single "gas" category.

    """
    co2ls = get_co2_levels(n_nodes=n_nodes)
    
    if co2_lvls_selected is not None:
        co2_in_list = list()
        for co2_r in sorted(co2_lvls_selected):
            diff_r = abs(co2ls - co2_r)
            idx_co2 = np.argmin(diff_r)
            if diff_r[idx_co2] < 1e-6:
                co2_in_list.append(co2ls[idx_co2])
        logger.info(f"Selected Co2 lvls {co2_in_list}")
        co2ls = co2_in_list        
        
    networks = {
        c:   load_pypsa_network(co2lvl=c, n_nodes=n_nodes)
        for c in co2ls
    }

    n_ref = networks[co2ls[0]]

    colors_gens = n_ref.carriers.color
    nice_names = n_ref.carriers.nice_name
    short_nice_names = {"battery": "Battery", 
                        "hydro": "Hydro", 
                        "PHS": "Pumped hydro"}
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
        colors_gens.loc['gas'] =   "#cc0099"
        nice_names.loc['gas'] = 'Gas'

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

    axes[0].set_ylabel(r"Operational cost [bn €/a]", fontsize=AXIS_LABEL_FONTSIZE)

    # CAPEX
    capex_df.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=colors_gens[capex_df.columns],
        legend=False

    )

    axes[1].set_ylabel(r"Investment cost [bn €/a]", fontsize=AXIS_LABEL_FONTSIZE)

    # STORAGE CAPEX
    storage_df.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        color=colors_gens[storage_df.columns],
    )

    axes[2].set_ylabel(r"Investment cost [bn €/a]", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].set_xlabel(r"CO$_2$ level [\% of 1990]", fontsize=AXIS_LABEL_FONTSIZE)


    handles, labels = axes[0].get_legend_handles_labels()

    new_labels = [nice_names.get(l, l).capitalize() for l in labels.copy()]

    axes[0].legend(handles, new_labels,
                    loc="upper right",
                    ncol=2,
                    fontsize=LEGEND_FONTSIZE,
    )

    handles, labels = axes[2].get_legend_handles_labels()

    new_labels = [nice_names.get(l, l).capitalize() for l in labels.copy()]

    axes[2].legend(handles, new_labels, 
                    loc="upper left",
                    ncol=2,
                    fontsize=LEGEND_FONTSIZE,
    )

    co2_order = np.array(sorted(co2ls))
    x_labels = [f"{get_actual_co2_level(c, percent=True):g}%" for c in co2_order]
    axes[2].set_xticklabels(x_labels, rotation=0)

    for idx, ax in enumerate(axes):
        add_panel_label(ax, idx, x_offset=-0.-0.03)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.invert_xaxis()
        
    fig.tight_layout()
    
    if save_prefix is not None:
        fname_fig = save_prefix + "_" + fname_fig

    save_figure(fig, fname_fig, path_to_figures_sclopf)

    return


_SEASONS = {"Winter": (12, 1, 2),
            "Spring": (3, 4, 5),
            "Summer": (6, 7, 8),
            "Autumn": (9, 10, 11)}


def _dispatch_carrier_style(n_ref):
    """Return (nice_names, colors_gens) for the dispatch plots.

    The carrier table of the network is patched with the aggregated carriers
    ('gas', 'offwind') and with explicit colours for the storage carriers.

    args:
        n_ref: PyPSA network used as reference for carrier names and colours.
    """
    nice_names = n_ref.carriers.nice_name.copy()
    colors_gens = n_ref.carriers.set_index("nice_name").color.copy()

    colors_gens["Offshore Wind"] = n_ref.carriers.color.get("offwind-dc", "#1f77b4")
    colors_gens['Hydrogen Storage'] = "green"
    colors_gens['Battery Storage'] = "tab:olive"
    nice_names.loc['gas'] = 'Gas'
    nice_names['offwind'] = 'Offshore Wind'
    colors_gens['Gas'] =   "#cc0099"

    return nice_names, colors_gens


def _agg_dispatch_carriers(df,
                           agg_solar: bool = True,
                           agg_gas: bool = True,
                           agg_offwind: bool = True,
                           drop_carriers: bool = True):
    """Aggregate the dispatch columns of ``df`` by technology family.

    args:
        df (pd.DataFrame): dispatch with one column per carrier.
        agg_solar (bool): aggregate solar and solar-hsat into "solar".
        agg_gas (bool): aggregate CCGT and OCGT into "gas".
        agg_offwind (bool): aggregate offwind-dc/-ac/-float into "offwind".
        drop_carriers (bool): drop geothermal, art_load and load.
    """
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


def _stacked_dispatch_plot(ax, df, colors, x_col=None, alpha=0.8, width=None,
                           align="center", edgecolor=None, linewidth=None):
    """Draw a stacked bar plot that stacks positive and negative values apart.

    args:
        ax: matplotlib axes to draw on.
        df (pd.DataFrame): values to stack, one column per carrier.
        colors (list): one colour per data column.
        x_col (str | None): column to use as x values, index if None.
        alpha (float): opacity of the bars.
        width (float | None): bar width in x units, matplotlib default if None.
        align (str): "center" to centre the bars on x, "edge" to let them start
            at x and extend over ``width`` (a snapshot stands for the interval
            that begins at its time stamp).
        edgecolor: colour of the bar outline, matplotlib default if None.
        linewidth (float | None): width of the bar outline, default if None.
    """
    # with an explicit width the bars are meant to tile the x-axis, so the
    # axis limits have to follow the bar edges instead of the default margin
    bar_width = 0.8 if width is None else width
    margin = 0.5 if width is None else 0.5 * bar_width

    # only override the bar styling when asked for, so that callers relying on
    # the matplotlib defaults keep their appearance
    bar_kw = {}
    if edgecolor is not None:
        bar_kw["edgecolor"] = edgecolor
    if linewidth is not None:
        bar_kw["linewidth"] = linewidth

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

        ax.bar(x, pos_vals, bottom=pos_bottoms, width=bar_width, align=align,
                label=col,  color=colors[i], alpha=alpha, **bar_kw)
        ax.bar(x, neg_vals, bottom=neg_bottoms, width=bar_width, align=align,
                color=colors[i], alpha=alpha, **bar_kw)

        pos_bottoms += pos_vals
        neg_bottoms += neg_vals

    ax.axhline(0, color="black", linewidth=1., linestyle="--")
    if align == "edge":
        ax.set_xlim(x[0], x[-1] + bar_width)
    else:
        ax.set_xlim(x[0] - margin, x[-1] + margin)


def _select_co2_levels(co2_lvls_selected, n_nodes: int):
    """Match the requested CO2 levels against the ones available on disk.

    args:
        co2_lvls_selected (tuple | None): requested levels, all if None.
        n_nodes (int): number of nodes of the networks to look for.
    """
    co2ls = get_co2_levels(n_nodes=n_nodes)
    if co2_lvls_selected is None:
        return list(co2ls)

    co2_in_list = list()
    for co2_r in co2_lvls_selected:
        diff_r = abs(co2ls - co2_r)
        idx_co2 = np.argmin(diff_r)
        if diff_r[idx_co2] < 1e-6:
            co2_in_list.append(co2ls[idx_co2])

    logger.info(f"Selected Co2 lvls {co2_in_list}")
    return co2_in_list


def plot_load_dispatch(n_nodes: int = 600,
                       agg_solar:bool = True,
                       agg_gas:bool = True, 
                       agg_offwind:bool = True, 
                       drop_carriers:bool = True,
                       co2_lvls_selected: tuple[float, ...] | None = (0.0 , 0.2, .6),
                       save_prefix: str | None = None):

    """Plot the average load and dispatch profiles for different seasons and co2 levels.

    This function loads the PyPSA networks for different CO2 levels, aggregates the generator and storage dispatch profiles by carrier, and plots the average profiles for each season (winter, spring, summer, autumn). The resulting figure shows how the supply and load profiles change with CO2 levels and seasons.

    args:
        n_nodes (int): The number of nodes in the PyPSA networks to load and analyze.
        agg_solar (bool): Whether to aggregate solar and solar-hsat profiles into a single "solar" category.
        agg_gas (bool): Whether to aggregate CCGT and OCGT profiles into a single "gas" category.
        agg_offwind (bool): Whether to aggregate offwind-dc, offwind-ac, and offwind-float profiles into a single "offwind" category.
        drop_carriers (bool): Whether to drop certain carriers (geothermal, art_load, load) from the profiles before plotting.
    """
    # Load networks for all CO2 levels
    co2ls = _select_co2_levels(co2_lvls_selected, n_nodes=n_nodes)

    networks = {
        c:   load_pypsa_network(co2lvl=c, n_nodes=n_nodes)
        for c in co2ls
    }

    n_ref = networks[co2ls[0]]

    nice_names, colors_gens = _dispatch_carrier_style(n_ref)

    idx = n_ref.snapshots

    season_masks = {name: idx.month.isin(months)
                    for name, months in _SEASONS.items()}

    def _agg_carriers(df):
        return _agg_dispatch_carriers(df,
                                      agg_solar=agg_solar,
                                      agg_gas=agg_gas,
                                      agg_offwind=agg_offwind,
                                      drop_carriers=drop_carriers)

    stackedplot = _stacked_dispatch_plot

    for co2, n in networks.items():
        fig, axes = plt.subplots(2, 2,sharex=True,
                                 sharey=True, figsize=FOUR_PANEL_SIZE)
        for ax, (title, season) in zip(axes.flatten(),
                                       season_masks.items()):
            gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
            sto = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T
            supply = sto.add(gen, fill_value=0)

            supply = _agg_carriers(supply)
            supply_season = supply.loc[season]

            hour = supply_season.index.hour
            avg_profile = supply_season.groupby(hour).mean()
            load = n_ref.loads_t.p.sum(axis=1)
            load_season = load.loc[season]
            avg_load = load_season.groupby(load_season.index.hour).mean()


            # plot supply and load
            stackedplot(ax, avg_profile/1e3,
                        colors=[colors_gens.get(c, "#333333") for c in nice_names[avg_profile.columns]], x_col=None, alpha=0.8)
            ax.plot(avg_load.index, avg_load/1e3, 
                    color="black", linewidth=2,label="Load")

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Hour of day", size=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Power [GW]", size=AXIS_LABEL_FONTSIZE)
            ax.set_title(f"{title}", size=TITLE_FONTSIZE)

        for idx, ax in enumerate(axes.flatten()):
            add_panel_label(ax, idx, x_offset=-0.05)
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

        # add legend
        legend_ax = fig.add_axes([0.1, -0.05, 0.8, 0.12])
        legend_ax.axis("off")
        handles, labels = axes[0,0].get_legend_handles_labels()
        
        labels = [nice_names.get(l, l) for l in labels]
        labels = [name.capitalize().replace("&", "\\&") for name in labels]

        legend_ax.legend(handles, labels, 
                        ncol=4, loc="center", 
                        frameon=True,
                        fontsize = LEGEND_FONTSIZE)

        save_prefix_str = save_prefix + "_" if save_prefix is not None else ""
        
        save_figure(fig, f"{save_prefix_str}load_dispatch_{n_nodes}_{co2}",
                    path_to_figures_sclopf)

    return


def _co2_row_label(co2, n_nodes: int) -> str:
    """Build the CO2 row label, falling back to the target level if needed.

    ``get_actual_co2_level`` is only available for the 600 node networks, so for
    every other resolution the target level of the scenario is shown instead.

    args:
        co2 (float): CO2 level of the scenario.
        n_nodes (int): number of nodes of the network.
    """
    try:
        lvl_percent = get_actual_co2_level(co2, n_nodes=n_nodes, percent=True)
    except NotImplementedError:
        lvl_percent = co2 * 100.

    return rf"CO$_2$ level $= {lvl_percent:g}\,\%$"


def plot_load_dispatch_grid(n_nodes: int = 600,
                            agg_solar: bool = True,
                            agg_gas: bool = True,
                            agg_offwind: bool = True,
                            drop_carriers: bool = True,
                            co2_lvls_selected: tuple[float, ...] | None = (0.6, 0.2, 0.0),
                            figsize: tuple[float, float] | None = None,
                            save_prefix: str | None = None):

    """Plot load and dispatch profiles as a single CO2-by-season grid figure.

    Same content as :func:`plot_load_dispatch`, but instead of one figure per CO2
    level this draws a single figure with one CO2 level per row (in descending
    order) and one season per column. All panels share the same y-axis, so the
    rows can be compared directly.

    args:
        n_nodes (int): The number of nodes in the PyPSA networks to load and analyze.
        agg_solar (bool): Whether to aggregate solar and solar-hsat profiles into a single "solar" category.
        agg_gas (bool): Whether to aggregate CCGT and OCGT profiles into a single "gas" category.
        agg_offwind (bool): Whether to aggregate offwind-dc, offwind-ac, and offwind-float profiles into a single "offwind" category.
        drop_carriers (bool): Whether to drop certain carriers (geothermal, art_load, load) from the profiles before plotting.
        co2_lvls_selected (tuple | None): CO2 levels to show as rows, all available ones if None.
        figsize (tuple | None): Figure size, derived from the grid shape if None.
        save_prefix (str | None): Prefix prepended to the file name.
    """
    # Load networks for all CO2 levels
    co2ls = _select_co2_levels(co2_lvls_selected, n_nodes=n_nodes)

    if not co2ls:
        logger.warning("No matching CO2 levels found, skipping load dispatch grid.")
        return

    # rows run from the highest to the lowest CO2 level, independent of the
    # order in which the levels were requested
    co2ls = sorted(co2ls, reverse=True)

    networks = {
        c:   load_pypsa_network(co2lvl=c, n_nodes=n_nodes)
        for c in co2ls
    }

    n_ref = networks[co2ls[0]]

    nice_names, colors_gens = _dispatch_carrier_style(n_ref)

    idx = n_ref.snapshots

    season_masks = {name: idx.month.isin(months)
                    for name, months in _SEASONS.items()}

    # the load is expected to be identical across the CO2 scenarios, which is
    # what makes the rows comparable -- verify instead of assuming it
    ref_load = n_ref.loads_t.p.sum(axis=1)
    for c, n in networks.items():
        if not np.allclose(n.loads_t.p.sum(axis=1), ref_load, rtol=1e-6):
            logger.warning(f"Load profile for Co2L{c} differs from the reference "
                           f"level {co2ls[0]}; rows are not directly comparable.")

    n_rows, n_cols = len(co2ls), len(season_masks)
    fig, axes = plt.subplots(n_rows, n_cols,
                             sharex=True, sharey=True, squeeze=False,
                             figsize=figsize or (2.6 * n_cols, 2.1 * n_rows))

    for row, co2 in enumerate(co2ls):
        n = networks[co2]

        gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
        sto = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T
        supply = _agg_dispatch_carriers(sto.add(gen, fill_value=0),
                                        agg_solar=agg_solar,
                                        agg_gas=agg_gas,
                                        agg_offwind=agg_offwind,
                                        drop_carriers=drop_carriers)
        load = n.loads_t.p.sum(axis=1)

        for col, (season_name, season) in enumerate(season_masks.items()):
            ax = axes[row, col]

            supply_season = supply.loc[season]
            avg_profile = supply_season.groupby(supply_season.index.hour).mean()

            load_season = load.loc[season]
            avg_load = load_season.groupby(load_season.index.hour).mean()

            # A snapshot stands for the interval that starts at its time
            # stamp and lasts snapshot_weightings hours (3 h at the 3H
            # resolution used here). The bars are therefore as wide as that
            # interval and start at the stamp, so that they tile the day.
            hours = avg_profile.index.values
            bar_width = float(np.diff(hours).min()) if len(hours) > 1 else 1.

            # plot supply and load
            _stacked_dispatch_plot(ax, avg_profile/1e3,
                                   colors=[colors_gens.get(c, "#333333")
                                           for c in nice_names[avg_profile.columns]],
                                   x_col=None, alpha=0.8,
                                   width=bar_width, align="edge",
                                   edgecolor="0.4", linewidth=0.3)

            # the load is an interval average as well, so it is drawn as a
            # step that is constant over each interval instead of a line
            # interpolating between the stamps
            load_x = np.append(avg_load.index.values, hours[-1] + bar_width)
            load_y = np.append(avg_load.values, avg_load.values[-1])
            ax.step(load_x, load_y/1e3, where="post",
                    color="black", linewidth=2, label="Load")

            ax.set_xticks(np.arange(hours[0], hours[-1] + bar_width + 1e-9, 6))
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

            # only label the outer edges of the grid
            if col == 0:
                ax.set_ylabel("Power [GW]", size=AXIS_LABEL_FONTSIZE)
            if row == n_rows - 1:
                ax.set_xlabel("Hour of day", size=AXIS_LABEL_FONTSIZE)
            if row == 0:
                # season as an annotation above the panel, not as an axes title
                ax.text(0.5, 1.03, season_name,
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout()

    # CO2 level of each row, placed left of everything the row already draws
    # (y-label and tick labels). The position is taken from the rendered
    # extent so that it holds for any figure size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    row_extents = [
        axes[row, 0].get_tightbbox(renderer).transformed(inv)
        for row in range(n_rows)
    ]
    x_label = min(bb.x0 for bb in row_extents) - 0.1 / fig.get_size_inches()[0]

    for row, co2 in enumerate(co2ls):
        pos = axes[row, 0].get_position()
        fig.text(x_label, 0.5 * (pos.y0 + pos.y1),
                 _co2_row_label(co2, n_nodes=n_nodes),
                 rotation=90, ha="right", va="center",
                 fontsize=AXIS_LABEL_FONTSIZE)

    # add legend
    # placed below the rendered extent of the bottom row -- using the tight
    # bounding box instead of the axes box keeps it clear of the tick labels
    # and the "Hour of day" labels whatever the font size and figure size are
    fig_h = fig.get_size_inches()[1]
    bottom_extent = min(
        axes[n_rows - 1, col].get_tightbbox(renderer).transformed(inv).y0
        for col in range(n_cols)
    )
    gap_in, legend_h_in = 0.3, 1.0

    legend_ax = fig.add_axes([0.02,
                              bottom_extent - (gap_in + legend_h_in) / fig_h,
                              0.96,
                              legend_h_in / fig_h])
    legend_ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()

    labels = [nice_names.get(l, l) for l in labels]
    labels = [name.capitalize().replace("&", "\\&") for name in labels]

    legend_ax.legend(handles, labels,
                     ncol=4, loc="center",
                     frameon=True,
                     fontsize=LEGEND_FONTSIZE)

    save_prefix_str = save_prefix + "_" if save_prefix is not None else ""

    save_figure(fig, f"{save_prefix_str}load_dispatch_seasons_grid_n{n_nodes}",
                path_to_figures_sclopf)

    return
