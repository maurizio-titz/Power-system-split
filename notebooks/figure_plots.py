# %%
import sys
import pickle
import copy
import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# import seaborn as sns
import matplotlib.lines as mlines
from sklearn.cluster import KMeans

import cartopy.crs as ccrs
import cartopy
import gzip


from tqdm.notebook import tqdm

%load_ext autoreload
%autoreload 2

root_path = '../' # This defaults to './', which should be the repository pathimport os
os.chdir(root_path)
sys.path.append(root_path)
from utils.visualization import get_actual_co2_level
from utils.config import path_to_pypsa_network_sclopf, path_to_cascade_results_sclopf, path_to_vis_results_sclopf, path_to_pre_outage_sclopf, path_to_evaluation_results_sclopf, path_to_figures_sclopf, path_to_sclopf_results
from utils import data_handling, cascade_simulation
from utils.config import path_to_indicator_vectors_sclopf
from utils.clustering_visualisation import truncate_colormap
from utils.alternative_split_indicator_vectors import get_indicator_vector_snapshot_weightings

n_nodes = 600

save_path = path_to_figures_sclopf
os.makedirs(save_path, exist_ok=True)

# %%
# import datetime

# dir = "/srv/data/jlange/power-system-split/no_extensions/results/sclopf/evaluation_results"
# for f_name in os.listdir(dir):
#     if "0.05" in f_name:
#         print(f_name)
#         mod_time = os.path.getmtime(os.path.join(dir, f_name))
#         new_var = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
#         print("Last modified:", new_var)
#         if new_var < "2025-07-10 00:00:00":
#             print("Deleting file:", f_name)
#             os.remove(os.path.join(dir, f_name))
#         else:
#             print("Keeping file:", f_name)

# %% [markdown]
# # todo:
# - replace all hard coded co2 lvls with the actual lvls

# %% [markdown]
# # Setup

# %%
# Load network graph and node positions
# (it is equal for all CO2 levels)
snet_index = 0
network = data_handling.load_pypsa_network_from_path(path_to_pypsa_network_sclopf +
                        f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L0.1-2920SEG.nc", True)
nx_graph = data_handling.build_networkx_graph(network, snet_index= snet_index)
pos = nx.get_node_attributes(nx_graph, 'pos')
I_m, B_d, num_parallels, line_limits = data_handling.get_matrices_from_nx_graph(nx_graph)
n_lines = nx_graph.number_of_edges()

# Determine number of simulations,
# i.e., total number of initial failures over the simulated period of one year
bridge_idxs = data_handling.nx_edges_to_matrix_indices(nx.bridges(nx_graph),
                                                           nx_graph)
n_2_failures = cascade_simulation.calc_possible_double_line_failures(num_parallels,
                                                                     ignored_idxs=bridge_idxs)
num_failures = len(n_2_failures)
num_failures_weighted = network.snapshot_weightings.objective.sum() * num_failures
n_snapshots = network.snapshots.shape[0]

# %%
from utils.visualization import get_co2_levels

co2ls = get_co2_levels(n_nodes)


# %%
networks = {co2l: data_handling.load_pypsa_network(n_nodes=600, co2lvl=co2l, use_sclopf=True) for co2l in co2ls}

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
warnings.simplefilter(action='ignore', category=FutureWarning)


try:
    actual_co2ls = get_actual_co2_level(co2ls, n_nodes)
except FileNotFoundError:
    print("No actual CO2 levels found, calculating them now...")

    relax = 1.0 # Relaxation factor for the CO2 constraint
    p_heurist = 0.7 # Heuristic for the maximum line loading
    NCLUSTERS = 600
    groupsize = 60
    TEMP_RESOLUTION = 3 #H
    load_shedding = True
    n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))




    # Co2_scenarios = ['0.0','0.05', '0.1','0.2', '0.3', '0.4', '0.5', '0.6'] #, '0.2'


    # networks_path = f"{root_path}/workflow/co2_rel_{relax}/s_max_pu_{p_heurist}/postnetworks/"



    root_path = '/srv/data/jlange/PYPSA3/sclopf-iter' # This defaults to './', which should be the repository path
    networks_path = f"{root_path}/workflow/co2_rel_{relax}/s_max_pu_{p_heurist}/postnetworks/"

    def get_emissions(n):

        gen = (
            n.generators_t.p
            .multiply(n.snapshot_weightings.objective,axis=0) # multiply by no. of hours (weight)
            .divide(n.generators.efficiency,axis=1).fillna(0) # efficiency and generation might be different (e.g. in the case of CCGT)
            .multiply(n.generators.carrier.map(n.carriers.co2_emissions)) # multiply by specific emissions
            .fillna(0).sum().sum()
        )

        stog = (
            n.storage_units_t.p
            .multiply(n.snapshot_weightings.objective,axis=0)
            .divide(n.storage_units.efficiency_dispatch,axis=1).fillna(0)
            .multiply(n.storage_units.carrier.map(n.carriers.co2_emissions))
            .fillna(0).sum().sum()
        )

        return gen+stog


    # os.makedirs('pics/lopf', exist_ok=True)
    Co2_scenarios = co2ls
    # Define the carriers
    renewable_carriers = ["solar", "solar-hsat", "onwind", "offwind-ac", "offwind-dc", "offwind-float", "hydro"]
    conventional_carriers = ["nuclear", "oil", "OCGT", "CCGT", "coal", "lignite", "geothermal", "biomass"]

    # Combine them into a single list for the DataFrame columns
    all_carriers = renewable_carriers + conventional_carriers#network.gener.index

    generation_by_carrier_and_co2l = pd.DataFrame(None,index=all_carriers, columns= Co2_scenarios)

    CO2_shadowprices = pd.Series(index = Co2_scenarios)
    CO2_values = pd.Series(index = Co2_scenarios)
    CO2_global = pd.Series(index = Co2_scenarios)
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
        num_bars = len(network.statistics.energy_balance().loc[:, :, "AC"].groupby(
            "carrier"
        ).sum())
        
        # colors
        cmap = matplotlib.colormaps.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, num_bars))
        
        # plot
        fig, ax_loss_lvl = plt.subplots(figsize=(10, 6),layout = "constrained")
        # tmp.rename({"-":"Load"},inplace=True)
        # data in plot
        # energy_balance_data = 
        network.statistics.energy_balance().loc[:, :, "AC"].groupby(
            "carrier"
        ).sum().rename({"-":"Load"}).to_frame().T.plot.bar(stacked=True, ax=ax_loss_lvl, color=colors, title=f"Energy Balance {Co2l}")
        
        
        ax_loss_lvl.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
        # fig.savefig(f"pics/lopf/Energymix_lopf_{Co2l}-{NCLUSTERS}.pdf")
        plt.close()


    # generation_fossil = generation_by_carrier_and_co2l.loc[["coal", "lignite", "CCGT", "OCGT"]].sum(axis=0)
    # fossil_percentage = generation_fossil.div(generation_by_carrier_and_co2l.sum(axis=0) - generation_fossil)*100

    # emission_by_carrier_by_co2l = generation_by_carrier_and_co2l.mul(network.carriers.co2_emissions, axis="index")
    # emission_by_carrier_by_co2l.dropna(axis=0, thresh=1, inplace=True)
    emission_by_co2l = CO2_values#emission_by_carrier_by_co2l.sum(axis=0)
    emission_by_co2l_perc = emission_by_co2l/emission_by_co2l[0.5]*50

    x_values = np.array(emission_by_co2l_perc.index, dtype=float)*100

    # plot red line
    plt.plot([60,0],[60,0], color = "r", alpha = 0.5)

    plt.scatter(x_values, emission_by_co2l_perc.values)

    plt.gca().invert_xaxis()
    plt.grid()
    plt.title("emission level normalized to 50%")
    plt.xlabel("intended emission lvl[%]")
    plt.ylabel("calculated emission lvl[%]")

    plt.figure()
    plt.scatter(x_values, emission_by_co2l.values,label = "Data")
    plt.scatter(x_values, CO2_global.values,label = "Global constraints",marker = "x")

    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel("intended emission lvl[%]")
    plt.ylabel("calculated emissions")
    # plt.savefig(f"pics/lopf/emissions.pdf")


    for cx,Co2l in enumerate(Co2_scenarios):
        print(f"{Co2l}: Actual CO2lvl: {emission_by_co2l.values[cx]/CO2_global.values[cx]*float(Co2l)*100}")
        

    actual_co2ls = {Co2l: emission_by_co2l.values[cx]/(CO2_global.values[cx]+10**-10)*float(Co2l) for cx,Co2l in enumerate(Co2_scenarios)}

    actual_co2ls = pd.DataFrame.from_dict(actual_co2ls, orient='index', columns=["actual_co2_levels"])
    actual_co2ls.sort_index(inplace=True)
    actual_co2ls.index.name = "co2_level"
    actual_co2ls.to_csv(path_to_sclopf_results + "actual_co2_levels.csv")

# %%
selected_co2ls = np.array([0.0, 0.2, 0.6])
print(co2ls)
print(selected_co2ls)

# %%
split_properties = pd.read_hdf(path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0)
split_properties.lost_load_share_blackout = split_properties.lost_load_share_blackout.astype(float)

# %%
actual_co2ls

# %% [markdown]
# # Figure 2: generation map

# %% [markdown]
# ## SCLOPF

# %%
generation_by_carrier_and_co2l = pd.DataFrame(index=network.carriers.index,
                                              columns=co2ls)
objectives = np.zeros(len(co2ls))
for i, level in enumerate(co2ls):
    network = networks[level]
    generation_by_carrier_and_co2l[level] = network.generators_t.p.mul(network.snapshot_weightings.generators, axis="index").mul(network.generators.sign).T.groupby(
        network.generators["carrier"]).sum().sum(axis=1)
    objectives[i] = network.objective
generation_by_carrier_and_co2l.dropna(axis=0, inplace=True)

plot_generator_type_threshold = 0.01
max_carrier_share = (generation_by_carrier_and_co2l/generation_by_carrier_and_co2l.sum(axis=0)).max(axis=1)
carrier_mask = max_carrier_share > plot_generator_type_threshold
dropped_carriers = carrier_mask[~carrier_mask].index

# %%
# drop the 0.7 and 0.8 columns because they carry no relevant information, see old plots
generation_by_carrier_and_co2l = pd.DataFrame(index=networks[0.6].carriers.index,
                                              columns=co2ls)
objectives = np.zeros(len(co2ls))
for i, level in enumerate(co2ls):
    network = networks[level]
    generation_by_carrier_and_co2l[level] = network.generators_t.p.mul(network.snapshot_weightings.generators, axis="index").mul(network.generators.sign).T.groupby(
        network.generators["carrier"]).sum().sum(axis=1)
    objectives[i] = network.objective
generation_by_carrier_and_co2l.dropna(axis=0, inplace=True)

plot_generator_type_threshold = 0.01
max_carrier_share = (generation_by_carrier_and_co2l/generation_by_carrier_and_co2l.sum(axis=0)).max(axis=1)
carrier_mask = max_carrier_share > plot_generator_type_threshold
carrier_mask["OCGT"] = True # keep gas carriers, add them together later
dropped_carriers = carrier_mask[~carrier_mask].index
print(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
generation_by_carrier_and_co2l = generation_by_carrier_and_co2l[carrier_mask]


mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

target_levels = [0.6, 0.0]

f = plt.figure(figsize=(24, 12))
# gs = GridSpec(2, 3, figure=f)
# ax1 = [f.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
#        for i in range(2)]
# ax2 = [f.add_subplot(gs[1, i], projection=ccrs.PlateCarree())
#        for i in range(2)]
# axs_clust = [ax1, ax2]
# ax3 = f.add_subplot(gs[0, 2])
outer = GridSpec(1, 2, width_ratios = [2, 1]) 
# gs = GridSpec(2, 3, figure=f)
gs_left = GridSpecFromSubplotSpec(2, 2, subplot_spec = outer[0], wspace=-0.0, hspace=0.05)
gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1], height_ratios=[1, 0.6])

ax1 = [f.add_subplot(gs_left[0, i], projection=ccrs.PlateCarree())
       for i in range(2)]
ax2 = [f.add_subplot(gs_left[1, i], projection=ccrs.PlateCarree())
       for i in range(2)]
axs_primary = [ax1, ax2]
ax3 = f.add_subplot(gs_right[0, 0])
ax3_legend = f.add_subplot(gs_right[1, 0])
# ax4 = f.add_subplot(outer[1, 1])
labels = [[r'\textbf{a}', r'', r'\textbf{c}'],
          [r'\textbf{b}', '', r'\textbf{d}', '', '']]

##### panal a and b #######

vmax = 3.2e2
months = [7, 12]

for i, target_level in enumerate(target_levels):
       # fpath = data_path + "European_networks_lopf/elec_s_800_ec_lv1.0_Co2L0.1-3H.nc"
       # n = data_handling.load_pypsa_network(fpath, False)
       
       # n = data_handling.load_pypsa_network(path_to_pypsa_network_sclopf +
       #                  f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{target_level}-2920SEG.nc", True)
       n = data_handling.load_pypsa_network(target_level, n_nodes, True)
       for ii, month in enumerate(months):

              axs_primary[i][ii].set_ylim([0, vmax])
              current_snapshots = n.snapshots[n.snapshots.month == month]
              current_gen = n.generators_t.p.mul(network.snapshot_weightings.generators, axis="index").loc[current_snapshots].sum().mul(network.generators.sign)
              current_gen = current_gen.groupby([n.generators["bus"], n.generators["carrier"]]).sum()
              n.plot(bus_sizes=current_gen/7e6,
                     line_colors='black',
                     link_colors='black',
                     link_widths=0.5,
                     line_widths=0.5,
                     ax=axs_primary[i][ii])

              if i == 0:
                     axs_primary[i][ii].set_title(current_snapshots.month_name()[0], fontsize=24)

              axs_primary[i][ii].text(0 - 0.1, 1+0.05,
                            labels[i][ii],
                            fontsize=36,
                            weight='bold',
                            verticalalignment='center',
                            transform=axs_primary[i][ii].transAxes)


       axs_primary[i][0].text(0 - 0.15, 0.4,
                     rf'CO$_2$={get_actual_co2_level(target_level, percent=True)}\% ',
                     fontsize=24,
                     verticalalignment='center',
                     transform=axs_primary[i][0].transAxes)

import itertools
marker = itertools.cycle((',', 'x', '.', 'o', '*')) 

colors_gens = network.carriers.set_index('nice_name').color
colors_gens["Open-Cycle Gas"] = "#cc0099"
colors_gens["Gas"] = "#cc0099"

co2s = generation_by_carrier_and_co2l.columns.astype('float')
index = np.argsort(co2s)
jitter = iter(np.linspace(-1,1,generation_by_carrier_and_co2l.shape[0]))

generation_by_carrier_and_co2l = generation_by_carrier_and_co2l.rename(index=network.carriers.nice_name)
gas_gen = generation_by_carrier_and_co2l.loc["Combined-Cycle Gas"] + generation_by_carrier_and_co2l.loc["Open-Cycle Gas"]
generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l.copy()
# generation_by_carrier_and_co2l_plot.index = generation_by_carrier_and_co2l_plot.index.str.apply(lambda x: x.upper())


for col in generation_by_carrier_and_co2l_plot.index:
    if "Gas" in col:
        generation_by_carrier_and_co2l_plot.drop(col, axis=0, inplace=True)
generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l_plot.rename(index=network.carriers.nice_name)
# generation_by_carrier_and_co2l_plot.drop("Oil", axis=0, inplace=True)
generation_by_carrier_and_co2l_plot.loc["Gas"] = gas_gen 
for ind,data in generation_by_carrier_and_co2l_plot.iterrows():
    # jit = next(jitter)
    jit = 0
    plot_inds = (data.values)[index]>0
    # print(plot_inds.sum())
    ax3.scatter((get_actual_co2_level(co2s[index])*100 + jit)[plot_inds],
                ((data.values)[index]*1e-6)[plot_inds],
                color=colors_gens.loc[ind],
                label=ind,
                s=100,
                marker= next(marker)
                )

    ax3.plot((get_actual_co2_level(co2s[index])*100 + jit)[plot_inds],
             ((data.values)[index]*1e-6)[plot_inds],
             color=colors_gens.loc[ind],
             alpha=0.8)
# plt.gca().set_yscale("log")
# ax3.scatter([], [], label=r"Oil$\textless 0.6$", marker="")


ax3.invert_xaxis()
ax3.tick_params(axis='both', which='both', labelsize=24)
ax3.set_xlabel(r'$\text{CO}_2$ level [\% of 1990]',  fontsize=24)
ax3.set_ylabel(r'Total annual generation [TWh]', fontsize=24)

ax3.grid(True)
ax3.yaxis.offsetText.set_fontsize(24)
ax3.text(0 - 0.15, 1+0.05,
         labels[0][2],
         fontsize=36,
         weight='bold',
         verticalalignment='center',
         transform=ax3.transAxes)
# ax3.plot([746/1251*100, 746/1251*100],[-2e1,1000], '--', lw=4, c='r', zorder=-1000) # TODO: Insert CO2 limit that corresponds to todays amount
# co2 emissions from https://www.umweltbundesamt.de/daten/klima/treibhausgas-emissionen-in-deutschland#emissionsentwicklung
# ax3.text(50,12e2*0.9, r'\textbf{2022`s emissions}', fontsize=24, color='r')
# ax3.text(746/1251*100,1050, r'\textbf{2022`s emissions}', fontsize=24, color='r', horizontalalignment='left')
ax3.set_ylim([-2e1,12.4e2])
# ax3.set_yscale('log')

# f.subplots_adjust(hspace=0.05, wspace=0.2)
h, l = ax3.get_legend_handles_labels()
ax3_legend.legend(h, l, borderaxespad=0, fontsize=20.5, ncol=2)
ax3_legend.axis("off")
# ax3.legend(fontsize=20.5, bbox_to_anchor=(1.02,0), ncol=2, loc="upper left")
# plt.tight_layout()
# ax3.legend(fontsize=20.5, ncol=2, columnspacing=0.5, handletextpad=0.1, borderpad=0.3)
plt.tight_layout()

plt.savefig(save_path+'generation_mix_versus_co2level.pdf', bbox_inches='tight')

# %%
def aggregate_carriers(current_gen, carrier_keyword_to_new_carrier, inplace=False):
    """
    Aggregate carriers in current_gen MultiIndex DataFrame by keywords.

    Parameters
    ----------
    current_gen : pd.Series or pd.DataFrame
        Indexed by (bus, carrier).
    carrier_keywords : list of str
        Keywords to aggregate, e.g. ["offwind", "solar"].

    Returns
    -------
    pd.Series or pd.DataFrame
        With carriers aggregated by keyword.
    """
    if inplace:
        current_gen = current_gen.copy()
    for keyword, new_carrier in carrier_keyword_to_new_carrier.items():
        mask = [carrier for bus, carrier in current_gen.index if keyword in carrier]
        summed = current_gen.loc[(slice(None), mask)].groupby(level="bus").sum()
        for bus in summed.index:
            current_gen.loc[(bus, new_carrier)] = summed[bus]
        current_gen = current_gen.drop(index=[(bus, carrier) for bus, carrier in current_gen.index if keyword in carrier and carrier != new_carrier])
    return current_gen

# %%
# drop the 0.7 and 0.8 columns because they carry no relevant information, see old plots
generation_by_carrier_and_co2l = pd.DataFrame(index=networks[0.6].carriers.index,
                                              columns=co2ls)
carriers = list(generation_by_carrier_and_co2l.index)

objectives = np.zeros(len(co2ls))
for i, level in enumerate(co2ls):
    network = networks[level]
    generation_by_carrier_and_co2l[level] = network.generators_t.p.mul(network.snapshot_weightings.generators, axis="index").mul(network.generators.sign).T.groupby(
        network.generators["carrier"]).sum().sum(axis=1)
    objectives[i] = network.objective
generation_by_carrier_and_co2l.dropna(axis=0, inplace=True)

plot_generator_type_threshold = 0.01
max_carrier_share = (generation_by_carrier_and_co2l/generation_by_carrier_and_co2l.sum(axis=0)).max(axis=1)
carrier_mask = max_carrier_share > plot_generator_type_threshold
carrier_mask["OCGT"] = True # keep gas carriers, add them together later
dropped_carriers = carrier_mask[~carrier_mask].index
print(f"Dropping {len(dropped_carriers)} carriers: {dropped_carriers}")
generation_by_carrier_and_co2l = generation_by_carrier_and_co2l[carrier_mask]


mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

target_levels = [0.6, 0.0]

f = plt.figure(figsize=(24, 12))
outer = GridSpec(1, 2, width_ratios = [2, 1]) 
# gs = GridSpec(2, 3, figure=f)
gs_left = GridSpecFromSubplotSpec(2, 2, subplot_spec = outer[0], wspace=-0.0, hspace=0.05)
gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1], height_ratios=[1, 0.6])

ax1 = [f.add_subplot(gs_left[0, i], projection=ccrs.PlateCarree())
       for i in range(2)]
ax2 = [f.add_subplot(gs_left[1, i], projection=ccrs.PlateCarree())
       for i in range(2)]
axs_primary = [ax1, ax2]
ax3 = f.add_subplot(gs_right[0, 0])
ax3_legend = f.add_subplot(gs_right[1, 0])
# ax4 = f.add_subplot(outer[1, 1])
labels = [[r'\textbf{a}', r'', r'\textbf{c}'],
          [r'\textbf{b}', '', r'\textbf{d}', '', '']]

##### panal a and b #######

vmax = 3.2e2
months = [7, 12]

carrier_keyword_to_new_carrier = {
       "offwind": "offwind-dc",
       "solar": "solar",
}
for i, target_level in enumerate(target_levels):
       # fpath = data_path + "European_networks_lopf/elec_s_800_ec_lv1.0_Co2L0.1-3H.nc"
       # n = data_handling.load_pypsa_network(fpath, False)
       
       # n = data_handling.load_pypsa_network(path_to_pypsa_network_sclopf +
       #                  f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{target_level}-2920SEG.nc", True)
       n = data_handling.load_pypsa_network(target_level, n_nodes, True)
       for ii, month in enumerate(months):

              axs_primary[i][ii].set_ylim([0, vmax])
              current_snapshots = n.snapshots[n.snapshots.month == month]
              current_gen = n.generators_t.p.mul(network.snapshot_weightings.generators, axis="index").loc[current_snapshots].sum().mul(network.generators.sign)
              current_gen = current_gen.groupby([n.generators["bus"], n.generators["carrier"]]).sum()

              current_gen = aggregate_carriers(current_gen, carrier_keyword_to_new_carrier, inplace=False)
              assert set(current_gen.index.get_level_values("carrier").unique()).issubset(set(carriers)), \
                     f"carrier is not in carriers: {set(current_gen.index.get_level_values('carrier').unique()) - set(carriers)}"
              # print(current_gen)
              n.plot(bus_sizes=current_gen/7e6,
                     line_colors='black',
                     link_colors='black',
                     link_widths=0.5,
                     line_widths=0.5,
                     ax=axs_primary[i][ii])

              if i == 0:
                     axs_primary[i][ii].set_title(current_snapshots.month_name()[0], fontsize=24)

              axs_primary[i][ii].text(0 - 0.1, 1+0.05,
                            labels[i][ii],
                            fontsize=36,
                            weight='bold',
                            verticalalignment='center',
                            transform=axs_primary[i][ii].transAxes)


       axs_primary[i][0].text(0 - 0.15, 0.4,
                     rf'CO$_2$={get_actual_co2_level(target_level, percent=True)}\% ',
                     fontsize=24,
                     verticalalignment='center',
                     transform=axs_primary[i][0].transAxes)

import itertools
marker = itertools.cycle((',', 'x', '.', 'o', '*')) 

colors_gens = network.carriers.set_index('nice_name').color
colors_gens["Open-Cycle Gas"] = "#cc0099"
colors_gens["Gas"] = "#cc0099"
colors_gens["Offshore Wind"] = n.carriers.color["offwind-dc"]

co2s = generation_by_carrier_and_co2l.columns.astype('float')
index = np.argsort(co2s)
jitter = iter(np.linspace(-1,1,generation_by_carrier_and_co2l.shape[0]))

generation_by_carrier_and_co2l = generation_by_carrier_and_co2l.rename(index=network.carriers.nice_name)
gas_gen = generation_by_carrier_and_co2l.loc["Combined-Cycle Gas"] + generation_by_carrier_and_co2l.loc["Open-Cycle Gas"]
offwind_carriers = [carrier for carrier in generation_by_carrier_and_co2l.index if "Off" in carrier]
offwind_gen = generation_by_carrier_and_co2l.loc[offwind_carriers].sum(axis=0)
solar_carriers = [carrier for carrier in generation_by_carrier_and_co2l.index if "solar" in carrier.lower()]
solar_gen = generation_by_carrier_and_co2l.loc[solar_carriers].sum(axis=0)

generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l.copy()


# generation_by_carrier_and_co2l_plot.index = generation_by_carrier_and_co2l_plot.index.str.apply(lambda x: x.upper())


for col in generation_by_carrier_and_co2l_plot.index:
    if "Gas" in col:
        generation_by_carrier_and_co2l_plot.drop(col, axis=0, inplace=True)
generation_by_carrier_and_co2l_plot = generation_by_carrier_and_co2l_plot.rename(index=network.carriers.nice_name)
# generation_by_carrier_and_co2l_plot.drop("Oil", axis=0, inplace=True)
generation_by_carrier_and_co2l_plot.loc["Gas"] = gas_gen 

generation_by_carrier_and_co2l_plot.loc["Offshore Wind"] = offwind_gen
generation_by_carrier_and_co2l_plot.drop(offwind_carriers, axis=0, inplace=True)

generation_by_carrier_and_co2l_plot.loc["Solar"] = solar_gen
generation_by_carrier_and_co2l_plot.drop([carrier for carrier in solar_carriers if carrier != "Solar"], axis=0, inplace=True)

for ind,data in generation_by_carrier_and_co2l_plot.iterrows():
    # jit = next(jitter)
    jit = 0
    plot_inds = (data.values)[index]>0
    # print(plot_inds.sum())
    ax3.scatter((get_actual_co2_level(co2s[index])*100 + jit)[plot_inds],
                ((data.values)[index]*1e-6)[plot_inds],
                color=colors_gens.loc[ind],
                label=ind,
                s=100,
                marker= next(marker)
                )

    ax3.plot((get_actual_co2_level(co2s[index])*100 + jit)[plot_inds],
             ((data.values)[index]*1e-6)[plot_inds],
             color=colors_gens.loc[ind],
             alpha=0.8)
# plt.gca().set_yscale("log")
# ax3.scatter([], [], label=r"Oil$\textless 0.6$", marker="")


ax3.invert_xaxis()
ax3.tick_params(axis='both', which='both', labelsize=24)
ax3.set_xlabel(r'$\text{CO}_2$ level [\% of 1990]',  fontsize=24)
ax3.set_ylabel(r'Total annual generation [TWh]', fontsize=24)

ax3.grid(True)
ax3.yaxis.offsetText.set_fontsize(24)
ax3.text(0 - 0.15, 1+0.05,
         labels[0][2],
         fontsize=36,
         weight='bold',
         verticalalignment='center',
         transform=ax3.transAxes)
# ax3.plot([746/1251*100, 746/1251*100],[-2e1,1000], '--', lw=4, c='r', zorder=-1000) # TODO: Insert CO2 limit that corresponds to todays amount
# co2 emissions from https://www.umweltbundesamt.de/daten/klima/treibhausgas-emissionen-in-deutschland#emissionsentwicklung
# ax3.text(50,12e2*0.9, r'\textbf{2022`s emissions}', fontsize=24, color='r')
# ax3.text(746/1251*100,1050, r'\textbf{2022`s emissions}', fontsize=24, color='r', horizontalalignment='left')
ax3.set_ylim([-2e1,12.4e2])
# ax3.set_yscale('log')

# f.subplots_adjust(hspace=0.05, wspace=0.2)
h, l = ax3.get_legend_handles_labels()
ax3_legend.legend(h, l, borderaxespad=0, fontsize=20.5, ncol=2)
ax3_legend.axis("off")
# ax3.legend(fontsize=20.5, bbox_to_anchor=(1.02,0), ncol=2, loc="upper left")
# plt.tight_layout()
# ax3.legend(fontsize=20.5, ncol=2, columnspacing=0.5, handletextpad=0.1, borderpad=0.3)
plt.tight_layout()

plt.savefig(save_path+'generation_mix_versus_co2level_aggregated.pdf', bbox_inches='tight')

# %% [markdown]
# # Storage map plots

# %%
# select whether to plot the storage OUTPUT capacity (capacity * output efficiency) or the storage capacity
for plot_output_capacity in [True]:

       mpl.style.use('default')
       plt.rc('text', usetex=True)
       plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

       target_levels = [0.0]
       unit = "GWh"
       unit_factor = 1e-3

       f = plt.figure(figsize=(15,10))

       gs_vertical = GridSpec(2, 2, width_ratios=[2,0.6], wspace=0.13)
       gs_maps = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs_vertical[0], wspace=-0.0, hspace=-0.2)
       gs_lines = GridSpecFromSubplotSpec(2, 1, subplot_spec = gs_vertical[1], height_ratios=[0.8,0.2], hspace=0.4)

       axs_maps = np.array([f.add_subplot(gs_maps[i], projection=ccrs.PlateCarree())
              for i in range(2)])
       ax_line = f.add_subplot(gs_lines[0])

       ##### panal a, b: maps #######

       buses = list(network.buses.index)
       storage_types = network.storage_units["carrier"].unique()
       multi_index = pd.MultiIndex.from_product([buses, storage_types], names=["bus", "carrier"])

       battery_color = "red"
       h2_color = "green"
       for i, target_level in enumerate(target_levels):
              for ii, (current_store_type, current_color) in enumerate(zip(["battery", "H2"], [battery_color, h2_color])):
                     n = networks[target_level]
                     storage_capacities = n.storage_units.max_hours * n.storage_units.p_nom
                     if plot_output_capacity:
                            storage_capacities = n.storage_units.max_hours * n.storage_units.p_nom * n.storage_units.efficiency_dispatch
                     storage_capacities_grouped = storage_capacities.groupby([n.storage_units["bus"], n.storage_units["carrier"]]).sum()
                     max_node_size = 0.85
                     max_capacity = storage_capacities_grouped[:,current_store_type].max()
                     max_capacity_rounded = round(max_capacity, -int(round(np.log10(max_capacity),0))+1)
                     n.plot(bus_sizes=storage_capacities_grouped[:,current_store_type]/max_capacity_rounded*max_node_size,
                            line_colors='black',
                            bus_colors=current_color,
                            link_widths=0.5,
                            line_widths=0.5,
                            ax=axs_maps[ii])
                     legend_relative_circle_size = np.array([0.25, 1])
                     legend_circle_size = [size*max_node_size for size in legend_relative_circle_size]
                     axs_maps[ii].legend(loc="upper left")
                     legend_circle_size = [round(size, -int(np.floor(np.log10(size)))) for size in legend_circle_size]
                     pypsa.plot.add_legend_circles(axs_maps[ii], sizes=legend_circle_size, labels=[f"{size*max_capacity_rounded*unit_factor:2g} {unit}" for size in legend_circle_size], patch_kw={"color":current_color})
                     

                     # set current_store_type to upper case
                     current_store_type_string = current_store_type[0].upper() + current_store_type[1:]
                     axs_maps[ii].set_title(current_store_type_string, fontsize=16)
                     
       axs_maps[0].text(0 - 0.1, 1+0.05,
                     r'\textbf{a}',
                     fontsize=24,
                     weight='bold',
                     verticalalignment='center',
                     transform=axs_maps[0].transAxes)
       # axs_maps[0].text(0 - 0.1, 1,
       #               r'\textbf{b}',
       #               fontsize=24,
       #               weight='bold',
       #               verticalalignment='center',
       #               transform=axs_maps[0].transAxes)


       axs_maps[0].text(0 - 0.15, 0.8,
                     fr'CO$_2$={get_actual_co2_level(target_levels[0], percent=True)}\% ',
                     fontsize=24,
                     fontweight='bold',
                     verticalalignment='center',
                     transform=axs_maps[0].transAxes)
       # axs_maps[0].text(0 - 0.15, 0.4,
       #               fr'CO$_2$={get_actual_co2_level(target_levels[1], percent=True)}\%',
       #               fontsize=16,
       #               verticalalignment='center',
       #               transform=axs_maps[0].transAxes)


       ###### panel c: total capacity line plot #####

       ax_line.text(0 - 0.1, 1+0.05,
                     r'\textbf{b}',
                     fontsize=24,
                     weight='bold',
                     verticalalignment='center',
                     transform=ax_line.transAxes)

       capacity_by_type_by_lvl = pd.DataFrame(index=storage_types, columns=co2ls[::-1])
       for target_level in co2ls:
              n = networks[target_level]
              print(f"sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{target_level}-2920SEG.nc")
              storage_capacities_all_lvl = pd.Series(index=multi_index, data=0)
              storage_capacities_lvl = n.storage_units.max_hours * n.storage_units.p_nom
              if plot_output_capacity:
                     storage_capacities_lvl = storage_capacities_lvl * n.storage_units.efficiency_dispatch
              storage_capacities_grouped_lvl = storage_capacities_lvl.groupby([n.storage_units["carrier"]]).sum()
              capacity_by_type_by_lvl.loc[storage_capacities_grouped_lvl.index,target_level] = storage_capacities_grouped_lvl

              capacity_by_type_by_lvl_plotting = capacity_by_type_by_lvl.T.copy()
       try:
              capacity_by_type_by_lvl_plotting.drop(index=[0.7,0.8], inplace=True)
       except:
              pass

       capacity_by_type_by_lvl_plotting.rename(columns={"battery": "Battery", "hydro": "Hydro", "PHS": "Pumped hydro"},inplace=True)
       colors = ["blue", "skyblue", h2_color, battery_color]
       ax_line.plot(get_actual_co2_level(capacity_by_type_by_lvl.columns,percent=True) ,capacity_by_type_by_lvl_plotting.loc[:,['Hydro', 'Pumped hydro', 'H2','Battery']].values*unit_factor, label=['Hydro', 'Pumped hydro', 'H2','Battery'])
       for feat_count in range(capacity_by_type_by_lvl_plotting.shape[1]):
              if "ydro" in capacity_by_type_by_lvl_plotting.columns[feat_count]:
                     continue
              plt.scatter(get_actual_co2_level(capacity_by_type_by_lvl.columns,percent=True) ,(capacity_by_type_by_lvl_plotting.loc[:,['Hydro', 'Pumped hydro', 'H2','Battery']].values*unit_factor)[:,feat_count], color=colors[feat_count])
       secax = ax_line.secondary_yaxis('right', functions=(lambda x: x*(1/6), lambda x: x*6))
       secax.set_ylabel('Battery power [GW]')#, weight='normal')
       # secax.tick_params(labelsize=16)
       # secax.tick_params(axis='y', which='both', labelsize=10, width=1)

       for i,ii in enumerate(ax_line.lines):
              ii.set_color(colors[i])

       ax_line.invert_xaxis()
       ax_line.grid(True)
       plt.yscale('log')
       if plot_output_capacity:
              plt.ylabel(f"Storage output capacity [{unit}]")
       else:
              plt.ylabel(f"Storage capacity [{unit}]")
       plt.xlabel(r"CO2 level [\% of 1990]")
       plt.legend()
       h, l = ax_line.get_legend_handles_labels()
       ax_line.get_legend().remove()
       ax_line_legend = f.add_subplot(gs_lines[1])
       ax_line_legend.legend(h, l, borderaxespad=0.2, fontsize=15, ncol=2, loc="upper left")
       ax_line_legend.axis("off")

       if plot_output_capacity:
              file_name = "storage_output_capacity_vs_co2level"
       else:
              file_name = "storage_capacity_vs_co2level"

       # plt.savefig(save_path+file_name + 'singleLvl.pdf', bbox_inches='tight')

# %% [markdown]
# # Figure 3: SPI

# %%
mean_consumption_vector = np.load(
    path_to_pre_outage_sclopf + f'mean_nodal_consumption_all_co2ls_{n_nodes}.npy')
weighted_mean_consumption_vector = np.load(
    path_to_pre_outage_sclopf + f'weighted_mean_nodal_consumption_all_co2ls_{n_nodes}.npy')
dipole_vector = np.load(path_to_pre_outage_sclopf +
                        f'dipole_vector_time_series_all_co2ls_{n_nodes}.npy')
vec_norm = np.load(path_to_pre_outage_sclopf + f'spi_time_series_all_co2ls_{n_nodes}.npy')

# %%
dipole_vector_abs = np.linalg.norm(dipole_vector, axis=1)

# %% [markdown]
# ## new figure, no map

# %%
inertia_time = np.load(path_to_pre_outage_sclopf +
                       f'inertia_time_series_all_co2ls_{n_nodes}.npy')/1000

# %%
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
labels = [[r'\textbf{a}', r'\textbf{b}'], [
    r'\textbf{c}', r'\textbf{d}', r'\textbf{e}', r'\textbf{f}']]

f = plt.figure(figsize=(20, 13), constrained_layout=True)
gs_vertical= GridSpec(2,1, figure=f, hspace=0.8, height_ratios=[1,1]) # outer grid
gs_horizontal_hists = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vertical[0])
ax_flow = f.add_subplot(gs_horizontal_hists[0])
ax_inertia = f.add_subplot(gs_horizontal_hists[1])
# axs_hist_legend = f.add_subplot(gs_horizontal_hists[2])

position_vector = np.array([pos[n] for n in nx_graph.nodes()])
mean_pos = [np.mean(position_vector[:, 0]), np.mean(position_vector[:, 1])]

gs_vectors = GridSpecFromSubplotSpec(3, 13, subplot_spec=gs_vertical[1], hspace=-0.6, )
ax_vectors_lvls = [f.add_subplot(gs_vectors[i, 0]) for i in range(3)]
for ax_loss_lvl in ax_vectors_lvls:
    ax_loss_lvl.axis('off')

gs_vecs_all = [[f.add_subplot(gs_vectors[j, i+1], projection=ccrs.PlateCarree())
       for i in range(12)] for j in range(3)]
gs_vecs_all = np.array(gs_vecs_all)
for ax_loss_lvl in gs_vecs_all.flatten():
    ax_loss_lvl.axis('off')
    ax_loss_lvl.set_aspect('equal')

scale_factor_spi = 2e4

################# panel a: total flow histograms ####################

# calculate line flows
cmap = plt.get_cmap('cividis_r')

unit_factor = 1e6

bins = np.arange(0, 1.02, 0.02)

selected_co2ls_spi = sorted(np.array([0.2, 0.05, 0.0]))  # CO2 levels to plot
hist_vals = np.zeros((len(selected_co2ls_spi), len(bins)-1))



flows_lvls = []
for count, co2l in enumerate(selected_co2ls_spi):

   
    n = networks[co2l]

    indices = nx.get_edge_attributes(nx_graph, 'line_index')

    # flows = n.lines_t.p0.abs().sum(axis=1)
    # distances = n.lines.length
    flow_distance = n.lines_t.p0.abs().mul(network.lines.length)/unit_factor
    # flows.hist(ax=axs_probs, bins=10, alpha=0.5, color=cmap(ind/len(co2ls), 0.8), label=r'{}\%'.format(int(100*co2l)))
    flows_lvls.append(flow_distance.sum(axis=1))

bins=np.histogram(np.hstack(flows_lvls), bins=40)[1]

mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

for count, co2l in enumerate(selected_co2ls_spi):

    ind = np.where(np.round(co2ls, 2) == co2l)[0][0]

    ax_flow.hist(flows_lvls[count],
            weights=network.snapshot_weightings.generators,
            bins=bins,
            histtype='step',
            label=r'{}\%'.format(int(round(100*co2l))),
            # density=True,
            linewidth=3,
            color=cmap(np.where(selected_co2ls_spi == co2l)[0][0]/(len(selected_co2ls_spi)-1)),
            alpha=1)
    
# ax_flow.legend(fontsize=16, title='CO$_2$ level [\% of 1990]')
ax_flow.tick_params(axis='both', which='both', labelsize=24)
ax_flow.set_xlabel(r'total power flow distance [TW$\cdot$km]', fontsize=26)
ax_flow.set_ylabel(r'Hours', fontsize=26)
ax_flow.grid(True)



################### panel b: rotational energy ####################


inertia_time = np.load(path_to_pre_outage_sclopf +
                       f'inertia_time_series_all_co2ls_{n_nodes}.npy')/1000
bins = np.linspace(0, inertia_time.max(), 40)

for count, co2l in enumerate(selected_co2ls_spi):

    ind = np.where(np.round(co2ls, 2) == co2l)[0][0]

    # ax.hist(np.repeat(inertia_time[ind], network.snapshot_weightings.generators)/1000,
    ax_inertia.hist(inertia_time[ind],
            weights=network.snapshot_weightings.generators,
            bins=bins,
            histtype='step',
            label=r'{} \%'.format(int(round(get_actual_co2_level(co2l)*100))),
            linewidth=3,
            color=cmap(np.where(selected_co2ls_spi == co2l)[0][0]/(len(selected_co2ls_spi)-1)),
            alpha=1)
# ax_inertia.xaxis.offsetText.set_fontsize(24)

leg = ax_inertia.legend(fontsize=22, title=r'CO$_2$ level [\% of 1990]')
# h, l = ax_inertia.get_legend_handles_labels()
# leg = axs_hist_legend.legend(h, l, borderaxespad=0, fontsize=20.5, title=r'CO$_2$ level [\% of 1990]')
# axs_probs_legend.legend(fontsize=24, title=r'CO$_2$ level [\% of 1990]')
# axs_hist_legend.axis("off")
# leg=ax_inertia.legend(fontsize=24, title=r'CO$_2$ level [\% of 1990]')

plt.setp(leg.get_title(),fontsize=24)
ax_inertia.tick_params(axis='both', which='both', labelsize=24)
ax_inertia.set_xlabel('Rotational energy [GWs]',  fontsize=26)
ax_inertia.set_ylabel('Hours', fontsize=26)
# ax_inertia.set_yscale('log')
ax_inertia.grid()
# ax_inertia.get_legend().remove()


################## panel c: SPI vectors ####################

# scale_factor_vecs = max(max(np.abs(x_vals)), max(np.abs(y_vals)))
scale_factor_vecs = 2e4
y_lims = np.array([31.707498915424157, 60.29928495567561])
# y_lims = y_lims/2
# co2lvls_spi_plot = [0.6, 0.0]
for lvl_count, current_level in enumerate(selected_co2ls_spi):
    old_ind = 0
    for ii in range(12):

        new_ind = old_ind + \
            len(network.snapshots[network.snapshots.month == int(ii+1)])

        ind = np.where(co2ls == current_level)[0][0]

        current_vec_0 = np.median(np.repeat(dipole_vector[ind, 0, old_ind:new_ind], network.snapshot_weightings.generators.values[old_ind:new_ind].astype(int))) # maybe take mean?!
        current_vec_1 = np.median(np.repeat(dipole_vector[ind, 1, old_ind:new_ind], network.snapshot_weightings.generators.values[old_ind:new_ind].astype(int))) # maybe take mean?!
        # for mean use np.average with weights
        cmap_val = ind/(len(co2ls)-0.999)
        print(f"current level: {current_level}, ind: {ind}, cmap_val: {cmap_val}")
        
        gs_vecs_all[-lvl_count-1,ii].arrow(mean_pos[0],
                                        mean_pos[1],
                                        current_vec_0/scale_factor_vecs,
                                        current_vec_1/scale_factor_vecs,
                                        head_width=2.0,
                                        width=1.0,
                                        head_length=2.0,

                                        #alpha  = .8,
                                        facecolor=cmap(np.where(selected_co2ls_spi == current_level)[0][0]/(len(selected_co2ls_spi)-1)),
        )
        gs_vecs_all[-lvl_count-1,ii].set_ylim(y_lims)
        gs_vecs_all[-lvl_count-1,ii].set_title(current_level)

        old_ind = new_ind
        
        ax_vectors_lvls[-lvl_count-1].text(0.5, 0.5, rf"{int(round(get_actual_co2_level(current_level)*100))}\%", fontsize=24, ha='center', )

for ii,ax_loss_lvl in enumerate(gs_vecs_all[0,:]):
    ax_loss_lvl.set_title(network.snapshots[network.snapshots.month == int(ii+1)].month_name()[0][:3], fontsize=24, pad=0)
    
ax_vectors_lvls[0].set_title('CO$_2$ level\n'+r'[\% of 1990]', fontsize=26, rotation=0, weight='bold', pad=-40)

# common grid behind spi vectors
axs_grid = f.add_subplot(gs_vectors[:,1:], zorder=-1)
for _, spine in axs_grid.spines.items():
    spine.set_visible(False)
axs_grid.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False, right=False )
axs_grid.set_title(r"\textbf{SPI vectors}", fontsize=30, pad=50)

ii = 7
ticks = [
    # 0.0,
 0.16666666666666666,
 0.515,
 0.86,
 ]
axs_grid.set_yticks(ticks)
#axs_grid.set_xlim([0,12])
#axs_grid.set_xticks(np.arange(1,12)-0.2)
axs_grid.grid(axis='y', alpha=0.5)

# plt.savefig(save_path+'flow_inertia_spi.pdf', bbox_inches='tight')

# %% [markdown]
# # Figure 7: line failures

# %% [markdown]
# ## split likelihood

# %% [markdown]
# Note: These are the likelihoods for ANY split that occured, i.e., also those with only 1 or 2 nodes being split off. Should we keep it like this?
# 
# Important: When setting vmin larger then the minimum probability, then edges with likelihood below vmin appear as gray, which might be confused with a line that never leads to a blackout. So: Check vmin/vmax configuration and maybe explain in figure legend. 

# %%
edge_likelihoods_primary = pickle.load(
    open(path_to_vis_results_sclopf+f'edge_likelihoods_primary_all_co2ls_n{n_nodes}.pickle', 'rb'))
edge_likelihoods_secondary = pickle.load(
    open(path_to_vis_results_sclopf+f'edge_likelihoods_secondary_all_co2ls_n{n_nodes}.pickle', 'rb'))


# %%
##### setup figure #####
f = plt.figure(figsize=(21, 10))
gs_vertical = GridSpec(1, 2, figure=f, width_ratios=[3, 0.05], wspace=-0.02)
gs_maps = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_vertical[0], wspace=-0.1, hspace=-0.1)
axs_primary = [f.add_subplot(gs_maps[0, i]) for i in range(3)]
axs_secondary = [f.add_subplot(gs_maps[1, i]) for i in range(3)]
gs_colorbars = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_vertical[1], height_ratios=[0.2, 1, 0.2, 1, 0.2])
axs_colorbars = [f.add_subplot(gs_colorbars[i]) for i in [1, 3]]
# f.subplots_adjust(hspace=-0.1, wspace=0.0)
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

##### setup parameters #####
labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']
vmaxvals = []
vminvals = []

#### Primary failures ######

cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
cmap.set_under('gainsboro', 1.0)

for count, co2l in enumerate(selected_co2ls[::-1]):

    level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

    print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    print('Max val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    
    vmax = 1e-3
    # vmax = 10**max(c_H_p_log)
    vmin =  1e-6

    nodes = nx.draw_networkx_nodes(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                node_color='black',
                                node_size=0)

    edges = nx.draw_networkx_edges(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                edge_color=c_H_p_log,
                                width=3.5,
                                edge_cmap=cmap,
                                edge_vmin=np.log10(vmin),
                                edge_vmax=np.log10(vmax))

    axs_primary[count].axis('off')


# cbar_ax = f.add_axes([0.9, 0.57, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
cb = f.colorbar(sm, cax=axs_colorbars[0])
cb.ax.tick_params(labelsize=26, width=1.0, which='both')
axs_colorbars[0].set_title(r'$\langle p_{\ell}^{\text{p}}\rangle$',
             fontsize=25,
             weight='bold',
             verticalalignment='center',
             pad=20)

###### Secondary failures #####

cmap = copy.copy(mpl.cm.get_cmap("viridis_r"))
cmap.set_under('gainsboro', 1.0)

for count, co2l in enumerate(selected_co2ls[::-1]):

    level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_s_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_s])
    
    print('Min val sec: {:e}'.format(np.amin(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
    print('Max val sec: {:e}'.format(np.amax(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
    # vmax = 1e-3
    # vmax = 10**max(c_H_s_log)
    vmax = 1e-3
    vmin = 1e-5
    vmaxvals.append(vmax)
    vminvals.append(vmin)

    nodes = nx.draw_networkx_nodes(nx_graph,
                                   pos=pos,
                                   ax=axs_secondary[count],
                                   node_color='black',
                                   node_size=0)

    edges = nx.draw_networkx_edges(nx_graph,
                                   pos=pos,
                                   ax=axs_secondary[count],
                                   edge_color=c_H_s_log,
                                   width=3.5,
                                   edge_cmap=cmap,
                                   edge_vmin=np.log10(vmin),
                                   edge_vmax=np.log10(vmax))

    axs_secondary[count].axis('off')

    axs_primary[count].set_title(r'CO$_2 =$ ' + '{} \%'.format(int(round(co2l*100))), fontsize=28)
    axs_primary[count].text(0+0.1, 1+0.05,
                    labels[count],
                    fontsize=30,
                    weight='bold',
                    verticalalignment='center',
                    transform=axs_primary[count].transAxes)


# cbar_ax = f.add_axes([0.9, 0.19, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
cb = f.colorbar(sm, cax=axs_colorbars[1])
# cb.set_ticks([1e-5,1e-4, 1e-3, vmax], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$1.5\cdot10^{-3}$'])
cb.set_ticks([1e-5,1e-4, 1e-3], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$'])
cb.ax.tick_params(labelsize=26, width=1.0, which='major')
axs_colorbars[1].set_title(r'$\langle p_{\ell}^{\textrm{s}}\rangle$',
             fontsize=25,
             weight='bold',
             verticalalignment='center',
             pad=20)
print(np.max(vmaxvals), np.min(vminvals))


plt.savefig(save_path+'line_failure_probs.pdf',bbox_inches = 'tight')

# %%
##### setup figure #####
f = plt.figure(figsize=(21, 10))
gs_vertical = GridSpec(1, 2, figure=f, width_ratios=[3, 0.05], wspace=-0.02)
gs_maps = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_vertical[0], wspace=-0.1, hspace=-0.1)
axs_primary = [f.add_subplot(gs_maps[0, i]) for i in range(3)]
axs_secondary = [f.add_subplot(gs_maps[1, i]) for i in range(3)]
gs_colorbars = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_vertical[1], height_ratios=[0.2, 1, 0.2, 1, 0.2])
axs_colorbars = [f.add_subplot(gs_colorbars[i]) for i in [1, 3]]
# f.subplots_adjust(hspace=-0.1, wspace=0.0)
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
level
##### setup parameters #####
labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']
vmaxvals = []
vminvals = []

#### Primary failures ######

cmap = copy.copy(mpl.cm.get_cmap("magma_r"))
cmap.set_under('gainsboro', 1.0)

probs_all = np.concatenate([np.array(list(edge_likelihoods_primary[level].values())) for level in selected_co2ls])
vmin = 0
vmax = probs_all.max()

for count, co2l in enumerate(selected_co2ls[::-1]):

    level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

    print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    print('Max val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    
    # vmax = 1e-3
    # # vmax = 10**max(c_H_p_log)
    # vmin =  1e-6

    nodes = nx.draw_networkx_nodes(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                node_color='black',
                                node_size=0)

    edges = nx.draw_networkx_edges(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                edge_color=c_H_p,
                                width=3.5,
                                edge_cmap=cmap,
                                edge_vmin=vmin,
                                edge_vmax=vmax)

    axs_primary[count].axis('off')


# cbar_ax = f.add_axes([0.9, 0.57, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mplcolors.Normalize(vmin=vmin, vmax=vmax))
cb = f.colorbar(sm, cax=axs_colorbars[0])
cb.ax.tick_params(labelsize=26, width=1.0, which='both')
axs_colorbars[0].set_title(r'$\langle p_{\ell}^{\text{p}}\rangle$',
             fontsize=25,
             weight='bold',
             verticalalignment='center',
             pad=20)

###### Secondary failures #####

cmap = copy.copy(mpl.cm.get_cmap("magma_r"))
cmap.set_under('gainsboro', 1.0)

# vmax = 1e-3
# vmin = 1e-5

probs_all = np.concatenate([np.array(list(edge_likelihoods_secondary[level].values())) for level in selected_co2ls])
vmin = 0
vmax = probs_all.max()


for count, co2l in enumerate(selected_co2ls[::-1]):

    level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_s_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_s])
    
    print('Min val sec: {:e}'.format(np.amin(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
    print('Max val sec: {:e}'.format(np.amax(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
    # vmax = 1e-3
    # vmax = 10**max(c_H_s_log)
    # vmax = 1e-3
    # vmin = 1e-5
    # vmaxvals.append(vmax)
    # vminvals.append(vmin)

    nodes = nx.draw_networkx_nodes(nx_graph,
                                   pos=pos,
                                   ax=axs_secondary[count],
                                   node_color='black',
                                   node_size=0)

    edges = nx.draw_networkx_edges(nx_graph,
                                   pos=pos,
                                   ax=axs_secondary[count],
                                   edge_color=c_H_s,
                                   width=3.5,
                                   edge_cmap=cmap,
                                   edge_vmin=vmin,
                                   edge_vmax=vmax)

    axs_secondary[count].axis('off')

    axs_primary[count].set_title(r'CO$_2 =$ ' + '{} \%'.format(int(round(co2l*100))), fontsize=28)
    axs_primary[count].text(0+0.1, 1+0.05,
                    labels[count],
                    fontsize=30,
                    weight='bold',
                    verticalalignment='center',
                    transform=axs_primary[count].transAxes)


# cbar_ax = f.add_axes([0.9, 0.19, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mplcolors.Normalize(vmin=vmin, vmax=vmax))
cb = f.colorbar(sm, cax=axs_colorbars[1])
# cb.set_ticks([1e-5,1e-4, 1e-3, vmax], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$1.5\cdot10^{-3}$'])
# cb.set_ticks([1e-5,1e-4, 1e-3], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$'])
cb.ax.tick_params(labelsize=26, width=1.0, which='major')
axs_colorbars[1].set_title(r'$\langle p_{\ell}^{\textrm{s}}\rangle$',
             fontsize=25,
             weight='bold',
             verticalalignment='center',
             pad=20)
# print(np.max(vmaxvals), np.min(vminvals))


plt.savefig(save_path+'line_failure_probs_linear.pdf',bbox_inches = 'tight')

# %% [markdown]
# inspect ration between primary and secondary failure probabilities

# %%
##### setup figure #####
f = plt.figure(figsize=(21, 10))
gs_vertical = GridSpec(1, 2, figure=f, width_ratios=[3, 0.05], wspace=-0.02)
gs_maps = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_vertical[0], wspace=-0.1, hspace=-0.1)
axs_primary = [f.add_subplot(gs_maps[0, i]) for i in range(3)]
# axs_secondary = [f.add_subplot(gs_maps[1, i]) for i in range(3)]
gs_colorbars = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_vertical[1], height_ratios=[0.2, 1, 0.2, 1, 0.2])
axs_colorbars = [f.add_subplot(gs_colorbars[i]) for i in [1, 3]]
# f.subplots_adjust(hspace=-0.1, wspace=0.0)
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

##### setup parameters #####
labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']
vmaxvals = []
vminvals = []

#### Primary failures ######

cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
cmap.set_under('gainsboro', 1.0)

for count, co2l in enumerate(selected_co2ls[::-1]):

    level = np.round(co2l, 2)
    
    # Load likelihoods as dictionary and transform into array
    c_H_p = [edge_likelihoods_primary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_p_sec = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
    c_H_p = [a/b if b!=0 else 0 for a,b in zip(c_H_p_sec,c_H_p)]
    c_H_p_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_p])

    print('Min val prim: {:e}'.format(np.amin(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    print('Mac_H_p_secx val prim: {:e}'.format(np.amax(np.array(c_H_p)[np.array(c_H_p)>1e-12])))
    
    # vmax = 1e-3
    # # vmax = 10**max(c_H_p_log)
    # vmin =  1e-6
    vmax = 10**max(c_H_p_log)
    vmin =  vmax * 1e-5

    nodes = nx.draw_networkx_nodes(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                node_color='black',
                                node_size=0)

    edges = nx.draw_networkx_edges(nx_graph,
                                pos=pos,
                                ax=axs_primary[count],
                                edge_color=c_H_p_log,
                                width=3.5,
                                edge_cmap=cmap,
                                edge_vmin=np.log10(vmin),
                                edge_vmax=np.log10(vmax))

    axs_primary[count].axis('off')


# cbar_ax = f.add_axes([0.9, 0.57, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
cb = f.colorbar(sm, cax=axs_colorbars[0])
cb.ax.tick_params(labelsize=26, width=1.0, which='both')
axs_colorbars[0].set_title(r'$\langle p_{\ell}^{\text{p}}\rangle$',
             fontsize=25,
             weight='bold',
             verticalalignment='center',
             pad=20)

# ###### Secondary failures #####

# cmap = copy.copy(mpl.cm.get_cmap("viridis_r"))
# cmap.set_under('gainsboro', 1.0)

# for count, co2l in enumerate(selected_co2ls[::-1]):

#     level = np.round(co2l, 2)
    
#     # Load likelihoods as dictionary and transform into array
#     c_H_s = [edge_likelihoods_secondary[level][(u, v)] for u, v in nx_graph.edges()]
#     c_H_s_log = np.array([np.log10(x) if x > 1e-12 else -np.inf for x in c_H_s])
    
#     print('Min val sec: {:e}'.format(np.amin(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
#     print('Max val sec: {:e}'.format(np.amax(np.array(c_H_s)[np.array(c_H_s)>1e-12])))
#     # vmax = 1e-3
#     # vmax = 10**max(c_H_s_log)
#     vmax = 1e-3
#     vmin = 1e-5
#     vmaxvals.append(vmax)
#     vminvals.append(vmin)

#     nodes = nx.draw_networkx_nodes(nx_graph,
#                                    pos=pos,
#                                    ax=axs_secondary[count],
#                                    node_color='black',
#                                    node_size=0)

#     edges = nx.draw_networkx_edges(nx_graph,
#                                    pos=pos,
#                                    ax=axs_secondary[count],
#                                    edge_color=c_H_s_log,
#                                    width=3.5,
#                                    edge_cmap=cmap,
#                                    edge_vmin=np.log10(vmin),
#                                    edge_vmax=np.log10(vmax))

#     axs_secondary[count].axis('off')

#     axs_primary[count].set_title(r'CO$_2 =$ ' + '{} \%'.format(int(co2l*100)), fontsize=28)
#     axs_primary[count].text(0+0.1, 1+0.05,
#                     labels[count],
#                     fontsize=30,
#                     weight='bold',
#                     verticalalignment='center',
#                     transform=axs_primary[count].transAxes)


# # cbar_ax = f.add_axes([0.9, 0.19, 0.005, 0.25]) # f.add_axes([0.89, 0.35, 0.005, 0.45])
# sm = plt.cm.ScalarMappable(
#     cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
# cb = f.colorbar(sm, cax=axs_colorbars[1])
# # cb.set_ticks([1e-5,1e-4, 1e-3, vmax], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$1.5\cdot10^{-3}$'])
# cb.set_ticks([1e-5,1e-4, 1e-3], labels= [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$'])
# cb.ax.tick_params(labelsize=26, width=1.0, which='major')
# axs_colorbars[1].set_title(r'$\langle p_{\ell}^{\textrm{s}}\rangle$',
#              fontsize=25,
#              weight='bold',
#              verticalalignment='center',
#              pad=20)
# cb.ax.yaxis.set_ticks([1e-6,1e-4,1e-2,1])

# minortcs = np.arange(1,10,1)
# minortcs = np.array([minortcs*1e-7,minortcs*1e-6,minortcs*1e-5,minortcs*1e-4,minortcs*1e-3]).flatten()
# minortcs = minortcs[(minortcs>vmin) & (minortcs<vmax)]
# cb.ax.yaxis.set_ticks(minortcs, minor=True)

# print(np.max(vmaxvals), np.min(vminvals))


# plt.savefig(save_path+'fig7.pdf',bbox_inches = 'tight')

# %% [markdown]
# # Nodal Blackout Prob

# %%
try:
    
    with open(path_to_evaluation_results_sclopf + f"bflackout_prob_dict_n{n_nodes}.pickle", "rb") as out:
        blackout_prob_dict = pickle.load(out)
    with open(path_to_evaluation_results_sclopf + f"edge_failure_prob_dict_n{n_nodes}.pickle", "rb") as out:
        edge_failure_prob_dict = pickle.load(out)
        
except FileNotFoundError:
    blackout_prob_dict = {}
    edge_failure_prob_dict = {}
    vmin_node = 1
    vmax_node = 0
    for target_level in selected_co2ls:
        print(target_level)
        
        path = (
                path_to_evaluation_results_sclopf
                + f"/indicator_vector_rocof_Co2L{target_level}_n{n_nodes}.pklz"
        )
        with gzip.open(path, "rb") as out:
            rocof_indicator_vectors = pickle.load(out)[-1]
        path = (
                path_to_evaluation_results_sclopf
                + f"/failed_edges_indicator_vector_Co2L{target_level}_n{n_nodes}.pklz"
        )
        with gzip.open(path, "rb") as out:
            failed_edges_indicator_vectors = pickle.load(out)[-1]

        weights = split_properties[split_properties.co2l==target_level].snapshot_weighting.values
        print(weights.shape, rocof_indicator_vectors.shape, failed_edges_indicator_vectors.shape)
        blackout_prob_dict[target_level] = (weights.reshape(1,-1) @  (rocof_indicator_vectors<-1)).squeeze()/num_failures_weighted
        
        edge_failure_prob_dict[target_level] = (weights.reshape(1,-1) @  failed_edges_indicator_vectors).squeeze()/num_failures_weighted
        
    with open(path_to_evaluation_results_sclopf + f"blackout_prob_dict_n{n_nodes}.pickle", "wb") as out:
        pickle.dump(blackout_prob_dict, out)
    with open(path_to_evaluation_results_sclopf + f"edge_failure_prob_dict_n{n_nodes}.pickle", "wb") as out:
        pickle.dump(edge_failure_prob_dict, out)


# %%
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

f = plt.figure(figsize=(15.5*1.1, 5*1.1))

gs_maps = GridSpec(1, len(selected_co2ls), figure=f)
gs_lvls = []
for i, target_level in enumerate(selected_co2ls):
    gs_lvl = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_maps[0, i], height_ratios=[15, 1], hspace=-0.05)
    gs_lvls.append(gs_lvl)

labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']

node_cmap = plt.get_cmap('plasma_r')
node_cmap = truncate_colormap(node_cmap, 0.1, 0.9, 1000)
node_cmap.set_under('gainsboro', 1.0)
node_cbar_label="blackout prob"

# vmax_node = -np.inf
# vmin_node = np.inf
# vmaxes = []

# for i,(gs_lvl,target_level) in enumerate(zip(gs_lvls,selected_co2ls[::-1])):
    
#     ax = f.add_subplot(gs_lvl[0])
#     ax_cbar = f.add_subplot(gs_lvl[1])

#     blackout_prob = blackout_prob_dict[target_level]
#     edge_failure_prob = edge_failure_prob_dict[target_level]
    
#     blackout_prob_log = np.array(
#         [np.log10(x) if x > 1e-12 else -np.inf for x in blackout_prob]
#     )
#     # vmax_node = blackout_prob.max()
#     # vmin_node = blackout_prob.min()
#     vmax_node = max(10**blackout_prob_log.max(), vmax_node)
#     vmin_node = min(10**blackout_prob_log.min(), vmin_node)
    
#     vmaxes.append(vmax_node)
#     # vmin_node = 10**blackout_prob_log.min()

# vmin_node = vmaxes[0]*1e-3

# vmin_node = 1e-4

for i,(gs_lvl,target_level) in enumerate(zip(gs_lvls,selected_co2ls[::-1])):
    
    ax_loss_lvl = f.add_subplot(gs_lvl[0])
    ax_cbar = f.add_subplot(gs_lvl[1])

    blackout_prob = blackout_prob_dict[target_level]
    edge_failure_prob = edge_failure_prob_dict[target_level]
    
    blackout_prob_log = np.array(
        [np.log10(x) if x > 1e-12 else -np.inf for x in blackout_prob]
    )
    # vmax_node = blackout_prob.max()
    # vmin_node = blackout_prob.min()
    vmax_node = 10**blackout_prob_log.max()
    vmin_node = 10**blackout_prob_log.min()

    # vmin_node = vmax * 10e-1
    print(f"level: {target_level}")
    print(vmax_node)
    print(vmin_node)
    # vmax_node = 10e-1
    # vmin_node = 10e-8

    nodes = nx.draw_networkx_nodes(
        nx_graph,
        pos=pos,
        ax=ax_loss_lvl,
        node_size=25,
        node_color=blackout_prob_log,
        cmap=node_cmap,
        vmax=np.log10(vmax_node),
        vmin=np.log10(vmin_node),

    )
    nodes.set_edgecolor("black")
    nodes.set_linewidth(0.2)

    edges = nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        ax=ax_loss_lvl,
        width=0.1,
        edge_color="black",
    )
    
    ax_loss_lvl.set_title(r'CO$_2 =$ ' + '{} \%'.format(int(get_actual_co2_level(target_level)*100)), fontsize=28)
    ax_loss_lvl.text(0, 1+0.1,
                    labels[i],
                    fontsize=30,
                    weight='bold',
                    verticalalignment='center',
                    transform=ax_loss_lvl.transAxes)
    
    # ax.set_title(f"CO2={int(target_level*100)}%", fontsize=24)
    ax_loss_lvl.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap=node_cmap, norm=mplcolors.LogNorm(vmin=vmin_node, vmax=vmax_node))
    cb = f.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    if target_level==0.0:
        # ticks = ax_cbar.get_xticks()
        # ax_cbar.set_xticks(ticks, labels=["" for i in range(len(ticks))])
        # ax_cbar.set_xticks([i*1e-4 for i in range(5,10)], labels=["$5\cdot 10^{-4}$", "$6\cdot10^{-4}$","$7\cdot 10^{-4}$","$8\cdot 10^{-4}$", "$9\cdot 10^{-4}$"])
        ax_cbar.set_xticks([i*1e-4 for i in range(5,10)], labels=["$5\cdot 10^{-4}$", "","$7\cdot 10^{-4}$","", "$9\cdot 10^{-4}$"])
    elif target_level==0.2:
        ax_cbar.set_xticks([5e-5,1e-4,2e-4,4e-4], labels=["$5\cdot 10^{-5}$", "$10^{-4}$","$2\cdot 10^{-4}$","$4\cdot 10^{-4}$"])
    ax_cbar.set_xlabel(r'$\langle p_{n}\rangle$', fontsize=18, labelpad=5)
    ax_cbar.tick_params(labelsize=18, width=1.0, which='both')


f.savefig(save_path+'nodal_blackout_prob.pdf',bbox_inches = 'tight')

# %% [markdown]
# # Figure 5: split statistics

# %%
# co2ls = np.arange(0.1,0.81,0.1).round(1)
# selected_co2ls = np.array([0.1, 0.3, 0.6, 0.8])

# Load split component properties
component_props = pd.read_hdf(path_to_vis_results_sclopf + f'component_properties_all_n{n_nodes}.h5')
component_props.time_stamp = pd.to_datetime(component_props.time_stamp)
# indicator_vectors = np.load(path_to_vis_results + f'indicator_vectors_all_n400.npy').astype(bool)
split_props = split_properties
# split_vectors = np.load(path_to_vis_results + f'split_vectors_all_n400.npy')

split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(float)

# %%
cmap = plt.get_cmap("cividis")

bins = np.linspace(0, 100, 6).astype(int)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
data = []
for co2l in co2ls[::-1]:
    vals = split_props[split_props.co2l==co2l].lost_load_share_blackout.values*100
    counts, _ = np.histogram(vals, bins=bins)
    data.append(counts)

data = np.stack(data)
data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)
data

# %%
split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(float)

# %%
split_props.rename(columns={"neglidgeable_blackout": "negligible"}, inplace=True)

# %%
#TODO: remove this when all splits were evaluated!!!
# Restricting analysis to minimum of available time steps

# n_steps_to_consider = component_props.groupby('co2l').apply(lambda x:x.time_stamp.unique().shape[0]).min()
# number_of_simulations = n_steps_to_consider*len(n_2_failures)
# component_props = component_props[pd.to_datetime(component_props.time_stamp)<network.snapshots[n_steps_to_consider]]
component_props["snapshot_weighting"] = network.snapshot_weightings.generators[component_props.time_stamp].values
# indicator_vectors = indicator_vectors[:component_props.shape[0]]

# split_props = split_props[pd.to_datetime(split_props.index.get_level_values('time_stamp'))<network.snapshots[n_steps_to_consider]]
# split_vectors = split_vectors[:split_props.shape[0]]

# n_steps_to_consider


# split_props["snapshot_weighting"] = network.snapshot_weightings.generators[split_props.index.get_level_values(1)].values
weighted_split_props_dict = dict({co2l:{category: split_props.loc[(split_props.category==category) & (split_props.index.get_level_values(0)==co2l), "snapshot_weighting"].sum() for category in split_props.category.unique()} for co2l in co2ls})
# for index,row in split_props.iterrows():
#         weighted_split_props_dict[index[0]][row.category] += row.snapshot_weighting

# %% [markdown]
# * **negligible**: there is a component with with >99% load that survives
# * **relevant split without blackout/ Non-severe**: Not negligible and not blackout.
# * **locally severe/ local blackout**: blackout of <99% of CE load
# * **globally severe/ global blackout**: Blackout of >99% of CE load.
# 
# 
# 

# %%
from cycler import cycler
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

fig = plt.figure(figsize=(9, 3.3))
gs_vertical = GridSpec(2, 1, figure=fig, hspace=0.6, height_ratios=[1,0.1])


# Panel c: split number and categories
n_cols = 3
gsTop = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_vertical[0,:], hspace=0, wspace=0.4)
gs_bottom = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_vertical[1,:])
ax1 = fig.add_subplot(gsTop[2])
ax1_legend = fig.add_subplot(gs_bottom[1:])


cmap = plt.get_cmap("cividis")

bins = np.linspace(0, 100, 6).astype(int)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
data = []
for co2l in co2ls[::-1]:
    vals = split_props[split_props.co2l==co2l].lost_load_share_blackout.values*100
    counts, _ = np.histogram(vals, bins=bins)
    data.append(counts)

data = np.stack(data)
data = pd.DataFrame(data, index=co2ls[::-1], columns=bin_centers)

plt.sca(ax1)
cmap = plt.get_cmap("inferno_r")
for i, bin_center in enumerate(bin_centers):
    actual_lvl = get_actual_co2_level(co2l)
    color = cmap(i/(len(bin_centers)+1)+(1/(len(bin_centers)+1)))
    counts = data.loc[:, bin_center]
    plt.plot(get_actual_co2_level(co2ls[::-1])*100, counts, label=rf'{bins[i]}-{bins[i+1]}\%', alpha=0.8, color=color, marker=".", markersize=10)
plt.xlabel(r'CO$_2$ level [\% of 1990]')
plt.ylabel('Number of System Splits')
plt.yscale("log")
plt.gca().invert_xaxis()
# plt.title('Lost load share distribution for different CO2 levels')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1, title='Loss of load share')

# Legend
# fig.tight_layout()
h, l = ax1.get_legend_handles_labels()
# ax1.get_legend().remove()
ax1_legend.legend(h, l, title='Loss of load share', loc='center', ncols=len(bin_centers), columnspacing=0.5)
ax1_legend.axis("off")
# ax1.get_legend().remove()

# filter out very small components
# component_props_filtered = component_props[component_props.load_share>1/200]
component_props_filtered = component_props

# Panel b: Inertia histograms
cmap = plt.get_cmap('cividis')

ax2 = fig.add_subplot(gsTop[1])
ax2_legend = fig.add_subplot(gs_bottom[0])
rot_energy = component_props_filtered.rot_energy/1000
bins = np.linspace(rot_energy.min(),rot_energy.max(),15)
intertia_hist_vals = []
for co2l in selected_co2ls:
    vals = ax2.hist(rot_energy[component_props_filtered.co2l==co2l],
            weights=component_props_filtered.snapshot_weighting[component_props_filtered.co2l==co2l],
            bins=bins,
            histtype='step',
            label=r'{} \%'.format(int(100*co2l)),
            linewidth=3,
            color=cmap(np.where(co2ls == co2l)[0][0]/(len(co2ls)-1)),
            alpha=0.8,
            density=False)
    intertia_hist_vals.append(vals[0])
ax2.set_yscale('log')
ax2.set_xlabel('Rotational energy [GWs]')
ax2.set_ylabel('Number of split components')

# Panel a: power imbalance histograms

ax3 = fig.add_subplot(gsTop[0])
power_imbalance = component_props_filtered.power_imbalance/1000
bins = np.linspace(power_imbalance.min(),power_imbalance.max(),15)
chandles = []
for co2l in selected_co2ls:
    h= ax3.hist(power_imbalance[component_props_filtered.co2l==co2l],
            weights=component_props_filtered.snapshot_weighting[component_props_filtered.co2l==co2l],
            bins=bins,
            histtype='step',
            linewidth=3,
            label=r'{} \%'.format(int(100*co2l)),
            color=cmap(np.where(co2ls == co2l)[0][0]/(len(co2ls)-1)),
            alpha=0.8,
            density=False)
ax3.set_yscale('log')
ax3.set_ylabel('Number of split components')
ax3.set_xlabel('Power imbalance [GW]')
ax3.set_xticks(np.arange(-50,51,step=25))


# h, l = ax4.get_legend_handles_labels()
h, l = ax2.get_legend_handles_labels()
# ax1.get_legend().remove()
ax2_legend.legend(h[::-1], l[::-1], title=r'CO$_2$ level [\% of 1990]', loc='center', ncols=3, columnspacing=0.5)
ax2_legend.axis("off")
# ax2.legend(bbox_to_anchor=(1,-0.35), ncols=4, title=r'CO$_2$ level [\% of 1990]')

# add title to subplots
# for ax,label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c','d']):
for ax_loss_lvl,label in zip([ax1, ax2, ax3], ['c', 'b', 'a']):
    # ax.set_title('')
    ax_loss_lvl.text(0 -0.25, 1+0.1,
                    label,
                    fontsize=30,
                    weight='bold',
                    verticalalignment='center',
                    transform=ax_loss_lvl.transAxes)

# plt.tight_layout()
plt.savefig(save_path+'split_statistics.pdf',bbox_inches = 'tight')

# %% [markdown]
# # storage time

# %%
types = ["discharging", "start_of_streak", "streak_id"]
n = networks[0.0]
multi_index = pd.MultiIndex.from_product([n.storage_units_t.p.columns,types], names=('units', "storage_state"))
df = pd.DataFrame(index=n.snapshots, columns=multi_index)

idx = pd.IndexSlice
df.loc[idx[:, idx[:,"discharging"]]]= (n.storage_units_t.p<0).values
df.loc[idx[:, idx[:,"start_of_streak"]]] = df.loc[idx[:, idx[:,"discharging"]]].ne(df.loc[idx[:, idx[:,"discharging"]]].shift()).values
df.loc[idx[:, idx[:,"streak_id"]]] = df.loc[idx[:, idx[:,"start_of_streak"]]].cumsum().values

# %%
df_streak_id = df.loc[idx[:, idx[:,"streak_id"]]].copy()
df_streak_id.columns = df_streak_id.columns.droplevel(level=1)
id_counts = df_streak_id.apply(lambda x: x.value_counts())
storage_type_list = [col.split(" ")[-1] for col in df_streak_id.columns]
streak_lenghts = id_counts.groupby(storage_type_list, axis=1).sum()

# %%
for target_level in selected_co2ls:
    n = networks[target_level].copy()
    discharging_ratios.columns = discharging_ratios.columns.droplevel(level=1)
    state_counts = discharging_ratios.apply(lambda x: x.value_counts())
    storage_type_list = [col.split(" ")[-1] for col in discharging_ratios.columns]
    discharging_ratios_types = discharging_ratios.groupby(storage_type_list, axis=1).mean().mean(axis=0)
    print(f"co2 lvl: {target_level}")
    print(discharging_ratios_types)

# %%
streak_lenghts.hist(bins=100)

# %%
# calculate line flows
cmap = plt.get_cmap('cividis_r')


bins = np.arange(0, 1.0, 0.02)


hist_vals = np.zeros((len(selected_co2ls), len(bins)-1))


storga_times_lvl = []
for count, co2l in enumerate(selected_co2ls):

   
    n = networks[co2l]

    indices = nx.get_edge_attributes(nx_graph, 'line_index')
    line_limits_dict = nx.get_edge_attributes(nx_graph, 's_nom')

    flows = n.lines_t.p0.abs().sum(axis=1)
    # flows.hist(ax=axs_probs, bins=10, alpha=0.5, color=cmap(ind/len(co2ls), 0.8), label=r'{}\%'.format(int(100*co2l)))
    storga_times_lvl.append(flows)

bins=np.histogram(np.hstack(storga_times_lvl), bins=40)[1]

# %% [markdown]
# # Inertia Mitigation

# %%
from utils.plot_mitigation_strategies import plot_map_inertia_placement_final, calc_inertia_placement_ref_loss, color1, color2

# %%
co2_lvl_ref = 0.6
inertia_needed_by_lvl = {}
ref_loss_factors = (1,2)
for ref_loss_factor in ref_loss_factors:
    inertia_at_ref_loss_by_lvl = calc_inertia_placement_ref_loss(n_nodes=600,     delta_Erot=1000, co2_lvl_ref=co2_lvl_ref, split_properties=split_properties, ref_loss_factor=ref_loss_factor, co2_lvls=co2ls)
    inertia_needed_by_lvl[ref_loss_factor] = inertia_at_ref_loss_by_lvl

# %%
# f.tight_layout()
mpl.style.use('default')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

colors = color1, color2
labels = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}']

co2_lvl_map = 0.1

f = plt.figure(figsize=(9, 6.2))
gs_vertical = GridSpec(2, 1, figure=f, height_ratios=[1, 1.5], hspace=0.2)
gs_lines = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vertical[0], wspace=0.4)
gs_maps = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vertical[1], wspace=0.1)
ax_loss_reduction = f.add_subplot(gs_lines[0])
ax_inertia_needed = f.add_subplot(gs_lines[1])
axs_maps = [f.add_subplot(gs_maps[i]) for i in range(2)]

unit = "GWs"
if unit == "GWs":
    unit_factor = 1e-3
elif unit == "MWs":
    unit_factor = 1
else:
    raise ValueError(f"Unit '{unit}' not known!")

# inertia needed by lvl
for i,ref_loss_factor in enumerate(ref_loss_factors):
    inertia_at_ref_loss_by_lvl = inertia_needed_by_lvl[ref_loss_factor] 
    label = "$R_{\\textrm{mit}}=ref_loss_factor R_{\\textrm{co2refString}}$"
    # label = label.replace("co2string", str(int(100 * co2_lvl_map)) + "\%")
    label = label.replace("co2refString", str(int(100 * get_actual_co2_level(co2_lvl_ref))) + "\%")
    if ref_loss_factor != 1:
        label = label.replace("ref_loss_factor", str(ref_loss_factor)+"\cdot")
    else:
        label = label.replace("ref_loss_factor", "")
    # label = label.replace("ref_loss_factor", str(ref_loss_factor))
    ax_inertia_needed.plot(np.array(list(inertia_at_ref_loss_by_lvl.keys()))*100, np.array(list(inertia_at_ref_loss_by_lvl.values()))*unit_factor, label=label, color=colors[i])
    
# ax_inertia_needed.plot(np.array(co2ls) * 100, np.array(inertia_at_ref_loss_by_lvl) *unit_factor)
# invert x axis
ax_inertia_needed.invert_xaxis()
ax_inertia_needed.legend(title="Reference loss factor")
# plt.legend(title="$\\Delta E_0 [MWhs]$")
ax_inertia_needed.set_ylabel(f"Inertia placed [{unit}]")
ax_inertia_needed.set_xlabel("CO2 level [\% of 1990]")
ax_inertia_needed.set_title("Inertia needed to reach reference lost load share")
ticks = np.array(list(inertia_at_ref_loss_by_lvl.keys()))*100
ax_inertia_needed.set_xticks(ticks, labels=[str(int(i)) for i in ticks])
if ticks[0]<ticks[-1]:
    ax_inertia_needed.invert_xaxis()
ax_inertia_needed.grid(True)
for i, ref_loss_factor in enumerate(ref_loss_factors):
    plot_curve = i==0
    plot_map_inertia_placement_final(
        axes=(axs_maps[1-i], ax_loss_reduction),
        co2_lvl = co2_lvl_map,
        nn=600,
        max_iter=10000,
        max_node_size=100,
        edge_width=0.2,
        delta_Erot=1000,
        rocof_thres=-1,
        l_share=0.0,
        resolve_strategy="random",
        show_step_number = False,
        plot_split_number = False,
        save_fig=False,
        co2_lvl_ref = co2_lvl_ref,
        ref_loss_factor=ref_loss_factors[i],
        unit = unit,
        color=colors[i],
        plot_curve=plot_curve,
        split_properties=split_properties,
        )
    
ax_loss_reduction.set_xlim(0, inertia_needed_by_lvl[1][co2_lvl_map]*unit_factor*1.2)

# f.tight_layout()
for ax_loss_lvl, label in zip([ax_loss_reduction, ax_inertia_needed], labels):
    ax_loss_lvl.text(0 - 0.2, 1+0.2,
            label,
            fontsize=36,
            weight='bold',
            verticalalignment='center',
            transform=ax_loss_lvl.transAxes)
# axs_maps[0].text( 0.5,70, rf"{int(100*get_actual_co2_level(co2_lvl_map))}\% CO$_2$ level", fontsize=19)
axs_maps[0].text(0 - 0.13, 1,
            labels[2],
            fontsize=36,
            weight='bold',
            verticalalignment='center',
            transform=axs_maps[0].transAxes)

# axs_maps[1].text(0.5,70, rf"{int(100*get_actual_co2_level(co2_lvl_map))}\% CO$_2$ level", fontsize=19)
axs_maps[1].text(0 - 0.05, 1,
            labels[3],
            fontsize=36,
            weight='bold',
            verticalalignment='center',
            transform=axs_maps[1].transAxes)
for ax_loss_lvl in axs_maps:
    ax_loss_lvl.legend(loc='upper left', )

ax_loss_reduction.set_title(f"Loss reduction {int(round(get_actual_co2_level(co2_lvl_map)*100))}\% CO$_2$ level")

# for i,ax in enumerate(axs_maps):
#     ax.title("Synthetic inertia placement")

# axs_maps[0].text(0.2, 1+0.06,
#         r'\textbf{a}',
#         fontsize=36,
#         weight='bold',
#         verticalalignment='center',
#         transform=axs_maps[0].transAxes)
# axs_maps[0].text(1 , 1+0.06,
#         r'\textbf{b}',
#         fontsize=36,
#         weight='bold',
#         verticalalignment='center',
#         transform=axs_maps[0].transAxes
#         )
# ax_map.text(1 , 0.5,
#         r'\textbf{c}',
#         fontsize=36,
#         weight='bold',
#         verticalalignment='center',
#         transform=ax_map.transAxes
#         )
    

# f.savefig(save_path+f'syn_inertia_mitigation_{co2_lvl_map}.pdf',bbox_inches = 'tight')

# %% [markdown]
# # line extension mitigation

# %% [markdown]
# ## inspecting induced loss by trigger line

# %%
co2_mitigation = 0.1
split_properties.lost_load_share_blackout = split_properties.lost_load_share_blackout.astype(float)
split_properties["co2l"] = split_properties.index.get_level_values("co2l")
split_properties_mitigation = split_properties[split_properties.co2l == co2_mitigation].copy()

trigger0 = split_properties_mitigation.loc[:,["lost_load_share_blackout","init_failure_0"],].groupby("init_failure_0").sum()
trigger1 = split_properties_mitigation.loc[:,["lost_load_share_blackout","init_failure_1"],].groupby("init_failure_1").sum()
trigger1.rename_axis("trigger", inplace=True)
trigger0.rename_axis("trigger", inplace=True)
loss_by_trigger = pd.Series(index=list(range(n_lines)), data=0)
loss_by_trigger.rename_axis("trigger", inplace=True)


# %% [markdown]
# ## mitigation

# %% [markdown]
# ## cost

# %%
from utils.cascade_simulation import LOOKUP_TABLE_NP
table_rounded = np.round(LOOKUP_TABLE_NP, 5)
num_par_paths = {}
for tup in table_rounded:
    current_tup = tup
    num_par_paths[tup[0]] = [current_tup[0]-current_tup[1]]
    while current_tup[1]!= 0:
        current_tup = table_rounded[np.where((table_rounded[:,0]==current_tup[1]))[0][0]]
        num_par_paths[tup[0]].append(current_tup[0]-current_tup[1])
max_num_par = {k: max(v) for k, v in num_par_paths.items()}


# %%
costs_ = networks[0.0].lines.copy()
costs_ = costs_.loc[:, ["bus0", "bus1", "capital_cost", "s_nom", "num_parallel"]].copy()
costs_["num_par_ext"] = costs_.num_parallel.apply(lambda x: max_num_par[np.round(x,5)] if x in max_num_par else 1)
costs_["extension_cost"] = costs_.capital_cost * costs_.s_nom * costs_.num_par_ext

# %%
branches = networks[0.6].lines.copy()
branches["cost_per_km"] = branches.capital_cost/branches.length

# %%
branches[["type", "cost_per_km"]].hist()

# %%
branches.capital_cost.hist()
plt.xlabel("Capital cost")
plt.ylabel("Frequency")
plt.title("Distribution of capital costs of lines in the network")

# %%
branches[["num_parallel", "x", "type", "length", "s_nom"]]

# %%
branches.s_nom.hist()

# %%
# def calculate_investment_cost(eac, r, n):
#     """
#     Calculate upfront investment cost.
    
#     Parameters:
#     eac (float): Equivalent Annual Cost
#     r (float): Discount rate (as decimal, e.g., 0.08 for 8%)
#     n (int): Lifespan in years

#     Returns:
#     float: Upfront investment cost
#     """
#     annuity_factor = (1 - (1 + r) ** -n) / r
#     return eac * annuity_factor

# # Example
# eac = 42  # annual cost
# r = 0.085    # 8% discount rate
# n = 40      # 10-year lifespan

# investment_cost = calculate_investment_cost(eac, r, n)
# print(f"Upfront Investment Cost: ${investment_cost:.2f}")

# %%
from utils.config import path_to_line_extension_mitigation_sclopf

os.makedirs(path_to_line_extension_mitigation_sclopf, exist_ok=True)

co2l_ref=0.6
split_properties_reference = split_properties[split_properties.co2l == co2l_ref].copy()
lost_load_reference_lvl = split_properties_reference.lost_load_share_blackout.sum()
extension_cost_per_MWkm = 445


for annualized_costs in [True, False]:
    
    cost_to_reach_ref_loss = {}
    cost_to_reach_double_ref_loss = {}
    
    for co2l in co2ls:
        if co2l==0.6:
            continue
        try:
            if annualized_costs:
                f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2l}_n{n_nodes}.pkl"
            
            else:
                f_name = f"heuristic_costMin_loss_mitigation_Co2L{co2l}_n{n_nodes}.pkl"
                
            reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(open(path_to_line_extension_mitigation_sclopf + f_name, "rb"))
        except FileNotFoundError:
            split_props_lvl = split_properties[split_properties.co2l == co2l].copy()
            split_props_lvl.lost_load_share_blackout = split_props_lvl.lost_load_share_blackout.astype(float)
            remaining_splits = split_props_lvl[split_props_lvl.lost_load_share_blackout>0]
            remaining_splits = remaining_splits.loc[:, ["lost_load_share_blackout","init_failure_0", "init_failure_1"]]
            costs_ = networks[0.6].lines.loc[:, ["bus0", "bus1", "capital_cost", "s_nom", "num_parallel", "length"]].copy()
            costs_["num_par_ext"] = costs_.num_parallel.apply(lambda x: max_num_par[np.round(x,5)] if x in max_num_par else 1)
            if annualized_costs:
                costs_["extension_cost"] = costs_.capital_cost * costs_.s_nom * costs_.num_par_ext
            else:
                costs_["extension_cost"] = extension_cost_per_MWkm * costs_.s_nom * costs_.num_par_ext * costs_.length

            loss_with_mitigation = [split_props_lvl.lost_load_share_blackout.sum()]
            reinforced_lines = []
            cost = []
            num_blackouts = [remaining_splits.shape[0]]

            while remaining_splits.shape[0]>0:
                trigger0 = remaining_splits.loc[:,["lost_load_share_blackout","init_failure_0"],].groupby("init_failure_0").sum()
                trigger1 = remaining_splits.loc[:,["lost_load_share_blackout","init_failure_1"],].groupby("init_failure_1").sum()
                trigger1.rename_axis("trigger", inplace=True)
                trigger0.rename_axis("trigger", inplace=True)
                loss_by_trigger = pd.Series(index=list(range(n_lines)), data=0)
                loss_by_trigger.rename_axis("trigger", inplace=True)
                loss_by_trigger = loss_by_trigger.add(trigger0.lost_load_share_blackout, fill_value=0)
                loss_by_trigger = loss_by_trigger.add(trigger1.lost_load_share_blackout, fill_value=0)
                loss_per_dollar = loss_by_trigger / costs_["extension_cost"].reset_index(drop=True)

                trigger = loss_per_dollar.idxmax()
                cost.append(costs_["extension_cost"][int(trigger)])
                remaining_splits = remaining_splits[(remaining_splits.init_failure_1!=trigger) & (remaining_splits.init_failure_0!=trigger)]
                loss_with_mitigation.append(remaining_splits.lost_load_share_blackout.sum())
                reinforced_lines.append(trigger)
                num_blackouts.append(remaining_splits.shape[0])

            with open(path_to_line_extension_mitigation_sclopf + f_name, "wb") as f:
                pickle.dump((reinforced_lines, loss_with_mitigation, num_blackouts, cost), f)
                
            num_lines_to_reach_ref_loss = np.where(loss_with_mitigation < lost_load_reference_lvl)[0][0]
            num_lines_to_reach_double_ref_loss = np.where(loss_with_mitigation < 2*lost_load_reference_lvl)[0][0]
            cost_to_reach_ref_loss[co2l] = sum(cost[:num_lines_to_reach_ref_loss])
            cost_to_reach_double_ref_loss[co2l] = sum(cost[:num_lines_to_reach_double_ref_loss])

            # latexreinforced_lines_selected = np.array(reinforced_lines[:num_lines_to_reach_ref_loss]).astype(int)
            # lines_not_extended = np.setdiff1d(np.arange(n_lines), reinforced_lines_selected)
            # reinforced_lines_full = list(reinforced_lines_selected) + list(lines_not_extended)

    cost_to_reach_ref_loss = pd.DataFrame.from_dict(cost_to_reach_ref_loss, orient='index', columns=['cost_ref_loss'])
    cost_to_reach_ref_loss.index.name = 'co2l'
    cost_to_reach_ref_loss["cost_double_ref_loss"] = pd.Series(cost_to_reach_double_ref_loss)
    fname = "cost_to_reach_ref_loss.csv"
    if annualized_costs:
        fname = "annualized_" + fname
    else:
        fname = "total_" + fname

    cost_to_reach_ref_loss.to_csv(save_path + fname)

# %%
# actual_co2_ref = int(round(get_actual_co2_level(co2l_ref) * 100))
# label = label.replace("co2string", str(int(100 * co2_lvl_map)) + "\%")
co2_ref_str = str(int(100 * get_actual_co2_level(co2l_ref))) + "\%"

plot_lvls = [0.1]
plot_cost= True
for annualized_costs in [True, False]:
    
    fname = "cost_to_reach_ref_loss.csv"
    if annualized_costs:
        fname = "annualized_" + fname
    else:
        fname = "total_" + fname

    cost_to_reach_ref_loss = pd.read_csv(save_path + fname, index_col=0)
    
    for co2l in plot_lvls:
        
        if annualized_costs:
            f_name = f"heuristic_costMin_loss_mitigation_annualized_Co2L{co2l}_n{n_nodes}.pkl"
        else:
            f_name = f"heuristic_costMin_loss_mitigation_Co2L{co2l}_n{n_nodes}.pkl"
        reinforced_lines, loss_with_mitigation, num_blackouts, cost = pickle.load(open(path_to_line_extension_mitigation_sclopf + f_name, "rb"))
        
        mpl.style.use('default')
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

        num_lines_to_reach_ref_loss = np.where(loss_with_mitigation < lost_load_reference_lvl)[0][0]
        num_lines_to_reach_double_ref_loss = np.where(loss_with_mitigation < 2*lost_load_reference_lvl)[0][0]

        # fig, (ax, ax_map) = plt.subplots(1,2, figsize=(12,5))
        fig = plt.figure(figsize=(11,4))
        
        # gs = GridSpec(1, 2, figure=f)
        if plot_cost:
            wspace=0.02
        else:
            wspace=-0.02
        gs_vertical = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.8,1.2], wspace=wspace)
        ax_loss_all = fig.add_subplot(gs_vertical[0])
        ax_loss_lvl = fig.add_subplot(gs_vertical[2])
        ax_map = fig.add_subplot(gs_vertical[1])
        
        ax_loss_lvl.plot(np.arange(len(loss_with_mitigation)),np.array(loss_with_mitigation)/lost_load_reference_lvl)

        right_xlim = np.where(loss_with_mitigation/lost_load_reference_lvl < 0.5)[0][0]
        
        if plot_cost:
            # add second y axis
            ax2 = ax_loss_lvl.twinx()
            ax2.plot(np.arange(len(cost)),np.cumsum(cost)/1e9, color='black', linestyle='dotted')
            # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if annualized_costs:
                y_lable_cost = 'Annualized construction cost [billion €]'
            else:
                y_lable_cost = 'Construction cost [billion €]'
            ax2.set_ylabel(y_lable_cost, fontsize=14)
            ax2.set_ylim((0, 1.1*sum(cost[:right_xlim])/1e9))
        
        ax_loss_lvl.plot([0, num_lines_to_reach_ref_loss], [1,1], '--', c='k')
        ax_loss_lvl.plot([num_lines_to_reach_ref_loss, num_lines_to_reach_ref_loss], [0,1], '--', c='k')
        ax_loss_lvl.text(num_lines_to_reach_ref_loss + right_xlim/200*5, 1, f"{num_lines_to_reach_ref_loss} lines", verticalalignment='bottom', horizontalalignment='left', zorder=np.inf, fontsize=14)
        ax_loss_lvl.plot([0, num_lines_to_reach_double_ref_loss], [2,2], '--', c='g')
        ax_loss_lvl.plot([num_lines_to_reach_double_ref_loss, num_lines_to_reach_double_ref_loss], [0,2], '--', c='g')
        ax_loss_lvl.text(num_lines_to_reach_double_ref_loss + right_xlim/200*5, 2, f"{num_lines_to_reach_double_ref_loss} lines", verticalalignment='bottom', horizontalalignment='left', zorder=np.inf, fontsize=14)
        ax_loss_lvl.set_xlim(0,right_xlim)
        ax_loss_lvl.set_ylim(bottom=00)
        # increase tick fontsize
        ax_loss_lvl.tick_params(axis='both', which='major', labelsize=14)
        
        ax_loss_lvl.set_xlabel('Number of reinforced lines')
        y_label = "$\\textrm{R}/\\textrm{R}_{\\textrm{co2_ref_str}}$"
        y_label = y_label.replace("co2_ref_str", co2_ref_str)
        if plot_cost:
            y_label += " (solid)"
        ax_loss_lvl.set_ylabel(y_label, fontsize=14)
        ax_loss_lvl.grid()
        # increase xlabel fontsize
        ax_loss_lvl.xaxis.label.set_size(14)

        fname = "costOpt_loss_reduction_blackouts_vs_reinforced_lines_costMin"
        if annualized_costs:
            fname += "_annualized"
        fname += f"_CO2{co2l}.pdf"
        
        reinforced_lines_selected = reinforced_lines[:num_lines_to_reach_ref_loss]
        lines_not_extended = np.setdiff1d(np.arange(n_lines), reinforced_lines_selected)
        reinforced_lines_full = reinforced_lines_selected + list(lines_not_extended)
        mitigated_loss = np.array([loss_with_mitigation[i] - loss_with_mitigation[i+1] for i in range(len(loss_with_mitigation)-1)])
        mitigated_loss = np.concatenate((mitigated_loss,-np.ones(n_lines - len(mitigated_loss))))
        mitigated_loss_sorted = mitigated_loss[np.argsort(reinforced_lines_full)]
        mitigated_loss_selected = -np.ones(n_lines)
        # Convert reinforced_lines_selected to integer indices
        reinforced_lines_selected_int = np.array(reinforced_lines_selected).astype(int)
        mitigated_loss_selected[reinforced_lines_selected_int] = mitigated_loss[:num_lines_to_reach_ref_loss]/lost_load_reference_lvl

        width= 2*(mitigated_loss_selected>0).astype(int) + 1
        # cmap = "cividis_r"
        # cmap = plt.get_cmap('cividis_r')
        cmap = copy.copy(mpl.cm.get_cmap("plasma_r"))
        cmap.set_under('gainsboro', 1.0)

        nodes = nx.draw_networkx_nodes(nx_graph,
                                    pos=pos,
                                    ax=ax_map,
                                    node_color='black',
                                    node_size=0)

        edges = nx.draw_networkx_edges(nx_graph,
                                    pos=pos,
                                    ax=ax_map,
                                    # edge_color=mitigated_loss_selected/mitigated_loss_selected.max(),
                                    edge_color=mitigated_loss_selected,
                                    width=width,
                                    edge_cmap=cmap,
                                    edge_vmin=0,
                                    # edge_vmax=np.log10(vmax)
                                    )
        # Add an axis for the legend at the bottom of ax_map
        bbox = ax_map.get_position()
        ax_map_legend = fig.add_axes([bbox.x0+bbox.width*0.1, bbox.y0 - 0.0, bbox.width*0.8, 0.03])
        y_label = "$\Delta \\textrm{R}_{{\\ell}}/\\textrm{R}_{\\textrm{co2refString}}$"
        y_label = y_label.replace("co2refString", str(int(100 * co2l_ref)) + "\%")
        cbar = plt.colorbar(edges, ax=ax_map, cax=ax_map_legend, label=y_label, shrink=0.5, orientation="horizontal")
        # increase cbar fontsize
        cbar.ax.tick_params(labelsize=14)
        # increase cbar label fontsize
        cbar.ax.set_xlabel(y_label, fontsize=14)
        
        ax_map.axis("off")

        ax_loss_all.text(0 - 0.2, 1+0.04,
                "a",
                fontsize=36,
                weight='bold',
                verticalalignment='center',
                transform=ax_loss_all.transAxes)
                    
        ax_loss_lvl.text(0 - 0.2, 1+0.04,
                "c",
                fontsize=36,
                weight='bold',
                verticalalignment='center',
                transform=ax_loss_lvl.transAxes)
        ax_loss_lvl.set_title(f"{int(round(get_actual_co2_level(co2l)*100))}\% CO$_2$ level \n Loss reduction")
        
        ax_map.text(0 + 0.15, 1+0.04,
                "b",
                fontsize=36,
                weight='bold',
                verticalalignment='center',
                transform=ax_map.transAxes)
        title = "{:.0f}\\% CO$_2$ level \n Grid extension to reach $R_{{co2_ref_str}}$".format(round(get_actual_co2_level(co2l)*100))
        title = title.replace("co2_ref_str", co2_ref_str)
        ax_map.set_title(title)


        ax_loss_all.plot(get_actual_co2_level(cost_to_reach_ref_loss.index), cost_to_reach_ref_loss.cost_ref_loss/10e9, label="$R_{{{:.0f}\\%}}$".format(actual_co2_ref))
        ax_loss_all.plot(get_actual_co2_level(cost_to_reach_ref_loss.index), cost_to_reach_ref_loss.cost_double_ref_loss/10e9, label="$2R_{{{:.0f}\\%}}$".format(actual_co2_ref))
        ax_loss_all.set_xlabel("CO2 level [\% of 1990]")
        if annualized_costs:
            ax_loss_all.set_ylabel("Anualized Cost [billion €]")
        else:
            ax_loss_all.set_ylabel("Cost [billion €]")
        ax_loss_all.legend()
        ax_loss_all.invert_xaxis()
        ax_loss_all.set_title("Grid extension \n to reach reference loss")

        plt.savefig(save_path+fname, bbox_inches='tight')


