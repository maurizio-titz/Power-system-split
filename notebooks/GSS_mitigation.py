#%%
%load_ext autoreload
%autoreload 2
#%%
import gzip
import os
import pickle
import sys

import pandas as pd

from utils.mitigation_visualization import load_inertia_placement_results
root_path = '../' # This defaults to './', which should be the repository path
sys.path.append(root_path)
#%%
os.environ["POWER_SYSTEM_DATASET"] = "current"
from utils import config
from utils import data_handling
#%%
net = data_handling.load_pypsa_network(n_nodes=600, co2lvl=0.6, use_sclopf=True, lopt=False)
#%%


from utils import data_handling

import pypsa
#%%

n_nodes = 600
co2ls = data_handling.get_co2_levels(n_nodes)
split_properties = pd.read_hdf(
    config.path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
)
component_properties = pd.read_hdf(
    config.path_to_vis_results_sclopf + f"component_properties_all_n{n_nodes}.h5", index_col=0
)

#%%
co2_lvl = 0.0
blackoutthreshold=0.8
max_iter=10000
delta_Erot=5000
rocof_thres=1
l_share=0.0
resolve_strategy="random"
delta_Erot_saved, res_tuple = load_inertia_placement_results(
            n_nodes,
            max_iter,
            delta_Erot,
            rocof_thres,
            l_share,
            resolve_strategy,
            blackoutthreshold=blackoutthreshold,
            path_to_inertia_mitigation_results=config.path_to_inertia_mitigation_results_sclopf,
            co2_lvl=co2_lvl,
        )
#%%
component_properties["split_props_idx"] = component_properties[["co2l", "time_stamp", "split_number"]].apply(tuple, axis=1)
# %%

idx_tuple = component_properties.loc[0,"split_props_idx"]
split_properties.loc[idx_tuple]
#%%
co2_lvl = 0.6
split_properties_lvl = split_properties[
    split_properties.co2l == co2_lvl
]
component_properties_lvl = component_properties[
    component_properties.co2l == co2_lvl
]
# %%
co2_lvl = 0.6

f_names = os.listdir(config.path_to_inertia_mitigation_results_sclopf)
f_names_filtered = [f for f in f_names if "blackoutthres0.8" in f and f"Co2{co2_lvl:g}_" in f]
f_name = f_names_filtered[0]
with gzip.open(
    config.path_to_inertia_mitigation_results_sclopf + f_name, "rb"
) as f:
    inertia_placement_results = pickle.load(f)
#         modified_comp_index, modified_comp_arr,
        # inertia_placed_loss_mitigated_ls, componont_mitigated_step,
        # resolve_equality_counter, still_used_random_node_choice
#%%
componont_mitigated_step = inertia_placement_results[3]
inertia_placed_loss_mitigated = np.array(inertia_placement_results[2])

# %%
def calc_inertia_placement_ref_GSS(split_properties_lvl, component_properties_lvl, componont_mitigated_step, target_value:float, blackout_threshold=0.8, target:str="num_GSS"):
    component_properties_lvl = component_properties_lvl[component_properties_lvl.rocof.abs() > 1]
    if not "split_props_idx" in component_properties_lvl.columns:
        component_properties_lvl["split_props_idx"] = component_properties_lvl[["co2l", "time_stamp", "split_number"]].apply(tuple, axis=1)
    component_properties_lvl["mitigated_in_step"] = componont_mitigated_step

    n_steps = componont_mitigated_step.max()
    split_properties_steps = split_properties_lvl[["lost_load_share_blackout", "total_weighting"]].copy()

    # Prepare all columns at once
    step_columns = {}
    step_columns[f"lost_load_share_blackout_step-1"] = split_properties_steps["lost_load_share_blackout"]

    for step in range(0, n_steps):
        comp_idxs_in_step = component_properties_lvl[
        component_properties_lvl.mitigated_in_step == step
    ].index
        split_props_idxs_in_step = component_properties_lvl.loc[
        comp_idxs_in_step, "split_props_idx"
    ]
    
    # Create new column based on previous step
        prev_col = step_columns[f"lost_load_share_blackout_step{step-1}"]
        new_col = prev_col.copy()
        new_col.loc[split_props_idxs_in_step] -= (
        component_properties_lvl.loc[comp_idxs_in_step, "load_share"]
    ).values
        step_columns[f"lost_load_share_blackout_step{step}"] = new_col

    # Remove the step-1 column and concatenate all at once
    del step_columns[f"lost_load_share_blackout_step-1"]
    split_properties_steps = pd.concat([split_properties_steps, pd.DataFrame(step_columns)], axis=1)
    
    if target == "num_GSS":
        weighted_number_of_GSS = (split_properties_steps["total_weighting"].values @ (split_properties_steps[[col for col in split_properties_steps.columns if "step" in col]] > blackout_threshold).values)
        steps_needed = np.argmax(weighted_number_of_GSS <= target_value)
    elif target == "lost_load":
        weighted_loss = (split_properties_steps["total_weighting"].values @ split_properties_steps[[col for col in split_properties_steps.columns if "step" in col]].values)
        steps_needed = np.argmax(weighted_loss <= target_value)
    else:
        raise ValueError("Target must be 'num_GSS' or 'lost_load'")
    
    return steps_needed

steps_needed = calc_inertia_placement_ref_GSS(split_properties_lvl, component_properties_lvl, componont_mitigated_step, blackout_threshold=0.8, target="num_GSS", target_value=27945)
steps_needed
# %%
from matplotlib import pyplot as plt
plt.plot(range(n_steps), weighted_number_of_GSS / weighted_number_of_GSS[0], label="Number of GSS")
plt.plot(range(n_steps), weighted_loss / weighted_loss[0], label="Lost Load")
plt.xlabel("Inertia Placement Steps")
plt.ylabel("Relative to initial")
plt.legend()

#%%
import numpy as np
ref_num_GSS = weighted_number_of_GSS[0] * 0.5  # Example: 50% of initial number
ref_loss = weighted_loss[0] * 0.5  # Example: 50% of initial loss
steps_needed_num_GSS = np.argmax(weighted_number_of_GSS <= ref_num_GSS)
steps_needed_loss_GSS = np.argmax(weighted_loss <= ref_loss)
print(f"Steps needed for num GSS: {steps_needed_num_GSS}, Steps needed for loss: {steps_needed_loss_GSS}")
# %%
co2_lvl = 0.6
df_comp_props = pd.read_hdf(
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5".format(co2_lvl, n_nodes),
        key="df",
    )
#%%
df_comp_props.head()
#%%
import numpy as np
#%%
np.array(inertia_placement_results[2]).shape

#%%
from matplotlib import pyplot as plt
componont_mitigated_step = pd.Series(inertia_placement_results[3])
mitigated_in_step = pd.Series(0, index=range(len(inertia_placement_results[2])))
mitigated_in_step_missing_idxs = componont_mitigated_step.value_counts().drop(-1)
# valid_idxs = mitigated_in_step_missing_idxs.index[mitigated_in_step_missing_idxs.index.isin(mitigated_in_step.index)]
# mitigated_in_step.loc[valid_idxs] = mitigated_in_step_missing_idxs.loc[valid_idxs].values

#%%
set(mitigated_in_step_missing_idxs.index) - set(mitigated_in_step.index)
#%%
(componont_mitigated_step!=-1).sum(), (split_properties_lvl.lost_load_share_blackout>0.8).sum()

#%%
idx_step = inertia_placed_loss_mitigated[:,0]
idx_node = inertia_placed_loss_mitigated[:,1]
delta_rot_energy_factor = inertia_placed_loss_mitigated[:,2]
max_change = inertia_placed_loss_mitigated[:,3]
count_beyond_threshold = inertia_placed_loss_mitigated[:,4]
# idx_step,
# idx_node,
# delta_rot_energy_factor,
# max_change,
# count_beyond_threshold,
#%%
plt.hist(delta_rot_energy_factor)
#%%
plt.scatter(delta_rot_energy_factor, idx_step)
plt.loglog()    
plt.plot([0, max(delta_rot_energy_factor)], [0, max(delta_rot_energy_factor)], 'r--')
#%%
a = np.array(idx_step)
b = np.concatenate([[0], np.array(idx_step)[:-1]])
diff = a - b
plt.scatter(delta_rot_energy_factor, diff)
plt.loglog()
plt.plot([0, max(delta_rot_energy_factor)], [0, max(delta_rot_energy_factor)], 'r--')
# %%
plt.scatter(np.arange(max_change.shape[0]), max_change, c=delta_rot_energy_factor)
plt.yscale('log')
plt.colorbar(label='Delta Rotational Energy Factor')
#%%

plt.scatter(np.arange(delta_rot_energy_factor.shape[0]), delta_rot_energy_factor, c=max_change)
plt.yscale('log')
plt.colorbar(label='loss mitigated')
#%%
set(idx_step) - set(componont_mitigated_step.unique()), set(componont_mitigated_step.unique()) - set(idx_step)
#%%
set(idx_step) ==( set(componont_mitigated_step.unique()) - {1})
#%%
idx_step.shape,componont_mitigated_step.nunique()
#%%
componont_mitigated_step.unique()
#%%

# index differences between component props of different lvls?
co2_lvl = 0.6

fpath_component_in = (
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5"
    )
component_df = pd.read_hdf(fpath_component_in, key="df")
# %%
co2_lvl = 0.4

fpath_component_in = (
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5"
    )
component_df04 = pd.read_hdf(fpath_component_in, key="df")
# %%

component_properties[component_properties.co2l==0.3
                     ].head()
#%%

split_properties[split_properties.co2l==0.3
                     ].index.get_level_values("split_number_snapshot").max()

#%%
# load rocof vectors
path_in = config.path_to_evaluation_results_sclopf
co2_lvl = 0.0

## Load Data
# Load indicator vector rocof
fpath_rocof_in = path_in + f"/indicator_vector_rocof_Co2L{co2_lvl}_n{n_nodes}.pklz"
with gzip.open(fpath_rocof_in, "rb") as fh_rocof_in:
    indi_vec_rocof = pickle.load(fh_rocof_in)[-1]
# %%
indi_vec_rocof.shape

# %%
import numpy as np
component_properties_lvl = component_properties[component_properties.co2l==co2_lvl]
split_properties_lvl = split_properties[split_properties.co2l==co2_lvl]
#%%
idx = 0
np.unique(indi_vec_rocof[idx,:], return_counts=True), component_properties_lvl.rocof.values[0]
# %%
component_properties_lvl.isna().sum()

#%%
co2_lvl = 0.5
n_nodes = 600
fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5"
#%%
fpath = "/srv/data/jlange/power-system-split/no_extensions//results/sclopf/evaluation_results/to2013-01-03 00:00/" + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5"
comp_props_new = pd.read_hdf(fpath, key="df")
#%%
os.path.exists(fpath)
# %%
fpath_old = fpath.replace("/results/", "/results_no_load_inertia/")
comp_props_old = pd.read_hdf(fpath_old, key="df")
# %%
comp_props_new.equals(comp_props_old)
# %%
comp_props_new.shape, comp_props_old.shape

#%%
for col in comp_props_old.columns:
    if not comp_props_new[col].equals(comp_props_old[col]):
        print(f"Column {col} differs")
# %%
comp_props_new.split_number.astype(float).equals(comp_props_old.split_number.astype(float)[comp_props_new.index])
#%%
(comp_props_new.split_number[comp_props_new.time_stamp==comp_props_new.time_stamp.unique()[1]] == comp_props_old.split_number[comp_props_new.time_stamp==comp_props_old.time_stamp.unique()[1]]).all()
# %%

comp_props_new.split_number[comp_props_new.time_stamp==comp_props_new.time_stamp.unique()[0]], comp_props_old.split_number[comp_props_old.time_stamp==comp_props_old.time_stamp.unique()[0]]

#%%
fpath = "/srv/data/jlange/power-system-split/no_extensions//results/sclopf/evaluation_results/rocof_indicator_vectors_Co2L0.5_n600.pklz"
with gzip.open(fpath, "rb") as fh_rocof_in:
    indi_vec_rocof_new = pickle.load(fh_rocof_in)
# %%
fpath = "/srv/data/jlange/power-system-split/no_extensions//results_no_load_inertia/sclopf/evaluation_results/rocof_indicator_vectors_Co2L0.5_n600.pklz"
with gzip.open(fpath, "rb") as fh_rocof_in:
    indi_vec_rocof_old = pickle.load(fh_rocof_in)
# %%
indi_vec_rocof_new.shape, indi_vec_rocof_old.shape
# %%
indi_vec_rocof_new[:15,0], indi_vec_rocof_old[:15,0]

#%%
import numpy as np
(np.abs(indi_vec_rocof_old) - np.abs(indi_vec_rocof_new)).min()