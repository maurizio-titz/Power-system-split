#%%
%load_ext autoreload
%autoreload 2
#%%
import os
import sys
import shutil
os.environ["POWER_SYSTEM_DATASET"] = "current"

import pandas as pd
root_path = '../' # This defaults to './', which should be the repository path

sys.path.append(root_path)
from utils import data_handling
from utils import config
import pypsa
#%%
co2_lvls = data_handling.get_co2_levels(600)
co2_lvls = [0.0]
for co2l in co2_lvls:
    print(60*"_")
    print(f"Validating results for CO2 level: {co2l}")
    n_nodes = 600
    fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
    # copy to a separate file for safekeeping
    backup_path = fpath.replace(".h5", "_broken_noLoadInertiaRoCoF.h5")
    shutil.move(fpath, backup_path)
    comp_props_new = pd.read_hdf(
                backup_path
            )
    
    fpath_old = fpath.replace("/results/", "/results_no_load_inertia/")

    # calculation of rocof without load inertia had a bug in the new evaluation script, so we recalculate it here for comparison
    rocof_ = 50 * comp_props_new.power_imbalance / ((comp_props_new.rot_energy_gen + 1e-8) * 2)
    comp_props_new["rocof_noLoadInertia"] = rocof_
    comp_props_new.to_hdf(fpath, key="df", mode="w")
    
#    #%%
    comp_props_old = pd.read_hdf(fpath_old, key="df")
#    #%%
    comp_props_new[comp_props_old.columns].equals(comp_props_old)
#    #%%
    # comp_props_new.time_stamp.head(), comp_props_old.time_stamp.head()
#    #%%
    if not comp_props_old.time_stamp.equals(comp_props_old.time_stamp.sort_values().reset_index(drop=True)):
        print("Old time_stamp is not sorted, sorting now.")
        comp_props_old = comp_props_old.sort_values(["time_stamp", "init_failure_0", "init_failure_1"]).reset_index(drop=True)
    # if not comp_props_new.time_stamp.equals(comp_props_new.time_stamp.sort_values().reset_index(drop=True)):
        # print("Old time_stamp is not sorted, sorting now.")
    if co2l==0.0:
        comp_props_new = comp_props_new.sort_values(["time_stamp", "init_failure_0", "init_failure_1"]).reset_index(drop=True)
#    #%%
    comp_props_new.time_stamp.equals(comp_props_new.time_stamp.sort_values().reset_index(drop=True))
#    #%% check data type differences
    import numpy as np
    dtype_diffs = comp_props_new[comp_props_old.columns].dtypes.compare(comp_props_old.dtypes)
    for col in dtype_diffs.index:
        print(f"Column {col} has different dtypes: new={comp_props_new[col].dtype}, old={comp_props_old[col].dtype}")
        # set to the more general dtype
        if set(dtype_diffs.loc[col].values)=={np.dtype('float64'), np.dtype('int64')}:
            comp_props_new[col] = comp_props_new[col].astype(float)
            comp_props_old[col] = comp_props_old[col].astype(float)
#    #%% collect mismatched columns
    mismatch_cols = set()
    for col in comp_props_old.columns:
        if not comp_props_new[col].equals(comp_props_old[col]):
            # print(f"Difference in column {col}")
            mismatch_cols.add(col)
#    #%% some mismatches are expected due to the rocof calculation change
    unexpected_mismatches = mismatch_cols - {'blackout_load_loss_share', 'rocof', 'rot_energy', 'shedding_load_loss_share', 'split_number', 'total_load_loss_share'}
    if unexpected_mismatches != set():
        print("Unexpected mismatches found: ", unexpected_mismatches)
    else:
        print("All mismatches are expected due to known changes.")
#    #%%
    # if not comp_props_new.rocof_noLoadInertia.equals(comp_props_old.rocof):
    #     raise ValueError("rocof_no_load_inertia does not match rocof from old results.")
    rocof_sig_mismatch_count = ((comp_props_new.rocof_noLoadInertia - comp_props_old.rocof).abs()>1e-6).sum()
    rel_num_mismatches = rocof_sig_mismatch_count / len(comp_props_new)
    print(f"Relative Number of significant rocof mismatches: {rel_num_mismatches}")
    # #%%

#%%
co2l = 0.5
n_nodes = 600
fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
comp_props_new = pd.read_hdf(fpath, key="df")
comp_props_old = pd.read_hdf(fpath.replace("/results/", "/results_no_load_inertia/"), key="df")
#%%
import gzip
import pickle
fname = "rocof_indicator_vectors_Co2L"
indicator_vectors_file_path = (
    config.path_to_evaluation_results_sclopf + f"{fname}{co2l}_n{n_nodes}.pklz"
)
with gzip.open(indicator_vectors_file_path, "rb") as fh_in_indi:
    indicator_vector_rocof = pickle.load(fh_in_indi)
# %%
import numpy as np
idx = np.arange(comp_props_new.shape[0])[comp_props_new.rocof==0]
vecs = indicator_vector_rocof[idx]
comps = comp_props_new[comp_props_new.rocof==0]
# %%
np.unique(vecs, axis=1, return_counts=True)
#%%
comp_props_new.rocof[:6], np.unique(indicator_vector_rocof[:3,:].round(5), axis=1, return_counts=True)
# %%
# comp_props_old[comp_props_old.power_imbalance==0]
#%%
(comp_props_new.power_imbalance==0).sum(), (comp_props_old.power_imbalance==0).sum()
#%%
(comp_props_new.rocof==0).sum(), (comp_props_old.rocof==0).sum()
#%%
rocof_ = 50 * comp_props_new.power_imbalance / ((comp_props_new.rot_energy + 1e-8) * 2)
#%%
(rocof_==0).sum()
# %%
comp_props_new[comp_props_new.rocof==0].head()

# %%
from utils.alternative_split_indicator_vectors import validate_rocofVec_splitProps_match
validate_rocofVec_splitProps_match(n_nodes, co2l, test_mode=True)