#%%
%load_ext autoreload
%autoreload 2
#%%
import os
import sys
import gzip
import pickle
import shutil
import numpy as np
os.environ["POWER_SYSTEM_DATASET"] = "current"

import pandas as pd
root_path = '../' # This defaults to './', which should be the repository path

sys.path.append(root_path)
from utils import data_handling
from utils import config
import pypsa
#%%
co2_lvls = data_handling.get_co2_levels(600)
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
co2l = 0.6
n_nodes = 600
fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
comp_props_new = pd.read_hdf(fpath, key="df")
#%%
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
#%%
splits_per_snapshot = [comp_props_new[comp_props_new.time_stamp==ts].split_number.iloc[-1] for ts in comp_props_new.time_stamp.unique()]
#%%
split_num_offset = [comp_props_new[comp_props_new.time_stamp==ts].split_number.iloc[0] for ts in comp_props_new.time_stamp.unique()]

# %%
sum(splits_per_snapshot)
#%%
from matplotlib import pyplot as plt
plt.plot(splits_per_snapshot)
#%%
comp_props_new.split_number
#%%
plt.plot(split_num_offset)
#%%

idxs_first_in_snapshot = [comp_props_new.index[comp_props_new.time_stamp==ts][0] for ts in comp_props_new.time_stamp.unique()]
# %%
offset = comp_props_new.split_number[idxs_first_in_snapshot]
# %%
import numpy as np

comp_props_new["offset"] = np.NaN
comp_props_new.loc[idxs_first_in_snapshot, "offset"] = offset
comp_props_new["offset"] = comp_props_new["offset"].fillna(method="ffill")
# %%
comp_props_new["split_number_corrected"] = comp_props_new["split_number"] - comp_props_new["offset"] + 1
# %%
comp_props_new.split_number_corrected.value_counts().hist()
#%%
split_num_offset_corrected = [comp_props_new[comp_props_new.time_stamp==ts].split_number_corrected.iloc[0] for ts in comp_props_new.time_stamp.unique()]
plt.plot(split_num_offset_corrected)
#%%

(comp_props_new.split_number_corrected.shift(-1) - comp_props_new.split_number_corrected).dropna().value_counts().values[2:].sum()
# %%
total_splits = comp_props_new.split_number_corrected.shift(-1)[(comp_props_new.split_number_corrected.shift(-1).fillna(0) > comp_props_new.split_number_corrected)].sum()# + comp_props_new.split_number_corrected.iloc[-1]
total_splits
#%%
total_splits = comp_props_new.split_number_corrected.shift(1)[(comp_props_new.split_number_corrected.shift(1).fillna(0) > comp_props_new.split_number_corrected)].sum() + comp_props_new.split_number_corrected.iloc[-1]
total_splits
# %%







#%%
print(1+1)
import numpy as np
def correct_split_numbers(comp_props_new):
    comp_props_new = comp_props_new.copy()
    # get indices of first occurrence of each unique time_stamp
    idxs_first_in_snapshot = [comp_props_new.index[comp_props_new.time_stamp==ts][0] for ts in comp_props_new.time_stamp.unique()]
    assert np.equal(comp_props_new.index, np.arange(len(comp_props_new))).all(), "Index of comp_props_new is not a simple range index."
    idxs_last_in_snapshot = sorted(comp_props_new.index[(np.array(idxs_first_in_snapshot) -1)])
    # get corresponding split_number values, which should be 1, i.e. here we get the incorrect offsets
    offset = comp_props_new.split_number[idxs_first_in_snapshot]

    comp_props_new["offset"] = np.NaN
    comp_props_new.loc[idxs_first_in_snapshot, "offset"] = offset
    comp_props_new["offset"] = comp_props_new["offset"].fillna(method="ffill")

    comp_props_new["split_number_snapshot"] = comp_props_new["split_number"] - comp_props_new["offset"] + 1
    comp_props_new = comp_props_new.drop(columns=["offset"])
    
    splits_per_snapshot = comp_props_new.split_number[idxs_last_in_snapshot]
    assert len(splits_per_snapshot) == len(comp_props_new.time_stamp.unique()), "Number of detected snapshot splits does not match number of unique time_stamps."
    comp_props_new["total_splits_offset"] = np.NaN
    comp_props_new.loc[splits_per_snapshot.index, "total_splits_offset"] = splits_per_snapshot.cumsum()
    comp_props_new.total_splits_offset = comp_props_new.total_splits_offset.fillna(method="ffill").fillna(0).astype(int)
    comp_props_new["split_number"] = comp_props_new["split_number_snapshot"] + comp_props_new["total_splits_offset"]
    comp_props_new = comp_props_new.drop(columns=["total_splits_offset"])
    
    return comp_props_new

def calculate_total_splits(comp_props_new):
    total_splits = comp_props_new.split_number.shift(1)[(comp_props_new.split_number.shift(1).fillna(0) > comp_props_new.split_number)].sum() + comp_props_new.split_number.iloc[-1] # get the last split number of each snapshot and add the last of the dataframe manually
    return total_splits
# # %%
# comp_props_new = correct_split_numbers(comp_props_new)
# total_splits = calculate_total_splits(comp_props_new)
# print(f"Total splits after correction: {total_splits}")
# #%%


co2_lvls = data_handling.get_co2_levels(600)
for co2l in co2_lvls:
    print(60*"_")
    print(f"Validating results for CO2 level: {co2l}")
    n_nodes = 600
    fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
    # copy to a separate file for safekeeping
    backup_path = fpath.replace("/evaluation_results/", "/evaluation_results_backup/")
    if not os.path.exists(backup_path):
        shutil.move(fpath, backup_path)
    comp_props_new = pd.read_hdf(
                backup_path
    )
    comp_props_new = correct_split_numbers(comp_props_new)
    # total_splits = calculate_total_splits(comp_props_new)
    
    
    component_ind_vectors_fpath = (
        config.path_to_evaluation_results_sclopf
        + f"rocof_indicator_vectors_Co2L{co2l}_n{n_nodes}.pklz"
    )
    import gzip
    import pickle
    with gzip.open(component_ind_vectors_fpath, "rb") as fh_in_indi:
        indicator_vectors = pickle.load(fh_in_indi)
    num_splits_from_vecs = indicator_vectors.shape[0]
    
    if total_splits != num_splits_from_vecs:
        raise ValueError(f"After correction, total_splits {total_splits} does not match number of indicator vectors {num_splits_from_vecs} for Co2L {co2l}.")

    comp_props_new.to_hdf(fpath, key="df", mode="w")
    
#%%
# fname = "rocof_indicator_vectors_Co2L"
# indicator_vectors_file_path = (
#         config.path_to_evaluation_results_sclopf + f"{fname}{co2l}_n{n_nodes}.pklz"
#     )
# with gzip.open(indicator_vectors_file_path, "rb") as fh_in_indi:
#     indicator_vector_rocof = pickle.load(fh_in_indi)
# # %%
# indicator_vector_rocof.shape
# get indices of first occurrence of each unique time_stamp
#%%
co2l = 0.6
n_nodes = 600
fpath = config.path_to_evaluation_results_sclopf + f"component_properties_Co2L{co2l}_n{n_nodes}.h5"
comp_props_new = pd.read_hdf(
            fpath
)
#%%

fname = "rocof_indicator_vectors_Co2L"
indicator_vectors_file_path = (
        config.path_to_evaluation_results_sclopf + f"{fname}{co2l}_n{n_nodes}.pklz"
    )
with gzip.open(indicator_vectors_file_path, "rb") as fh_in_indi:
    indicator_vector_rocof = pickle.load(fh_in_indi)
#%%
idxs_first_in_snapshot = [comp_props_new.index[comp_props_new.time_stamp==ts][0] for ts in comp_props_new.time_stamp.unique()]
assert np.equal(comp_props_new.index, np.arange(len(comp_props_new))).all(), "Index of comp_props_new is not a simple range index."
#%%
idxs_last_in_snapshot = sorted(comp_props_new.index[(np.array(idxs_first_in_snapshot) -1)])
# get corresponding split_number values, which should be 1, i.e. here we get the incorrect offsets
offset = comp_props_new.split_number[idxs_first_in_snapshot]

comp_props_new["offset"] = np.NaN
comp_props_new.loc[idxs_first_in_snapshot, "offset"] = offset
comp_props_new["offset"] = comp_props_new["offset"].fillna(method="ffill")

comp_props_new["split_number_snapshot"] = comp_props_new["split_number"] - comp_props_new["offset"] + 1
comp_props_new = comp_props_new.drop(columns=["offset"])
assert comp_props_new.split_number_snapshot[idxs_first_in_snapshot].unique().astype(int)==1, "After correction, first split_number_snapshot in each snapshot is not 1."

splits_per_snapshot = comp_props_new.split_number_snapshot[idxs_last_in_snapshot]
assert len(splits_per_snapshot) == len(comp_props_new.time_stamp.unique()), "Number of detected snapshot splits does not match number of unique time_stamps."
comp_props_new["total_splits_offset"] = np.NaN
comp_props_new.loc[splits_per_snapshot.index, "total_splits_offset"] = splits_per_snapshot.cumsum()
comp_props_new.total_splits_offset = comp_props_new.total_splits_offset.fillna(method="ffill").fillna(0).astype(int)
#%%
comp_props_new["split_number"] = comp_props_new["split_number_snapshot"] + comp_props_new["total_splits_offset"]
comp_props_new = comp_props_new.drop(columns=["total_splits_offset"])


# %%
increments = (comp_props_new["split_number_snapshot"]- comp_props_new["split_number_snapshot"].shift(1).fillna(0))
#%%
increments[idxs_last_in_snapshot]=0
increments.value_counts()
# %%
increments = (comp_props_new["split_number"]- comp_props_new["split_number"].shift(1).fillna(0))
# increments[idxs_last_in_snapshot]=0
increments.value_counts()

#%%
comp_props_new.split_number[1562147-3:1562147+3]
#%%

comp_props_new.head()


#%%
np.equal(comp_props_new[["split_number"]][comp_props_new.component_number==0].values.squeeze(), np.arange((comp_props_new.component_number==0).sum())+1).all()
#%%
comp_props_new[["split_number_snapshot"]][comp_props_new.component_number==0].values == np.arange(len(comp_props_new))[comp_props_new.component_number==0][:, None]
#%%
comp_props_new[["split_number"]][comp_props_new.component_number==0].values.squeeze()


#%%
n_jobs = 30
timestamp_list = comp_props_new.time_stamp.unique().tolist()
chunk_size = max(1, 2920 // n_jobs)
timestamp_chunks = [
    timestamp_list[i : i + chunk_size]
    for i in range(0, 2920, chunk_size)
]
len(timestamp_chunks)
# %%
timestamp_list = np.arange(2920).tolist()
timestamp_chunks = np.array_split(timestamp_list, n_jobs)
# Filter out empty chunks (can happen if n_jobs > len(timestamp_list))
timestamp_chunks = [chunk for chunk in timestamp_chunks if len(chunk) > 0]
n_chunks = len(timestamp_chunks)
n_chunks
# %%

rocof_vec = pd.DataFrame(indicator_vector_rocof)
# %%
rocof_vec.unique()
#%%
comp_props_new.rocof

#%%
test_inds = 1000
rocofs_vecs = [np.unique(indicator_vector_rocof[i,:].round(5)) for i in range(test_inds)]
rocofs_props = [comp_props_new.rocof[comp_props_new.split_number==i].unique() for i in range(test_inds)]
# %%
rocofs_vecs[:10], rocofs_props[:10]