#%%
%load_ext autoreload
%autoreload 2
#%%
import gzip
import os
import sys
root_path = '../' # This defaults to './', which should be the repository path

sys.path.append(root_path)
from utils import data_handling
from utils import config
import pypsa
import pandas as pd
# %%
n_nodes = 600
split_props = pd.read_hdf(
        config.path_to_vis_results_sclopf + f"split_properties_all_n{n_nodes}.h5", index_col=0
    )
# %%
component_props = pd.read_hdf(
        config.path_to_vis_results_sclopf + f"component_properties_all_n{n_nodes}.h5"
    )
component_props.reset_index(inplace=True, drop=False)
# %%

split_props.head()
# %%
component_props.head()
# %%
component_props.split_number.max() / component_props.shape[0]
# %% replace index of component_props with multi index like index of split_props
component_props = component_props.set_index(
    ['co2l', 'time_stamp', 'split_number']
)
#%%
split_props.columns
# %%
catastrophic_splits_idx = split_props[split_props.lost_load_share_blackout>0.99].index
# %%
components_filtered = component_props.loc[catastrophic_splits_idx]
split_props_filtered = split_props.loc[catastrophic_splits_idx]
# %%

co2_lvl = 0.6
fpath_component_in = (
        config.path_to_evaluation_results_sclopf
        + f"component_properties_Co2L{co2_lvl}_n{n_nodes}.h5"
)
component_props = pd.read_hdf(fpath_component_in, key="df")
split_props = data_handling.load_split_props(
    n_nodes=n_nodes, co2l=co2_lvl, use_sclopf=True
)
# %%
component_props.reset_index(inplace=True, drop=False)
component_props["co2l"] = co2_lvl
component_props = component_props.set_index(
    ['co2l', 'time_stamp', 'split_number']
)
#%%
blackout_threshold = 0.99
catastrophic_splits_idx = split_props[split_props.lost_load_share_blackout>blackout_threshold].index
# %%
components_filtered = component_props.loc[catastrophic_splits_idx]
split_props_filtered = split_props.loc[catastrophic_splits_idx]
# %%
f_path = config.path_to_inertia_mitigation_results_sclopf + os.listdir(config.path_to_inertia_mitigation_results_sclopf)[0]
with gzip.open(
    f_path, "rb"
) as f:
    mitigation_res = pd.read_pickle(f)
# %%
len(mitigation_res[0])
# %%
