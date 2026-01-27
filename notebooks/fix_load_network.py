#%%
%load_ext autoreload
%autoreload 2
#%%
import os
import sys
root_path = '../' # This defaults to './', which should be the repository path

sys.path.append(root_path)
from utils import data_handling
from utils import config
import pypsa
#%%
f_names = os.listdir(config.path_to_pypsa_network_sclopf)
f_name_0 = f_names[0] # 0%
f_name_6 = f_names[-1] # 60%
#%%
net6 = pypsa.Network(config.path_to_pypsa_network_sclopf + f_name_6)
net0 = pypsa.Network(config.path_to_pypsa_network_sclopf + f_name_0)
#%%
def drop_dummy_lines(net):
    dummy_lines = net.lines[net.lines.s_nom == 0]
    net.mremove("Line", dummy_lines.index)

    return net
net6 = drop_dummy_lines(net6)
net0 = drop_dummy_lines(net0)
# %%
lines6 = net6.lines
lines0 = net0.lines
cols = ["s_nom", "s_nom_opt", "x", "num_parallel", "partition", "orig_partition"]
# %% unfortnunately, there are some differences in the columns:
set(lines0.columns).difference(set(lines6.columns)), set(lines6.columns).difference(set(lines0.columns))

#%% let's check partitioning
line_number_groups = lines6.groupby(lines6.index.str.split("_").str[0])
# %%
import numpy as np
sorted_ind = np.argsort(list(lines6.index.str.split("_").str[0]))
lines6 = lines6.iloc[sorted_ind]
#%% and here we can see that s_nom is not partitioned properly:
lines6[cols].head()
# line 1 has the same s_nom for both lines.

#%% lets check the same thing for lines0
sorted_ind0 = np.argsort(list(lines0.index.str.split("_").str[0]))
lines0 = lines0.iloc[sorted_ind0]
cols = ["s_nom", "s_nom_opt", "x", "num_parallel", "partition"]
lines0[cols].head()
# same thing here. S_nom is the same for all parallel lines.

#%% what about x?
(lines6.x/lines6.length*lines6.num_parallel).round(3).nunique(), (lines0.x/lines0.length*lines0.num_parallel).round(3).nunique()

# x seems to be correctly partitioned in both cases. 