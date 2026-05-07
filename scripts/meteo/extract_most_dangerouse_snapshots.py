# %%
# #!/usr/bin/env python3
"""
Split Statistics Plot - Figure 5: split statistics
Creates histograms of power imbalance, rotational energy, and loss of load share distributions.
"""

import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


sys.path.append("./")
import pandas as pd
from utils.config import (
    path_to_vis_results_sclopf,
)

n_nodes = 600
split_props = pd.read_hdf(
    path_to_vis_results_sclopf + f"/split_properties_all_n{n_nodes}.h5",
    index_col=0,
)
# split_props.lost_load_share_blackout = split_props.lost_load_share_blackout.astype(
#     float)
most_dangerous_snapshots_by_co2l = {}
# %%
for co2l in split_props.co2l.unique():
    split_props_co2l = split_props[split_props.co2l == co2l]
    split_props_co2l = split_props_co2l[split_props_co2l.lost_load_share_blackout > 0.8]
    # split_props_co2l.groupby(split_props_co2l.index.get_level_values("time_stamp")).total_weighting.sum().sort_values(ascending=False).head(10)
    most_dangerous_snapshots = (
        split_props_co2l.groupby(split_props_co2l.index.get_level_values("time_stamp"))
        .total_weighting.sum()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    most_dangerous_snapshots_by_co2l[co2l] = most_dangerous_snapshots
# save most dangerous snapshots to csv
most_dangerous_snapshots_df = pd.DataFrame.from_dict(
    most_dangerous_snapshots_by_co2l, orient="index"
).transpose()
most_dangerous_snapshots_df.to_csv(
    path_to_vis_results_sclopf + f"/most_dangerous_snapshots_by_co2l_n{n_nodes}.csv",
    index=False,
)
