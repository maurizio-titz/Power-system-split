
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network(snakemake.input.pre)

n_subnetworks = int(2920 / snakemake.config["groupsize"])

for i in range(0, n_subnetworks):

    fn_i = snakemake.input.sub.split('.nc')[0][:-2]+f'{i}.nc'

    sub_n = pypsa.Network(fn_i)

    snapshots = sub_n.snapshots
    
    for key in n.lines_t.keys():
        n.lines_t[key].loc[snapshots, sub_n.lines_t[key].columns] = (
            sub_n.lines_t[key].loc[snapshots, :]
        )

    for key in n.links_t.keys():
        n.links_t[key].loc[snapshots, :] = sub_n.links_t[key].loc[snapshots, :]

    for key in n.generators_t.keys():
        n.generators_t[key].loc[snapshots, :] = sub_n.generators_t[key].loc[snapshots, :]

    for key in n.storage_units_t.keys():
        n.storage_units_t[key].loc[snapshots, :] = sub_n.storage_units_t[key].loc[snapshots, :]

    for key in n.loads_t.keys():
        n.loads_t[key].loc[snapshots, :] = sub_n.loads_t[key].loc[snapshots, :]

n.export_to_netcdf(snakemake.output[0])
