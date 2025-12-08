
import pypsa
import pandas as pd
import numpy as np

NCLUSTERS = 100 #100#200
groupsize = 60
TEMP_RESOLUTION = 3 #H
load_shedding = True


n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))

for Co2l in ['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']:
    

    n = pypsa.Network(("../workflow/prenetworks/elec_s_" + str(NCLUSTERS) + f"_ec_lv1.0_Co2L{Co2l}_prepared.nc"))

    # n_subnetworks = int(2920 / snakemake.config["groupsize"])

    for i in range(0, n_subnetworks):

        fn_i =  f"../workflow/postnetworks/Co2L{Co2l}/sclopf-elec_s_" + str(NCLUSTERS) + "_ec_lv1.0-" + f'{i}.nc'

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

    n.export_to_netcdf("../workflow/postnetworks/sclopf-elec_s_" + str(NCLUSTERS) + f"_ec_lv1.0_Co2L{Co2l}.nc")
