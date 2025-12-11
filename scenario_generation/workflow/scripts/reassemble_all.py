
import pypsa
import pandas as pd
import numpy as np

NCLUSTERS = 600
groupsize = 60
TEMP_RESOLUTION = 3 #H

from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.config import path_to_sclopf_data

Co2levels = ['0.0','0.05', '0.1','0.2', '0.3', '0.4', '0.5', '0.6']

# added by Jan
opts = 'copt'


n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))

    

def reassemble(path,NCLUSTERS,Co2l,n_subnetworks,opts):
    lopf_network = pypsa.Network(f"{path_to_sclopf_data}prenetworks/elec_s_" + str(NCLUSTERS) + f"_ec_l{opts}_Co2L{Co2l}_prepared.nc")
    
    # Initialize a new combined network and set up snapshots
    output_network = lopf_network # lopf_network
    

    for i in range(n_subnetworks):

        # solved sclopf network for one timewindow
        sub_n = pypsa.Network(f"{path_to_sclopf_data}postnetworks/Co2L{Co2l}/sclopf-elec_s_" + str(NCLUSTERS) + f"_ec_l{opts}-" + f'{i}' + ".nc")

        # snapshots in timewindow
        snapshots = sub_n.snapshots
        
        # add data for specific timewindow 
        for key in sub_n.lines_t.keys():
            output_network.lines_t[key].loc[snapshots, sub_n.lines_t[key].columns] = (
                sub_n.lines_t[key].loc[snapshots, :]
            )

        for key in sub_n.links_t.keys():
            output_network.links_t[key].loc[snapshots, :] = sub_n.links_t[key].loc[snapshots, :]

        for key in sub_n.generators_t.keys():
            output_network.generators_t[key].loc[snapshots, :] = sub_n.generators_t[key].loc[snapshots, :]

        for key in sub_n.storage_units_t.keys():
            output_network.storage_units_t[key].loc[snapshots, :] = sub_n.storage_units_t[key].loc[snapshots, :]

        for key in sub_n.loads_t.keys():
            output_network.loads_t[key].loc[snapshots, :] = sub_n.loads_t[key].loc[snapshots, :]

    # export network data
    output_network.export_to_netcdf(f"{path_to_sclopf_data}/postnetworks/sclopf-elec_s_{NCLUSTERS}_ec_l{opts}_Co2L{Co2l}.nc")



for Co2l in Co2levels:
    reassemble(NCLUSTERS=NCLUSTERS, Co2l=Co2l, n_subnetworks=n_subnetworks, opts=opts)
    print(f"Finished reassemble for Co2 level {Co2l}")




