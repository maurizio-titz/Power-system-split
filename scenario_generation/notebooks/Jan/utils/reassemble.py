
import pypsa
import pandas as pd
import numpy as np

def reassemble(path,NCLUSTERS,Co2l,n_subnetworks):
    lopf_network = pypsa.Network(f"../workflow{path}/prenetworks/elec_s_" + str(NCLUSTERS) + f"_ec_lv1.0_Co2L{Co2l}_prepared.nc")
    # lopf_network = pypsa.Network(f"../workflow/prenetworks/elec_s_{NCLUSTERS}_ec_lv1.0_Co2L{Co2l}_prepared.nc")
    first_sclopf = pypsa.Network(f"../workflow{path}/postnetworks/Co2L{Co2l}/sclopf-elec_s_" + str(NCLUSTERS) + "_ec_lv1.0-0.nc")
    
    # Initialize a new combined network and set up snapshots
    output_network = lopf_network # lopf_network
    
    # if load_shedding:
            # output_network.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
            # buses_i = output_network.buses.index
            # output_network.madd(
            #     "Generator",
            #     buses_i,
            #     " load",
            #     bus=buses_i,
            #     carrier="load",
            #     sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            #     marginal_cost=1e2,  # Eur/kWh
            #     p_nom=1e9,  # kW
            # )


    for i in range(n_subnetworks):

        # fn_i = snakemake.input.sub.split('.nc')[0][:-2]+f'{i}.nc'
        # solved sclopf network for one timewindow
        sub_n = pypsa.Network(f"../workflow{path}/postnetworks/Co2L{Co2l}/sclopf-elec_s_" + str(NCLUSTERS) + "_ec_lv1.0-" + f'{i}' + ".nc")

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
    output_network.export_to_netcdf(f"../workflow{path}/postnetworks/sclopf-elec_s_{NCLUSTERS}_ec_lv1.0_Co2L{Co2l}.nc")
