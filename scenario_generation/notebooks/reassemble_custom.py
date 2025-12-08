
import pypsa
import pandas as pd
import numpy as np

from Jan.utils import reassemble

NCLUSTERS = 400
groupsize = 60
TEMP_RESOLUTION = 3 #H
load_shedding = True


n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))




Co2levels = ['0.05', '0.1','0.2', '0.3', '0.4', '0.5', '0.6'] #, '0.2'

# added by Jan
path_syntax = 'new'
# 
# added by Jan
s_max_pu = 0.7#config["lines"]["s_max_pu"]

co2_relaxation = 1.0 #config['co2_relaxation']

n_subnetworks = int(np.ceil(8760 / TEMP_RESOLUTION / groupsize))

# path = "/2.0"

if path_syntax == 'new':
    # added by Jan
    s_max_pu_path = f"s_max_pu_{s_max_pu}/" if s_max_pu != "" else "" 


    co2_relaxation_path = f"co2_rel_{co2_relaxation}/" if co2_relaxation else ""
    path =  '/'+co2_relaxation_path + s_max_pu_path
    
elif path_syntax == 'old':
    s_max_pu_path = ''
    path = f"{co2_relaxation}"
    
    
for Co2l in Co2levels:
    reassemble.reassemble(path=path,NCLUSTERS=NCLUSTERS,Co2l=Co2l,n_subnetworks=n_subnetworks)