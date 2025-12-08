import sys
import pickle
import copy
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.cluster import KMeans

import cartopy.crs as ccrs
import cartopy
import gzip


from tqdm.notebook import tqdm

import Johannes

from Johannes.utils import data_handling
from Johannes.check_n1_stab import check_n1_stab
import pandas as pd


import concurrent.futures # parallel execution of for loops


co2ls = ['0.3'] #['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']#np.arange(0.1,0.61,0.1).round(1)

n_nodes = 100
# %%
for level in co2ls:
    current_directory = os.getcwd()

    print("The current working directory is:", current_directory)
    path_to_pypsa_network = (f"../postnetworks/sclopf-elec_s_{n_nodes}_ec_lv1.0_Co2L{level}.nc") #sclopf network
#    Load PyPSA network


    assert (
os.path.isfile(path_to_pypsa_network) == True
    ), f'File "{path_to_pypsa_network}" does not exist'
    # network = pypsa.Network(path_to_pypsa_network)
    check_n1_stab(path_to_pypsa_network = path_to_pypsa_network, use_sclopf=True)
    