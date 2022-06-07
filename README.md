# Power System Split
Code accompanying the mansucript "Blabla". Preprint: https://arxiv.org/abs/blabla .

This project evaluates cascading failures of transmission lines in the European power grid. The project uses PyPSA  
to simulate future renewable power grids. 

TODO: Insert title and arxiv link. 

## Installation

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```[python]
conda env create -f environment.yml 
conda activate system_split
```

## Content

The `scripts` folder contains scripts to reproduce the paper results. The `notebook` folder contains a notebook to produce the paper figures from the results. The `power_system_split` folder contains the relevant utilities for cascade simulation and evaluation, as well as functions for visualisation.

## Usage

The `scripts` contain our workflow with five stages:

- `1_calc_pre_outage_data.py`: Calculate inertia and other properties from the (solved) PyPSA networks.
- `2_run_cascade_code.py` : Run the cascade algorithm on the PyPSA networks.
- `3_evaluate_cascade.py`: Evaluating the cascade results by calculating inertia and load imbalance for each split.
- `4_prepare_split_visualization.py`: Prepare data from the results, that we use for visualisation, e.g., prototypical clusters of system splits.
- `5_calc_inertia_placement.py` : Determine optimal inertia placement to mitigate the impact of system splits. 

All code assume that your PYTHONPATH contains the repository directory and the code is executed in there, too. 

TODO: add script parameters and outputs,


## Input data and results

All the input data to run the scripts is publicly available. We have uploaded the input data and the results of our workflow [on zenodo](https://zenodo.org/record/blabla).

TODO: Insert zenodo link and complete info on data below

- Input data: Put the solved pypsa networks in `data/European_networks/`. 
- Results:  

## Runtime

These runtimes are estimated on a machine with 32 CPUs of type "Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz" and 504 GB of memory.

- `1_calc_pre_outage_data.py`: ~ 30 min 
- `2_run_cascade_code.py`: ~ 12 hour for one Co2 level
- `3_evaluate_cascade.py`: ~ 24 hours for one Co2 level 
- `4_prepare_split_visualization.py`: ~ 4h 
- `5_calc_inertia_placement.py` : 

#TODO: check runtime for first and last script

