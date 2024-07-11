# Power System Split
Code accompanying the mansucript "XXX". Preprint: https://arxiv.org/abs/XXXX .

This project evaluates cascading failures of transmission lines in the Continental European power grid. The project uses PyPSA to simulate future renewable power grids. 

## Installation

The code is written in Python (tested with python 3.8). To install the required dependencies execute the following commands:

```[python]
conda env create -f environment.yml 
conda activate system_split
```

### Config
To post message to mattermost, please set the incoming 'mattermost_url' in [utils/config.py](utils/config.py).


## Content

The `scripts` folder contains scripts to reproduce the paper results. The `notebook` folder contains a notebook to produce the paper figures from the results. The `utils` folder contains the relevant utilities for cascade simulation and evaluation, as well as functions for visualisation and data handling. 

## Usage

The `scripts` contain our workflow with five stages:

- `1_calc_pre_outage_data.py`: Calculate inertia and other properties from the (solved) PyPSA networks.
- `run_cascade_code.py` : Run the cascade algorithm on the PyPSA networks.
- `evaluate_cascade.py`: Evaluating the cascade results, e.g., inertia and load imbalance for each split, indicator vectors.
- `4_prepare_split_visualization.py`: Prepare data from the results, that we use for visualisation, e.g., prototypical clusters of system splits.
- `5_calc_inertia_placement.py` : Determine optimal inertia placement to mitigate the impact of system splits. 

All code assume that your PYTHONPATH contains the repository directory and the code is executed in there, too. 

## Input data and results

All the input data to run the scripts is publicly available. We have uploaded the input data and the results of our workflow [on zenodo](https://zenodo.org/record/XXXX).

TODO: Insert zenodo link and complete info on data below

- Input data: Put the solved pypsa networks in `data/European_networks/`. 
- Results:  


