# Power System Split
[![pipeline status](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/pipeline.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)
[![coverage report](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/coverage.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)

Code accompanying the mansucript "Blabla". Preprint: https://arxiv.org/abs/blabla

TODO: Insert title and arxiv link

## Installation

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```[python]
conda env create -f environment.yml 
conda activate system_split
```

## Content

The `scripts` folder contains scripts to reproduce the paper results. The `notebook` folder contains a notebook to produce the paper figures from the results and an overview notebook of the workflow. The `power_system_split` folder contains the relevant utilities for cascade simulation and evaluation, as well as functions for visualisation.

## Usage

The `scripts` contain our workflow with three stages:

- `1_run_cascade_code.py` : Running the cascade algorithm.
- `2_evaluate_cascade.py`: Evaluating the cascade results by calculating inertia and load imbalance for each split component and split. 
- `3_calc_visualization_data.py`: Prepare data from the results, that we use for visualisation, e.g., prototypical clusters of system splits. 

TODO: add script parameters and outputs,

The entire workflow is again summarised examplarily in a [jupyter notebook](notebooks/Example_of_whole_workflow.ipynb).

## Input data and results

All the input data to run the scripts is publicly available. We have uploaded the input data and the results of our workflow [on zenodo](https://zenodo.org/record/blabla).

TODO: Insert zenodo link and complete info on data below

- Input data: Put the solved pypsa networks in `data/European_networks/`. 
- Results:  

## Runtime

These runtimes are estimated on a machine with 32 CPUs of type "Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz" and 504 GB of memory.

- `1_run_cascade_code.py`: ~ 12 hour for one Co2 level
- `2_evaluate_cascade.py`: ~ 24 hours for one Co2 level 
- `3_cluster_split_components.py`: ~ 4h 

## Illustration

The following image displays nodes that remain connected after a large system split
with 99 % probability:
![alt text](img/Likely_splits_node_based_2013_threshold_0.99_edges.png "Nodes that remain connected likely after a large system split")
