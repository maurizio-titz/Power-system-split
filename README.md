# Power System Split
Code accompanying the manuscript "Cascading Failures and Critical Infrastructures in Future Renewable European Power Systems" on [ArXiv](https://arxiv.org/abs/2603.24529) and the accompanying publication. 

This project evaluates cascading failures of transmission lines in the Continental European power grid. The project uses PyPSA to simulate future renewable power grids. 

## Installation

The code is written in Python (tested with python 3.11.8). To clone the repository and install the required dependencies, e.g., by using a conda environment, execute the following commands:

```[shell]
user:dir$ git clone --recurse-submodules https://link/to/repository
user:dir$ conda create system_split_pyenv python=3.11.8 
user:dir$ conda activate system_split_env
(system_split_env)user:dir$ pip install -e .
```

This installs the package in editable mode using the dependencies defined in [pyproject.toml](./pyproject.toml). 

### Config
The config at [utils/config.py](./utils/config.py) contains all paths that generally do not need to be changed but you might want to change the root path, if you want to store both the scenario data and the results in a different location.
To post message to mattermost for the calculation of the cascades and the evaluation of the results, please set the  'mattermost_url' to the one you configured in your mattermost instance in [utils/config.py](utils/config.py). 

You can setup the basic folder structure by running `python utils/config.py`. Note, this folder structure by default will be identical to the one used in the prepared data set that is mentioned below.

## Content

The `scripts` folder contains scripts to reproduce the paper results. The `scripts/plots` folder contains a scripts to produce the paper figures from these results. The `utils` folder contains the functions for cascade simulation and evaluation, as well as functions for visualization and data handling. 

## Usage
Start by creating the scenarios by following the workflow outlined in the folder `scenario_generation/README.md` or downloading the already generated scenarios from the zenodo link you see below.

Note that this code was run using a workstation Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz, with 32 logical cores and around 500GB of memory.

Before running the scripts, be sure to have setup the folder structure following the description in the `Config` section.
The `scripts` contain our workflow and needs to be executed in roughly this order:
- [`extract_grid_data.py`](./scripts/extract_grid_data.py): Saves grid properties to disk. Run before running `run_cascade_code.py`.
- [`calc_pre_outage_data.py`](./scripts/calc_pre_outage_data.py): Calculate inertia and other properties from the (solved) PyPSA networks.
- [`run_cascade_code.py`](./scripts/run_cascade_code.py): Run the cascade algorithm on the PyPSA networks. This can take a few weeks depending on compute power.
- [`evaluate_cascade.py`](./scripts/evaluate_cascade.py): Evaluating the cascade results, e.g., inertia and load imbalance for each split, indicator vectors. Run only after `run_cascade_code.py` has finished.
- [`prepare_split_visualization.py`](./scripts/prepare_split_visualization.py): Prepare data from the results, that we use for visualisation, e.g., prototypical clusters of system splits. Run only after `evaluate_cascade.py` has finished.
- [`calc_inertia_placement.py`](./scripts/calc_inertia_placement.py) : Determine optimal inertia placement to mitigate the impact of system splits. Run only after `prepare_split_visualization.py` has finished.
- [`cluster_blackouts.py`](./scripts/cluster_blackouts.py): Performs the clustering of splits using a composite metric taking both nodes and edges features into account as described in the main manuscript.
- Create plots by running the scripts in `scripts/plotting` individually or run the bash script `generate_all_plots.sh` to generate all plots in one go.
- [`plot_all_manuscript_figures.py`](./scripts/plot_all_manuscript_figures.py): Can be called to create only the figures used in the main manuscript and the supplementary material accompanying the manuscript. The memory usage of some functions exceed 32GB. If your machine can not deal with this, please call the script as `python scripts/plot_all_manuscript_figures.py --low_memory` or with the argument `low_memory=True` if you use the function via a python interpreter. This way these memory hungry figures will be excluded.

All code assume that your PYTHONPATH contains the repository directory and the code is executed in there, too. 

The scenario data is created in `scenario_generation`. A separate conda enviroment is used there, more details under `Input data and results` and the respective README.

## Input data and results

All the input data to run the scripts is publicly available and can be generated using the repository. 

We have uploaded the version of this repo used in the publication to zenodo with the doi []() 

Additionally, you can find a version of the data on zenodo that includes the CO2 scenarios generated using PyPSA and the results generated using the code in this repository. Unpacking this data set into the main folder of the repo on you local machine, you can plot the figures that are presented in the manuscript.

If you want to place the files in another location, please modify the path in [utils/config.py](./utils/config.py) accordingly.

The scenario data can also be generated by using the [`scenario_genertaion`](./scenario_generation/) subfolder. Installation and usage are explained in more detail in the README there.

## Contributors
- Maurizio Titz [Orcid](https://orcid.org/0000-0002-0249-6244)
- Franz Kaiser [Orcid](https://orcid.org/0000-0002-7089-2249)
- Johannes Kruse [Orcid](https://orcid.org/0000-0002-3478-3379)
- Philipp C. Böttcher [Orcid](https://orcid.org/0000-0002-3240-0442)
- Jan Lange [Orcid](https://orcid.org/0009-0009-5985-4848)
- Martha Frysztacki [Orcid](https://orcid.org/0000-0002-0788-1328)