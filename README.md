# Power System Split
Code accompanying the article: Titz et al., Cascading failures and system splits in decarbonized European power systems, Joule (2026), DOI: [10.1016/j.joule.2026.102742](https://doi.org/10.1016/j.joule.2026.102742). Preprint: [arXiv:2603.24529](https://arxiv.org/abs/2603.24529).

This project evaluates cascading failures of transmission lines in the Continental European power grid. It uses PyPSA-Eur to simulate future renewable power system scenarios, which are subsequently used to perform cascade experiments.

## Installation

The code is written in Python (tested with Python 3.11.8). To clone the repository and install the required dependencies, e.g., in a conda environment, execute the following commands:

```shell
user:dir$ git clone --recurse-submodules https://github.com/maurizio-titz/Power-system-split
user:dir$ cd Power-system-split
user:Power-system-split$ conda create -n system_split_env python=3.11.8
user:Power-system-split$ conda activate system_split_env
(system_split_env)user:Power-system-split$ pip install -e .
```

Alternatively, create the environment with a prefix, which stores it in a local folder (here `./pyenv` inside the repository) instead of conda's default environment directory:

```shell
user:Power-system-split$ conda create -p ./pyenv python=3.11.8
user:Power-system-split$ conda activate ./pyenv
(pyenv)user:Power-system-split$ pip install -e .
```

This installs the Python package in editable mode together with the dependencies defined in [pyproject.toml](./pyproject.toml).

### Config
The config at [utils/config.py](./utils/config.py) contains all paths, which generally do not need to be changed. You might want to change the root path if you want to store the scenario data and the results in a different location than the repository.
To post messages to Mattermost about the progress of the cascade calculation and the evaluation of the results, set `mattermost_url` in [utils/config.py](./utils/config.py) to the webhook URL you configured in your Mattermost instance.

You can set up the basic folder structure by running `python utils/config.py`. This default folder structure is identical to the one used in the prepared data set mentioned below.

## Content

The `scripts` folder contains the scripts to reproduce the paper results. The `scripts/plots` folder contains the scripts to produce the paper figures from these results. The `utils` folder contains the functions for the cascade simulation and evaluation, as well as functions for visualization and data handling.

## Usage
All scripts assume that your `PYTHONPATH` contains the repository directory and that they are run from there. Before running any script, set up the folder structure as described in the `Config` section.

### A. Reproduce the figures with the Zenodo data

With the data from Zenodo, no simulation is needed:

1. Download both data archives from Zenodo (see `Input data and results`) and unpack them into the main folder of the repository, e.g., `tar -xzf system_split_results.tar.gz -C path/to/Power-system-split`. This creates the folders `data` and `results` next to `scripts`. 
Unpacking needs about 30 GB of free disk space.
2. Run `python scripts/plot_all_manuscript_figures.py`. It creates all figures of the main manuscript and the Supplemental Information in `results/figures`. Some figures need more than 32 GB of memory. On smaller machines, add `--low_memory` to skip them, or pass `low_memory=True` if you call the functions from a Python interpreter.

The scripts in part B are not needed for this.

### B. Full workflow without the Zenodo data

Only needed if you want to recompute the scenarios and results yourself. Note that this code was run on a workstation with an Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz, 32 logical cores and around 500 GB of memory.

Run the steps in this order, each only after the previous one has finished:

0. Create the scenarios with [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) (included as a submodule), following `scenario_generation/README.md` (separate conda environment).
1. [`extract_grid_data.py`](./scripts/extract_grid_data.py): Saves grid properties to disk.
2. [`calc_pre_outage_data.py`](./scripts/calc_pre_outage_data.py): Calculates inertia and other properties from the (solved) PyPSA networks.
3. [`run_cascade_code.py`](./scripts/run_cascade_code.py): Runs the cascade algorithm on the PyPSA networks. This can take a few weeks, depending on the available compute power.
4. [`evaluate_cascade.py`](./scripts/evaluate_cascade.py): Evaluates the cascade results, e.g., inertia and load imbalance for each split, and indicator vectors.
5. [`prepare_split_visualization.py`](./scripts/prepare_split_visualization.py): Prepares the data from the results that we use for visualization, e.g., prototypical clusters of system splits.
6. [`calc_inertia_placement.py`](./scripts/calc_inertia_placement.py): Determines the optimal inertia placement to mitigate the impact of system splits.
7. [`cluster_blackouts.py`](./scripts/cluster_blackouts.py): Clusters the splits using a composite metric that takes both node and edge features into account, as described in the main manuscript.
8. [`compare_ac_with_dc_snapshots.py`](./scripts/compare_ac_with_dc_snapshots.py): Compares DC with AC power flows for representative snapshots of all CO2 levels. Needs only the PyPSA networks, so it can run at any time after step 0. Use `--n_processes` to run in parallel and `--resume` to continue an interrupted run.

Afterwards, create the figures of the manuscript as in part A, step 2, with [`plot_all_manuscript_figures.py`](./scripts/plot_all_manuscript_figures.py). Further plots can be created by running the scripts in `scripts/plots`, but only if the intermediate data was generated with the steps in this section.

## Input data and results

All input data needed to run the scripts is publicly available or can be generated with this repository.

The version of this repository used in the publication is archived on Zenodo: [10.5281/zenodo.24158671](https://doi.org/10.5281/zenodo.24158671). 

The data set used in the publication is also available on Zenodo: [10.5281/zenodo.24157827](https://doi.org/10.5281/zenodo.24157827). It includes the CO2 scenarios generated with PyPSA-Eur and the results generated with the code in this repository. After unpacking it into the main folder of the repository on your local machine, you can plot the figures presented in the manuscript as described in part A of the `Usage` section.

If you want to place the files in another location, modify the paths in [utils/config.py](./utils/config.py) accordingly.

## Use of AI tools
Claude Code (Anthropic) was used to assist with code development. The authors reviewed the output and are responsible for the code and results.

## License
This project is licensed under the MIT License, see [LICENSE](./LICENSE).

## Authors
- Maurizio Titz [ORCID](https://orcid.org/0000-0002-0249-6244)
- Franz Kaiser [ORCID](https://orcid.org/0000-0002-7089-2249)
- Johannes Kruse [ORCID](https://orcid.org/0000-0002-3478-3379)
- Philipp C. Böttcher [ORCID](https://orcid.org/0000-0002-3240-0442)
- Jan Lange [ORCID](https://orcid.org/0009-0009-5985-4848)
- Martha Frysztacki [ORCID](https://orcid.org/0000-0002-0788-1328)