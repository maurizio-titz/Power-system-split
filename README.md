# Power System Split
Code accompanying the article: Titz, M., et al. (2027). Cascading failures and system splits in decarbonized European power systems. Joule 11, 102742. DOI: [10.1016/j.joule.2026.102742](https://doi.org/10.1016/j.joule.2026.102742). Preprint: [arXiv:2603.24529](https://arxiv.org/abs/2603.24529).

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
Start by creating the scenarios, following the workflow described in `scenario_generation/README.md`, or download the already generated scenarios from Zenodo (see below).

Note that this code was run on a workstation with an Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz, 32 logical cores and around 500 GB of memory.

Before running the scripts, make sure you have set up the folder structure as described in the `Config` section.
The scripts in `scripts` contain our workflow and need to be executed in roughly this order:
- [`extract_grid_data.py`](./scripts/extract_grid_data.py): Saves grid properties to disk. Run before `run_cascade_code.py`.
- [`calc_pre_outage_data.py`](./scripts/calc_pre_outage_data.py): Calculates inertia and other properties from the (solved) PyPSA networks.
- [`run_cascade_code.py`](./scripts/run_cascade_code.py): Runs the cascade algorithm on the PyPSA networks. This can take a few weeks, depending on the available compute power.
- [`evaluate_cascade.py`](./scripts/evaluate_cascade.py): Evaluates the cascade results, e.g., inertia and load imbalance for each split, and indicator vectors. Run only after `run_cascade_code.py` has finished.
- [`prepare_split_visualization.py`](./scripts/prepare_split_visualization.py): Prepares the data from the results that we use for visualization, e.g., prototypical clusters of system splits. Run only after `evaluate_cascade.py` has finished.
- [`calc_inertia_placement.py`](./scripts/calc_inertia_placement.py): Determines the optimal inertia placement to mitigate the impact of system splits. Run only after `prepare_split_visualization.py` has finished.
- [`cluster_blackouts.py`](./scripts/cluster_blackouts.py): Clusters the splits using a composite metric that takes both node and edge features into account, as described in the main manuscript.
- [`compare_ac_with_dc_snapshots.py`](./scripts/compare_ac_with_dc_snapshots.py): Compares DC with AC power flows for representative snapshots of all CO2 levels. Requires only the PyPSA networks. Use `--n_processes` to run in parallel and `--resume` to continue an interrupted run.
- Create plots by running the scripts in `scripts/plots` individually, or run the bash script [`generate_all_plots.sh`](./scripts/generate_all_plots.sh) to generate all plots in one go.
- [`plot_all_manuscript_figures.py`](./scripts/plot_all_manuscript_figures.py): Creates only the figures of the main manuscript and of the Supplemental Information. Some functions need more than 32 GB of memory. If your machine cannot provide this, call the script as `python scripts/plot_all_manuscript_figures.py --low_memory`, or pass `low_memory=True` if you call the functions from a Python interpreter. This skips the memory-intensive figures.

All code assumes that your `PYTHONPATH` contains the repository directory and that the code is executed from there.

The scenario data is created in `scenario_generation`, which uses a separate conda environment. More details are given under `Input data and results` and in the README there.

## Input data and results

All input data needed to run the scripts is publicly available or can be generated with this repository.

The version of this repository used in the publication is archived on Zenodo: [10.5281/zenodo.20529176](https://doi.org/10.5281/zenodo.20529176).

The data set is also available on Zenodo ([10.5281/zenodo.20528959](https://doi.org/10.5281/zenodo.20528959)). It includes the CO2 scenarios generated with PyPSA-Eur and the results generated with the code in this repository. After unpacking it into the main folder of the repository on your local machine, you can plot the figures presented in the manuscript.

If you want to place the files in another location, modify the paths in [utils/config.py](./utils/config.py) accordingly.

The scenario data can also be generated with the [`scenario_generation`](./scenario_generation/) subfolder. Its installation and usage are explained in the README there.

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