<!--
SPDX-FileCopyrightText: Martha Frysztacki (KIT, OET)

SPDX-License-Identifier: MIT
-->

# Repository Structure

- `workflow` contains the all relevant data, configuration files, scripts, and submodules.
- `workflow/configs` contains all relevant configuration files. `config.yaml` is the configuration file to create networks within PyPSA-Eur, `config.sclopf.yaml` is the configuration file used for the sclopf. 
- `workflow/data` contains all relevant data: the lookup tabe for the sclopf workflow for N-1 failures, and a list of conventional powerplants used for PyPSA-Eur 
- `workflow/scripts` contains scripts that modify the LOPF networks retrieved from PyPSA-Eur to run the SC-LOPF
- `workflow/submodules` contains the PyPSA-Eur submodule
- `workflow/Snakefile` manages the SC-LOPF workflow

# Installation and Usage

## 0. Snakemake flags
Snakemake is a workflow management used here to streamline the optimization process in parallel.
Every command is started with `snakemake`. Dependencies and execution instructions are detailed in the `Snakefile`.


The following flags can be added to a snakemake prompt:

Force execution even if nothing has changed:

    <command> -F

If an error occurs in base task, still continue:

    <command> --keep-going

If runs before where terminated (e.g. by keyboard interupt), you need to use

    <command> --rerun-incomplete


## 1. Installation

Clone the repository including its submodules:

    git clone --recurse-submodules https://github.com/martacki/sclopf-iter

Install the necessary dependencies using `conda` or `mamba`:

    mamba env create -f submodules/pypsa-eur/envs/environment.yaml

Activate `pypsa-eur` environment:

    conda activate pypsa-eur

Navigate into the main Snakemake workflow directory of `PyPSA-Eur`:

    cd workflow/submodules/pypsa-eur

## 2. Running scenarios

Before running all scenarios, check your spatial and temporal resolution set in the `workflow/confings/config.yaml` file. The spatial and temporal resoltion can be set at the respective sections, see below. 

    scenario:
      clusters:
        - 50 # change for a different spatial resolution

    clustering:
      temporal:
        resolution_elec: 2H # change for a different temporal resolution

And make sure to copy the custom powerplants to the right place in pypsa-eur

    cp workflow/data/custom_powerplants.csv workflow/submodules/pypsa-eur/data/

**Note!** Running the scenarios requires a high-performance computing environment, as well as a [Gurobi license](https://www.gurobi.com/downloads/gurobi-software/).

### A. Running all LOPF scenarios using the *automated Snakemake workflow*

To create and solve all scenarios (all different Co2 Limits), switch to the PyPSA-Eur repository

    cd workflow/submodules/pypsa-eur

and run the following command:

    snakemake -call -j1 solve_elec_networks --configfile ../../configs/config.yaml 

Please follow the documentation of PyPSA-Eur for more details.

### B. Running all SC-LOPF scenarios using the *automated Snakemake workflow*

After all LOPF results are successfully created in `results/networks/elec_s_200_ec_lv1.0_Co2L*.nc`, navigate back to the SC-LOPF workflow

    cd ../..

To run all sc-lopf simulations, run

    snakemake run_all

### C. Running the SC-LOPF using the *faster linopy solver interface* [STANDARD]

Unfortunately, releases of `pypsa<=0.28.0` contain a bug for the faster `linopy` solver interface when preparing the `sclopf` constraints (in short, wrong constaints are written).

As of today (03.07.2024 09:00 CET), the bug is fixed in upstream, but the fix is not available in any `pypsa` release. If you still want to use the fast `linopy` interface, navigate to `config.sclopf.yaml` and change the following setting to `False`:

    network_sclopf: False # if False, jumps to faster formulation using linopy

**Note!** Using the updates requires an update in your environment, which can be installed using

    pip install git+https://github.com/PyPSA/PyPSA.git@master

Unfortunately, this version does not support the old `sclopf` formulation. If you want to revert back, install `pypsa v0.28.0` by running (again)

    pip install pypsa==0.28.0

You don't have to uninstall, all updates should be treaded automatically.