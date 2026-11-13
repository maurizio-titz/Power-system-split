
# Structure of the scenario generation

- `workflow` contains all relevant data, configuration files, scripts, and submodules.
- `workflow/configs` contains all relevant configuration files. `config.yaml` is the configuration file to create networks with PyPSA-Eur, `config.sclopf.yaml` is the configuration file used for the SCLOPF.
- `workflow/data` contains all relevant data: the lookup table for the SCLOPF workflow for N-1 failures, and a list of conventional power plants used for PyPSA-Eur.
- `workflow/scripts` contains scripts that modify the LOPF networks retrieved from PyPSA-Eur to run the SCLOPF.
- `workflow/submodules` contains the PyPSA-Eur submodule, pinned to commit [`299a8ed`](https://github.com/PyPSA/pypsa-eur/commit/299a8ed65bf3adf1700dbca9b725470c4221bd8b) (v0.11.0 + 94 commits). If you downloaded the code without git (e.g., from Zenodo), clone [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) into `workflow/submodules/pypsa-eur` and check out this commit.
- `workflow/Snakefile` manages the SCLOPF workflow.

# Installation and Usage

## 0. Snakemake flags
Snakemake is a workflow management system, used here to run the optimizations in parallel.
Every command is started with `snakemake`. Dependencies and execution instructions are detailed in the `Snakefile`.


The following flags can be added to a Snakemake command:

Force execution even if nothing has changed:

    <command> -F

Continue with independent tasks if a task fails:

    <command> --keep-going

If previous runs were terminated (e.g., by a keyboard interrupt), you need to use

    <command> --rerun-incomplete


## 1. Installation
Navigate into `workflow`:
    cd workflow
<!-- go to workflow -->
Install the necessary dependencies using `conda` or `mamba`:

    mamba env create -f env.yaml
<!-- check if conda channels are all available to your workstation -->
Activate the `pypsa-eur` environment:

    conda activate pypsa-eur

## 2. Running scenarios

Before running all scenarios, check the spatial and temporal resolution set in `configs/config.yaml`. They can be set in the respective sections, see below.

    scenario:
      clusters:
        - 600 # change for a different spatial resolution

    clustering:
      temporal:
        resolution_elec: 2H # change for a different temporal resolution

Also copy the custom power plants to the right place in PyPSA-Eur:

    cp data/custom_powerplants.csv submodules/pypsa-eur/data/

**Note!** Running the scenarios requires a high-performance computing environment, as well as a [Gurobi license](https://www.gurobi.com/downloads/gurobi-software/).

### A. Running all LOPF scenarios using the *automated Snakemake workflow*

To create and solve all scenarios (all CO2 limits), switch to the PyPSA-Eur repository

    cd submodules/pypsa-eur

and run the following command:

    snakemake -call -j1 solve_elec_networks --configfile ../../configs/config.yaml

When running the scenarios for the first time, set `retrieve = true`, and it is advised to increase the allowed latency with the `--latency-wait 20` flag.

Please follow the documentation of PyPSA-Eur for more details.

### B. Running all SCLOPF scenarios using the *automated Snakemake workflow*

After all LOPF results are successfully created in `results/networks/elec_s_600_ec_lv1.0_Co2L*.nc`, navigate back to the SCLOPF workflow

    cd ../..

To run all SCLOPF simulations, run

    snakemake run_all

After solving the security-constrained optimization, reassemble the networks by executing the `reassemble_all.py` script in `scripts/`.