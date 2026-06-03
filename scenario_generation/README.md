
# Structure of Scenario 

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
Navigate into `workflow`:
    cd workflow
<!-- go to workflow -->
Install the necessary dependencies using `conda` or `mamba`:

    mamba env create -f env.yaml
<!-- check if conda channels are all availible to your workstation -->
Activate `pypsa-eur` environment:

    conda activate pypsa-eur

## 2. Running scenarios

Before running all scenarios, check your spatial and temporal resolution set in the `workflow/confings/config.yaml` file. The spatial and temporal resoltion can be set at the respective sections, see below. 

    scenario:
      clusters:
        - 600 # change for a different spatial resolution

    clustering:
      temporal:
        resolution_elec: 2H # change for a different temporal resolution

And make sure to copy the custom powerplants to the right place in pypsa-eur

    cp data/custom_powerplants.csv submodules/pypsa-eur/data/

**Note!** Running the scenarios requires a high-performance computing environment, as well as a [Gurobi license](https://www.gurobi.com/downloads/gurobi-software/).

### A. Running all LOPF scenarios using the *automated Snakemake workflow*

To create and solve all scenarios (all different Co2 Limits), switch to the PyPSA-Eur repository

    cd workflow/submodules/pypsa-eur

and run the following command:

    snakemake -call -j1 solve_elec_networks --configfile ../../configs/config.yaml 
When running the scenarios the first time, one needs to set the `retrieve = true` and it is advised to increase the allowed latency using the `--latency-wait 20` flag.

Please follow the documentation of PyPSA-Eur for more details.

### B. Running all SC-LOPF scenarios using the *automated Snakemake workflow*

After all LOPF results are successfully created in `results/networks/elec_s_200_ec_lv1.0_Co2L*.nc`, navigate back to the SC-LOPF workflow

    cd ../..

To run all sc-lopf simulations, run

    snakemake run_all

After solving the security constrained optimization, one needs to reassemble the networks by executing `reassemble_all.py` script in `\scripts`.