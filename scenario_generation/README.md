
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

Before running all scenarios, check your spatial and temporal resolution set in `workflow/configs/config.yaml`. The spatial and temporal resolution can be set at the respective sections, see below.

    clustering:
      cluster_network:
        n_clusters: 50 # change for a different spatial resolution
      temporal:
        averaging: 3h # change for a different temporal resolution

The CO2 budget sweep itself is not set directly in `config.yaml` anymore: `run.name` lists the scenario names to run (e.g. `co2-0.6`, `co2-0.5`, ...), and each name is defined in `workflow/configs/scenarios.yaml`, which overrides `co2_budget.upper`/`lower` per scenario. To add, remove, or change a CO2 target, edit `scenarios.yaml` (and keep `run.name` and `config.sclopf.yaml`'s `Co2-scenarios` list in sync with it).

And make sure to copy the custom powerplants to the right place in pypsa-eur

    cp data/custom_powerplants.csv submodules/pypsa-eur/data/

**Note!** Running the scenarios requires a high-performance computing environment, as well as a [Gurobi license](https://www.gurobi.com/downloads/gurobi-software/).

### A. (Re-)running all LOPF scenarios using the *automated Snakemake workflow*

To create and solve all scenarios (all different CO2 limits from `run.name`/`scenarios.yaml`), switch to the PyPSA-Eur repository

    cd workflow/submodules/pypsa-eur

and run the following command:

    snakemake -call -j1 solve_networks --configfile ../../configs/config.yaml

When running the scenarios the first time, one needs to set `retrieve = true` and it is advised to increase the allowed latency using the `--latency-wait 20` flag. This produces, per scenario, `results/co2-<X>/networks/solved_2050.nc`.

To re-run a scenario from scratch (e.g. after a config change that affects it, such as the CO2 budget scope), force Snakemake to rebuild it even though the output already exists:

    snakemake -call -j1 solve_networks --configfile ../../configs/config.yaml -F

Please follow the documentation of PyPSA-Eur for more details.

### B. (Re-)running all SC-LOPF scenarios using the *automated Snakemake workflow*

After all LOPF results are successfully created in `results/co2-*/networks/solved_2050.nc`, navigate back to the SC-LOPF workflow

    cd ../..

To run all sc-lopf simulations (rolling-window security-constrained redispatch) and reassemble the resulting per-window networks into one network per scenario, run

    snakemake run_all

`run_all` triggers, for every CO2 target listed in `config.sclopf.yaml`'s `Co2-scenarios`: `prepare_sclopf` → `solve_sclopf` (once per rolling window) → `reassemble_all`, which merges all windows back into a single network at `data/European_networks_sclopf/sclopf-elec_s_<NCLUSTERS>_ec_lv1.0_Co2L<X>.nc`. Reassembly now happens automatically as part of `run_all` — there is no separate manual step.

To re-run after upstream (LOPF) results changed, use `-F` to force a full rebuild, or `--rerun-incomplete` if a previous run was interrupted:

    snakemake run_all -F
    snakemake run_all --rerun-incomplete