# Power System Split
[![pipeline status](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/pipeline.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)
[![coverage report](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/coverage.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)

## Installation

All the requirements can be found in `requirements.txt` and the package can be installed using `pip install -r requirements.txt`. Tested for python 3.7.

To get the code running, copy the German data into a folder `power_system_split/data/Germany/`. The European data should lie under `power_system_split/data/European_networks/`.

## Usage
The typical procedure consists of 
1. Running the cascade algorithm (see [example file](power_system_split/Example_run_cascades.py) )
1. Evaluating the cascade results by calculating inertia and load imbalance for each split component and split (see [example file](power_system_split/Example_evaluate_cascade_results.py))
1. Visualising the results and calculating additional features, such as prototypical, clustered splits (see [example file](power_system_split/Example_visualise_results.py))

The entire workflow is summarised examplarily in a [jupyter notebook](Example_run.ipynb).

## Illustration
The following image displays nodes that remain connected after a large system split
with 99 % probability:
![alt text](img/Likely_splits_node_based_2013_threshold_0.99_edges.png "Nodes that remain connected likely after a large system split")
