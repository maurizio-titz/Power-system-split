# Power System Split
[![pipeline status](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/pipeline.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)
[![coverage report](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/coverage.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)

## Installation

All the requirements can be found in `requirements.txt` and the package can be installed using `pip install -r requirements.txt`. Tested for python 3.7.

To get the code running, copy the German data into a folder `power_system_split/data/Germany/`. The European data should lie under `power_system_split/data/European_networks/`.

## Usage

The following image displaces the nodes that remain connected after a large system split
with 99 % probability:
![alt text](img/Likely_splits_node_based_2013_threshold_0.99_edges.png "Nodes that remain connected likely after a large system split")
