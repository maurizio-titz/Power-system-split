# Power System Split
[![pipeline status](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/pipeline.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)
[![coverage report](https://jugit.fz-juelich.de/network-science-group/power-system-split/badges/master/coverage.svg)](https://jugit.fz-juelich.de/network-science-group/power-system-split/-/commits/master)

## Installation

All the requirements can be found in `requirements.txt` and the package can be installed using `pip install .`.

NOTE: You might have to manually install xarray and its io tools. To do this run `pip install xarray` and `pip install "xarray[io]"` to install the input/output dependencies of [xarray](http://xarray.pydata.org/en/stable/getting-started-guide/installing.html) that pypsa uses for imports. 

## Usage
The following image displaces the nodes that remain connected after a large system split
with 99 % probability:
![alt text](img/Likely_splits_node_based_2013_threshold_0.99_edges.png "Nodes that remain connected likely after a large system split")
