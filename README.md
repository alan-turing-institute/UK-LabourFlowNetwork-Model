# UK Labour Flow Network (LFN) Model

Simulation and model-fitting code for manuscript by Fair & Guerrero (2022). Model emerges the LFN describing the patterns of labour mobility (i.e. job-to-job transitions) within the UK labour market. The associated preprint is posted at TKTKTK.

To run a set of simulations, make all required parameter selections (detailed within each script) and launch script. Expected runtime varies between scripts, but an individual simulation should run within 1 minute.

In order to run simulations, the user will need to generate input datasets from the UK Labour Force Survey longitudinal dataset, as well as the O-NET database as these data cannot be posted publicly in this repository.

## Scripts

* **covidHierV48_Github_parmfit.R** runs fitting procedure for obtaining parameter sets, outputs files containing the best parameter set as determined by that run of the fitting algorithms.

## Input Files

* **simsFITPARMS_v48.rds** contains best-fit parameter sets (generated from covidHierV46_2Github_parmfit.R and covidHierV46_2Github_basesim.R), used for all counterfactual simulations

## System & Hardware Requirements

* macOS Monterey version 12.6 (or similar)

* Sufficient RAM to support storage of data structures during simulations

## Installation Guides

All software should install within a few minutes on a standard computer, the versions listed here are those the scripts have been tested on.

* Python Version 3.9.12 https://www.python.org/downloads/
* Spyder Version 5.3.3 (IDE for R) https://www.spyder-ide.org/
