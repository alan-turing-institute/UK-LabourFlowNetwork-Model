# UK Labour Flow Network (LFN) Model **[UNDER CONSTRUCTION]**

Simulation and model-fitting code for manuscript by Fair & Guerrero (2022). Model emerges the LFN describing the patterns of labour mobility (i.e. job-to-job transitions) within the UK labour market. The associated preprint is posted at TKTKTK.

To run a set of simulations, make all required parameter selections (detailed within each script) and launch script. Expected runtime varies between scripts, but an individual simulation with 3500 agents should run in approximately 1 minute.

In order to run simulations, the user will need to generate input datasets from the UK Labour Force Survey (LFS) longitudinal dataset, as well as the O-NET database as these data cannot be posted publicly in this repository.

## Repository Structure

```
├── LICENSE
├── README.md                           <- The top-level README for users of this project.
│
├── data                                <- All data (that can be made publicly available in this repository) required to run the model.
│
├── code               
│   ├── preprocessing    
│   │   ├── LFS data collection         <- Python scripts used to collect data from LFS dataset.
│   │   ├── non-LFS data collection     <- Python scripts used to collect data from other sources.
│   │   └── pre-simulation processing   <- Python scripts used to further process data collected from LFS.
│   │
│   └── simulation                      <- Python scripts and notebooks used to run model simulations.
└──
```

## System & Hardware Requirements

* macOS Monterey version 12.6 (or similar)

* Sufficient RAM to support storage of data structures during simulations

## Installation Guides

All software should install within a few minutes on a standard computer, the versions listed here are those the scripts have been tested on.

* Python Version 3.9.12 https://www.python.org/downloads/
* Spyder Version 5.3.3 (IDE for R) https://www.spyder-ide.org/
