# UK Labour Flow Network (LFN) Model **[UNDER CONSTRUCTION]**

This public repository contains code and data associated with the UK LFN model. This model emerges the LFN describing the pattern of labour mobility (i.e. job-to-job transitions) within the UK labour market. The associated preprint is posted at TKTKTK.

This project was developed by [K.R. Fair](https://www.turing.ac.uk/people/researchers/kathyrn-r-fair) and [O.A. Guerrero](http://oguerr.com/), within the Public Policy Programme at the [Alan Turing Institute](https://www.turing.ac.uk/). ***acknowledge the EPSRC grant, and has benefitted greatly from interactions between the researchers and [BEIS](https://www.gov.uk/government/organisations/department-for-business-energy-and-industrial-strategy). Please direct all correspondence to K.R. Fair (kfair@turing.ac.uk).

To run a set of simulations, make all required parameter selections (detailed within each script) and launch the script. The expected runtime varies between scripts, but an individual simulation with 3500 agents should run in approximately 1 minute.

**NOTE:** In order to run simulations, the user will need to generate input datasets from the UK Labour Force Survey (LFS) longitudinal dataset, as well as the O-NET database as these data cannot be posted publicly in this repository.

## Repository Structure

```
├── LICENSE
├── README.md                           <- README for project users.
│
├── data                                <- All publicly available data required to run the model.
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

## Tutorial Notebooks

We provide several annotated Jupyter notebooks in the code/simulation folder with examples of different uses of the model. These tutorials use some synthetic data, as not all data used in our simulations can be made publicly available.

1. [BasicSimulation.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/BasicSimulation.ipynb) performs a single simulation of the model and presents a visualisation of the LFNs generated, along with associated statistics comparing the simulated and observed LFNs.
2. [Calibration.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/Calibration.ipynb) demonstrates the algorithm used to calibrate the model's free parameters, and includes visualisations of calibration process.
3. [ShockSimulation.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/ShockSimulation.ipynb) performs a suite of Monte Carlo simulations of the model, incorporating a shock.

## System & Hardware Requirements

* macOS Monterey version 12.6 (or similar)

* Sufficient RAM to support storage of data structures during simulations

## Installation Guides

All software should install within a few minutes on a standard computer, the versions listed here are those the scripts have been tested on.

* Python Version 3.9.12 https://www.python.org/downloads/
* Spyder Version 5.3.3 (IDE for R) https://www.spyder-ide.org/

<!--## Acknowledgements

We thank Andy Jones and his team at BEIS, Áron Pap, and Dr. Alden Conner for their contributions to the development of this model and associated repository.-->
