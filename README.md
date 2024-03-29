# UK Labour Flow Network (LFN) Model **[UNDER CONSTRUCTION]**

This public repository contains code and data associated with the UK LFN model. This model emerges the LFN describing the pattern of labour mobility (i.e. job-to-job transitions) within the UK labour market. The associated pre-print is posted [here](https://arxiv.org/abs/2301.07979).

This project was developed by [K.R. Fair](https://www.turing.ac.uk/people/researchers/kathyrn-r-fair) and [O.A. Guerrero](http://oguerr.com/), within the Public Policy Programme at the [Alan Turing Institute](https://www.turing.ac.uk/). Please direct all correspondence to K.R. Fair (kfair@turing.ac.uk).

You will find all instructions for running simulations [here](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/README_CODE.md) within the code folder.

**NOTE:** In order to run simulations, the user will need to generate input datasets from the UK Labour Force Survey (LFS) longitudinal dataset, as well as the O-NET database as these data cannot be posted publicly in this repository. All data that can be posted publicly is described [here](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/data/README_data.md) within the data folder.

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
* Spyder Version 5.3.3 (IDE for python) https://www.spyder-ide.org/

## Acknowledgements

This work was supported by Wave 1 of The UKRI Strategic Priorities Fund under the EPSRC Grant EP/W006022/1, particularly the "Shocks and Resilience" theme within that grant & The Alan Turing Institute. We would like to thank Áron Pap for conducting preliminary explorations of this topic, and Andy Jones and his team at [BEIS](https://www.gov.uk/government/organisations/department-for-business-energy-and-industrial-strategy) for their interactions with us throughout this project. We also thank Dr. Alden Conner for her contributions to the development of this repository.
