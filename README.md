# UK Labour Flow Network (LFN) Model

Simulation and model-fitting code for manuscript by Fair & Guerrero (2022). Model emerges the LFN describing the patterns of labour mobility (i.e. job-to-job transitions) within the UK labour market. The associated preprint is posted at TKTKTK.

To run a set of simulations, make all required parameter selections (detailed within each script) and launch script. Expected runtime varies between scripts, but an individual simulation should run within 1 minute.

In order to run simulations, the user will need to generate input datasets from the UK Labour Force Survey longitudinal dataset, as well as the O-NET database as these data cannot be posted publicly in this repository.

## Repository Structure

├── LICENSE
├── README.md          <- The top-level README for users of this project.
├── CODE_OF_CONDUCT.md <- Guidelines for users and contributors of the project.
├── CONTRIBUTING.md    <- Information on how to contribute to the project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── project_management <- Meeting notes and other project planning resources
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│  └── visualisation  <- Scripts to create exploratory and results oriented visualisations
│       └── visualise.py
└──

## System & Hardware Requirements

* macOS Monterey version 12.6 (or similar)

* Sufficient RAM to support storage of data structures during simulations

## Installation Guides

All software should install within a few minutes on a standard computer, the versions listed here are those the scripts have been tested on.

* Python Version 3.9.12 https://www.python.org/downloads/
* Spyder Version 5.3.3 (IDE for R) https://www.spyder-ide.org/
