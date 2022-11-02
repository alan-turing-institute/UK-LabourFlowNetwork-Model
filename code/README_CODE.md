# Code for UK Labour Flow Network Model

This README file contains descriptions of all scripts used to run simulations, and to generate datasets required to run simulations. Throughout we use LFS to refer to the UK Labour Force Survey.

## Folder Structure

```               
├── preprocessing    
│   ├── LFS data collection         <- Python scripts used to collect data from LFS dataset.
│   │    ├── ActivationRateAnalysis.py                <- Produces estimates of relative rates of activation for employed and unemployed individuals.
│   │    ├── IndividualAttributesAnalysis.py          <- Compiles datasets describing attributes of individuals within the UK labour force.
│   │    ├── JobDistributionAnalysis.py               <- Compiles datasets describing the UK job distribution.
│   │    └── TranstionMatrixGeneration.py             <- Produces job to job transition density matrices for the UK.
│   ├── non-LFS data collection     <- Python scripts used to collect data from other sources.
│   │    ├── region_similarity.py                     <- Calculates the similarity between geographical regions based on distance.
│   │    ├── sic_similarity.py                        <- Calculate the similarity between industries, based on industry-level input-output table.
│   │    ├── soc_similarity.py                        <- Calculates the similarity between occupations, based on the skills associated with each occupation.
│   │    └── soc_skillgetter.py                        <- Collects O-NET (Occupational Information Network) skills data associated with occupations.
│   └── pre-simulation processing   <- Python scripts used to further process data collected from LFS.
│        ├── DataReweighter.py                        <- Re-weights data extracted from LFS using the LFS longitudinal weighting variable.
│        └── ExpandedSimilarityMatrixGeneration.py    <- Generates expanded (i.e. node-level) similarity matrices for use in model. 
│
└── simulation                    
│   ├── ABMrun.py                   <- Contains all functions needed to run model.
│   ├── BasicSimulation.ipynb       <- Provides an overview of how to run a basic simulation of the model.
│   ├── BasicSimulation.py          <- Runs a basic simulation of the model.
│   ├── Calibration.ipynb           <- Provides an overview of how to calibrate the model.
│   ├── Calibration.py              <- Runs model calibration procedure.
│   ├── ShockSimulation.ipynb       <- Provides an overview of how to run simulations of the model where a shock is introduced.
│   └── ShockSimulation.py          <- Runs simulations of the model where a shock is introduced.
└──
```

## Run sequence

Below we detail the suggested run order for these scripts.

Pre-processing:

1. LFS data collection: this should *always* be run before non-LFS data collection.
  1.  ActivationRateAnalysis.py
  2. IndividualAttributesAnalysis.py
  3. JobDistributionAnalysis.py
  4. TransitionMatrixGeneration.py
2. Non-LFS data collection: should *always* be run after LFS data collection, and before pre-simulation processing, as scripts depend on files produced during the LFS data collection step.
  1. region_similarity.py
  2. sic_similarity.py 
  3. soc_skillgetter.py
  4. soc_similarity.py - should *always* be run after soc_skillgetter.py, as dependent on a file produced by that script.
3. Pre-simulation processing: should *always* be run after both data collection stages, as scripts depend on files produced at those stages.
  1. DataReweighter.py
  2. ExpandedSimilarityMatrixGeneration.py
