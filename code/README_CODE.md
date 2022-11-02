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

### Pre-processing

1. **LFS data collection:** this should *always* be run before non-LFS data collection.
    1.  ActivationRateAnalysis.py
    2. IndividualAttributesAnalysis.py
    3. JobDistributionAnalysis.py
    4. TransitionMatrixGeneration.py
2. **Non-LFS data collection:** should *always* be run after LFS data collection, and before pre-simulation processing, as scripts depend on files produced during the LFS data collection step.
    1. region_similarity.py
    2. sic_similarity.py 
    3. soc_skillgetter.py
    4. soc_similarity.py - should *always* be run after soc_skillgetter.py, as dependent on a file produced by that script.
3. **Pre-simulation processing:** should *always* be run after both data collection steps, as scripts depend on files produced at those stages.
    1. DataReweighter.py
    2. ExpandedSimilarityMatrixGeneration.py

### Simulation

1. **Calibration:** run one of Calibration.py, Calibration.ipynb - both perform calibration routine.
2. **Basic simulation:** run one of BasicSimulation.py, BasicSimulation.ipynb - both run a single simulation using the parameters calibrated in the previous step. This can be used as a quick ''sense check'' on the results of the calibration procedure.
3. **Shock simulation:** run one of ShockSimulation.py, ShockSimulation.ipynb - both run a set of simulations where a shock has been introduced.

ABMrun.py is not run as a standalone, but is called within the abovementioned simulation scripts.

## File Dependencies

The following files, produced during the pre-processing stage, should be placed in the data folder before running any simulations or calibration, as they are necessary inputs. The placeholders regvar, sicvar, and socvar indicate the variables the user has chosen to describe geographical region, industry, and occupation, and can be defined within the scripts.

- activation_dict.txt
- income_dict_LFS_{regvar}_{sicvar}_{socvar}.txt
- region_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- sic_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- soc_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- reg_expanded_similaritymat_LFS.sav
- sic_expanded_similaritymat_LFS.sav
- soc_expanded_similaritymat_LFS.sav
- positiondist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv
- age_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv
- consumptionpref_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv

**Note:** in order to run BasicSimulation.py/.ipynb or ShockSimulation.py/.ipynb you will also require the files containing the calibrated parameters, namely:
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_reg_scost_mat_LFS.sav
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_sic_scost_mat_LFS.sav
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_soc_scost_mat_LFS.sav

All other required files are provided in the data folder of this repository.
