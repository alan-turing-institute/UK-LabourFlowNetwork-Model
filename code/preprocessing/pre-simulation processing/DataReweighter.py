#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:14:05 2022

Code to re-weight data extracted from the UK Labour Force Survey (LFS) using the LGWT18 (longitudinal weighting) variable
NB: Throughout we use the acronyms SIC for "Standard Industrial Classification", and SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair
"""

#Load necessary libraries
import numpy as np
import pandas as pd
import os

# Set working directory
home =  os.getcwd()[:-4]

# Choose the variables of interest
regvar = "GORWKR" #geographical region 
sicvar = "Inds07m" #industry (SIC)
socvar = "SC10MMJ" #occupation (SOC)

### REWEIGHT JOB DISTRIBUTION ###

### Read in data
df_positions = pd.read_csv(open(f'{home}data/positiondist_LFS_{regvar}_{sicvar}_{socvar}.csv', 'rb'), header=0,index_col=0)
df_positions = df_positions[df_positions.reg_id!=22].copy()

### Perform weighted counts of jobs based on (region, sic, soc) groupings
df_summary = df_positions.groupby(['pos_node_ids', 'reg_id', 'sic_id', 'soc_id'])['LGWT18'].agg(weight_sum='sum').reset_index().copy()
df_summary['perc_jobs'] = 100*(df_summary.weight_sum/df_summary.weight_sum.sum())
df_summary['count_jobs'] = (df_summary['perc_jobs']/df_summary['perc_jobs'].min()).round().astype(int)

### Expand summarised dataset back out to the disaggreate version (i.e. to 1 row entry per observation)   
df=pd.melt(df_summary.replace("",0), id_vars=['pos_node_ids', 'reg_id', 'sic_id', 'soc_id'], value_vars=['count_jobs']).sort_values(by='pos_node_ids')
df_positions_revised = (pd.DataFrame(np.repeat(df.values,df.value.astype(int),axis=0))).rename(columns={0:'pos_node_ids',1:'reg_id',2:'sic_id',3:'soc_id',4:'varname',5:'count_jobs'})

### Write re-weighted job distribution to file
df_positions_revised[['pos_node_ids', 'reg_id', 'sic_id', 'soc_id']].to_csv(f'{home}data/positiondist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv', index=False)

# Generate summary data frame grouped by industry
df_ind = df_positions.groupby(['reg_id'])['LGWT18'].agg(weight_sum='sum').reset_index().copy()
df_ind['normalised_sum'] = df_ind.weight_sum.copy()/df_ind.weight_sum.sum()

### Write re-weighted jobs by industry list to file
df_ind.to_csv(f'{home}data/positionsbyindustry_LFS_{regvar}_{sicvar}_{socvar}.csv', index=False)

### REWEIGHT CONSUMPTION PREFERENCE DISTRIBUTION ###

### Read in data
df_cpr = pd.read_csv(open(f'{home}data/consumptionpref_dist_LFS.csv'))
df_cpr = df_cpr[df_cpr>0]

### Perform weighted counts of consumption preference values
df_summary = df_cpr.groupby(['consumption_pref'])['LGWT18'].agg(weight_sum='sum').reset_index().copy()
df_summary['perc_cpref'] = 100*(df_summary.weight_sum/df_summary.weight_sum.sum())
df_summary['count_cpref'] = (df_summary['perc_cpref']/df_summary['perc_cpref'].min()).round().astype(int)

### Expand summarised dataset back out to the disaggreate version (i.e. to 1 row entry per observation)      
df=pd.melt(df_summary.replace("",0), id_vars=['consumption_pref'], value_vars=['count_cpref']).sort_values(by='consumption_pref')
df_cpr_revised = (pd.DataFrame(np.repeat(df.values,df.value.astype(int),axis=0))).rename(columns={0:'consumption_pref',1:'varname',2:'count_cpref'})

### Write re-weighted job distribution to file
df_cpr_revised[['consumption_pref']].to_csv(f'{home}data/consumptionpref_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv', index=False)

### REWEIGHT AGE DISTRIBUTION ###

### Read in data
df_age = pd.read_csv(open(f'{home}data/age_dist_LFS.csv'), dtype="float64")

### Perform weighted counts of age values
df_summary = df_age.groupby(['AGE'])['LGWT18'].agg(weight_sum='sum').reset_index().copy()
df_summary['perc_age'] = 100*(df_summary.weight_sum/df_summary.weight_sum.sum())
df_summary['count_age'] = (df_summary['perc_age']/df_summary['perc_age'].min()).round().astype(int)

### Expand summarised dataset back out to the disaggreate version (i.e. to 1 row entry per observation)      
df=pd.melt(df_summary.replace("",0), id_vars=['AGE'], value_vars=['count_age']).sort_values(by='AGE')
df_age_revised = (pd.DataFrame(np.repeat(df.values,df.value.astype(int),axis=0))).rename(columns={0:'AGE',1:'varname',2:'count_age'})

### Write re-weighted job distribution to file
df_age_revised[['AGE']].to_csv(f'{home}data/age_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv', index=False)
