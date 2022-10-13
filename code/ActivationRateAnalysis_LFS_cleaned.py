#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:04:28 2021

@author: kfair

Based on script developed by Áron Pap
"""

#Load necessary libraries
import pandas as pd
import numpy as np
import os
import glob

# Set working directory
home =  os.getcwd()

### Generate the longitudinal dataframe containing data on job seeking behaviour of employed and unemployed individiuals

#Indicate which variables (columns) we want to keep - Refer to LFS documentaiton for variable definitions
vois = ['PERSID', 'AGE1', 'AGE2', 'AGE3', 'AGE4', 'AGE5', 'DIFJOB1','DIFJOB2','DIFJOB3','DIFJOB4','DIFJOB5', 'ADDJOB1', 'ADDJOB2', 'ADDJOB3', 'ADDJOB4', 'ADDJOB5', 'LOOK41', 'LOOK42', 'LOOK43', 'LOOK44', 'LOOK45', 'ILODEFR1', 'ILODEFR2', 'ILODEFR3', 'ILODEFR4', 'ILODEFR5', 'LGWT18']

filelist = glob.glob(home+'\CSVs\LGWT*.csv') # Create a list of all LGWT files from LFS

for i in range(len(filelist)):

    # Read in files one at a time
    filename = filelist[i]
    tmp = pd.read_csv(filename, usecols = vois)
    tmp['source'] = filename[-13:-4]
    
    # Get labels for years and quarters
    startmonth = filename[-13:-11]
    startyear= int(filename[-11:-9])
    
    if startmonth == 'JM':
        quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1']
        years = [2000 + startyear, 2000 + startyear, 2000 + startyear, 2000 + startyear, 2000 + startyear+1]

    if startmonth == 'AJ':
        quarters = ['Q2', 'Q3', 'Q4', 'Q1', 'Q2']
        years = [2000 + startyear, 2000 + startyear, 2000 + startyear, 2000 + startyear + 1, 2000 + startyear + 1]
        
    if startmonth == 'JS':
        quarters = ['Q3', 'Q4', 'Q1', 'Q2', 'Q3']
        years = [2000 + startyear, 2000 + startyear, 2000 + startyear + 1, 2000 + startyear + 1, 2000 + startyear + 1]    
    
    if startmonth == 'OD':
        quarters = ['Q4', 'Q1', 'Q2', 'Q3', 'Q4']
        years = [2000 + startyear, 2000 + startyear + 1, 2000 + startyear + 1, 2000 + startyear + 1, 2000 + startyear + 1]  
    
    # Cycle through quarters
    for j in np.arange(1,6):
        df = tmp[['PERSID', 'AGE' + f'{j}', 'DIFJOB' + f'{j}', 'ADDJOB' + f'{j}', 'LOOK4' + f'{j}', 'ILODEFR' + f'{j}', 'source', 'LGWT18']] # subset to specific quarter

        #Rename columns so we have consistency across all years/quarters (i.e. get rid of character prefixes)
        cnames=df.columns.tolist()
        df.rename(columns={cnames[0]:'PERSID', cnames[1]:'AGE', cnames[2]:'DIFJOB', cnames[3]:'ADDJOB', cnames[4]:'LOOK4', cnames[5]:'ILODEFR'}, inplace=True)

         # Add year and quarter columns
        df['year'] = years[j-1]
        df['quarter'] = quarters[j-1]
        
        if j==1:
            df_cmb=df
        else:
            frames = [df_cmb, df]
            df_cmb = pd.concat(frames)       
    
    print(i)
    
    if i==0:
        df_cmb_tot=df_cmb
    else:
        frames = [df_cmb_tot, df_cmb]
        df_cmb_tot = pd.concat(frames)

#Drop all workers under age 18, as we assume youngest worker is 18
df_cmb_tot = df_cmb_tot.loc[df_cmb_tot['AGE']>=18].copy()

# Display descriptive statistics for data frame
print(df_cmb_tot.describe())

### Generate counts 
# NB: 1=yes, 2=no, except for ADDJOB where 1=new (replacement) job, 2=additional job, ILODEFR where 1=in employment, 2= ILO unemployed, 3=inactive, 4=under16
employed_active = len(df_cmb_tot[(df_cmb_tot['DIFJOB']==1) & (df_cmb_tot['ADDJOB']==1) & (df_cmb_tot['ILODEFR']==1)]) #employed individuals actively seeking a new (replacement) job
employed_active_weight = df_cmb_tot.LGWT18[(df_cmb_tot['DIFJOB']==1) & (df_cmb_tot['ADDJOB']==1) & (df_cmb_tot['ILODEFR']==1)].sum()
employed = len(df_cmb_tot[(df_cmb_tot['DIFJOB'].isin([1,2])) & (df_cmb_tot['ILODEFR']==1)]) #employed individuals asked if they were seeking a new job
employed_weight = df_cmb_tot.LGWT18[(df_cmb_tot['DIFJOB'].isin([1,2]))  & (df_cmb_tot['ILODEFR']==1)].sum()
unemployed_active = len(df_cmb_tot[(df_cmb_tot['LOOK4']==1) & (df_cmb_tot['ILODEFR']==2)]) #unemployed individuals actively looking for work
unemployed_active_weight = df_cmb_tot.LGWT18[(df_cmb_tot['LOOK4']==1) & (df_cmb_tot['ILODEFR']==2)].sum()
unemployed = len(df_cmb_tot[(df_cmb_tot['LOOK4'].isin([1,2])) & (df_cmb_tot['ILODEFR']==2)]) #unemployed individuals asked if they were looking for work
unemployed_weight = df_cmb_tot.LGWT18[(df_cmb_tot['LOOK4'].isin([1,2]))  & (df_cmb_tot['ILODEFR']==2)].sum()
### Calculate frequencies of activation
employed_freq = employed_active/employed
unemployed_freq = unemployed_active/unemployed

### Save all info in a dictionary
activation_dict = {'activation_dict': {'employed_active': employed_active, 'employed_active_weight': employed_active_weight, \
                                       'employed': employed, 'employed_weight': employed_weight, \
                            'unemployed_active': unemployed_active, 'unemployed_active_weight': unemployed_active_weight, \
                                'unemployed': unemployed, 'unemployed_weight': unemployed_weight
                             }}

with open('activation_dict.txt', 'w') as f:
    print(activation_dict, file=f)
