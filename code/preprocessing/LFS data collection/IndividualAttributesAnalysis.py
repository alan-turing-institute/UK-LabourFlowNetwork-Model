#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:59:56 2021

Code compiling datasets describing attributes of individuals within the UK labour force from UK Labour Force Survey (LFS)

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

#Load necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# Set working directory
home =  os.getcwd()

### Generate the longitudinal dataframe containing data (ages, hours worked) used to inform agent characteristics (initial age, consumption preference)

#Indicate which variables (columns) we want to keep
vois = ['PERSID', 'INCAC051','INCAC052','INCAC053','INCAC054','INCAC055', 'AGE1', 'AGE2', 'AGE3', 'AGE4', 'AGE5', 'BUSHR1', 'BUSHR2', 'BUSHR3', 'BUSHR4', 'BUSHR5', 'LGWT18']

filelist = glob.glob(home+'\CSVs\LGWT*.csv')  # Create a list of all longitudial weighted (LGWT) files from LFS

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
        filter_var = 'INCAC05' + f'{j}'
        df0 = tmp[['PERSID', 'INCAC05' + f'{j}', 'AGE' + f'{j}', 'BUSHR' + f'{j}', 'source', 'LGWT18']] # subset to specific quarter
        df = df0.loc[(df0[filter_var].isin([1,2]))] # keep only individuals who were employees or self employed

        #Rename columns so we have consistency across all years/quarters (i.e. get rid of character prefixes)
        cnames=df.columns.tolist()
        df.rename(columns={cnames[0]:'PERSID', cnames[1]:'INCAC05', cnames[2]:'AGE', cnames[3]:'BUSHR'}, inplace=True)
        # Drop column containing info on employment status
        df = df[['PERSID', 'AGE','BUSHR', 'source', 'LGWT18']]
        
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

# Create data frame containing age data        

#Drop all workers under age 18, as we assume youngest worker is 18 in the model
dft_age = df_cmb_tot.loc[df_cmb_tot['AGE']>=18].copy()

# Check counts for ages
dft_age['count'] = dft_age.groupby('AGE')['AGE'].transform('count')
# Drop all rows corresponding to ages with <10 observations
dft_agefin = dft_age.loc[dft_age['count'] >= 10, ].copy()

print('Removing <10 counts leaves us with this fraction of total observations for age:')
print(len(dft_agefin)/len(dft_age))

# Shuffle order of data frame and reset index to prevent matching of this array with the consumption preference array
dft_agefin = dft_agefin.sample(frac=1).reset_index(drop=True).copy()

# Store age distribution, this will be used to create a weighted data set to assign agent ages in the model
dft_agefin[['AGE', 'count', 'LGWT18']].to_csv('age_dist_LFS.csv')

# Create data frame containing consumption preference data
dft = df_cmb_tot.loc[~df_cmb_tot.BUSHR.isin([-8,-9])].copy()
#Drop all workers under age 18, as we assume youngest worker is 18 in the model
dft = dft.loc[dft['AGE']>=18].copy()

# Check counts for weekly hours
dft['count'] = dft.groupby('BUSHR')['BUSHR'].transform('count')
# Drop all rows corresponding to weekly hours with <10 observations
dft_fin = dft.loc[dft['count'] >= 10, ].copy()

print('Removing <10 counts leaves us with this fraction of total observations for consumption preference:')
print(len(dft_fin)/len(dft))

### Calculate consumption preference        
dft_fin['leisurefrac'] = 1 - (52*dft_fin['BUSHR'].copy())/8760 # Fraction of week that constitutes leisure time = 1 - fraction of week spent working 
dft_fin['consumption_pref'] = 1 - dft_fin['leisurefrac'] # Consumption preference = 1 - leisurefrac = fraction of week spent working

### Display descriptive statistics for data frame
print(dft_fin.describe())

# Shuffle order of data frame and reset index to prevent matching of this array with the age array
dft_fin = dft_fin.sample(frac=1).reset_index(drop=True).copy()

# Store consumption preference distribution, this will be used to create a weighted data set to assign agent consumption preferences in the model
dft_fin[['consumption_pref', 'count', 'LGWT18']].to_csv('consumptionpref_dist_LFS.csv')

### Visualise data

# Plot of unemployment statistics
plt.figure(1, figsize=(9,3))

plt.subplot(121)
plt.hist(dft_agefin['AGE'])
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(122)
plt.hist(dft_fin['consumption_pref'])
plt.xlabel("Consumption preference")
plt.ylabel("Count")

plt.tight_layout()
plt.show()




