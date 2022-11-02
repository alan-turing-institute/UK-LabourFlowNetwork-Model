#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 10:37:21 2021

Code producing job to job transition density matrices from UK Labour Force Survey (LFS)

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""
#Load necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob 

# Set working directory
home =  os.getcwd()

# Create dataframe to use for mapping months to quarters
monthmapper = pd.DataFrame(list(zip([1,2,3,4,5,6,7,8,9,10,11,12], ['Q1','Q1','Q1','Q2','Q2','Q2','Q3','Q3','Q3','Q4','Q4','Q4'])), columns=['CONMON', 'quarter_CONMON'])

### Generate the longitudinal dataframe containing industry (SIC), occupation (SOC), geographical region, and wage data

# define desired region, sic, soc variables
select_vars = ['GORWKR', 'Inds07m', 'SC10MMJ'] 
#Indicate which variables (columns) we want to keep for creating job to job transition matrices
vois = ['PERSID','AGE1', 'AGE2', 'AGE3', 'AGE4', 'AGE5', 
        select_vars[0] + '1', select_vars[0] + '2', select_vars[0] + '3', select_vars[0] + '4', select_vars[0] + '5',
        select_vars[2] + '1', select_vars[2] + '2', select_vars[2] + '3', select_vars[2] + '4', select_vars[2] + '5', 
        select_vars[1] + '1', select_vars[1] + '2', select_vars[1] + '3', select_vars[1] + '4', select_vars[1] + '5',
        'INCAC051','INCAC052','INCAC053','INCAC054','INCAC055',
        'CONMPY1', 'CONMPY2', 'CONMPY3', 'CONMPY4', 'CONMPY5', 'CONSEY1', 'CONSEY2', 'CONSEY3', 'CONSEY4', 'CONSEY5', 
        'CONMON1', 'CONMON2', 'CONMON3', 'CONMON4', 'CONMON5', 'LGWT18']

filelist = glob.glob(home+'\CSVs\LGWT*.csv') # Create a list of all longitudial weighted (LGWT) files from LFS

for i in range(len(filelist)):

    # Read in files one at a time
    filename = filelist[i]
    tmp = pd.read_csv(filename, usecols = vois)
    
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
    for j in np.arange(1,5):
        
        # Get "from" dataframe
        filter_var_from = 'INCAC05' + f'{j}'
        df_from0 = tmp[['PERSID', 'AGE' + f'{j}', 'INCAC05' + f'{j}', select_vars[0] + f'{j}', select_vars[2] + f'{j}', select_vars[1] + f'{j}', 'CONMPY' + f'{j}', 'CONSEY' + f'{j}', 'CONMON' + f'{j}', 'LGWT18']] # subset to specific quarter
        df_from = df_from0.loc[(df_from0[filter_var_from].isin([1,2]))] # keep only individuals who were employees or self employed
        df_from = df_from.merge(monthmapper, left_on='CONMON'+f'{j}', right_on='CONMON',how = 'inner')
        
        # Add year and quarter columns
        df_from['year_from'] = years[j-1]
        df_from['quarter_from'] = quarters[j-1]
    
        # Get "to" dataframe
        filter_var_to = 'INCAC05' + f'{j+1}'
        df_to0 = tmp[['PERSID', 'AGE' + f'{j+1}', 'INCAC05' + f'{j+1}', select_vars[0] + f'{j+1}', select_vars[2] + f'{j+1}', select_vars[1] + f'{j+1}', 'CONMPY' + f'{j+1}', 'CONSEY' + f'{j+1}', 'CONMON' + f'{j+1}', 'LGWT18']] # subset to specific quarter
        df_to = df_to0.loc[(df_to0[filter_var_to].isin([1,2]))] # keep only individuals who were employees or self employed
        df_to = df_to.merge(monthmapper, left_on='CONMON'+f'{j+1}', right_on='CONMON',how = 'inner')
        
        # Add year and quarter columns
        df_to['year_to'] = years[j]
        df_to['quarter_to'] = quarters[j]
        
        #Match on PERSID
        df_match = df_to.merge(df_from, left_on='PERSID', right_on='PERSID',how = 'inner')
        
        #Rename columns so we have consistency across all years/quarters (i.e. get rid of character prefixes)
        cnames=df_match.columns.tolist()
        
        df_match.rename(columns={cnames[0]:'PERSID', cnames[1]:'AGE_to', cnames[2]:'INCAC05_to', cnames[3]:select_vars[0] + '_to', cnames[4]:select_vars[2] + '_to' , \
                                 cnames[5]:select_vars[1] + '_to' , cnames[6]:'CONMPY_to', cnames[7]:'CONSEY_to', cnames[8]:'CONMON_to', cnames[9]: 'LGWT18_to', \
                                     cnames[11]:'quarter_CONMON_to', cnames[14]:'AGE_from', cnames[15]:'INCAC05_from', cnames[16]:select_vars[0] + '_from', \
                           cnames[17]:select_vars[2] + '_from', cnames[18]:select_vars[1] + '_from', cnames[19]:'CONMPY_from', cnames[20]:'CONSEY_from', \
                           cnames[21]:'CONMON_from' , cnames[22]: 'LGWT18_from', cnames[24]:'quarter_CONMON_from'}, inplace=True)
        
        if j==1:
            df_cmb=df_match
        else:
            frames = [df_cmb, df_match]
            df_cmb = pd.concat(frames)       
    
    print(i)
    
    if i==0:
        df_cmb_tot_all=df_cmb
    else:
        frames = [df_cmb_tot_all, df_cmb]
        df_cmb_tot_all = pd.concat(frames)
     
#Drop all workers under age 18, as we assume youngest worker is 18 in the model
df_cmb_tot = df_cmb_tot_all.loc[(df_cmb_tot_all['AGE_from']>=18)].copy()

# Finding job-to-job moves
# First drop all entries where the newest entry for 'year the job was started' (i.e CONMPY_to or CONSEY_to) doesn't match either of the year from or the year to
df_cmb_tot = df_cmb_tot.loc[(df_cmb_tot.CONMPY_to ==  df_cmb_tot.year_to) | (df_cmb_tot.CONMPY_to == df_cmb_tot.year_from) | (df_cmb_tot.CONSEY_to == df_cmb_tot.year_to) | (df_cmb_tot.CONSEY_to == df_cmb_tot.year_from)]
# then drop all entries where similarly the quarter the job was started does not match either the quarter from or quarter to
df_cmb_tot = df_cmb_tot.loc[(df_cmb_tot.quarter_CONMON_to ==  df_cmb_tot.quarter_to) | (df_cmb_tot.quarter_CONMON_to == df_cmb_tot.quarter_from)]

# Then keep only the variables we care about 
df_jtj = df_cmb_tot[[select_vars[0] + '_to', select_vars[2] + '_to' , select_vars[1] + '_to', select_vars[0] + '_from', select_vars[2] + '_from' , select_vars[1] + '_from', 'LGWT18_from']]


### Regional transition matrix ###
#Get rid of rows with missing data for region, or where region is 'Outside UK'
df_jtj_reg = df_jtj.loc[~(df_jtj[select_vars[0] + '_from'].isin([-8,-9,22])) & ~(df_jtj[select_vars[0] + '_to'].isin([-8,-9,22]))]
#Keep only the voi we're interested in (region)
df_m = df_jtj_reg[[select_vars[0] + '_from',select_vars[0] + '_to', 'LGWT18_from']]
# Cross-tabulation to generate adjacency matrices
df_reg_unweighted = pd.crosstab(index=df_m[select_vars[0] + '_from'], columns = df_m[select_vars[0] + '_to'])
df_reg = pd.crosstab(index=df_m[select_vars[0] + '_from'], columns = df_m[select_vars[0] + '_to'], values=df_m['LGWT18_from'], aggfunc='sum')
df_reg[np.isnan(df_reg)] = 0
# Store region transition counts
df_reg_unweighted.sum(axis=1).to_csv(f'region_unweightedtransitionrowcounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
df_reg_unweighted.to_csv(f'region_unweightedtransitioncounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
# Normalizing based on sum across rows (i.e. out degree from each region aka each row)
print(f'The lowest out-degree for a region is: {min(df_reg_unweighted.sum(axis=1))}')

df_reg_density = df_reg.div(df_reg.sum().sum())

# Generate plot of transition matrix
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(df_reg_density, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(df_reg_density.index.tolist())))
ax.set_xticklabels(df_reg_density.index.tolist(), rotation = 90)
ax.set_yticks(np.arange(len(df_reg_density.index.tolist())))
ax.set_yticklabels(df_reg_density.index.tolist())
ax.set_xlabel("Next region", fontsize=14)
ax.set_ylabel("Previous region", fontsize=14)
ax.set_title("Transition densities for regions (movers only)", fontsize=16)

#Generate fig with layout
plt.grid(False)
plt.savefig(f'regional_transitiondensities_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.png', bbox_inches="tight")
plt.show()

# Store region matrix
df_reg_density.to_csv(f'region_transitiondensity_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')

### SOC (occupation) transition matrix ###
#Get rid of rows with missing data for soc
df_jtj_soc = df_jtj.loc[(df_jtj[select_vars[2] + '_from'] > 0) & (df_jtj[select_vars[2] + '_to'] > 0)].copy()
#Keep only the variables we're interested in
df_m = df_jtj_soc[[select_vars[2] + '_from',select_vars[2] + '_to', 'LGWT18_from']]
# Cross-tabulation to generate weighted adjacency - check for issues with low counts
df_soc_unweighted = pd.crosstab(index=df_m[select_vars[2] + '_from'], columns = df_m[select_vars[2] + '_to'])
df_soc = pd.crosstab(index=df_m[select_vars[2] + '_from'], columns = df_m[select_vars[2] + '_to'], values=df_m['LGWT18_from'], aggfunc='sum')
df_soc[np.isnan(df_soc)] = 0
# Store SOC transition counts
df_soc_unweighted.sum(axis=1).to_csv(f'soc_unweightedtransitionrowcounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
df_soc_unweighted.to_csv(f'soc_unweightedtransitioncounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
# Normalizing based on sum across rows (i.e. out degree from each soc aka each row)
print(f'The lowest out-degree for a soc is: {min(df_soc_unweighted.sum(axis=1))}')

df_soc_density = df_soc.div(df_soc.sum().sum())

# Generate plot of transition matrix
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(df_soc_density, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(df_soc_density.index.tolist())))
ax.set_xticklabels(df_soc_density.index.tolist(), rotation = 90)
ax.set_yticks(np.arange(len(df_soc_density.index.tolist())))
ax.set_yticklabels(df_soc_density.index.tolist())
ax.set_xlabel("Next SOC", fontsize=14)
ax.set_ylabel("Previous SOC", fontsize=14)
ax.set_title("Transition densities for SOC (movers only)", fontsize=16)

#Generate fig with layout
plt.grid(False)
plt.savefig(f'occupation_transitiondensities_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.png', bbox_inches="tight")
plt.show()

# Store soc matrix
df_soc_density.to_csv(f'soc_transitiondensities_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')

### SIC (industry) transition matrix ###

# #Get rid of rows with missing data for sic
df_jtj_sic = df_jtj.loc[(df_jtj[select_vars[1] + '_from'] > 0) & (df_jtj[select_vars[1] + '_to'] > 0)].copy()

#Keep only the variables we're interested in
df_m = df_jtj_sic[[select_vars[1] + '_from',select_vars[1] + '_to', 'LGWT18_from']]
# Cross-tabulation to generate weighted adjacency - check for issues with low counts
df_sic_unweighted = pd.crosstab(index=df_m[select_vars[1] + '_from'], columns = df_m[select_vars[1] + '_to'])
df_sic = pd.crosstab(index=df_m[select_vars[1] + '_from'], columns = df_m[select_vars[1] + '_to'], values=df_m['LGWT18_from'], aggfunc='sum')
df_sic[np.isnan(df_sic)] = 0
# Store SIC transition counts
df_sic_unweighted.sum(axis=1).to_csv(f'sic_unweightedtransitionrowcounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
df_sic_unweighted.to_csv(f'sic_unweightedtransitioncounts_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
# Normalizing based on sum across rows (i.e. out degree from each sic aka each row)
print(f'The lowest out-degree for a sic is: {min(df_sic_unweighted.sum(axis=1))}')

df_sic_density = df_sic.div(df_sic.sum().sum())

# Generate plot of transition matrix
fig=plt.figure(figsize=(6,6))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(df_sic_density, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(df_sic_density.index.tolist())))
ax.set_xticklabels(df_sic_density.index.tolist(), rotation = 90)
ax.set_yticks(np.arange(len(df_sic_density.index.tolist())))
ax.set_yticklabels(df_sic_density.index.tolist())
ax.set_xlabel("Next SIC", fontsize=14)
ax.set_ylabel("Previous SIC", fontsize=14)
ax.set_title("Transition densities for SIC (movers only)", fontsize=16) 

#Generate fig with layout
plt.grid(False)
plt.savefig(f'industry_transitiondensities_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.png', bbox_inches="tight")
plt.show()

# Store sic matrix
df_sic_density.to_csv(f'sic_transitiondensities_empirical_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv') 

# NB: in all cases the output matrix should sum to 1, and we should have an n x n matrix where n is the number of categories for the variable
