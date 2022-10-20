#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:42:19 2021

Code compiling datasets describing the UK job distribution from UK Labour Force Survey (LFS)

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

#Load necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import glob
from statsmodels.stats.weightstats import DescrStatsW

# Set working directory
home =  os.getcwd()

### Generate the longitudinal dataframe containing industry (SIC), occupation (SOC), geographical region, and wage data

# define desired region, sic, soc variables
select_vars = ['GORWKR', 'Inds07m', 'SC10MMJ']
# Create list of all required variables
vois = ['PERSID', 'AGE1', 'AGE2', 'AGE3', 'AGE4', 'AGE5', 
        select_vars[0] + '1', select_vars[0] + '2', select_vars[0] + '3', select_vars[0] + '4', select_vars[0] + '5', 
        select_vars[2] + '1', select_vars[2] + '2', select_vars[2] + '3', select_vars[2] + '4', select_vars[2] + '5',  
        select_vars[1] + '1', select_vars[1] + '2', select_vars[1] + '3', select_vars[1] + '4', select_vars[1] + '5',
        'INCAC051','INCAC052','INCAC053','INCAC054','INCAC055', 'NETWK1' , 'NETWK2' , 'NETWK3' , 'NETWK4' , 'NETWK5', 'LGWT18']

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
        df0 = tmp[['PERSID', 'AGE' + f'{j}', 'INCAC05' + f'{j}', select_vars[0] + f'{j}', select_vars[2] + f'{j}', select_vars[1] + f'{j}', 'NETWK' + f'{j}', 'LGWT18', 'source']] # subset to specific quarter
        df = df0.loc[(df0[filter_var].isin([1,2]))] # keep only individuals who were employees or self employed

        #Rename columns so we have consistency across all years/quarters (i.e. get rid of character prefixes)
        cnames=df.columns.tolist()
       
        df.rename(columns={cnames[0]:'PERSID', cnames[1]:'AGE', cnames[2]:'INCAC05', cnames[3]:select_vars[0], cnames[4]:select_vars[2], cnames[5]:select_vars[1], cnames[6]:'NETWK'}, inplace=True)
        # Drop column containing info on employment status
        keepvars = ['PERSID', 'AGE', select_vars[0], select_vars[2], select_vars[1], 'NETWK', 'LGWT18', 'source']
        df = df[keepvars]
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
        
#Drop all workers under age 18, as we assume youngest worker is 18 in the model
df_cmb_tot = df_cmb_tot.loc[df_cmb_tot['AGE']>=18].copy()

# Exclude all rows without a complete (region, sic, soc) tuple
dft = df_cmb_tot.loc[(~df_cmb_tot[select_vars[0]].isin([-8,-9,22]) & (df_cmb_tot[select_vars[2]] > 0) & (df_cmb_tot[select_vars[1]] > 0))] 

# Keep only necessary variables
df_fin = dft[[select_vars[0], select_vars[1], select_vars[2],'NETWK', 'LGWT18', 'source', 'year', 'quarter']].copy()
# Set all missing or zero-value income data to nan, set to a float type
df_fin.loc[(df_fin['NETWK']<=0), 'NETWK'] = float("nan")
df_fin.loc[:,'NETWK'] = df_fin['NETWK'].astype(float).values.copy()

#  Adjust wages for inflation
#  Read in UK CPIH (Consumer Prices Index including owner occupiers' housing costs) data (from https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23)
cpih = pd.read_csv('uk_cpih_quarterly.csv',sep=",")
# Merge on year to get (year, quarter)-specific CPIH values
df_fin = df_fin.merge(cpih, left_on=['year', 'quarter'], right_on=['year', 'quarter'],how = 'inner')
# Create deflator variable                      
df_fin['deflator'] = 100/df_fin['CPIH'].copy()
# Calculate inflation-adjusted income
df_fin['income_adj'] = df_fin['NETWK']*df_fin['deflator'].copy()
# Calculate annual inflation-adjusted income
df_fin['annincome_adj'] = 52*df_fin['income_adj'].copy()
                    
# Generate integer values for the non-integer characteristics (region, SIC section)
reg_int = pd.DataFrame(list(zip(sorted(df_fin[select_vars[0]].unique()), np.arange(1,len(df_fin[select_vars[0]].unique())+1).tolist())), columns=[select_vars[0], 'reg_id'])
sic_int = pd.DataFrame(list(zip(sorted(df_fin[select_vars[1]].unique()), np.arange(1,len(df_fin[select_vars[1]].unique())+1).tolist())), columns=[select_vars[1], 'sic_id'])
soc_int = pd.DataFrame(list(zip(sorted(df_fin[select_vars[2]].unique()), np.arange(1,len(df_fin[select_vars[2]].unique())+1).tolist())), columns=[select_vars[2], 'soc_id'])

# Save keys indicating what these integer valyes correspond to
reg_int.to_csv(f'reg_int_key_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
sic_int.to_csv(f'sic_int_key_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')
soc_int.to_csv(f'soc_int_key_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')

#Merge in sic and region integer IDs for use in model
df_fin = df_fin.merge(reg_int, left_on=select_vars[0], right_on=select_vars[0],how = 'inner')
df_fin = df_fin.merge(sic_int, left_on=select_vars[1], right_on=select_vars[1],how = 'inner')
df_fin = df_fin.merge(soc_int, left_on=select_vars[2], right_on=select_vars[2],how = 'inner')

# Count number of soc and sic categories
num_soc = len(df_fin[select_vars[2]].unique())
num_sic = len(df_fin[select_vars[1]].unique()) #len(sic_int)
# Create node IDs based on (region, sic, soc) that will match up with the indices in the similarity matrix used in the model
df_fin['pos_node_ids'] = (df_fin['reg_id']-1)*(num_sic)*(num_soc) + (df_fin['sic_id']-1)*(num_soc) + (df_fin['soc_id'] - 1)

# Generate a copy of the dataframe to use for later income calculations
df_inc = df_fin.copy()

print('This fraction of observations are missing NETWK wage data:')
print(((len(df_inc) - len(df_inc[df_inc.NETWK>0]))/len(df_inc)))

# Check counts for (reg, sic, soc) tuples 
df_fin['counts'] = df_fin.groupby('pos_node_ids')['pos_node_ids'].transform('count')
# Drop all rows corresponding to (reg, sic, soc) tuples with <10 observations
df_out = df_fin.loc[df_fin['counts'] >= 10, ].copy()

print('Removing <10 counts leaves us with this fraction of total observations for jobs:')
print(len(df_out)/len(df_fin))

# Shuffle order of data frame and reset index to prevent matching of this array with other LFS-derived arrays
df_out = df_out.sample(frac=1).reset_index(drop=True).copy()

# Store simplfied (region, sic division, 1-digit soc) distribution, this will be used to generate a weighted version of the job distribution for use in the model
df_out[['reg_id', 'sic_id','soc_id','pos_node_ids', 'counts', 'LGWT18']].to_csv(f'positiondist_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')          
                                             
### Get summary info for income 
# Calculate global mean, standard deviation, min, and max values (min/max rounded to nearest 1,000,000) for income
weighted_stats = DescrStatsW(df_inc.annincome_adj[df_inc.annincome_adj>=0], weights=df_inc.LGWT18[df_inc.annincome_adj>=0], ddof=0)
global_mean = weighted_stats.mean
global_std = weighted_stats.std
global_min = round(df_inc.annincome_adj.min(), -6)
global_max = round(df_inc.annincome_adj.max(), -6)

# Store min/max income info in a dictionary
income_dict = {'income_dict': {'min_annincome': global_min, 'max_annincome': global_max}}

with open(f'income_dict_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.txt', 'w') as f:
    print(income_dict, file=f)

# Get mean and standard deviation values for the wages associated with each node
for i in np.arange(0,len(df_inc.pos_node_ids.unique())):

    tmp0 = df_inc[(df_inc.pos_node_ids==df_inc.pos_node_ids.unique()[i])]
    mean_annincome = DescrStatsW(tmp0.annincome_adj[tmp0.annincome_adj>=0], weights=tmp0.LGWT18[tmp0.annincome_adj>=0], ddof=0).mean
    std_annincome = DescrStatsW(tmp0.annincome_adj[tmp0.annincome_adj>=0], weights=tmp0.LGWT18[tmp0.annincome_adj>=0], ddof=0).std
    counts = len(tmp0[tmp0.annincome_adj>=0])
    
    tmp1 = np.asarray([int(tmp0.pos_node_ids.unique()), int(tmp0.reg_id.unique()), int(tmp0.sic_id.unique()), int(tmp0.soc_id.unique()), mean_annincome, std_annincome, counts])
 
    if i==0:
        df_sum = tmp1
    else:
        df_sum = np.vstack([df_sum, tmp1])

df_sum = pd.DataFrame(df_sum)
cnames=df_sum.columns.tolist()
df_sum.rename(columns={cnames[0]:'pos_node_ids', cnames[1]:'reg_id', cnames[2]:'sic_id', cnames[3]: 'soc_id', \
                   cnames[4]: 'mean_annincome', cnames[5]: 'std_annincome', cnames[6]: 'counts'}, inplace=True)

# check for each node what percentage of observations contain wage data
df_fin_check = df_fin.groupby('pos_node_ids')['pos_node_ids'].agg(counts_all='count')
df_check = df_fin_check.merge(df_sum, left_on='pos_node_ids', right_on='pos_node_ids', how='inner')
df_check['frac_wage_obvs'] = df_check['counts'].values.copy()/df_check['counts_all'].values.copy()

# plot histogram of the fraction of each nodes observations that contain wage data
plt.figure(figsize=(8,8))
df_check.frac_wage_obvs.hist(bins=np.arange(0,1,0.01))
plt.show()

print('Replacing income data for wages where, for a given job nodes wage observations, there are <10 counts means this fraction of nodes incomes have to be set to the global mean/std:')
print(len(df_sum.loc[df_sum.counts<10])/len(df_sum.counts))
print('That means this fraction of positions incomes have to be set to the global mean/std:')
print(df_sum.loc[df_sum.counts<10, 'counts'].sum()/df_sum.counts.sum())

# Replace all mean & std deviation data based on <10 observations with the global mean and standard deviation
df_sum.loc[df_sum['counts']<10, 'mean_annincome'] = global_mean
df_sum.loc[df_sum['counts']<10, 'std_annincome'] = global_std

# add in column to store Kolmogorov-smirnov (KS) test results
df_sum['ks_stat'] = np.nan
df_sum['ks_pval'] = np.nan

# Use KS test to check whether the distribution of synthetic wage observations (generated using a normal distribution based on the previously calculated mean and std dev) resemble the distribution of wage observations in the LFS data
for i in np.arange(0,len(df_sum)):

    df_emp = df_fin.loc[(df_fin.pos_node_ids==df_sum.iloc[i].pos_node_ids)]
    distinfo_emp = df_sum.loc[(df_sum.pos_node_ids==df_sum.iloc[i].pos_node_ids)]

    if distinfo_emp['counts'].values>0:
        
        df_synth = pd.DataFrame(np.minimum(np.maximum(global_min, np.random.normal(loc=distinfo_emp.mean_annincome,
                                scale=distinfo_emp.std_annincome, size=int(distinfo_emp['counts']))),global_max), columns=["annincome_adj"])
        
        #KS test on distributions
        test = stats.ks_2samp(df_emp.annincome_adj[df_emp.annincome_adj>=0], df_synth.annincome_adj)
        df_sum.loc[(df_sum.pos_node_ids==df_sum.iloc[i].pos_node_ids), 'ks_stat'] = test[0]
        df_sum.loc[(df_sum.pos_node_ids==df_sum.iloc[i].pos_node_ids), 'ks_pval'] = test[1]

        
# Replace all KS test data based on <10 observations with nan
df_sum.loc[df_sum['counts']<10, 'ks_stat'] = np.nan
df_sum.loc[df_sum['counts']<10, 'ks_pval'] = np.nan
#  Replace all <10 counts with nan
df_sum.loc[df_sum['counts']<10, 'counts'] = np.nan

# Plot KS test p-values
plt.figure(figsize=(8,8))
df_sum.ks_pval.hist(bins=np.arange(0,1,0.01))
plt.show()

# Plot relationship between p-values and sample size
plt.figure(figsize=(8,8))
plt.plot(df_sum.counts, df_sum.ks_pval, 'o')
plt.xlabel("Count")
plt.ylabel("p-value")
plt.show()
   
## Output
# Set the pos_node_id as the dataframe index
df_sum.set_index('pos_node_ids', inplace=True)

# Store simplfied (reg,sic,soc) grouped wage distribution data, this will be used to generate wages within the model
df_sum[['reg_id', 'sic_id','soc_id', 'mean_annincome', 'std_annincome', 'ks_stat', 'ks_pval' , 'counts']].to_csv(f'incomedist_LFS_{select_vars[0]}_{select_vars[1]}_{select_vars[2]}.csv')