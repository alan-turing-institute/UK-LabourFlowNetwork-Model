"""
Created on Thu Oct  7 14:40:32 2021

Code to calculate the similarity between industries, based on industry-level input-output table (IOT)
NB: Throughout we use the acronym SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

#Load in required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Set working directory
home =  os.getcwd()[:-4]

# Choose the variables of interest
regvar = "GORWKR" #geographical region 
sicvar = "Inds07m" #industry (SIC)
socvar = "SC10MMJ" #occupation (SOC)

# Read in IOT
df_iot = pd.read_excel('%sdata/IOT_2018.xlsx' % home, sheet_name = 'IOT', header=None)

# Generate matrix where entry (i,j) is the value of inputs to industry i from industry j, as a fraction of all inputs to i
df_frac = (df_iot.T/df_iot.sum(axis=1)).T

# Some industries do not have any stated inputs (so the sum across the row corresponding to that SIC will = 0, introducing divide by 0 issue above), these rows we set entirely to 0 
df_frac.loc[df_frac.index[df_frac.isnull().all(1)]]=0

# Generate plot of similarity matrix (based on fraction of inputs)
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(df_frac, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xlabel("SIC from (output)", fontsize=14)
ax.set_ylabel("SIC to (input)", fontsize=14)
ax.set_title("SIC similarties (fraction of inputs)", fontsize=16)

### Create aggregated version of similarities at sic section level (alpha-groupings, equivalent to NACE rev.2 A21) using fractional values (df_fracnorm)

# Read in sic mapping codes and labels
df_conversion = pd.read_excel('%sdata/IOT_2018.xlsx' % home, sheet_name = 'conversion_key')
df_labels = pd.read_excel('%sdata/IOT_2018.xlsx' % home, sheet_name = 'label_key')

#Add in dummy row/col of zeros for group U, which exists in the codes but is not present in the iot data
df_frac = df_frac.append(pd.Series(0, index=df_frac.columns), ignore_index=True) #Dummy row
df_frac[len(df_frac)+1] = float(0) #Dummy col

# Map df_fracnorm values to the SIC code/section they correspond to
data = []

n,d = df_frac.shape

for i in range(0,n):
    for j in range (0,d):
        data.append([df_conversion.iloc[i,0], df_conversion.iloc[i,1], df_conversion.iloc[j,0], df_conversion.iloc[j,1], df_frac.iloc[i,j]])
        
df_ann = pd.DataFrame(data, columns=['to_A21','to_iotlabel',  'from_A21', 'from_iotlabel', 'value'])

# Aggregate values based on the SIC sections (alph-values)
df_agg = df_ann.groupby(['from_A21', 'to_A21']).sum().reset_index()

# Pivot to create a aggregate similarity matrix
df_agg = df_agg.pivot(index='from_A21', columns='to_A21', values='value')

# Normalize to span [0,1]
df_agg = (df_agg - df_agg.min().min())/(df_agg.max().max() - df_agg.min().min())

# Generate labels for plot
labs = df_labels.short_label

# Generate plot of similarity matrix (based on fraction of inputs)
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(df_agg, cmap='magma_r')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(labs)))
ax.set_xticklabels(labs, rotation = 90)
ax.set_yticks(np.arange(len(labs)))
ax.set_yticklabels(labs)
ax.set_xlabel("SIC section to", fontsize=14)
ax.set_ylabel("SIC section from", fontsize=14)
ax.set_title("SIC similarties (aggregated to section-level)", fontsize=16)

# Load empirical SIC transition probability matrix
tmat = pd.read_csv(open(f'{home}data/20220520 KF PrePub 2001646/sic_transitiondensities_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv', 'rb'), header=0,index_col=0)

#Compare empirical transition density matrix with similarity matrix scaled to have the same maximum value as that contained in the transition matrix
print(np.corrcoef(np.asarray(tmat).flatten(),np.asarray(tmat.max().max()*df_agg).flatten()))
# Frobeneius norm
print(np.linalg.norm(np.asarray(tmat).flatten()-np.asarray(tmat.max().max()*df_agg).flatten()))

# Store SIC similarity matrix (converting to a numpy array for use with ABM script)
pickle.dump(df_agg.to_numpy(), open('%sdata/sic_similaritymat_LFS.sav' % home, 'wb'))

df_agg.to_csv('%sdata/sic_similaritymat_LFS.csv' % home)