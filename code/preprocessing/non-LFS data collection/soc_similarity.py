#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:40:26 2021

Code to calculate the similarity between occupations, based on the skills associated with each occupation
NB: Throughout we use the acronym SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

#Load in required libraries
from scipy import spatial
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory
home =  os.getcwd()[:-4]

# Choose the variables of interest
regvar = "GORWKR" #geographical region 
sicvar = "Inds07m" #industry (SIC)
socvar = "SC10MMJ" #occupation (SOC)

# Define function that calculates cosine similarity between all possible pairs of SOC skill vectors
def soc_sim_matrix(df): 
    
    # Get number of SOCs
    d = df.shape[0]
    
    # Initialize matrix to store the results
    sim_matrix = np.ones((d,d))

    for i in range(0,d):
        for j in range(i+1,d):
            
            # Calculate cosine similarity (1 - cosine distance)
            sim_val = 1 - spatial.distance.cosine(np.asarray(df.iloc[i]),np.asarray(df.iloc[j]))
            
            # Add similarity score to similarity matrix
            sim_matrix[i,j] = sim_val
            sim_matrix[j,i] = sim_val
       
    return sim_matrix

# Load in list with skill vector data for SOC codes
soc_skill_list = pickle.load(open('%sdata/soc_skill_list_LFS.sav' % home, 'rb'))

# Reshape to pandas data frame
tmp = []
for skvec in soc_skill_list:
    for soc, soc_entry in skvec.items():
        soc_entry['SOC'] = soc
        soc_entry['SC10MMJ'] = str(soc)[0]
    if len(soc_entry)>2:
        tmp.append(soc_entry)
        
soc_skill_df = pd.DataFrame.from_dict(tmp, orient='columns')
skill_df = soc_skill_df.drop(['SOC', 'SC10MMJ'],axis=1)

# Replace NaN skill entries with zeros for cosine similarity calculations
skill_df = skill_df.fillna(0)

# Calculate SOC similarity based on cosine similarity of skill vectors
sim_mat = soc_sim_matrix(skill_df)
    
# Generate plot of similarity matrix
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(sim_mat, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xlabel("SOC", fontsize=14)
ax.set_ylabel("SOC", fontsize=14)
ax.set_title("SOC similarties", fontsize=16)

plt.grid(False)
plt.show()

#Create aggregated version of similarity matrix
skill_df = soc_skill_df.drop(['SOC'],axis=1).groupby(['SC10MMJ']).mean()

# Replace NaN skill entries with zeros for cosine similarity calculations
skill_df = skill_df.fillna(0)

# Calculate SOC similarity based on cosine similarity of skill vectors
sim_mat = soc_sim_matrix(skill_df)

# Normalize to span [0,1]
sim_mat = (sim_mat - sim_mat.min().min())/(sim_mat.max().max() - sim_mat.min().min())

# Generate plot of similarity matrix
fig=plt.figure(figsize=(15,15))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(sim_mat, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(skill_df.index.tolist())))
ax.set_xticklabels(skill_df.index.tolist(), rotation = 90)
ax.set_yticks(np.arange(len(skill_df.index.tolist())))
ax.set_yticklabels(skill_df.index.tolist())
ax.set_xlabel("SOC", fontsize=14)
ax.set_ylabel("SOC", fontsize=14)
ax.set_title("SOC similarties (simplfied to SOC major groups - 1 digits)", fontsize=16)

plt.show()

# Load empirical SOC transition probability matrix
tmat = pd.read_csv(open(f'{home}data/soc_transitiondensities_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv', 'rb'), header=0,index_col=0)

#Compare empirical transition density matrix with similarity matrix scaled to have the same maximum value as that contained in the transition matrix
#Pearson correlation
print(np.corrcoef(np.asarray(tmat).flatten(),np.asarray(tmat.max().max()*sim_mat).flatten()))
# Frobeneius norm
print(np.linalg.norm(np.asarray(tmat).flatten()-np.asarray(tmat.max().max()*sim_mat).flatten()))

# Store SOC similarity matrix
pickle.dump(sim_mat, open('%sdata/soc_similaritymat_LFS.sav' % home, 'wb'))
pd.DataFrame(sim_mat).to_csv('%sdata/soc_similaritymat_LFS.csv' % home)
