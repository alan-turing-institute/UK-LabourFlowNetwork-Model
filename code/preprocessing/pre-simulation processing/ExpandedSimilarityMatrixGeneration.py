#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:18:18 2021

Code to generate expanded (i.e. node-level) similarity matrices for use in UK labour market model 
Requires as inputs the similarity matrices generated using "region_similarlity.py", "sic_similarlity.py", "soc_similarlity.py"
NB: Throughout we use the acronyms SIC for "Standard Industrial Classification", and SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

import numpy as np
import pickle
import itertools
import os

# Set working directory
home =  os.getcwd()[:-4]

### Generate similarity matrices for determining similarity between positions based on their (region, sic, soc)

# Read in base similarity matrices
filename = f'{home}data/region_similaritymat_LFS.sav'
reg_sim_base = pickle.load(open(filename, 'rb')) # Region (geographical) similarity
filename = f'{home}data/sic_similaritymat_LFS.sav'
sic_sim_base = pickle.load(open(filename, 'rb')) # SIC (industry) similarity
filename = f'{home}data/soc_similaritymat_LFS.sav'
soc_sim_base = pickle.load(open(filename, 'rb')) # SOC (occupation) similarity

# Create list of arrays containing all possible values of the integers associated with the regions, SIC sections, and 1-digit SOC codes
iterables = [np.arange(1,reg_sim_base.shape[0]+1), np.arange(1, sic_sim_base.shape[0]+1), np.arange(1, soc_sim_base.shape[0]+1)]

# Generate all possible combinations of these (region, SIC, SOC) integers (each corresponding to a potential node)
combos = list(itertools.product(*iterables))

# Create dictionary of (region, SIC, SOC) IDs for these nodes, with associated integer index values
node_dict = {}
for i in range(0,len(combos)):
    node_dict[i] = combos[i] #Key is the index, value is the ID 
# Initialize expanded matrices covering all possible combinations of (reg, sic, soc)
n = len(node_dict.keys())
node_reg_sim_mat = np.zeros((n,n))
node_sic_sim_mat = np.zeros((n,n))
node_soc_sim_mat = np.zeros((n,n))
# Loop over nodes and compute the values
for i in range(0,n):
    for j in range(0,n):
        node_reg_sim_mat[i,j] = reg_sim_base[node_dict[i][0]-1,node_dict[j][0]-1]
        node_sic_sim_mat[i,j] = sic_sim_base[node_dict[i][1]-1,node_dict[j][1]-1]
        node_soc_sim_mat[i,j] = soc_sim_base[node_dict[i][2]-1,node_dict[j][2]-1]    
        
# Store expanded similarity matrices
filename = f'{home}data/reg_expanded_similaritymat_LFS.sav'
pickle.dump(node_reg_sim_mat, open(filename, 'wb'))
filename = f'{home}data/sic_expanded_similaritymat_LFS.sav'
pickle.dump(node_sic_sim_mat, open(filename, 'wb'))        
filename = f'{home}data/soc_expanded_similaritymat_LFS.sav'
pickle.dump(node_soc_sim_mat, open(filename, 'wb'))