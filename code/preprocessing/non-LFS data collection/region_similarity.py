#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 09:40:54 2021

Code to calculate the similarity between geographical regions based on distance

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

#Load in required libraries
from geopy.geocoders import Nominatim
from geopy import distance
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

# Set working directory
home =  os.getcwd()[:-4]

# Choose the variables of interest
regvar = "GORWKR" #geographical region 
sicvar = "Inds07m" #industry (SIC)
socvar = "SC10MMJ" #occupation (SOC)

# Read in list of regions and associated population centers
df_regions = pd.read_excel(f'{home}data/RegionKey_GORWKR_LFS.xlsx', sheet_name = 'LFSGORWKRRegions')

# Convert back to lists post-sort
centerlist = df_regions['center'].tolist()
regionlist = df_regions['region'].tolist()

### Get (latitute, longitude) pairs for all centers

#Specify header for API requests to OpenStreetMap
geolocator = Nominatim(user_agent="llgttr_OSM")

#Define function that gets longitude, latitute data for a center
def llgttr(center):
    
    location = geolocator.geocode(center)
    return (location.latitude, location.longitude)

#Apply function to centers
lat_lon = list(map(llgttr, centerlist))

### Calculate distances between centers

# Initialize 2D array for storing distances
n = len(centerlist)
dist_mat = np.zeros((n,n))

# Store distances for all possible combinations of centers
for i in range(0,n):
    for j in range(i+1,n):

        loc1 = lat_lon[i]
        loc2 = lat_lon[j]
        gc_dist = distance.great_circle(loc1,loc2).km
        dist_mat[i,j] = dist_mat[j,i] = gc_dist
        
### Calculate similarity between regions based on distance between them
sim_mat = 1 - (dist_mat - dist_mat.min().min())/(dist_mat.max().max() - dist_mat.min().min())

    
# Generate plot of similarity matrix
fig=plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1])
sp=ax.imshow(sim_mat, cmap='YlOrRd')
fig.colorbar(sp, ax=ax)
ax.set_xticks(np.arange(len(centerlist)))
ax.set_xticklabels(regionlist, rotation = 90)
ax.set_yticks(np.arange(len(centerlist)))
ax.set_yticklabels(regionlist)
ax.set_xlabel("Region", fontsize=14)
ax.set_ylabel("Region", fontsize=14)
ax.set_title("Region similarties", fontsize=16)

plt.grid(False)
plt.show()


# Load empirical region transition probability matrix
tmat = pd.read_csv(open(f'{home}data/region_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv', 'rb'), header=0,index_col=0)

#Compare empirical transition density matrix with similarity matrix scaled to have the same maximum value as that contained in the transition matrix
#Pearson correlation (should match above pearson corr value - just a sense check)
print(np.corrcoef(np.asarray(tmat).flatten(),np.asarray(tmat.max().max()*sim_mat).flatten()))
# Frobeneius norm
print(np.linalg.norm(np.asarray(tmat).flatten()-np.asarray(tmat.max().max()*sim_mat).flatten()))

# Store region similarity matrix
pickle.dump(sim_mat, open('%sdata/region_similaritymat_LFS.sav' % home, 'wb'))

pd.DataFrame(sim_mat).to_csv('%sdata/region_similaritymat_LFS.csv' % home)
