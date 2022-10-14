#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:11:08 2021

Code to get O*NET (Occupational Information Network) skills data associated with occupations
NB: Throughout we use the acronyms SIC for "Standard Industrial Classification", and SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""
#Load in required libraries
import requests
import json
import pickle
import os
import pandas as pd

# Set working directory
home =  os.getcwd()[:-4]

#Define function that obtains mean skill values for a given SOC
def aggregator(list_of_dicts):

    # Get list length to use when normalizing
    list_len = len(list_of_dicts)

    # Initialize global dictionary to store skill values for all O*NET codes corresponding to a given SOC code
    global_dict = {}
    
    # Aggregate skill values across all O*NET codes; take sum of values (normalized by list_len) to get a mean skill value
    for selected_dict in list_of_dicts:
        for key, value in selected_dict.items():
            # If skill is already in global dictionary, add the current value
            if key in global_dict.keys():
                global_dict[key] += value/list_len
            # Else create new dictionary entry
            else:
                global_dict[key] = value/list_len

    return global_dict

#Define function that creates a vector of skills corresponding to a given SOC
def skills_soc(soc_4digit, scales):

    # Get SOC to O*NET crosswalk
    dat = json.loads(requests.get('http://api.lmiforall.org.uk/api/v1/o-net/soc2onet/{}'.format(soc_4digit), headers={'Accept': 'application/json',}).text)

    # Get O-NET codes corresponding to the provided SOC code
    onet_codes = []
    for i in dat["onetCodes"]:
        onet_codes.append(i['code'])

    # Initialize list to store dictionaries of normalized skill values
    values = []

    for onet_code in onet_codes:

        # Get skill data for an O-NET code
        dat = json.loads(requests.get('http://api.lmiforall.org.uk/api/v1/o-net/skills/{}'.format(onet_code), headers={'Accept': 'application/json',}).text)
        
        # Get the skill levels and normalizers for the O-NET code
        for j in dat["scales"]:

            # Initialize temporary dictionary
            temp_dict = {}

            # Get measurement ID
            selected_id = j['id']

            # Get dictionary of skills using that measurement
            associated_skills = j['skills']
        
            # Get the corresponding scales for that measurement
            for scale in scales:
                if scale['id']==selected_id: # If there is a pre-defined scale
                    normalizer = scale['max']-scale['min']
                    break
                else:
                    normalizer = 1


            for selected_skill in associated_skills:

                # Get ID of current skill
                skill_id = selected_skill['id']

                # Store normalized value of skill
                temp_dict[skill_id] = selected_skill['value']/normalizer
                
                if normalizer!=7:
                    print(normalizer)

            # Append dictionary of normalized skill values to list
            values.append(temp_dict)

    # Aggregate to obtain mean skill values for a given SOC
    mean_skills = aggregator(values)

    return mean_skills

# Choose the variables of interest
regvar = "GORWKR" #geographical region 
sicvar = "Insc07m" #industry (SIC)
socvar = "SOC10M" #occupation (SOC)

# Load list of 4-digit SOCs
df = pd.read_csv(open(f'{home}data/soc_int_key_LFS_{regvar}_{sicvar}_{socvar}.csv', 'rb'), header=0,index_col=0)
soc_4digit_list = df['SOC10M'].tolist()

# Get measurement scales
scales = json.loads(requests.get('http://api.lmiforall.org.uk/api/v1/o-net/scales', headers={'Accept': 'application/json',}).text)

# Initialize list to store the results for all SOCs
soc_skill_list = []

# Get skills info
for current_soc in soc_4digit_list:

    temp_dict = {}

    # Get skill values
    mean_skill_vals = skills_soc(current_soc, scales)

    # Create nested dictionary structure
    temp_dict[current_soc] = mean_skill_vals

    # Append it to the global list
    soc_skill_list.append(temp_dict)

# Store list
pickle.dump(soc_skill_list, open('%sdata/soc_skill_list_LFS.sav' % home, 'wb'))
