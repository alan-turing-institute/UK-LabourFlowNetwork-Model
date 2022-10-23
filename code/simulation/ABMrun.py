#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:22:57 2021

Code running simulations for the UK labour market model
NB: Throughout we use the acronyms SIC for "Standard Industrial Classification", and SOC for "Standard Occupational Classification"

@author: Kathyrn R Fair

Based on script developed by Áron Pap
"""

### Import necessary packages
import numpy as np
import pandas as pd
import time
import os

# Set working directory
home =  os.getcwd()[:-4]

# Start run timer
t0 = time.perf_counter()

### Initialize our agent classes (workers, positions, statistics office)

def init_workers(N, P, age_dist, cpr_dist, pos_node_ids):
    
    """Initialise vectors containing info on workers
    
    Inputs:
        N = # of workers in model (integer)
        P = # of positions in model (integer)
        age_dist = array containing age data
        cpr_dist = array containing consumption preference data
        pos_node_ids = IDs indicating the node the position is located on (1D numpy array)
    
    Outputs:
        wor_vectors = tuple of 1D arrays characterising workers in model (i.e. containing info on age, wages, etc.)    
    """
    
    ## Vectors related to worker's locations within the system
    # Generate worker IDs
    wor_ids = np.arange(0,N,1)
    # Generate vector of initial job positions for workers (by randomly assigning the N workers to the P positions within the model)
    wor_jobs = np.random.choice(a = range(P), size = N, replace = True)
    # Generate the associated vector of nodes these positions are located on (hence the node the worker is initially located at)
    wor_job_node_ids = pos_node_ids[wor_jobs]
  
    ## Vectors related to worker's characteristics (age, consumption preference)
    # Generate agent ages by sampling empirical distribution
    age_sample = age_dist.sample(n=N,replace=True)
    # Assign age to each workers
    wor_ages = age_sample.to_numpy()
    # Generate agent consumption preferences by sampling empirical distribution
    cpr_sample = cpr_dist.sample(n=N,replace=True)
    # Assign consumption preference value to each worker
    wor_consumption_prefs = cpr_sample.to_numpy()

    # Assign worker wages (all workers are initially unemployed and thus initally have wages of 0)
    wor_wages = np.zeros(N)
    # Assign non-labour income (all set to 0 for current version of model)
    wor_nonlabour_incomes = np.zeros(N)
    # Generate vector of initial employment statuses for workers (0 if unemployed, 1 if employed); we assume all workers are initially unemployed
    wor_employmentstatus = np.zeros(N,dtype=np.int64)
    
    ## Vectors related to data collection
    # Assign length of worker unemployment spells (all initially have unemployment spells of length 0)
    wor_unemp_spells= np.zeros(N,dtype=np.int64)

    # Collect all worker info into a tuple
    wor_vectors = (wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs,
                   wor_wages,wor_nonlabour_incomes,wor_unemp_spells,wor_employmentstatus)
    
    return wor_vectors
    
 
def init_positions(P,pos_dist,inc_dist, wage_min, wage_max,num_sic,num_soc):
    
    """Initializes vectors containing info on positions
    
        Inputs:
            P = # of positions in model (integer)
            pos_dist = array containing observed (region, SIC, SOC) tuples to sample
            inc_dist = array containing summary statistics for observed wage data, grouped by (region, SIC, SOC)
            wage_min = minimum possible annual wage
            wage_max = maximum possible annual wage
            num_sic = number of unique SIC codes
            num_soc = number of unique SOC codes
            
        Outputs:
            pos_vectors = tuple of 1D arrays characterising positions in the model (i.e. containing info on their industry, region, etc.)
    """
        
    ## Vectors related to position's characteristics
    # Generate vector of position statuses (0 = unfilled, 1 = filled), we assume all positions are initially unfilled
    pos_status = np.zeros(P,dtype=np.int64)
    # Generate vector of assigned_worker IDs (all initally set to -1 as positions are initally unfilled and thus have no assigned worker)
    pos_worker_ids = np.ones(P,dtype=np.int64)*(-1)
    # Generate position (region, sic, soc, wage) tuples by sampling empirical distribution
    pos_sample = pos_dist.sample(n=P,replace=True)
    # Assign region, sic, soc to their corresponding vectors
    pos_reg, pos_sic, pos_soc = pos_sample['reg_id'].to_numpy(), pos_sample['sic_id'].to_numpy(), pos_sample['soc_id'].to_numpy()
    
    ## Vectors related to position's locations within the system
    # Generate position IDs
    pos_ids = np.arange(0,P,1)
    # Vector of node IDs based on (region, sic, soc) that will match up with the indices in the similarity matrix (XI)
    pos_node_ids = (pos_reg-1)*(num_sic)*(num_soc) + (pos_sic-1)*(num_soc) + (pos_soc - 1)

    # Generate wages associated with positions based on their (reg, sic, soc)
    # Vector of summary statistics describing wages
    wage_df = pd.merge(inc_dist, pd.DataFrame(pos_node_ids, columns = ["pos_node_ids"]), on="pos_node_ids")
    
    if (wage_df.shape[0]!=len(pos_node_ids)):
        print("Issue with matching wage data")
    
    # Create vectors of mean and standard deviation values for each position (based on region, sic, soc characteristics)
    wage_means =  wage_df.mean_annincome
    wage_stdevs = wage_df.std_annincome
    
    # Generate wages from normal distribution parametrised with the aforementioned mean and standard deviation values
    pos_wages = np.minimum(np.maximum(wage_min, np.random.normal(loc=wage_means,
                            scale=wage_stdevs, size=None)),wage_max)
    
    # Collect all position info into a tuple
    pos_vectors = (pos_ids,pos_node_ids,pos_status,pos_worker_ids,
                   pos_reg,pos_sic,pos_soc,pos_wages)
    
    return pos_vectors

def init_statistical_office(reg_trans_mat,sic_trans_mat,soc_trans_mat,sic,soc,reg):
    
    """Initializes objects for tracking model outputs
    
        Inputs:
            region_trans_mat = matrix of empirical region to region job transitions
            sic_trans_mat = matrix of empirical SIC to SIC job transitions
            soc_trans_mat = matrix of empirical SOC to SOC job transitions
            reg = list of region codes
            sic = list of SIC codes
            soc = list of SOC codes
            
        Outputs:
            statoff_vectors = tuple of objects containing info on model outputs (unemployment rate, transition matrices, etc.)
    """
    
    ### Generate lists for storing output timeseries
    statoff_u_rates = [] # unemployment rate
    statoff_jtj_moves = [] # number of job-to-job transitions per timestep
    statoff_num_vacancies = [] # number of vacancies
    statoff_u_durations = [] # unemployment durations
    statoff_active_searches = [] # number of actively searching workers'
    
    # Generate list to store timeseries for objective function
    statoff_obj_vals = []
    
    # Initialize arrays for storing model-generated transition matrices (for movers only)
    statoff_reg_transition_matrix = np.zeros(reg_trans_mat.shape)
    statoff_sic_transition_matrix = np.zeros(sic_trans_mat.shape)
    statoff_soc_transition_matrix = np.zeros(soc_trans_mat.shape)
    
    # Collect all statistical office info into a tuple
    statoff_vectors = (statoff_u_rates,statoff_u_durations,statoff_jtj_moves,
                       statoff_num_vacancies,statoff_reg_transition_matrix,
                       statoff_sic_transition_matrix,statoff_soc_transition_matrix,
                       statoff_obj_vals,statoff_active_searches)
    
    return statoff_vectors

### Define functions for use within simulation runs

def position_sampling(XI_samp,vac_ids,active_wor_ids,active_wor_node_ids,
                      nodes,counts,sample_size):
    
    """Generates the candidate positions sampled by each active worker (based on similarity to current position)
    
        Inputs:
            XI_samp = subset of similarity matrix pertaining to currently open positions
            vac_ids = IDs of vacant positions
            active_wor_ids = IDs of workers actively seeking a new job ("active workers")
            active_wor_node_ids = IDs of the nodes on which active workers are located
            nodes = list of nodes on which active workers are located
            counts = list of the number of active workers located on each of the aforementioned nodes
            sample_size = number of jobs for each active worker to sample (i.e. be matched with)
            
        Outputs:
            active_wor_ids, wor_sampled_pos = 1D arrays, the first containing the IDs of active workers, with the corresponding entry in the second corresponding to the position that worker was matched with
    """
    
    # Stack ID vectors so that each ID appears [sample_size] times
    active_wor_ids = np.repeat(active_wor_ids,sample_size)
    
    # Initialize array to store the sampled positions
    wor_sampled_pos = np.zeros(len(active_wor_ids), dtype=int)
    n_sample = len(wor_sampled_pos)
    
    # Loop over Region-SIC-SOC triplets
    counter = 0
    group_counter = 0
    
    while counter < n_sample:
        
        # Get node ID (current group)
        group = nodes[group_counter]
        # Get number of actively searching workers in the current group (in positions on this node)
        group_count = counts[group_counter]
        
        # Sample candidate positions based on probabilties conditional on their current node
        # NB: Replacement is theoretically invalid but in practice makes little difference for a sufficiently large sample size
        sample = np.random.choice(a = vac_ids, p = XI_samp[group,:],
                                   size = group_count*sample_size,
                                   replace = True)
                 
        # Store sampled positions
        wor_sampled_pos[counter:(counter+sample_size*group_count)] = sample
         
        # Update counters
        counter += sample_size*group_count
        group_counter += 1
        
    return active_wor_ids, wor_sampled_pos

def switch_calc(gamma,granularity,employmentstatus,ages,wages,nonlabour_incomes,
                consumption_prefs,cand_wages,switch_costs):
    
    """Calculate current/prospective expected utilities + switching costs and check inequality to decide whether to apply to job
    
        Inputs:
            gamma = discounting factor
            granularity = temporal granulairty of model
            employmentstatus = array containing info on the employment status of workers
            ages = array containing info on worker ages
            wages = array containing info on the wage obtained by each worker
            nonlabour_incomes = array containing info on the non-labour incomes obtained by each worker
            consumption_prefs = array containing info on the consumption preferences of each worker
            cand_wages = array containing info on the wages associated with the open position a worker has been matched with ("candidate position")
            switch_costs = array containing info on the switching costs associated with a worker moving from their current position to the candidate position
            
            
        Outputs:
            incentive_comp = 1D array containing TRUE/FALSE values for whether the inequality governing whether a worker applies to the position they've been matched with (based on the expected utility calculation)
    """

    # Initialize vector of utilities for current positions
    curr_utilities = np.zeros(len(employmentstatus))
    
    # Get Boolean vector for currently employed agents
    cond = employmentstatus == 1
    
    # Current expected utility if worker is employed
    part0 = (gamma**(ages[cond]))/(1-gamma)
    part1 = (wages[cond] + nonlabour_incomes[cond])
    part2 = consumption_prefs[cond]**consumption_prefs[cond]
    part3 = ((1-consumption_prefs[cond])/(wages[cond]))**(1-consumption_prefs[cond])
    # Update current expected utilities for employed workers
    curr_utilities[cond] = part0*part1*part2*part3
    
    # Current expected utility if worker is unemployed
    # NB: =0 right now because we assume non-labour income = 0 and don't consider government transfers
    curr_unemployed_utility = ((gamma**(ages[~cond]))/(1-gamma))*\
        ((nonlabour_incomes[~cond])**consumption_prefs[~cond])
    
    # Update current expected utilities for unemployed workers
    curr_utilities[~cond] = curr_unemployed_utility
    
    # Initialize vector of expected utilities for candidate positions
    cand_utilities = np.zeros(len(employmentstatus))

    # Candidate expected utility if worker is hired (and thus employed)
    part0 = (gamma**(ages))/(1-gamma)
    part1 = (cand_wages + nonlabour_incomes)
    part2 = consumption_prefs**consumption_prefs
    part3 = ((1-consumption_prefs)/(cand_wages))**(1-consumption_prefs)
 
    # Update candidate expected utilities
    cand_utilities = part0*part1*part2*part3

    # Check if incentive compatibility holds (resulting in worker applying for candidate position)
    incentive_comp = cand_utilities-switch_costs > curr_utilities
       
    return incentive_comp

def normalized_transitions(matrix):
    
    """Normalize transition matrix
    
        Inputs:
            matrix = the matrix the user wishes to normalize

        Outputs:
            matrix_temp = matrix containing normalized values
    """
    
    matrix_temp = matrix.copy()   
    matrix_temp = matrix_temp/matrix_temp.sum().sum() #Normalize by dividing all cells by largest cell
    
    return matrix_temp

def survival(worker_survival_rates, wor_ages, granularity):
    
    
    """Simulate worker death (and thus determine which workers survive to next timestep)
    
        Inputs:
            worker_survival_rates = 1D array containing age-specific survival rates for each worker
            wor_ages = 1D array containing worker ages
            granularity = temporal granularity of model
            
        Outputs:
            wor_survived = 1D array containing TRUE/FALSE values for whether each worker survived to the next age (or died, and thus needs to be replaced by a worker of the youngest age)
    """
    
    # Get number of workers
    n = len(wor_ages)
    
    # Convert workers' ages to integers
    wor_ages = np.asarray(wor_ages,dtype=np.int64)

    # For survival calculations, assume all workers with age > 100 have age 100 as this is highest age for which we have a reported survival rate
    wor_ages[wor_ages > 100] = 100
    
    # Get age-specific surivival rates for each worker
    wor_survival_rates = worker_survival_rates[wor_ages]

    # Convert survival rate from annual to match the model's current granularity
    wor_survival_rates = wor_survival_rates**(1/granularity)

    # Perform Bernoulli trials to determine which workers survive
    wor_survived = np.random.uniform(low=0,high=1,size=n) < wor_survival_rates
    
    return wor_survived



def run_iteration(t,input_data_dict):
    
    """Run an iteration of the model (i.e. simulate a single timestep within the model)
        
        Inputs:
            t = integer corresponding to the previous timestep simulated using run_iteration()
            input_data_dict = dictionary containing model parameters
            
        Outputs:
            t, ss_dif = integer corresponding to the timestep simulated using run_iteration(), value used to determine steady state convergence
            
    """ 
    
    # Unpack input dictionary
    N = input_data_dict['N']
    P = input_data_dict['P']
    PD = input_data_dict['PD']
    PC = input_data_dict['PC']
    S = input_data_dict['S']
    granularity = input_data_dict['granularity']
    new_worker_init_age = input_data_dict['new_worker_init_age']
    worker_survival_rates = input_data_dict['worker_survival_rates']
    activation_rate_unemployed = input_data_dict['activation_rate_unemployed']
    activation_rate_employed = input_data_dict['activation_rate_employed']
    sample_size = input_data_dict['sample_size']
    node_dict = input_data_dict['node_dict']
    gamma = input_data_dict['gamma']
    wage_max = input_data_dict['wage_max']
    wage_min = input_data_dict['wage_min']
    lag = input_data_dict['lag']
    avg_length = input_data_dict['avg_length']
    t_ss = input_data_dict['t_ss']
    num_reg = input_data_dict['num_reg']
    num_sic = input_data_dict['num_sic']
    num_soc = input_data_dict['num_soc']
    pos_dist = input_data_dict['pos_dist']
    inc_dist = input_data_dict['inc_dist']
    age_dist = input_data_dict['age_dist']
    cpr_dist = input_data_dict['cpr_dist']
    reg_trans_mat = input_data_dict['reg_trans_mat']
    sic_trans_mat = input_data_dict['sic_trans_mat'] 
    soc_trans_mat = input_data_dict['soc_trans_mat']
    XI = input_data_dict['XI']
    pos_ids = input_data_dict['pos_ids']
    pos_status = input_data_dict['pos_status']
    pos_worker_ids = input_data_dict['pos_worker_ids']
    pos_node_ids = input_data_dict['pos_node_ids']
    pos_reg = input_data_dict['pos_reg']
    pos_sic = input_data_dict['pos_sic']
    pos_soc = input_data_dict['pos_soc']
    pos_wages = input_data_dict['pos_wages']
    wor_ids = input_data_dict['wor_ids']
    wor_employmentstatus = input_data_dict['wor_employmentstatus']
    wor_unemp_spells = input_data_dict['wor_unemp_spells']
    wor_jobs = input_data_dict['wor_jobs']
    wor_job_node_ids = input_data_dict['wor_job_node_ids']
    wor_ages = input_data_dict['wor_ages']
    wor_consumption_prefs = input_data_dict['wor_consumption_prefs']
    wor_nonlabour_incomes = input_data_dict['wor_nonlabour_incomes']
    wor_wages = input_data_dict['wor_wages']
    statoff_u_rates = input_data_dict['statoff_u_rates']
    statoff_u_durations = input_data_dict['statoff_u_durations']
    statoff_jtj_moves = input_data_dict['statoff_jtj_moves']
    statoff_num_vacancies = input_data_dict['statoff_num_vacancies']
    statoff_reg_transition_matrix = input_data_dict['statoff_reg_transition_matrix']
    statoff_sic_transition_matrix = input_data_dict['statoff_sic_transition_matrix']
    statoff_soc_transition_matrix = input_data_dict['statoff_soc_transition_matrix']
    statoff_obj_vals = input_data_dict['statoff_obj_vals']
    statoff_active_searches = input_data_dict['statoff_active_searches']
    
    ### Update unemployment info (wor_employmentstatus = 1 if employed, = 0 if unemployed)
    # Increment length of most recently unemployment spell based on current employment status
    wor_unemp_spells += (1-wor_employmentstatus)
    
    ### Update aging
    # Increment depends on model granularity (in relation to a year), so for weekly timesteps granularity = 52, etc.
    wor_ages += float(1/granularity)
  
    ### Destroy jobs
    # Randomly select positions to destroy
    idx = np.random.permutation(np.arange(P))[:PD]
    # Get ID-s of workers whose positions were destroyed
    new_unemployed_wor_ids = pos_worker_ids[idx]
    
    ### Update worker vectors with newly unemployed workers
    # Filter for destroyed positions which were occupied by workers
    cond = new_unemployed_wor_ids[(new_unemployed_wor_ids != -1)]
    # Update vectors
    wor_employmentstatus[cond] = 0
    wor_unemp_spells[cond] = 0
    wor_wages[cond] = 0
    
    ### Create jobs
    # Create new positions to replace those destroyed
    new_pos = init_positions(PC,pos_dist, inc_dist, wage_min, wage_max, num_sic,num_soc)
    # Unpack new position vectors 
    new_pos_ids,new_pos_node_ids,new_pos_status,new_pos_worker_ids, \
    new_pos_reg,new_pos_sic,new_pos_soc,new_pos_wages = new_pos   

    ### Update position vectors with newly created positions
    pos_status[idx] = new_pos_status
    pos_worker_ids[idx] = new_pos_worker_ids
    pos_reg[idx] = new_pos_reg
    pos_sic[idx] = new_pos_sic
    pos_soc[idx] = new_pos_soc
    pos_wages[idx] = new_pos_wages
    pos_node_ids[idx] = new_pos_node_ids

    ### Update position vacancy info
    vac_ids = pos_ids[pos_status==0]
    vac_node_ids = pos_node_ids[pos_status==0]
    # Collect vacancy statistics
    statoff_num_vacancies.append(len(vac_ids))

    ### Generate sampling probabilities based on similarity between current and candidate nodes in terms of (region, SIC, SOC)
    # Store columns from similarity matrix (XI) corresponding to nodes with vacant positions
    XI_samp = XI[:,vac_node_ids]
    # Renormalize to get sampling probabilities
    XI_samp = XI_samp / XI_samp.sum(axis=1)[:, np.newaxis]
    
    if (XI_samp.sum(axis=1) > 1+1e-10).any():
        print("Issue with sampling probabilities exceeding 1")
        print(XI_samp.sum(axis=1))
        print(XI_samp)

    ### Get actively searching workers 
    # Initialize array
    search_probs = np.zeros(N)
    # Get Boolean for employed workers
    cond = wor_employmentstatus == 1
    # Compute the activation probability
    search_probs[~cond] = activation_rate_unemployed # Unemployed workers become active (i.e. actively seek a position this timestep) based on the unemployed activation rate
    search_probs[cond] = activation_rate_employed # Employed workers become active (i.e. actively seek a new position this timestep) based on the employed activation rate
    # Bernoulli trial for active job-seeking
    wor_activations = np.random.uniform(low=0.0, high=1.0,size=len(search_probs)) < search_probs
    
    # Collect active worker statistics
    statoff_active_searches.append(np.sum(wor_activations))
    # Get ID-s of active workers and the nodes their postions are located on
    active_wor_ids = wor_ids[wor_activations]
    active_wor_node_ids = wor_job_node_ids[wor_activations]
    
    # Sort arrays according to node ID
    active_wor_ids = active_wor_ids[np.argsort(active_wor_node_ids)]
    active_wor_node_ids = active_wor_node_ids[np.argsort(active_wor_node_ids)]
    
    # Get groups of nodes and their number of occurences
    nodes, counts = np.unique(active_wor_node_ids,return_counts = True)
    
    # Sample open positions (based on similarity to current position)
    active_wor_ids, wor_sampled_pos = position_sampling(XI_samp=XI_samp,vac_ids=vac_ids,
                                    active_wor_ids=active_wor_ids,
                                    active_wor_node_ids=active_wor_node_ids,
                                    nodes=nodes,counts=counts,
                                    sample_size=sample_size)

    ### Collect info on active workers, their current position, and their candidate position(s)
    active_wor_wages = wor_wages[active_wor_ids]
    active_wor_employmentstatus = wor_employmentstatus[active_wor_ids]
    active_wor_ages = wor_ages[active_wor_ids]
    active_wor_nonlabour_incomes = wor_nonlabour_incomes[active_wor_ids]
    active_wor_consumption_prefs = wor_consumption_prefs[active_wor_ids]
    cand_wages = pos_wages[wor_sampled_pos]
    
    # Collect all switching costs (currently assume these = 0)
    switch_costs = 0
    
    ### Get position similarity scores
    # Update node IDs for obtaining position similarity scores
    active_wor_node_ids = np.repeat(active_wor_node_ids,sample_size)
    cand_pos_node_ids = pos_node_ids[wor_sampled_pos]
    # Store position similarity scores
    pos_sim_scores = S[active_wor_node_ids,cand_pos_node_ids]

    ### Calculate lifetime utilities in current and candidate positions, and check if incentive compatibility holds
    active_wor_incentive_comps = switch_calc(gamma=gamma,granularity=granularity,
                                    employmentstatus=active_wor_employmentstatus,
                                    ages=active_wor_ages,wages=active_wor_wages,
                                    nonlabour_incomes=active_wor_nonlabour_incomes,
                                    consumption_prefs=active_wor_consumption_prefs,
                                    cand_wages=cand_wages, switch_costs = switch_costs)
        
    ### Filter for workers for whom the incentive compatibility holds (who will apply to candidate positions)
    active_wor_ids = active_wor_ids[active_wor_incentive_comps]
    wor_sampled_pos = wor_sampled_pos[active_wor_incentive_comps]
    active_wor_node_ids = active_wor_node_ids[active_wor_incentive_comps]
    cand_pos_node_ids = cand_pos_node_ids[active_wor_incentive_comps]  
    pos_sim_scores = pos_sim_scores[active_wor_incentive_comps]
    
    ### Rank the strength of these worker's applications
    # Create random submission order
    submission_order = np.random.permutation(len(active_wor_ids))
    # Create full ranking array 
    ranking_array = np.asarray([pos_sim_scores,submission_order]).T
   
    # Sort arrays in descending order (by similarity, break ties using submission order)
    active_wor_ids = active_wor_ids[np.lexsort((ranking_array[:,1],ranking_array[:,0]))][::-1] 
    wor_sampled_pos = wor_sampled_pos[np.lexsort((ranking_array[:,1],ranking_array[:,0]))][::-1]
    
    ### Filter to find best candidate (worker) for each position and first offer of a position for each worker
    # Get best candidate for each position 
    wor_sampled_pos, wor_sampled_pos_ids = np.unique(wor_sampled_pos, return_index=True)
    # Filter workers based on best candidates for each vacant position
    active_wor_ids = active_wor_ids[wor_sampled_pos_ids]
                
    # Get first offer of a position for each worker
    active_wor_ids, active_wor_ids_ix = np.unique(active_wor_ids, return_index=True)                
    # Filter positions based on first offers (to get final list of successful job-to-job transitions)                
    wor_sampled_pos = wor_sampled_pos[active_wor_ids_ix]
                
    # Collect number of job-to-job transitions for statistics office
    statoff_jtj_moves.append(len(wor_sampled_pos))
    # Collect additions to transition matrices for statistics office
    prev_nodes = wor_job_node_ids[active_wor_ids]
    next_nodes = pos_node_ids[wor_sampled_pos]
    for i in range(0,len(prev_nodes)):
        statoff_reg_transition_matrix[node_dict[prev_nodes[i]][0]-1,\
         node_dict[next_nodes[i]][0]-1] += 1  #Update regional transition matrix                                 
        statoff_sic_transition_matrix[node_dict[prev_nodes[i]][1]-1,\
         node_dict[next_nodes[i]][1]-1] += 1 #Update SIC transition matrix
        statoff_soc_transition_matrix[node_dict[prev_nodes[i]][2]-1,\
         node_dict[next_nodes[i]][2]-1] += 1 #Update SOC transition matrix

    ### Update info on vacancies, employment following job transitions
    # Vacancy updates
    pos_status[wor_sampled_pos] = 1
    pos_worker_ids[wor_sampled_pos] = active_wor_ids 
    # Updates relating to active workers who were already employed
    cond = wor_employmentstatus[active_wor_ids] == 1
    employed_active_wor_ids = active_wor_ids[cond]
    unemployed_active_wor_ids = active_wor_ids[~cond]
    prev_jobs = wor_jobs[employed_active_wor_ids]
    pos_status[prev_jobs] = 0
    pos_worker_ids[prev_jobs] = -1
    # Collect unemployment durations for previously unemplpyed workers who were hired this time-step for the statistics office
    statoff_u_durations += list(wor_unemp_spells[unemployed_active_wor_ids])
    # Updates to all active workers
    wor_jobs[active_wor_ids] = wor_sampled_pos
    wor_wages[active_wor_ids] = pos_wages[wor_sampled_pos]
    wor_job_node_ids[active_wor_ids] = pos_node_ids[wor_sampled_pos]
    wor_employmentstatus[active_wor_ids] = 1
    
    ### Worker turnover (deaths and replacements)
    # Bernoulli trial for survival
    wor_survived = survival(worker_survival_rates, wor_ages, granularity)

    # Get the ID-s of dead workers
    wor_dead = wor_ids[~wor_survived]
                
    # Update positions corresponding to dead workers to be vacant
    employed_wor_dead = wor_ids[(wor_employmentstatus == 1) & (~wor_survived)]
    prev_jobs = wor_jobs[employed_wor_dead]
    pos_status[prev_jobs] = 0
    pos_worker_ids[prev_jobs] = -1
    ## Generate new workers to replace the dead ones
    new_wor_vectors = init_workers(len(wor_dead), P, age_dist, cpr_dist, pos_node_ids)
    # Unpack new worker vectors
    new_wor_ids,new_wor_jobs,new_wor_job_node_ids,new_wor_ages,new_wor_consumption_prefs,\
    new_wor_wages,new_wor_nonlabour_incomes,new_wor_unemp_spells,new_wor_employmentstatus = new_wor_vectors
    # Assign ages of new workers (default should be the min age)
    new_wor_ages = np.ones(len(wor_dead))*new_worker_init_age
    # Update worker vectors with the new workers' characteristics 
    # NB: do not update wor_id, as this will create duplicate wor_id values because of how init_workers function assigns them
    wor_jobs[wor_dead] = new_wor_jobs
    wor_job_node_ids[wor_dead] = new_wor_job_node_ids
    wor_ages[wor_dead] = new_wor_ages
    wor_consumption_prefs[wor_dead] = new_wor_consumption_prefs
    wor_wages[wor_dead] = new_wor_wages
    wor_nonlabour_incomes[wor_dead] = new_wor_nonlabour_incomes
    wor_unemp_spells[wor_dead] = new_wor_unemp_spells
    wor_employmentstatus[wor_dead] = new_wor_employmentstatus

    ### Collect additional unemployment info for statistics office
    statoff_u_rates.append(1-np.nanmean(wor_employmentstatus)) #unemployment rate
    
    ### Calculate objective function
    # Normalize simulated transition matrices
    simulated_reg = normalized_transitions(statoff_reg_transition_matrix)
    simulated_sic = normalized_transitions(statoff_sic_transition_matrix)
    simulated_soc = normalized_transitions(statoff_soc_transition_matrix)

    # Compute region/SIC/SOC components of objective value, calculated as Frobenius norm of difference between empirical and simulated transition matrices
    reg_val = np.linalg.norm(reg_trans_mat.flatten()-simulated_reg.flatten())
    sic_val = np.linalg.norm(sic_trans_mat.flatten()-simulated_sic.flatten())
    soc_val = np.linalg.norm(soc_trans_mat.flatten()-simulated_soc.flatten())
    
    #Compute combined objective value (currently equally weight region, SIC, SOC)
    obj_val = (1/3)*(reg_val + sic_val + soc_val)
    
    #Store objective value with statistics office
    statoff_obj_vals.append(obj_val)
    
    ### Calculate steady state convergence criteria
    if (t-t_ss)>(lag+avg_length):
        ss_dif = abs(np.mean(statoff_obj_vals[-avg_length:])\
                -np.mean(statoff_obj_vals[-avg_length-lag:-lag]))
        
    ### Increment time
    t += 1
    
    ### Return time and steady state convergence criteria
    if (t-t_ss-1)>(lag+avg_length):
        return t, ss_dif
    else:
        return t, 1

def run_simulation(input_data_dict):
    
    """Run a simulation to the point where the steady state has been reached
    
    
        Inputs:
            input_data_dict = dictionary containing model parameters
            
        Outputs:
            Tuple containing outputs at end of simulation
            
    """
    
    # Unpack input dictionary
    N = input_data_dict['N']
    P = input_data_dict['P']
    PD = input_data_dict['PD']
    PC = input_data_dict['PC']
    granularity = input_data_dict['granularity']
    new_worker_init_age = input_data_dict['new_worker_init_age']
    worker_survival_rates = input_data_dict['worker_survival_rates'] 
    activation_rate_unemployed = input_data_dict['activation_rate_unemployed']
    activation_rate_employed = input_data_dict['activation_rate_employed']
    sample_size = input_data_dict['sample_size']
    node_dict = input_data_dict['node_dict']
    n = input_data_dict['n']
    gamma = input_data_dict['gamma']
    node_reg_sim_mat = input_data_dict['node_reg_sim_mat'] 
    node_sic_sim_mat = input_data_dict['node_sic_sim_mat']
    node_soc_sim_mat = input_data_dict['node_soc_sim_mat']
    reg = input_data_dict['reg']
    sic = input_data_dict['sic']
    soc = input_data_dict['soc']
    wage_max = input_data_dict['wage_max']
    wage_min = input_data_dict['wage_min']
    ss_threshold = input_data_dict['ss_threshold']
    lag = input_data_dict['lag']
    avg_length = input_data_dict['avg_length']
    t_ss = input_data_dict['t_ss']
    num_reg = input_data_dict['num_reg']
    num_sic = input_data_dict['num_sic']
    num_soc = input_data_dict['num_soc']
    pos_dist = input_data_dict['pos_dist']
    inc_dist = input_data_dict['inc_dist']
    age_dist = input_data_dict['age_dist']
    cpr_dist = input_data_dict['cpr_dist']
    reg_trans_mat = input_data_dict['reg_trans_mat']
    sic_trans_mat = input_data_dict['sic_trans_mat']
    soc_trans_mat = input_data_dict['soc_trans_mat']
    
    ### Peform calculations for similarity, simiarlity-based position sampling
    # NB: could pre-compute S for experiments (i.e. once nu-values are calibrated), but do not currently do this
    S = (node_reg_sim_mat)*(node_sic_sim_mat)*(node_soc_sim_mat)

    # Generate simularity matrix for use in calculations, adding small positive value to account for zero-entries in S
    XI=S + 1e-10
    
    ### Assign parameters for tracking iterations
    t = 0 #Initialize timestep
    ss_dif = 1 # Initialize steady-state tracking metric (to a value above the threshold)
    
    ### Set up similarity data

    ### Initialize model environment
    # Initialize position vectors
    pos_vectors_0 = init_positions(P,pos_dist,inc_dist, wage_min, wage_max, num_sic,num_soc)
    # Unpack position vectors 
    pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
    pos_reg,pos_sic,pos_soc,pos_wages = pos_vectors_0    
    # Initialize worker vectors
    wor_vectors_0 = init_workers(N, P, age_dist, cpr_dist, pos_node_ids)    
    # Unpack worker vectors
    wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus = wor_vectors_0
    # Initialize statistical office vectors
    statoff_vectors_0 = init_statistical_office(reg_trans_mat,sic_trans_mat,
                                                soc_trans_mat,sic,soc,reg)
    # Unpack statistical office vectors
    statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches = statoff_vectors_0
                    
    ### Generate input dictionary for use within each iteration (timestep)
    with open('%sdata/build_dict.txt' % home, 'r') as file:
        data = file.read()    
    exec(data)
    # Update input dictionary to include objects generated by this function
    input_data_dict['S'] = S
    input_data_dict['XI'] = XI
    input_data_dict['pos_ids'] = pos_ids
    input_data_dict['pos_node_ids'] = pos_node_ids
    input_data_dict['pos_status'] = pos_status
    input_data_dict['pos_worker_ids'] = pos_worker_ids
    input_data_dict['pos_reg'] = pos_reg
    input_data_dict['pos_sic'] = pos_sic
    input_data_dict['pos_soc'] = pos_soc
    input_data_dict['pos_wages'] = pos_wages
    input_data_dict['wor_ids'] = wor_ids
    input_data_dict['wor_employmentstatus'] = wor_employmentstatus
    input_data_dict['wor_unemp_spells'] = wor_unemp_spells
    input_data_dict['wor_jobs'] = wor_jobs
    input_data_dict['wor_job_node_ids'] = wor_job_node_ids
    input_data_dict['wor_ages'] = wor_ages
    input_data_dict['wor_consumption_prefs'] = wor_consumption_prefs
    input_data_dict['wor_nonlabour_incomes'] = wor_nonlabour_incomes
    input_data_dict['wor_wages'] = wor_wages
    input_data_dict['statoff_u_rates'] = statoff_u_rates
    input_data_dict['statoff_u_durations'] = statoff_u_durations
    input_data_dict['statoff_jtj_moves'] = statoff_jtj_moves
    input_data_dict['statoff_num_vacancies'] = statoff_num_vacancies
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    input_data_dict['statoff_obj_vals'] = statoff_obj_vals
    input_data_dict['statoff_active_searches'] = statoff_active_searches
       
    ### Run successive iterations of model until the simulation reaches a steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)

    print(f"It took {t} timesteps to reach the steady state.")
    
    ### Collect and aggregate outputs (just spit out our initialization for now)
    sim_output = (wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus, \
                   pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
                   pos_reg,pos_sic,pos_soc,pos_wages, \
                   statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches)
    
    return sim_output


def extended_run_simulation(input_data_dict):
    
    """Run extended (original run to steady state + second period to capture steady state LFNs) simulation
    
        Inputs:
            input_data_dict = dictionary containing model parameters
            
        Outputs:
            Tuple containing outputs at end of simulation
    
    """
    
    # Unpack input dictionary
    N = input_data_dict['N']
    P = input_data_dict['P']
    PD = input_data_dict['PD']
    PC = input_data_dict['PC']
    granularity = input_data_dict['granularity']
    new_worker_init_age = input_data_dict['new_worker_init_age']
    worker_survival_rates = input_data_dict['worker_survival_rates'] 
    activation_rate_unemployed = input_data_dict['activation_rate_unemployed']
    activation_rate_employed = input_data_dict['activation_rate_employed']
    sample_size = input_data_dict['sample_size']
    node_dict = input_data_dict['node_dict']
    n = input_data_dict['n']
    gamma = input_data_dict['gamma']
    node_reg_sim_mat = input_data_dict['node_reg_sim_mat'] 
    node_sic_sim_mat = input_data_dict['node_sic_sim_mat']
    node_soc_sim_mat = input_data_dict['node_soc_sim_mat']
    reg = input_data_dict['reg']
    sic = input_data_dict['sic']
    soc = input_data_dict['soc']
    wage_max = input_data_dict['wage_max']
    wage_min = input_data_dict['wage_min']
    ss_threshold = input_data_dict['ss_threshold']
    lag = input_data_dict['lag']
    avg_length = input_data_dict['avg_length']
    t_ss = input_data_dict['t_ss']
    num_reg = input_data_dict['num_reg']
    num_sic = input_data_dict['num_sic']
    num_soc = input_data_dict['num_soc']
    pos_dist = input_data_dict['pos_dist']
    inc_dist = input_data_dict['inc_dist']
    age_dist = input_data_dict['age_dist']
    cpr_dist = input_data_dict['cpr_dist']
    reg_trans_mat = input_data_dict['reg_trans_mat']
    sic_trans_mat = input_data_dict['sic_trans_mat']
    soc_trans_mat = input_data_dict['soc_trans_mat']

    ### Peform calculations for similarity, simiarlity-based position sampling
    # NB: could pre-compute S for experiments (i.e. once nu-values are calibrated), but do not currently do this
    S = (node_reg_sim_mat)*(node_sic_sim_mat)*(node_soc_sim_mat)

    # Generate simularity matrix for use in calculations, adding small positive value to account for zero-entries in S
    XI=S + 1e-10
  
    ### Assign parameters for tracking iterations
    t = 0 #Initialize timestep
    ss_dif = 1 # Initialize steady-state tracking metric (to a value above the threshold)
    
    ### Set up similarity data

    ### Initialize model environment
    # Initialize position vectors
    pos_vectors_0 = init_positions(P,pos_dist, inc_dist, wage_min, wage_max,num_sic,num_soc)
    # Unpack position vectors 
    pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
    pos_reg,pos_sic,pos_soc,pos_wages = pos_vectors_0    
    # Initialize worker vectors
    wor_vectors_0 = init_workers(N, P, age_dist, cpr_dist, pos_node_ids)    
    # Unpack worker vectors
    wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus = wor_vectors_0
    # Initialize statistical office vectors
    statoff_vectors_0 = init_statistical_office(reg_trans_mat,sic_trans_mat,
                                                soc_trans_mat,sic,soc,reg)
    # Unpack statistical office vectors
    statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches = statoff_vectors_0
                    
    
    ### Generate input dictionary for use within each iteration (timestep)
    with open('%sdata/build_dict.txt' % home, 'r') as file:
        data = file.read()    
    exec(data)
    # Update input dictionary to include objects generated by this function
    input_data_dict['S'] = S
    input_data_dict['XI'] = XI
    input_data_dict['pos_ids'] = pos_ids
    input_data_dict['pos_node_ids'] = pos_node_ids
    input_data_dict['pos_status'] = pos_status
    input_data_dict['pos_worker_ids'] = pos_worker_ids
    input_data_dict['pos_reg'] = pos_reg
    input_data_dict['pos_sic'] = pos_sic
    input_data_dict['pos_soc'] = pos_soc
    input_data_dict['pos_wages'] = pos_wages
    input_data_dict['wor_ids'] = wor_ids
    input_data_dict['wor_employmentstatus'] = wor_employmentstatus
    input_data_dict['wor_unemp_spells'] = wor_unemp_spells
    input_data_dict['wor_jobs'] = wor_jobs
    input_data_dict['wor_job_node_ids'] = wor_job_node_ids
    input_data_dict['wor_ages'] = wor_ages
    input_data_dict['wor_consumption_prefs'] = wor_consumption_prefs
    input_data_dict['wor_nonlabour_incomes'] = wor_nonlabour_incomes
    input_data_dict['wor_wages'] = wor_wages
    input_data_dict['statoff_u_rates'] = statoff_u_rates
    input_data_dict['statoff_u_durations'] = statoff_u_durations
    input_data_dict['statoff_jtj_moves'] = statoff_jtj_moves
    input_data_dict['statoff_num_vacancies'] = statoff_num_vacancies
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    input_data_dict['statoff_obj_vals'] = statoff_obj_vals
    input_data_dict['statoff_active_searches'] = statoff_active_searches
       
    ### Run successive iterations of model until the simulation reaches a steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)
    
    t_ss = t
    input_data_dict['t_ss'] = t_ss
    
    print(f"It took {t_ss} timesteps to reach the steady state.")
           
    # Once the steady-state is reached, clear the data on flows and continue
    # running the model until this new set of flows stabilises (i.e. a new steady state is reached), collect the 
    # objective function value from this second time period only
    
    # Re-initialize the data storage location for the model's transition matrices
    statoff_reg_transition_matrix = np.zeros(reg_trans_mat.shape)
    statoff_sic_transition_matrix = np.zeros(sic_trans_mat.shape)
    statoff_soc_transition_matrix = np.zeros(soc_trans_mat.shape)
    
    # Update data dictionary to match with agent info at end of initial stage of sim
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    
    ss_dif = 1 # Re-initialize steady-state tracking metric (to a value above the threshold)
    
    ### Run successive iterations of model until the simulation once again reaches a steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)
    
    print(f"It took {t-t_ss} additional timesteps for the new flows to stabilise.")
    
    ### Collect and aggregate outputs (just spit out our initialization for now)
    sim_output = (wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus, \
                   pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
                   pos_reg,pos_sic,pos_soc,pos_wages, \
                   statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches)
    
    return sim_output

def shock_run_simulation(shock_type, impacted_industries, input_data_dict):
       
    """Run shock (original run to steady state + second period to capture steady state LFNs + shock + third period to capture new post-shock LFNs) simulation
    
        Inputs:
            shock_type = type of shock to perform; position ("position"), wage increase ("wageincr"), wage decrease ("wagedecr")
            impacted_industries = array containing information on which industries are impacted by the shock
            input_data_dict = dictionary containing model parameters
            
        Outputs:
            Tuple containing outputs at end of simulation
    
    """
    
    # Unpack input dictionary
    N = input_data_dict['N']
    P = input_data_dict['P']
    PD = input_data_dict['PD']
    PC = input_data_dict['PC']
    granularity = input_data_dict['granularity']
    new_worker_init_age = input_data_dict['new_worker_init_age']
    worker_survival_rates = input_data_dict['worker_survival_rates'] 
    activation_rate_unemployed = input_data_dict['activation_rate_unemployed']
    activation_rate_employed = input_data_dict['activation_rate_employed']
    sample_size = input_data_dict['sample_size']
    node_dict = input_data_dict['node_dict']
    n = input_data_dict['n']
    gamma = input_data_dict['gamma']
    node_reg_sim_mat = input_data_dict['node_reg_sim_mat'] 
    node_sic_sim_mat = input_data_dict['node_sic_sim_mat']
    node_soc_sim_mat = input_data_dict['node_soc_sim_mat']
    reg = input_data_dict['reg']
    sic = input_data_dict['sic']
    soc = input_data_dict['soc']
    wage_max = input_data_dict['wage_max']
    wage_min = input_data_dict['wage_min']
    ss_threshold = input_data_dict['ss_threshold']
    lag = input_data_dict['lag']
    avg_length = input_data_dict['avg_length']
    t_ss = input_data_dict['t_ss']
    num_reg = input_data_dict['num_reg']
    num_sic = input_data_dict['num_sic']
    num_soc = input_data_dict['num_soc']
    pos_dist = input_data_dict['pos_dist']
    inc_dist = input_data_dict['inc_dist']
    age_dist = input_data_dict['age_dist']
    cpr_dist = input_data_dict['cpr_dist']
    reg_trans_mat = input_data_dict['reg_trans_mat']
    sic_trans_mat = input_data_dict['sic_trans_mat']
    soc_trans_mat = input_data_dict['soc_trans_mat']

    ### Peform calculations for similarity, simiarlity-based position sampling
    # NB: could pre-compute S for experiments (i.e. once nu-values are calibrated), but do not currently do this
    S = (node_reg_sim_mat)*(node_sic_sim_mat)*(node_soc_sim_mat)

    # Generate simularity matrix for use in calculations, adding small positive value to account for zero-entries in S
    XI=S + 1e-10
  
    ### Assign parameters for tracking iterations
    t = 0 #Initialize timestep
    ss_dif = 1 # Initialize steady-state tracking metric (to a value above the threshold)
    
    ### Set up similarity data

    ### Initialize model environment
    # Initialize position vectors
    pos_vectors_0 = init_positions(P,pos_dist, inc_dist, wage_min, wage_max,num_sic,num_soc)
    # Unpack position vectors 
    pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
    pos_reg,pos_sic,pos_soc,pos_wages = pos_vectors_0    
    # Initialize worker vectors
    wor_vectors_0 = init_workers(N, P, age_dist, cpr_dist, pos_node_ids)    
    # Unpack worker vectors
    wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus = wor_vectors_0
    # Initialize statistical office vectors
    statoff_vectors_0 = init_statistical_office(reg_trans_mat,sic_trans_mat,
                                                soc_trans_mat,sic,soc,reg)
    # Unpack statistical office vectors
    statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches = statoff_vectors_0
                    
    
    ### Generate input dictionary for use within each iteration (timestep)
    with open('%sdata/build_dict.txt' % home, 'r') as file:
        data = file.read()    
    exec(data)
    # Update input dictionary to include objects generated by this function
    input_data_dict['S'] = S
    input_data_dict['XI'] = XI
    input_data_dict['pos_ids'] = pos_ids
    input_data_dict['pos_node_ids'] = pos_node_ids
    input_data_dict['pos_status'] = pos_status
    input_data_dict['pos_worker_ids'] = pos_worker_ids
    input_data_dict['pos_reg'] = pos_reg
    input_data_dict['pos_sic'] = pos_sic
    input_data_dict['pos_soc'] = pos_soc
    input_data_dict['pos_wages'] = pos_wages
    input_data_dict['wor_ids'] = wor_ids
    input_data_dict['wor_employmentstatus'] = wor_employmentstatus
    input_data_dict['wor_unemp_spells'] = wor_unemp_spells
    input_data_dict['wor_jobs'] = wor_jobs
    input_data_dict['wor_job_node_ids'] = wor_job_node_ids
    input_data_dict['wor_ages'] = wor_ages
    input_data_dict['wor_consumption_prefs'] = wor_consumption_prefs
    input_data_dict['wor_nonlabour_incomes'] = wor_nonlabour_incomes
    input_data_dict['wor_wages'] = wor_wages
    input_data_dict['statoff_u_rates'] = statoff_u_rates
    input_data_dict['statoff_u_durations'] = statoff_u_durations
    input_data_dict['statoff_jtj_moves'] = statoff_jtj_moves
    input_data_dict['statoff_num_vacancies'] = statoff_num_vacancies
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    input_data_dict['statoff_obj_vals'] = statoff_obj_vals
    input_data_dict['statoff_active_searches'] = statoff_active_searches

    ### Run successive iterations of model until the simulation reaches a steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)
    
    t_ss = t
    input_data_dict['t_ss'] = t_ss
    
    print(f"It took {t_ss} timesteps to reach the steady state.")
           
    # Once the steady-state is reached, clear the data on flows and continue
    # running the model until this new set of flows stabilises (i.e. a new steady state is reached), collect the 
    # objective function value from this second time period only
    
    # Re-initialize the data storage location for the model's transition matrices
    statoff_reg_transition_matrix = np.zeros(reg_trans_mat.shape)
    statoff_sic_transition_matrix = np.zeros(sic_trans_mat.shape)
    statoff_soc_transition_matrix = np.zeros(soc_trans_mat.shape)
    
    # Update data dictionary to match with agent info at end of initial stage of sim
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    
    ss_dif = 1 # Re-initialize steady-state tracking metric (to a value above the threshold)
    
    ### Run successive iterations of model until the simulation reaches a new steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)
        
    print(f"It took {t-t_ss} additional timesteps for the new flows to stabilise.")
    
    t_ss = t
    input_data_dict['t_ss'] = t_ss
    
    # Once the steady-state flows stabilise, clear the data on flows, shock the system, 
    # and continue running the model until this new set of flows stabilises (i.e. a new steady state is reached), 
    # collect the objective function value from this third time period only
    
    # Re-initialize the data storage location for the model's transition matrices
    statoff_reg_transition_matrix = np.zeros(reg_trans_mat.shape)
    statoff_sic_transition_matrix = np.zeros(sic_trans_mat.shape)
    statoff_soc_transition_matrix = np.zeros(soc_trans_mat.shape)
    
    # Update data dictionary to match with agent info at end of initial stage of sim
    input_data_dict['statoff_reg_transition_matrix'] = statoff_reg_transition_matrix
    input_data_dict['statoff_sic_transition_matrix'] = statoff_sic_transition_matrix
    input_data_dict['statoff_soc_transition_matrix'] = statoff_soc_transition_matrix
    
    # Implement shock
    if shock_type=="wagedecr":
        
        # Perform adjustment to wage data (shift mean wage values down by 2 standard deviations)
        inc_dist_homogenised = inc_dist.copy()
        inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'mean_annincome'] = inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'mean_annincome'] - 2*inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'std_annincome'] 
        input_data_dict['inc_dist'] = inc_dist_homogenised
        
    # Implement shock
    if shock_type=="wageincr":
        
        # Perform adjustment to wage data (shift mean wage values up by 2 standard deviations)
        inc_dist_homogenised = inc_dist.copy()
        inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'mean_annincome'] = inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'mean_annincome'] + 2*inc_dist_homogenised.loc[inc_dist_homogenised.sic_id.isin(impacted_industries),'std_annincome'] 
        input_data_dict['inc_dist'] = inc_dist_homogenised
    
    if shock_type == "position":
        
        # Perform adjustment to position data
        pos_dist_homogenised = pos_dist.copy()
        
        for i in np.arange(0,len(impacted_industries)):
            
            selected_industry=impacted_industries[i]
        
            # Get information on the occupations and regions associated with that industry
            df_select = pos_dist[pos_dist.sic_id==(selected_industry)].copy()
            pairs_unique = df_select[['soc_id','reg_id']].drop_duplicates()
            
            # Perform homogenisation (i.e. replace SOC and region pair with a SOC and region pair randomly drawn from all possible such pairs within the industry)
            pairs_sample = pairs_unique.sample(df_select.shape[0], replace=True)
            pos_dist_homogenised.loc[pos_dist_homogenised.sic_id==(selected_industry),'soc_id'] = np.array(pairs_sample['soc_id'])
            pos_dist_homogenised.loc[pos_dist_homogenised.sic_id==(selected_industry),'reg_id'] = np.array(pairs_sample['reg_id'])

        input_data_dict['pos_dist'] = pos_dist_homogenised
        
    ss_dif = 1 # Re-initialize steady-state tracking metric (to a value above the threshold)
    
    ### Run successive iterations of model until the simulation reaches a post-shock steady state
    while (ss_dif>ss_threshold):    
        # Run an iteration of the model
        t,ss_dif = run_iteration(t,input_data_dict)
    
    print(f"It took {t-t_ss} additional timesteps for the shocked flows to stabilise.")
    
    ### Collect and aggregate outputs (just spit out our initialization for now)
    sim_output = (wor_ids,wor_jobs,wor_job_node_ids,wor_ages,wor_consumption_prefs, \
                   wor_wages,wor_nonlabour_incomes, \
                   wor_unemp_spells,wor_employmentstatus, \
                   pos_ids,pos_node_ids,pos_status,pos_worker_ids, \
                   pos_reg,pos_sic,pos_soc,pos_wages, \
                   statoff_u_rates,statoff_u_durations,statoff_jtj_moves, \
                    statoff_num_vacancies,statoff_reg_transition_matrix, \
                    statoff_sic_transition_matrix,statoff_soc_transition_matrix, \
                    statoff_obj_vals,statoff_active_searches,impacted_industries)
    
    return sim_output
