#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 8 09:37:50 2022

@author: jeremynachison
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

########################### INITIAL CONDITIONS ################################

samps = 2000

N = 6e4
i01, v_0  = 1, 1e-5
t0 = N
e0 = 0
init1 = (t0,i01,v_0,e0)

wt = 0.5
# run simulation for one year
days = 365
spd = 10

t1 = np.linspace(0,days,spd*days)
v0_vect = np.zeros(7001)

params_dict = {"delta" : 0.8, 
               "p": 4.5e2, "c": 5e-2, "beta":2e-5,
               "d":0.065, "dX":1e-4, "r":0.01, "alpha":1.2}

############################## STORING THE DATA ################################

# Create a dataframe storing each particle
# Particles are assigned an id
# Particles consist of a combination of each parameter randomly smapled within 10% of original value
param_df = pd.DataFrame()
param_df["id"] = pd.Series(range(0,samps)).astype(str)
for param in params_dict.keys():
    param_df[param] = np.random.uniform(0.95*params_dict[param],1.05*params_dict[param],samps)
    

# Create a list of T, I, V, E repeating for the number of samples (for multi-index)
tive_letters = ["T","I","V","E"]
tive_list = []
for i in range(samps):
    tive_list += tive_letters
 
# create an array of ids, with each id repeating 4 times (to store T, I, V, E solutions for each particle)
# This is for multi-index purposes
id_array_temp = np.array(pd.Series(range(0,samps)).astype(str))
id_array = np.repeat(id_array_temp,4)

# For multi index, store ids, and repeating tive in a list
arrays = [id_array, np.array(tive_list)]
# Create a multi-indexed dataframe using the list for multi-indexed columns above
# The Multi-index works as follows:
    # The first level of the multi-index is particle id
    # for each particle id there are 4 sub columns storing solutions to TIVE equation of corresponding particle
tive_multi = pd.DataFrame(np.zeros((len(t1), 4*samps)), columns=arrays)
# Add in a column to keep track of what times the equation is evaluating
tive_multi.insert(0,"time",t1)
# Label the levels of the multi indexed column
tive_multi.columns = tive_multi.columns.rename("Particle ID", level=0)
tive_multi.columns = tive_multi.columns.rename("TIVE Sols", level=1)

# Create a dataframe storing the sample statistics for each particle
    # Sample stats include: 
        #number of infection cycles (how many times does num infected cells cross 100 from above)
        # average length pf infection: if all infection cycles are fully cleared by the end of simulation, 
        # what is average duration of infection cycle
sample_stats = pd.DataFrame()
sample_stats["id"] = pd.Series(range(0,samps)).astype(str)
sample_stats["infection_cycles"] = np.zeros(samps)
sample_stats["avg_infection_length"] = np.full((samps,1), np.nan)

# Define equation of within-host dynamics (TIVE Model)
def tive(time, x, trigger=0, beta=0, alpha=0,delta=0,dX=0,p=0,c=0,r=0,d=0):
    t,i,v,e = x
    time_int = int((np.round(time,2) * spd))
    dx = np.zeros(4)
    if (v >= v_0):
        dx[0] = - beta * v * t + alpha*t*(1-(t + i)/N) 
        dx[1] = (beta * v * t - delta * i - dX * i * e ) 
        dx[2] = p * i - c * v * e
        dx[3] = r*i - d*e
    else:
        dx[0] = alpha*t*(1-(t + i)/N)
        dx[1] = 0 
        dx[2] = 0 + trigger + v0_vect[time_int]
        dx[3] = r*i - d*e
    return dx   

# For each particle id, evaluate the tive model with the particle's parameters
for col_id in param_df["id"]:
    row = param_df.loc[param_df["id"] == col_id].index.item()
    beta = param_df.loc[row,"beta"]
    alpha = param_df.loc[row,"alpha"]
    delta = param_df.loc[row,"delta"]
    dX = param_df.loc[row,"dX"]
    p = param_df.loc[row,"p"]
    c = param_df.loc[row,"c"]
    r = param_df.loc[row,"r"]
    d = param_df.loc[row,"d"]
    # create a vector storing the external triggering of infection from outside sources
    # host is exposed once a day to infection
    v0_vect = np.zeros(spd*days+1001)
    for i in range(spd*days+1001):
        if (i % spd == 0) and (i != 0):
            # exposure amount (percentage of initial viral load) is a parameter, wt
            v0_vect[i] = wt*v_0
        else:
            v0_vect[i] = 0
    # solve tive equation for given particle       
    sol = solve_ivp(tive, (0,days), init1, t_eval = t1, args=(0,beta,alpha,delta,dX,p,c,r,d)).y
    y = np.transpose(sol)
    # store solutions for T, I, V, E curves in respective column for given ID
    tive_multi[col_id,"T"] = y[:,0]
    tive_multi[col_id,"I"] = y[:,1]
    # create logical vector checking when infected cells are under 100
    under100 = tive_multi[col_id,"I"] < 100
    # store indices when infected cells transition from below 100 to over 100
    beg_infcycle = under100.index[under100.shift(fill_value=True) & ~under100]
    #  store indices when infected cells transition from over 100 to below 100
    end_infcycle = under100.index[~under100.shift(fill_value=True) & under100]
    # length of end_infcycle will count the numer of inction cycles
    sample_stats.loc[sample_stats["id"] == col_id, "infection_cycles"] =  len(end_infcycle)
    # if infection all infection cycles are cleared by the end of simulation, take average duration of infection cycles (in days)
    if len(beg_infcycle) == len(end_infcycle):
        sample_stats.loc[sample_stats["id"] == col_id, "avg_infection_length"] = np.mean((end_infcycle - beg_infcycle )/10)
    tive_multi[col_id,"V"] = y[:,2]
    tive_multi[col_id,"E"] = y[:,3]

############################### EXPORTING THE DATA ############################

# writing the multi_indexed dataframe
compression_opts = dict(method='zip',
                        archive_name='tive_multi.csv') 
tive_multi.to_csv('dense_tive.zip',compression=compression_opts)

# writing the particle dataframe
compression_opts2 = dict(method='zip',
                        archive_name='param_df.csv')
param_df.to_csv("dense_param.zip",compression=compression_opts2)

# writing the sample statistics dataframe
compression_opts3 = dict(method='zip',
                        archive_name='sample_stats.csv')
sample_stats.to_csv("dense_stats.zip",compression=compression_opts3)



