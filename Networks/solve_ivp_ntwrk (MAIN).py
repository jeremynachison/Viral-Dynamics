#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:34:31 2022

@author: jeremynachison
"""

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import random

### Within 

def host_ntwrk(G,num_inf_init, T, trans_rate, user_fct, viral_load_ind, uninf_init, inf_init, viral_max):
    """ function that runs a simulation of a within host model across a network.
    Args:
        G (Networkx Graph): Network disease spreads over
        
        num_inf_init (int): Initial number of infected individuals
        
        T (int): Length of time (in days) simulation runs for
        
        trans_rate (float): Between host transmission rate
        
        user_fct (function): User defined within-host differential equations model
        
        viral_load_ind(int): The index of the output of user_fct corresponding to viral load
        
        uninf_init (list): Initial conditions for uninfected individuals
        
        inf_init (list): Initial conditions for infected individuals
        
        viral_max (int or float): Approximate maximum viral load a node can carry
    """
    # set initial conditions for uninfected nodes
    target = np.array([uninf_init])
    # Create uninfected nodes
    for u in G.nodes():
        # create node attributes
        G.nodes[u]["state"] = target
        G.nodes[u]["AgeInfected"] = 0
        G.nodes[u]["Neighbors"] = [n for n in G.neighbors(u)]
        G.nodes[u]["TimeInfected"] = np.nan
        G.nodes[u]["Updated"] = [(0,"Initialized")]
        G.nodes[u]["Exposure"] = np.nan
    # initialize infected nodes
    init = random.sample(list(G.nodes()), num_inf_init)
    # set initial conditions for infected nodes
    infected = np.array([inf_init])
    for u in init:
        # create infected node attributes
        G.nodes[u]["state"] = infected
        G.nodes[u]["AgeInfected"] = 0.1
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["Exposure"] = inf_init[viral_load_ind]
    # running simulation
    dict_check = {} # probability of node u getting infected by its neighbors
    for int_time in range(1,10*T):
        time = int_time / 10
        # sample 10 times per day
        for u in G.nodes:
            # update state of infected nodes first (check if node is infected)
            if (float(G.nodes[u]["state"][int_time - 1,viral_load_ind]) >= float(infected[0,viral_load_ind])):
                # update age of infection and differential equation model
                G.nodes[u]["AgeInfected"] += 0.1
                update = solve_ivp(user_fct, (time-0.1,time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2)).y
                # add new row of solutions to differential equation array
                #A
                G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(update)[-1]))
                G.nodes[u]["Updated"] += [(time,"A")]
        # check contagion after updating all infected nodes
        for u in G.nodes:
            # update state of uninfected nodes second (check if node is uninfected)
            if (float(G.nodes[u]["state"][int_time-1,viral_load_ind]) < float(infected[0,viral_load_ind])):
                # sum total viral load (at time t) across all infected neighbors
                virus_tot = 0
                # count infected neighbors
                inf_neighbors = 0
                for n in G.nodes[u]["Neighbors"]:
                    # check if neighbor infected
                    if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
                        # if neighbor infected, add viral load to virus_tot and add 1 to inf_neighbors
                        virus_tot += G.nodes[n]["state"][-1, viral_load_ind]
                        inf_neighbors += 1
                # probability of node "u" getting infected is determined by the pooled viral load of its neighbors
                # Normalized by global maximum viral load an individual can have
                if inf_neighbors > 0:
                    probability = trans_rate * (virus_tot / (inf_neighbors * viral_max))
                else: 
                    probability = 0
                # probability check (debugging purposes)
                if u in dict_check.keys():
                    dict_check[u] += [probability]
                else:
                    dict_check[u] = [probability]
                if np.random.rand() < probability:
                    # node is exposed to (5e-9)% of average viral load from neighbors
                    exposure = (1e-4,)#(inf_init[viral_load_ind],)
                    inf_update = solve_ivp(user_fct, (time-0.1,time) ,G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2), args = exposure).y
                    # B
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(inf_update)[-1]))
                    G.nodes[u]["TimeInfected"] = time
                    G.nodes[u]["Updated"] += [(time,"B")]
                    G.nodes[u]["Exposure"] = exposure
                # node remains uninfected, add another uninfected row to diff eq array
                else:
                    # C
                    uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2)).y
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(uninf_update)[-1]))
                    G.nodes[u]["Updated"] += [(time,"C")]
    return G, dict_check 





def host_ntwrk_fixed(G,initial_n, T, trans_rate, user_fct, viral_load_ind, uninf_init, inf_init, viral_max):
    """ function that runs a simulation of a within host model across a network.
    This version allows users to specify the initially infected nodes
    Args:
        G (Networkx Graph): Network disease spreads over
        
        initial_n (List): List of Initial infected individuals
        
        T (int): Length of time (in days) simulation runs for
        
        trans_rate (float): Between host transmission rate
        
        user_fct (function): User defined within-host differential equations model
        
        viral_load_ind(int): The index of the output of user_fct corresponding to viral load
        
        uninf_init (list): Initial conditions for uninfected individuals
        
        inf_init (list): Initial conditions for infected individuals
        
        viral_max (int or float): Approximate maximum viral load a node can carry
    """
    # set initial conditions for uninfected nodes
    target = np.array([uninf_init])
    # Create uninfected nodes
    for u in G.nodes():
        # create node attributes
        G.nodes[u]["state"] = target
        G.nodes[u]["AgeInfected"] = 0
        G.nodes[u]["Neighbors"] = [n for n in G.neighbors(u)]
        G.nodes[u]["TimeInfected"] = np.nan
        G.nodes[u]["Updated"] = [(0,"Initialized")]
        G.nodes[u]["Exposure"] = np.nan
    # set initial conditions for infected nodes
    infected = np.array([inf_init])
    # initialize infected nodes
    for u in initial_n:
        # create infected node attributes
        G.nodes[u]["state"] = infected
        G.nodes[u]["AgeInfected"] = 0.1
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["Exposure"] = inf_init[viral_load_ind]
    # running simulation
    dict_check = {} # probability of node u getting infected by its neighbors
    for int_time in range(1,10*T):
        time = int_time / 10
        # sample 10 times per day
        for u in G.nodes:
            # update state of infected nodes first (check if node is infected)
            if (float(G.nodes[u]["state"][int_time - 1,viral_load_ind]) >= float(infected[0,viral_load_ind])):
                # update age of infection and differential equation model
                G.nodes[u]["AgeInfected"] += 0.1
                update = solve_ivp(user_fct, (time-0.1,time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2)).y
                # add new row of solutions to differential equation array
                #A
                G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(update)[-1]))
                G.nodes[u]["Updated"] += [(time,"A")]
        # check contagion after updating all infected nodes
        for u in G.nodes:
            # update state of uninfected nodes second (check if node is uninfected)
            if (float(G.nodes[u]["state"][int_time-1,viral_load_ind]) < float(infected[0,viral_load_ind])):
                # sum total viral load (at time t) across all infected neighbors
                virus_tot = 0
                # count infected neighbors
                inf_neighbors = 0
                for n in G.nodes[u]["Neighbors"]:
                    # check if neighbor infected
                    if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
                        # if neighbor infected, add viral load to virus_tot and add 1 to inf_neighbors
                        virus_tot += G.nodes[n]["state"][-1, viral_load_ind]
                        inf_neighbors += 1
                # probability of node "u" getting infected is determined by the pooled viral load of its neighbors
                # Normalized by global maximum viral load an individual can have
                if inf_neighbors > 0:
                    probability = trans_rate * (virus_tot / (inf_neighbors * viral_max))
                else: 
                    probability = 0
                # probability check (debugging purposes)
                if u in dict_check.keys():
                    dict_check[u] += [probability]
                else:
                    dict_check[u] = [probability]
                if np.random.rand() < probability:
                    # node is exposed to (5e-9)% of average viral load from neighbors
                    exposure = (1e-4,)#(inf_init[viral_load_ind],)
                    inf_update = solve_ivp(user_fct, (time-0.1,time) ,G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2), args = exposure).y
                    # B
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(inf_update)[-1]))
                    G.nodes[u]["TimeInfected"] = time
                    G.nodes[u]["Updated"] += [(time,"B")]
                    G.nodes[u]["Exposure"] = exposure
                # node remains uninfected, add another uninfected row to diff eq array
                else:
                    # C
                    uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2)).y
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(uninf_update)[-1]))
                    G.nodes[u]["Updated"] += [(time,"C")]
    return G, dict_check 


def neighbor_vload(node):
    global G, viral_load_ind
    inf_neighbors = 0
    for n in node["Neighbors"]:
        # check if neighbor infected
        if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
            # if neighbor infected, add viral load to virus_tot and add 1 to inf_neighbors
            virus_tot += G.nodes[n]["state"][-1, viral_load_ind]
            inf_neighbors += 1









