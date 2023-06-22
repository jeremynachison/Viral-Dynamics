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

def inf_prob(x, viral_max):
    """ Calculates the probability of infection given viral load and an 
    estimated maximum viral load
    
    x (float): neighborhood viral load
    viral_max (float): approximate maximum viral load that can be achieved by an individual"""
    return 1/(1+(np.e)**((-2/viral_max)*(x-viral_max)))

def host_ntwrk(G,num_inf_init, T, trans_rate, user_fct, 
               viral_load_ind, uninf_init, inf_init, viral_max, spd, Init_Nodes = None):
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
        
        spd (int): samples per day. How many times a day are the epidemic dynamics updated.
        
        Init_Nodes (None or List): If a list of ints (correspondning to 
        individual nodes) is given, these nodes in the network will be seeded 
        with virus in the network. If None, random nodes in the network will be 
        selected for initial seeding
        
        Returns: Netorkx graph containing all nodes with updated within-host data
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
    # initialize infected nodes, either randomized or as passed by the Init_Nodes argument
    if Init_Nodes == None:
        init = random.sample(list(G.nodes()), num_inf_init)
    else:
        init = Init_Nodes
    # set initial conditions for infected nodes
    infected = np.array([inf_init])
    for u in init:
        # create infected node attributes
        G.nodes[u]["state"] = infected
        G.nodes[u]["AgeInfected"] = 1/spd
        G.nodes[u]["TimeInfected"] = 0
    # running simulation
    for int_time in range(1,spd*T):
        time = int_time / spd
        # sample spd times per day
        for u in G.nodes:
            # update state of infected nodes first (check if node is infected)
            if (float(G.nodes[u]["state"][int_time - 1,viral_load_ind]) >= float(infected[0,viral_load_ind])):
                # update age of infection and differential equation model
                G.nodes[u]["AgeInfected"] += (1/spd)
                update = solve_ivp(user_fct, (time-(1/spd),time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-(1/spd),time,2)).y
                # add new row of solutions to differential equation array
                G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(update)[-1]))
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
                if inf_neighbors > 0:
                    probability = trans_rate * inf_prob(virus_tot, viral_max)
                else: 
                    probability = 0
                if np.random.rand() < probability:
                    # node is exposed to constant viral load, regardless of neighborhood vload
                    exposure = (1e-5,) #(inf_init[viral_load_ind],)
                    inf_update = solve_ivp(user_fct, (time-0.1,time) ,G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2), args = exposure).y
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(inf_update)[-1]))
                    G.nodes[u]["TimeInfected"] = time
                    G.nodes[u]["Exposure"] = exposure
                # node remains uninfected, add another uninfected row to diff eq array
                else:
                    uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[u]["state"][-1], t_eval = np.linspace(time-0.1,time,2)).y
                    G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(uninf_update)[-1]))
    return G 









