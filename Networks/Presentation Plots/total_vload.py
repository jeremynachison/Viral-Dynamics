#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:02:05 2022

@author: jeremynachison
"""

from solve_ivp_ntwrk import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import pandas as pd
plt.rcParams.update(plt.rcParamsDefault)


### Defining TIVE model

N = 6e4
v_0 = 1e-5

delta = 0.8
p = 7.9e-3
c = 5e-2
beta = 2e-5
s = 10
d = 0.065
dX = 1e-4
r = 0.01
n = 200
p = 4.5e2
alpha = 1.2
tau = 0.67


def tive(time, x, trigger=0):
    t,i,v,e = x
    dx = np.zeros(4)
    if (v >= v_0):
        dx[0] = - beta * v * t + alpha*t*(1-(t + i)/N) 
        dx[1] = (beta * v * t - delta * i - dX * i * e ) 
        dx[2] = p * i - c * v * e
        dx[3] = r*i - d*e
    else:
        dx[0] = alpha*t*(1-(t + i)/N)
        dx[1] = 0
        dx[2] = 0 + trigger
        dx[3] = r*i - d*e
    return dx

##############

T=100
N_nodes = 1000


#### LINE GRAPH ######
G1 = nx.path_graph(N_nodes)

test, prob = host_ntwrk(G1, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G1_df = pd.DataFrame()
for i in test.nodes:
    G1_df[i] = test.nodes[i]["state"][:,2]    
G1_df.to_csv("line_allnodes.csv", index=False)

# sample_sums = pd.DataFrame()
# for samp in range(10):
#     test, prob = host_ntwrk(G1, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in test.nodes:
#         viraldf[i] = test.nodes[i]["state"][:,2]    
#     sample_sums[samp] = viraldf.sum(axis=1)

# sample_sums.to_csv("linegraph.csv", index = False)

##### ERDOS RENYI ######

### 0.05
G2 = nx.erdos_renyi_graph(N_nodes, 0.05)

erdos1, prob = host_ntwrk(G2, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G2_df = pd.DataFrame()
for i in erdos1.nodes:
    G2_df[i] = erdos1.nodes[i]["state"][:,2]  

G2_df.to_csv("erdos05_allnodes.csv", index=False)

# sample_sumsERDOS1 = pd.DataFrame()
# for samp in range(10):
#     erdos1, prob = host_ntwrk(G2, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in erdos1.nodes:
#         viraldf[i] = erdos1.nodes[i]["state"][:,2]    
#     sample_sumsERDOS1[samp] = viraldf.sum(axis=1)

# sample_sumsERDOS1.to_csv("erdos05.csv", index = False)

#### 0.01
G3 = nx.erdos_renyi_graph(N_nodes, 0.1)

erdos2, prob = host_ntwrk(G3, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G3_df = pd.DataFrame()
for i in erdos2.nodes:
    G3_df[i] = erdos2.nodes[i]["state"][:,2]  

G3_df.to_csv("erdos01_allnodes.csv", index=False)

# sample_sumsERDOS2 = pd.DataFrame()
# for samp in range(10):
#     erdos2, prob = host_ntwrk(G3, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in erdos2.nodes:
#         viraldf[i] = erdos2.nodes[i]["state"][:,2]    
#     sample_sumsERDOS2[samp] = viraldf.sum(axis=1)

# sample_sumsERDOS2.to_csv("erdos01.csv", index = False)

####### BARABASI ALBERT GRAPH #######

##### 2

G4 = nx.barabasi_albert_graph(N_nodes, 2)

barabasi1, prob = host_ntwrk(G4, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G4_df = pd.DataFrame()
for i in barabasi1.nodes:
    G4_df[i] = barabasi1.nodes[i]["state"][:,2]  

G4_df.to_csv("barabasi2_allnodes.csv", index=False)

# sample_sumsBARABASI1 = pd.DataFrame()
# for samp in range(10):
#     barabasi1, prob = host_ntwrk(G4, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in barabasi1.nodes:
#         viraldf[i] = barabasi1.nodes[i]["state"][:,2]    
#     sample_sumsBARABASI1[samp] = viraldf.sum(axis=1)

# sample_sumsBARABASI1.to_csv("barabasi2.csv", index = False)

####### 6

G5 =  nx.barabasi_albert_graph(N_nodes, 5)

barabasi2, prob = host_ntwrk(G5, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G5_df = pd.DataFrame()
for i in barabasi2.nodes:
    G5_df[i] = barabasi2.nodes[i]["state"][:,2]  

G5_df.to_csv("barabasi5_allnodes.csv", index=False)

# sample_sumsBARABASI2 = pd.DataFrame()
# for samp in range(10):
#     barabasi2, prob = host_ntwrk(G5, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in barabasi2.nodes:
#         viraldf[i] = barabasi2.nodes[i]["state"][:,2]    
#     sample_sumsBARABASI2[samp] = viraldf.sum(axis=1)

# sample_sumsBARABASI2.to_csv("barabasi6.csv", index = False)

######## CAVEMAN GRAPH ########

G6 = nx.connected_caveman_graph(100,10)

caveman, prob = host_ntwrk(G6, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G6_df = pd.DataFrame()
for i in caveman.nodes:
    G6_df[i] = caveman.nodes[i]["state"][:,2]  

G6_df.to_csv("caveman_allnodes.csv", index=False)

# sample_sumsCAVEMAN = pd.DataFrame()
# for samp in range(10):
#     caveman, prob = host_ntwrk(G6, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in caveman.nodes:
#         viraldf[i] = caveman.nodes[i]["state"][:,2]    
#     sample_sumsCAVEMAN[samp] = viraldf.sum(axis=1)

# sample_sumsCAVEMAN.to_csv("caveman.csv", index = False)

# ####### BLOCK MODEL #######

x_upperlim = N_nodes
while True:
    pick = random.sample(range(25, 350), 5)
    if sum(pick) == x_upperlim:
        break
result = pick

probs = [[0.4,0.1,0.2,0.05,0.15], [0.1,0.3,0.25,0.05,0.3], [0.2,0.25,0.4,0.05,0.1], [0.05,0.05,0.05,0.55,0.3], [0.15,0.3,0.1,0.3,0.15]]

G7 = nx.stochastic_block_model(result, probs)

block, prob = host_ntwrk(G7, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G7_df = pd.DataFrame()
for i in block.nodes:
    G7_df[i] = block.nodes[i]["state"][:,2]  

G7_df.to_csv("block_allnodes.csv", index=False)


# sample_sumsBLOCK = pd.DataFrame()
# for samp in range(10):
#     block, prob = host_ntwrk(G7, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in block.nodes:
#         viraldf[i] = block.nodes[i]["state"][:,2]    
#     sample_sumsBLOCK[samp] = viraldf.sum(axis=1)

# sample_sumsBLOCK.to_csv("block.csv", index = False)


############ Florentine Families ####

G8 = nx.florentine_families_graph()

flor, prob = host_ntwrk(G8, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)

G8_df = pd.DataFrame()
for i in flor.nodes:
    G8_df[i] = flor.nodes[i]["state"][:,2]  

G8_df.to_csv("flor_allnodes.csv", index=False)

# sample_sumsFLOR = pd.DataFrame()
# for samp in range(10):
#     flor, prob = host_ntwrk(G8, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
#     viraldf = pd.DataFrame()
#     for i in block.nodes:
#         viraldf[i] = flor.nodes[i]["state"][:,2]    
#     sample_sumsFLOR[samp] = viraldf.sum(axis=1)

# sample_sumsFLOR.to_csv("florentine.csv", index = False)

######### BARABASI WITH FIXED INITIAL CONDITIONS ########

G9 = nx.barabasi_albert_graph(N_nodes, 2)

sample_sumsBARABASI3 = pd.DataFrame()
for samp in range(10):
    print("rand",samp)
    barabasi3, prob = host_ntwrk(G9, 1, T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
    viraldf = pd.DataFrame()
    for i in barabasi3.nodes:
        viraldf[i] = barabasi3.nodes[i]["state"][:,2]
    viraldf.to_csv("barabasi_rand"+str(samp)+".csv", index=False)
    sample_sumsBARABASI3[samp] = viraldf.sum(axis=1)

sample_sumsBARABASI3.to_csv("barabasi_rand_sums.csv", index = False)

sample_sumsBARABASI4 = pd.DataFrame()
for samp in range(10):
    print("fixed",samp)
    barabasi4, prob = host_ntwrk_fixed(G9,[int(N_nodes/2)], T, 0.80, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.5e6)
    viraldf = pd.DataFrame()
    for i in barabasi4.nodes:
        viraldf[i] = barabasi4.nodes[i]["state"][:,2]
    viraldf.to_csv("barabasi_fixed"+str(samp)+".csv", index=False)
    sample_sumsBARABASI4[samp] = viraldf.sum(axis=1)

sample_sumsBARABASI4.to_csv("barabasi_fixed_sums.csv", index = False)


















