#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:32:01 2023

@author: jeremynachison
"""

from NetworkSimulation import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import pandas as pd
from colour import Color
plt.rcParams["figure.figsize"] = (20,20)

### Defining TIVE model

N = 6e4
v_0 = 1e-5

delta = 0.8
p = 4.5e2
c = 5e-2
beta = 2e-5
d = 0.065
dX = 1e-4
r = 0.01
n = 200
#p = 4.5e2
alpha = 1.2


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
spd = 10
N_nodes=10
G = nx.path_graph(N_nodes)
nx.draw_networkx(G,pos=nx.spring_layout(G), node_color=range(N_nodes), cmap=plt.cm.plasma, with_labels = True)
test = host_ntwrk(G, 1, T, 0.5, tive, 2, [6e4,0,0,0], [6e4,1,1e-5,0], 3.8e6, spd)

observation = test.nodes[5]

node_dict = {}
for i in test.nodes:
    node_dict[i] = test.nodes[i]["state"]
timeinf_dict = {}
for i in test.nodes:
    timeinf_dict[i] = test.nodes[i]["TimeInfected"]
time_if_dict = {}
for i in test.nodes:
    time_if_dict[i] = test.nodes[i]["Updated"]
exposure_dict = {}
for i in test.nodes:
    exposure_dict[i] = test.nodes[i]["Exposure"]
    
see = test.nodes[9]["state"]
node_color = range(20)
plt.cm.plasma
cmap = plt.cm.plasma

t1 = np.linspace(0,T,spd*T)
purple = Color("#341d5c")
colors = pd.Series(purple.range_to(Color("#FF3E19"), N_nodes))
plt.style.use('fivethirtyeight')
plt.title("Does this work")
plt.subplot(4, 1, 1)
for i in test.nodes:
    test.nodes[i]["color"] = colors.astype(str)[i]
    plt.plot(t1, test.nodes[i]["state"][:,2], color=test.nodes[i]["color"])
#plt.yscale("log")
plt.ylabel("Viral Load")
ax = plt.gca()
#ax.set_ylim([10e-4, 10e8])
#ax.set_xlim([0, 40])
plt.subplot(4, 1, 2)
for i in test.nodes:
    plt.plot(t1, test.nodes[i]["state"][:,1], color=test.nodes[i]["color"])
plt.ylabel("# of Infected Cells")
ax = plt.gca()
#ax.set_xlim([0, 40])
plt.subplot(4,1,3)
for i in test.nodes:
    plt.plot(t1, test.nodes[i]["state"][:,0], color=test.nodes[i]["color"])
plt.ylabel("# of Target Cells")
plt.subplot(4,1,4)
for i in test.nodes:
    plt.plot(t1, test.nodes[i]["state"][:,3], color=test.nodes[i]["color"])
plt.ylabel("Immune Response")
#plt.yscale("log")
ax = plt.gca()
#ax.set_xlim([0, 40])
plt.show()  







inf_init = [6e4,1,1e-5,0]
viral_load_ind = 2
inf_init[viral_load_ind]

check = node_dict[5]

node8state = test.nodes[0]["state"]
t = node8state[:,0]
i = node8state[:,1]
v = node8state[:,2]
e = node8state[:,3]
t1 = np.linspace(0,T,3*T)

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()


twin2.spines.right.set_position(("axes", 1.2))
twin3.spines.right.set_position(("axes", 1.4))

transp = 0.8
p1, = ax.plot(t1,t,"dodgerblue" , label="Target Cells", alpha = transp)
p2, = twin1.plot(t1, i,"orangered",label="Infective", alpha = transp)
p3, = twin2.plot(t1,v,color="limegreen",label="Viral Load", alpha = transp)
p4, = twin3.plot(t1,e,color = "purple",label="Immune Response", alpha = transp)

ax.set_xlim(0,100)
ax.set_ylim(0,np.max(t))
twin1.set_ylim(0,np.max(t))
twin2.set_ylim(0,np.max(v))
twin3.set_ylim(0,np.max(e))

ax.set_xlabel("Time (Days)")
ax.set_ylabel("Number of Traget Cells")
twin1.set_ylabel("Number of Infective Cells")
twin2.set_ylabel("Viral Load")
twin3.set_ylabel("Immune Response")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
twin3.yaxis.label.set_color(p4.get_color())

tkw = dict(size=4,width=1.5)
ax.tick_params(axis="y",colors=p1.get_color(), **tkw)
twin1.tick_params(axis="y",colors=p2.get_color(), **tkw)
twin2.tick_params(axis="y",colors=p3.get_color(), **tkw)
twin3.tick_params(axis="y",colors=p4.get_color(), **tkw)
#plt.vlines(x=v0_index, ymin=0, ymax = np.max(t_m), colors = "black", alpha = 0.2)


ax.tick_params(axis="x", **tkw)

ax.legend(handles=[p1, p2, p3, p4])

#plt.plot(t1, i1,color="orangered",label="Infective")
#plt.plot(t1,t_m + i1, color="black",label="sum")
# plt.plot(t1,v,color="limegreen",label="Viral Load")
# plt.plot(t1,t_m,color="dodgerblue",label="Target Cells")
#plt.plot(t1,e1,color = "purple",label="Immune Response")
plt.title("TIVE Model (solve_ivp)")
# plt.xlabel("Days")
# plt.ylabel("Number of Cells")
#twin2.set_yscale("log")
# plt.legend()
plt.show()

viral_df = pd.DataFrame()
for node in node_dict:
    node_dict[node][:,]



