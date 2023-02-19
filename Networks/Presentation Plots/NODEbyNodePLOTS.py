#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:55:29 2022

@author: jeremynachison
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


T = 100
barabasi_deg = list(G4.degree)

min_deg = 100
node = None
for i in barabasi_deg:
    if i[1] < min_deg:
        min_deg = i[1]
        node = i[0]
barabasi_all = pd.read_csv("data/barabasi2_allnodes.csv")
G4.nodes[550]["Neighbors"]

barabasi_allday10 = barabasi_all[90:-1]


timeinf_dict = {}
for i in G4.nodes:
    timeinf_dict[i] = test.nodes[i]["TimeInfected"]
    

# maximum degree node is node 2 w degree 66
# node 948 has degree 2

node2_neighbors = [0,3,4,5,6,7,9,10,11,17,19,20,30,35,43,53,58,74,86,91,93,99,105,
                111,117,133,142,153,162,172,182,189,201,210,218,221,228,235,239,
                253,261,263,288,329,336,351,353,361,368,423,424,425,454,563,588,
                609,656,660,711,728,739,767,818,893,995,996]

node948_neighbors = [512,249]

barabasi_node2Neighbors = pd.DataFrame()
for i in barabasi_all:
    if int(i) in node2_neighbors:
        barabasi_node2Neighbors[i] = barabasi_all[i]
node2_vload = barabasi_node2Neighbors.sum(axis=1)


barabasi_node948Neighbors = pd.DataFrame()
for i in barabasi_all:
    if int(i) in node948_neighbors:
        barabasi_node948Neighbors[i] = barabasi_all[i]
node948_vload = barabasi_node948Neighbors.sum(axis=1)



time = np.linspace(0,T,10*T)
barabasi_sum = barabasi_all.sum(axis=1)

# plt.plot(time,barabasi_sum, color = "limegreen")
# plt.style.use('fivethirtyeight')
# for i in barabasi_all:
#     plt.plot(time,barabasi_all[i], color="dimgray",alpha=0.02)
# plt.xlim(0,20)

fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()
twin2 = ax.twinx()

twin2.spines.right.set_position(("axes", 1.2))
twin2.yaxis.get_offset_text()

transp = 0.1
p1, = ax.plot(time,barabasi_sum, color="limegreen", label="Population Viral Load", linewidth=3)
#p2, = twin1.plot(time,node2_vload, color="orangered", label="Neighbor Viral Load (Highest Degree)")
plist = []
for i in barabasi_all:
    plist += twin1.plot(time,barabasi_all[i],color="darkslateblue", label="Node Viral Load", alpha = transp, linewidth=0.8)
p3, = twin2.plot(time,node2_vload, color="orangered", label="Neighbor Viral Load (High Degree)", linewidth=2)
p4, = twin2.plot(time,node948_vload, color="turquoise", label="Neighbor Viral Load (Low Degree)", linewidth=3)
#p3, = ax.plot(time,mean_barabasi2, color="darkorange", label="Barabasi", alpha = transp)

ax.set_xlim(0,20)
ax.set_ylim(0,np.max(barabasi_sum))
twin1.set_ylim(0,np.max(barabasi_all["0"])*8)
twin2.set_ylim(10, np.max(node2_vload)*2)
#twin2.set_yscale("log")
tkw = dict(size=4,width=1.5)
twin1.tick_params(axis="y", **tkw)
twin2.tick_params(axis="y", **tkw)

ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load")
twin1.set_ylabel("Viral Load (Individual Node)")
twin2.set_ylabel("Viral Load (Neighborhood)")

leg = ax.legend(handles=[p1, p3, p4, plist[0]])
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

plt.title("Population, Nieghborhood, and Individual Viral Load Across a Barabasi Network", y=1.08)
plt.show()


####### Diff Networks #####

line_all = pd.read_csv("data/line_allnodes.csv")
line_sum = line_all.sum(axis=1)

erdos05_all = pd.read_csv("data/erdos05_allnodes.csv")
erdos05_sum = erdos05_all.sum(axis=1)

caveman_all = pd.read_csv("data/caveman_allnodes.csv")
caveman_sum = caveman_all.sum(axis=1)

block_all = pd.read_csv("data/block_allnodes.csv")
block_sum = block_all.sum(axis=1)

plt.plot(time, caveman_sum)
plt.plot(time, block_sum)
plt.plot(time,barabasi_sum)
#plt.plot(time, erdos05_sum)
plt.plot(time,line_sum)
plt.xlim(0,20)


fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()


transp = 0.7
width = 3
p1, = ax.plot(time,erdos05_sum, color="darkslateblue", label="Erdos Renyi", alpha = transp, linewidth = width)
p2, = twin1.plot(time,line_sum,color="limegreen", label="Line Graph", alpha = transp, linewidth = width)
p3, = twin1.plot(time,caveman_sum,color="tomato", label="Caveman Graph", alpha = transp, linewidth = width)
p4, = ax.plot(time,barabasi_sum, color="darkorange", label="Barabasi", alpha = transp, linewidth = width)

ax.set_xlim(0,20)
ax.set_ylim(0,np.max(erdos05_sum))
twin1.set_ylim(0,np.max(caveman_sum)*2)


ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load")
twin1.set_ylabel("Viral Load (Line and Caveman Graph)")

ax.legend(handles=[p1, p2, p3, p4])

plt.title("Population Viral Load Across Different Networks", y=1.04)
plt.show()

# filelist_rand = glob.glob("data/barabasi_trials/random/*.csv")
# filelist_fixed = glob.glob("data/barabasi_trials/fixed/*.csv")

# rand_df = pd.DataFrame()
# for file in filelist_rand:
#     newdf = pd.read_csv(file)
#     newdf["file"] = file[28:-4]
#     rand_df = pd.concat([rand_df,newdf])

rand_barabasi = pd.read_csv("data/barabasi_trials/barabasi_rand_sums.csv")
fixed_barabasi = pd.read_csv("data/barabasi_trials/barabasi_fixed_sums.csv")
                            
fig, ax = plt.subplots()
plt.style.use('fivethirtyeight')
rand_list = []
for i in rand_barabasi:
    rand_list += ax.plot(time,rand_barabasi[i],color="limegreen", label = "Random Seed", alpha=0.3, linewidth = 3)

fixed_list = [] 
for j in fixed_barabasi:
    fixed_list += ax.plot(time,fixed_barabasi[i],color="darkorange", label = "Fixed Seed", alpha=0.3, linewidth = 2)
    
plt.title("Population Viral Load with Random vs. Fixed Seeding", y=1.08)
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load (TCID50/mL)")
ax.legend(handles=[rand_list[0], fixed_list[0]])
#ax.set_yscale("log")
ax.set_xlim(0,20)
plt.show()


check = np.round(fixed_barabasi["0"],3) == np.round(fixed_barabasi["2"],3)







