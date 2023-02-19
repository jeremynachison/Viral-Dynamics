#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:04:19 2022

@author: jeremynachison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy

T = 100
line = pd.read_csv("linegraph.csv")
mean_line = line.mean(axis=1)

erdos01 = pd.read_csv("erdos01.csv")
mean_erdos01 = erdos01.mean(axis=1)

erdos05 = pd.read_csv("erdos05.csv")
mean_erdos05 = erdos05.mean(axis=1)


barabasi2 = pd.read_csv("barabasi2.csv")
mean_barabasi2 = barabasi2.mean(axis=1)

viral_load = pd.concat([mean_line, mean_erdos01, mean_erdos05, mean_barabasi2])
network = pd.concat([pd.Series(["Line"]*len(mean_line)), 
                     pd.Series(["Erdos (0.1)"]*len(mean_erdos01)),
                     pd.Series(["Erdos (0.05)"]*len(mean_erdos05)), 
                     pd.Series(["Barabasi (2)"]*len(mean_barabasi2))], ignore_index=True)

viral_df = pd.DataFrame()
viral_df["viral_load"] = viral_load
viral_df["network"] = network


viral_data = pd.DataFrame()
viral_data["Erdos (0.1)"] = mean_erdos01
viral_data["Erdos (0.05)"] = mean_erdos05
viral_data["Barabasi 2"] = mean_barabasi2
viral_data["Line"] = mean_line

time = np.linspace(0,T,10*T)


fig, ax = plt.subplots()
ax.plot(time,mean_line,color="limegreen")
plt.title("Population Viral Load through Time")
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load (TCID50/mL)")
#ax.set_yscale("log")
ax.set_xlim(0,20)


fig, ax = plt.subplots()
ax.plot(time,mean_line,color="limegreen")
plt.title("Population Viral Load through Time")
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load (TCID50/mL)")
#ax.set_yscale("log")
ax.set_xlim(0,20)

fig, ax = plt.subplots()
plt.style.use('fivethirtyeight')
for i in barabasi2:
    ax.plot(time,barabasi2[i],color="limegreen", alpha=0.3)
plt.title("Population Viral Load through Time (Barabasi)", y=1.08)
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load (TCID50/mL)")
#ax.set_yscale("log")
ax.set_xlim(0,20)
plt.show()




fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()


transp = 0.7
p1, = ax.plot(time,mean_erdos05, color="cornflowerblue", label="Erdos Renyi", alpha = transp)
p2, = twin1.plot(time,mean_line,color="limegreen", label="Line Graph", alpha = transp)
p3, = ax.plot(time,mean_barabasi2, color="darkorange", label="Barabasi", alpha = transp)

ax.set_xlim(0,20)
ax.set_ylim(0,np.max(mean_erdos05))
twin1.set_ylim(0,np.max(mean_line + 1e3))


ax.set_xlabel("Time (Days)")
ax.set_ylabel("Viral Load")
twin1.set_ylabel("Viral Load (Line Graph)")

ax.legend(handles=[p1, p2, p3])

plt.title("Population Viral Load Across Different Networks", y=1.08)
plt.show()








