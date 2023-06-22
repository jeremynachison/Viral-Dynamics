#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:32:20 2022

@author: jeremynachison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
plt.rcParams["figure.figsize"] = (4,6)
plt.rcParams.update({'font.size': 16})

### This file a sample of an individual TIVE model using scipy's solve_ivp

# Initializing all parameters
N = 6e4
i01, v_0  = 1, 1e-5
t0 = N
e0 = 0
init1 = (t0,i01,v_0,e0)

delta = 0.8
p = 4.5e2
c = 5e-2
beta = 2e-5
d = 0.065
dX = 1e-4
r = 0.01
alpha = 1.2

# Defining the diff eq
def tive(time, x):
    t,i,v,e = x
    # Convert the time to an integer, use to index the list containing external triggers
    time_int = int((np.round(time,2) * 10))
    dx = np.zeros(4)
    if (v >= v_0):
        dx[0] = - beta * v * t + alpha*t*(1-(t + i)/N) 
        dx[1] = (beta * v * t - delta * i - dX * i * e ) 
        dx[2] = p * i - c * v * e
        dx[3] = r*i - d*e
    else:
        dx[0] = alpha*t*(1-(t + i)/N)
        dx[1] = 0 #- delta * i - dX * i * e
        dx[2] = 0 + v0_vect[time_int] #-c*v*e
        dx[3] = r*i - d*e
    return dx
# Create time to evaluate the system through
t1 = np.linspace(0,600,6000)
# Populate a vector containing external triggers
v0_vect = np.zeros(7001)
for i in range(7001):
    if (i % 10 == 0) and (i != 0):
        v0_vect[i] = 0.5*v_0
    else:
        v0_vect[i] = 0
# Solve the system
sol = solve_ivp(tive, (0,600), init1, t_eval = t1)
y = sol.y

# Extract individual curves from output array
t_m = y[0,:]
i1 = y[1,:]
v = y[2,:]
e1 = y[3,:]

# plot
#plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()


twin2.spines.right.set_position(("axes", 1.33))
twin3.spines.right.set_position(("axes", 1.56))

transp = 0.8
width = 3
p1, = ax.plot(t1,t_m,"dodgerblue" , 
              label="Target Cells", alpha = transp, linewidth = width)
p2, = twin1.plot(t1, i1,"orangered",
                 label="Number of Infective",
                 alpha = transp, linewidth = width)
p3, = twin2.plot(t1,v,color="limegreen",
                 label="Viral Load", alpha = transp, linewidth = width)
p4, = twin3.plot(t1,e1,color = "purple",
                 label="Immune Response", alpha = transp, linewidth = width)

ax.set_xlim(0,50)
ax.set_ylim(0,np.max(t_m))
twin1.set_ylim(0,np.max(t_m))
twin2.set_ylim(0,np.max(v))
twin3.set_ylim(0,np.max(e1))

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


ax.tick_params(axis="x", **tkw)

ax.legend(handles=[p1, p2, p3, p4], prop={"size":12})
plt.title("Sample TIVE Model")
plt.show()
