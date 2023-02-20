#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:11:49 2022

@author: jeremynachison
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from scipy.integrate import odeint
N = 6e4
i01, v0 = 1/N, 10e-4
t0 = N
z0 = 0
init1 = [t0,i01,v0]

delta = 1.5
p = 4.5e2
c =  6.7e-1
beta = 5e-7
s = 10
alpha = 0.1
def tiv(x,t1):
    t,i,v = x
    dx = np.zeros(3)
    dx[0] = -beta * v * t
    dx[1] = beta * v * t - delta * i
    dx[2] = p * i - c * v
    return dx

t1 = np.linspace(0,300,3000)
y = odeint(tiv, init1, t1)

t_m = y[:,0]
i1 = y[:,1]
v = y[:,2]

np.max(v) / 8.1394e6

v[-1]
# plot
plt.plot(t1,i1,color="orangered",label="Infective")
#plt.plot(t1,v,color="limegreen",label="Viral Load")
plt.plot(t1,t_m,color="dodgerblue",label="Target Cells")
plt.title("Sample TIV Model")
plt.xlabel("Days")
plt.ylabel("Number of Cells")
#plt.yscale("log")
plt.legend()
plt.show


# probability plot
figure, ax = plt.subplots()
prob = (0.756 * (v / 8.1394e6))
plt.plot(t1,prob)
ax.set_xlim(0,50)
plt.show

prob_ser = pd.Series(prob)
prob_ser.sort_values(ascending = False)
