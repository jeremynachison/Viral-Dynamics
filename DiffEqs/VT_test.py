#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:07:07 2022

@author: jeremynachison
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint

t0 = 1e6
v0 = 1
p = 1.62
k = 1e9
c_t = 4.88e-8
c = 0.6
r = 0.96
delta_t = 2e-1
m = 2
k_t = 8.58e4

init1 = [v0,t0]

s_t = delta_t * t0

def vt(x,t1):
    v,t = x
    dx = np.zeros(2)
    dx[0] =  p*v*(1-(v/k)) - c_t * v * t - c * v
    dx[1] = s_t + r*t*((v**m)/(v**m + k_t**m)) - delta_t * t
    return dx

t1 = np.linspace(0,40,1000)
y = odeint(vt, init1, t1)

v1 = y[:,0]
t = y[:,1]

# plot
plt.plot(t1,v1,color="limegreen",label="Viral Load")
plt.plot(t1,t,color="dodgerblue",label="Immune Response")
plt.title("Sample TIV Model")
plt.xlabel("Days")
plt.ylabel("Number of Cells")
plt.yscale("log")
plt.legend()
plt.show
