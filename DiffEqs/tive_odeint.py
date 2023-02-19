#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:55:42 2022

@author: jeremynachison
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint

N = 6e4
i01, v0  = 1/N, 10e-6
t0 = N
z0 = 0
e0 = 0
init1 = [t0,i01,v0,e0]

delta = 1.5
p = 4.5e2
c =  6.7e-3
beta = 3e-5
s = 10
d = 0.025
dX = 1e-3
r = 0.1
n = 200
p = 4.5e2
alpha = 0.2
tau = 0.67


def tive(x,time):
    t,i,v,e = x
    dx = np.zeros(4)
    dx[0] = alpha*t*(1-(t + i)/N) - beta * v * t 
    dx[1] = (beta * v * t - delta * i - dX * i * e )
    dx[2] = p * i - c * v * e
    dx[3] = r*i - d*e
    return dx

t1 = np.linspace(0,300,3000)
v0_vect = np.zeros(5000)
for i in range(0,5000):
    if (i % 100 == 0) and (i != 0):
        v0_vect[i] = v0
    else:
        v0_vect[i] = 0
v0 = dict(zip(t1,v0_vect))
y,out = odeint(tive, init1, t1, full_output =1)

t_m = y[:,0]
i1 = y[:,1]
v = y[:,2]
e1 = y[:,3]
cumi = np.cumsum(i1)
np.max(v)

v[-1]
# plot

fig, ax = plt.subplots()
fig.subplots_adjust(right = 0.95)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()


twin2.spines.right.set_position(("axes", 1.2))
twin3.spines.right.set_position(("axes", 1.4))


p1, = ax.plot(t1,t_m,"dodgerblue" , label="Target Cells")
p2, = twin1.plot(t1, i1,"orangered",label="Infective")
p3, = twin2.plot(t1,v,color="limegreen",label="Viral Load")
p4, = twin3.plot(t1,e1,color = "purple",label="Immune Response")

ax.set_xlim(0,300)
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

ax.legend(handles=[p1, p2, p3, p4])

#plt.plot(t1, i1,color="orangered",label="Infective")
#plt.plot(t1,t_m + i1, color="black",label="sum")
# plt.plot(t1,v,color="limegreen",label="Viral Load")
# plt.plot(t1,t_m,color="dodgerblue",label="Target Cells")
#plt.plot(t1,e1,color = "purple",label="Immune Response")
plt.title("Sample TIVE Model")
# plt.xlabel("Days")
# plt.ylabel("Number of Cells")
# plt.yscale("log")
# plt.legend()
plt.show()
