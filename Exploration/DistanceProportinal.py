#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:38:35 2023

@author: jeremynachison
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.patches as mpatches

# curve 1: (x+1)^2-(7/10)
# curve 2: sin((pi/2)x)

# function defining curve 1
def curve1(x):
    return (x+1)**2-(7/10)

# function defining curve 2
def curve2 (x):
    return np.sin((np.pi/2)*x)

# distance from a point (a,b) to curve 1
def d1(x,a,b):
    return np.sqrt((x-a)**2+((x+1)**2-7/10-b)**2)

# maximum distance from a point in region 1 to curve1
max_dist1 = optimize.fmin(d1,0,full_output=1, args=(-1,1))[1]

# distance from a point (a,b) to curve 2
def d2(x,a,b):
    return np.sqrt((x-a)**2+(np.sin((np.pi/2)*x)-b)**2)

max_dist2 = optimize.fmin(d2,0,full_output=1, args=(1,-1))[1]

def prob_transformation(x,b):
    # more emphasis on larger probabilities for b close to 1 (but hgreater than 1)
    return (np.log((b-x)/x))/np.log((b-1)/b)
# this transformation is aisgmoid function
def prob_transformation2(x,b):
    return 1/(1+np.e**(-b*(x-0.5)))

def generate_points(n):
    xcoords1 = []
    ycoords1 = []
    # Region 1: points between y=1 and curve 1
    while len(xcoords1) < n:
        # generate random points in the region
        x = random.uniform(-1,0.30384)
        y = random.uniform(curve1(x),1)
        # calculate distance from point to curve 1
        distance = optimize.fmin(d1,0,full_output=1, args=(x,y))[1]
        # probability of generating a point is 
        probability = 1-(distance/(max_dist1))
        if random.uniform(0,1)<prob_transformation2(probability,10):
            xcoords1.append(x)
            ycoords1.append(y)
    xcoords2 = []
    ycoords2 = []
    while len(xcoords2) < n:
        # generate random points in the region
        x = random.uniform(-1,1)
        y = random.uniform(curve2(x),min(1,curve1(x)))
        # calculate distance from point to curve 1
        distance1 = optimize.fmin(d1,0,full_output=1, args=(x,y))[1]
        # calculate distance from point to curve 2
        distance2 = optimize.fmin(d2,0,full_output=1, args=(x,y))[1]
        # use closest curve to measure distance
        distance = min(distance1, distance2)
        # probability of generating a point
        # 0.15 is about the 1/2 of the maximum distance between each curve
        probability = 1-(distance/(0.16))
        if random.uniform(0,1)<prob_transformation2(probability,8):
            xcoords2.append(x)
            ycoords2.append(y)
    xcoords3 = []
    ycoords3 = []
    while len(xcoords3) < n:
        # generate random points in the region
        x = random.uniform(-1,1)
        y = random.uniform(-1,curve2(x))
        # calculate distance from point to curve 1
        distance = optimize.fmin(d2,0,full_output=1, args=(x,y))[1]
        # probability of generating a point is 
        probability = 1-(distance/(max_dist2))
        if random.uniform(0,1)<prob_transformation2(probability,10):
            xcoords3.append(x)
            ycoords3.append(y)
        
    return {"x1":xcoords1,"y1":ycoords1,
            "x2":xcoords2,"y2":ycoords2,
            "x3":xcoords3,"y3":ycoords3}
        

test = generate_points(1000)

plt.style.use('seaborn-v0_8')
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(test["x1"],test["y1"],alpha=0.6, 
            marker=".", c="darkorange")
plt.scatter(test["x2"],test["y2"],alpha=0.6, 
            marker=".",c="limegreen")
plt.scatter(test["x3"],test["y3"],alpha=0.6,
            marker=".",c="slateblue")
plt.xlim([-1,1])
plt.ylim([-1,1])
orange_patch =  mpatches.Patch(color='darkorange', label='Class A')
green_patch = mpatches.Patch(color='limegreen', label='Class B')
purp_patch = mpatches.Patch(color='slateblue', label='Class C')
ax.legend(handles=[orange_patch, green_patch, purp_patch])
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_aspect('equal', adjustable='box')
plt.show()



