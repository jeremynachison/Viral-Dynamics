#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:16:00 2023

@author: jeremynachison
"""

import pandas as pd
import numpy as np

param_df = pd.read_csv("d_param_df.csv",index_col=0)
param_df["id"] = param_df["id"].astype(str)
tive_multi = pd.read_csv("d_tive_multi.csv",header=[0,1],index_col=0)
stats = pd.read_csv("d_sample_stats.csv",index_col=0)
stats['id'] = stats['id'].astype(str)

samp_stats = pd.merge(param_df, stats.iloc[:,[0,1]],on="id")
params = samp_stats.iloc[:,1:10]
cycles = samp_stats.iloc[:,-1]

samps = len(param_df["id"])
xy_letters = ["x","y"]
xy_list = []
for i in range(samps):
    xy_list += xy_letters
    
id_array_temp = np.array(pd.Series(range(0,samps)).astype(str))
id_array = np.repeat(id_array_temp,2)
arrays = [id_array, np.array(xy_list)] 

empty = np.empty((len(tive_multi["time"]), 2*samps))
empty[:] = np.nan

xy = pd.DataFrame()
xy["x"], xy["y"] = None, None

xy_multi = pd.DataFrame(empty, columns=arrays)
xy_multi.columns = xy_multi.columns.rename("Particle ID", level=0)
xy_multi.columns = xy_multi.columns.rename("xy_sols", level=1)


for col_id in param_df["id"]:
    prev_step = tive_multi[col_id,"I"].shift(fill_value=0)
    next_step = tive_multi[col_id,"I"].shift(fill_value=0,periods = -1)
    current = tive_multi[col_id,"I"]
    peak = (current > prev_step) & (current > next_step) & (current > 100)
    x = pd.Series(tive_multi["time"][peak][1:].iloc[:,0])
    y = tive_multi[col_id,"I"][peak][1:]
    d = {'x':x, 'y':y}
    addition = pd.DataFrame(data = d)
    xy = pd.concat([xy.loc[:,["x","y"]], addition], ignore_index = True)
    xy_multi[col_id, "x"].iloc[x.index] =  x
    xy_multi[col_id, "y"].iloc[y.index] =  y
    
xy_multi.to_csv('d_xy_multi.csv', index = False)
