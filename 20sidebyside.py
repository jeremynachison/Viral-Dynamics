#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:41:43 2023

@author: jeremynachison
"""


import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sb
import random

plt.rcParams["figure.figsize"] = (50,50)

random.seed(0)

param_df = pd.read_csv("d_param_df.csv",index_col=0)
multi = pd.read_csv("d_tive_multi.csv",header=[0,1],index_col=0)
stats = pd.read_csv("d_sample_stats.csv",index_col=0)
#num = 100

samp_params = param_df[:50]

samp_stats = pd.merge(samp_params,stats.iloc[:,[0,1]],on="id")

samp_ids = list(samp_params["id"].astype(str))

my_cmap = sb.color_palette("rocket", as_cmap=True)

val_list = list(samp_stats["infection_cycles"])
minima = min(val_list)
maxima = max(val_list)
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=my_cmap)

sns.pairplot(samp_params, hue="id", palette=["black"])

ct = 0
fig, axes = plt.subplots(4, 5)
for row in range(4):
    for col in range(5):
        part_id = samp_ids[ct]
        ct+=1
        axes[row,col].plot(multi["time"], multi[part_id]["I"],
                           c= mapper.to_rgba(samp_stats.loc[samp_stats["id"] == int(part_id), "infection_cycles"]))
for ax in fig.get_axes():
    ax.label_outer()
plt.show()
    