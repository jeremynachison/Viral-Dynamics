#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:30:47 2023

@author: jeremynachison
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import LabelMaker as lm
import random
# Set figure size
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams.update({'font.size': 25})
# Set random seed
random.seed(0)
# Read in data
param_df = pd.read_csv("data/dense/d_param_df.csv",index_col=0)
# Convert id to string (for formatting purposes with the tive data)
param_df["id"] = param_df["id"].astype(str)
# Read in tive data
multi = pd.read_csv("data/dense/d_tive_multi.csv",header=[0,1],index_col=0)

# Label all of the TIVE curves
labels = param_df["id"].apply(lm.LabelMaker).reset_index()
# Convert ids to strings, rename output of apply to be more descriptive
labels['index'] = labels['index'].astype(str)
labels = labels.rename(columns={"index": "id", "id" : "label"})

# merge parameter dataframe with labels from apply function
labeled_df = pd.merge(param_df, labels, on="id")

# Create a datframe stroing the ids of the first three examples of curves 
# belonging to each label
color_labels = np.array(["sparse", "slow", "fast"])
samp_df = pd.DataFrame()
for i in color_labels:
    # Grab first three examples of curves from each type of label
    curve_type = (labeled_df.loc[labeled_df["label"]==i,"id"][:3].
                  reset_index(drop=True))
    samp_df[i] = curve_type
    
# Set color palette
colors = sns.color_palette("Dark2", 5)
color_map = dict(zip(color_labels, colors))

fig, axes = plt.subplots(3, 3)
for row in range(3):
    for col in range(3):
        part_id = samp_df.iloc[row, col]
        axes[row,col].plot(multi["time"], 
                           multi[part_id]["I"], 
                           c = labeled_df.loc[labeled_df["id"]==part_id,
                                              "label"].map(color_map).tolist()[0],
                           linewidth=3)
        axes[0, col].set_title(color_labels[col])
for ax in fig.get_axes():
    ax.label_outer()
# =============================================================================
# slow = mpatches.Patch(color=color_map["slow"], label='Slow')
# fast = mpatches.Patch(color=color_map["fast"], label='Fast')
# sparse = mpatches.Patch(color=color_map["sparse"], label='Sparse')
# handles=[sparse, slow, fast]
# =============================================================================
# =============================================================================
# fig.legend(handles=handles, 
#            loc="upper center", 
#            shadow = True, ncol = 5, 
#            bbox_to_anchor=(0.5,0.97), columnspacing = 5)
# =============================================================================
fig.suptitle("Categorization Scheme for TIVE System", 
            y =1.0)
plt.show()