#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:41:43 2023

@author: jeremynachison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import LabelMaker as lm
import random

plt.rcParams["figure.figsize"] = (50,50)

random.seed(0)

param_df = pd.read_csv("Data/dense/d_param_df.csv",index_col=0)
param_df["id"] = param_df["id"].astype(str)
multi = pd.read_csv("Data/dense/d_tive_multi.csv",header=[0,1],index_col=0)

samp_params = param_df[:50]


labels = samp_params["id"].apply(lm.LabelMaker).reset_index()
labels['index'] = labels['index'].astype(str)
labels = labels.rename(columns={"index": "id", "id" : "label"}) 

labeled_df = pd.merge(samp_params, labels, on="id")

sns.pairplot(labeled_df, hue="label", palette="Dark2", diag_kind=None)
