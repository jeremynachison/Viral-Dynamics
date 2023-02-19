#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:00:21 2023

@author: jeremynachison
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

xy_multi = pd.read_csv('Data/dense/d_xy_multi.csv', header=[0,1])

def converge_check(xdata,ydata):
    """ This function checks if a curve converges to a constant value 
    eventually. If it does, then it will label the curve with the "converges" 
    label. Otherwise, the function returns 0. """
    # Create a shifted series of the x and y data up one row. This will allow to 
    # compare the time duration between peaks
    shifted_x = xdata.shift(periods=-1)
    shifted_y = ydata.shift(periods=-1)
    # Calculate difference between subsequent x (time) and y (cells) peak
    xtime_diff = abs(shifted_x - xdata)
    abs_cell_diff = abs(ydata - shifted_y)
    # Set delta and epsilon terms, use def of continuity to check if curve is flat
    delta = 9
    epsilon = 60
    # Some cases there was a peak at the very end of the constant curve (because of numerical approximation)
    # Second condition accounts for these scenarios
    condition1 = (sum(xtime_diff[-6:-1] < delta) == 5) and (sum(abs_cell_diff[-6:-1] < epsilon) == 5)
    condition2 = (sum(xtime_diff[-7:-2] < delta) == 5) and (sum(abs_cell_diff[-7:-2] < epsilon) == 5)
    if condition1 or condition2:
        return "converges"
    else:
        return 0
    
def regression_check(xdata, ydata):
    """ This function checks if an exponential model or linear model fits the 
    peak curves better, or if the peaks remain constant."""
    ### EXPONENTIAL REGRESSION ###
    # Use linear regression on y data transformed by logarithm
    transformed_y = np.log(ydata)
    exp = LinearRegression().fit(xdata, transformed_y)
    # Get R^2 and pval for slope estimate to compare to linear model
    exp_r = exp.score(xdata, transformed_y)
    exp_pval = float(f_regression(xdata, transformed_y)[1])
    ##### LINEAR REGRESSION ######
    lin = LinearRegression().fit(xdata, ydata)
    # Get R^2 and pval for slope estimate to compare to exponential model
    lin_r = lin.score(xdata, ydata)
    lin_pval = float(f_regression(xdata, ydata)[1])
    # For curves where linear and exponential regression perform similarly, 
    # label as linear
    if abs(lin_r - exp_r) <= 0.01 and lin_pval < 0.001:
        return "linear"
    # If exponential performs better than linear, label as exponential
    elif (exp_r > lin_r) and (exp_pval < 0.001):
        return "exponential"
    # For remaining curves (linear performs better), if slope significant
    # label as linear
    elif lin_pval < 0.001:
        return "linear"
    # for curves where slope is insignificant, label as constant
    else:
        # c stands for constant
        return "constant"


def LabelMaker(ID):
    # store relevant data
    data = xy_multi[ID]
    # remove emprty rows
    data = data.dropna()
    # extract series of x and y coordinates individually
    true_x = pd.Series(data["x"])
    true_y = pd.Series(data["y"])
    # store last half of data, use array dtype for sklearn 
    half = int(np.floor(len(data)/2))
    half_data = data[half:]
    half_x = np.array(half_data["x"]).reshape(-1, 1)
    half_y = np.array(half_data["y"])
    # first check curves where there are enough peaks to use exponential/linear/convergent check
    if len(data) > 6:
        # Check for convergence
        converges = converge_check(true_x, true_y)
        if converges != 0:
            return converges
        # Check for linear/exponential/constant
        regressiontype = regression_check(half_x, half_y)
        return regressiontype
    # If data is too sparse to use other labeling methods, label data as sparse
    else:
        return "sparse"




