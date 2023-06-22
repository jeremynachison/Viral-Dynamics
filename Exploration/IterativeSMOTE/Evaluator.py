#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:00:53 2023

@author: jeremynachison
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull

########################### INITIAL CONDITIONS ################################

N = 6e4
i01, v_0  = 1, 1e-5
t0 = N
e0 = 0
init1 = (t0,i01,v_0,e0)

wt = 0.5
# run simulation for one year
days = 365
spd = 10

t1 = np.linspace(0,days,spd*days)
v0_vect = np.zeros(7001)

params_dict = {"delta" : 0.8, 
               "p": 4.5e2, "c": 5e-2, "beta":2e-5,
               "d":0.065, "dX":1e-4, "r":0.01, "alpha":1.2}

############################## ALL FUNCTIONS ################################

def initial_params(samps, params_dict):
    """ Function to draw initial parameter samples. 
    
    samps (int): number of samples to draw for each parameter
    params_dict (dict): Dictionary mapping parameter names to value samples are 
    centered around"""
    # Creates and returns dataframe of parameter samples
    param_df = pd.DataFrame()
    # Creates a column to label parameter combos by ID
    param_df["id"] = pd.Series(range(0,samps)).astype(str)
    # Draw samples randomly within 5% of value passed to function
    for param in params_dict.keys():
        param_df[param] = np.random.uniform(0.95*params_dict[param],
                                            1.05*params_dict[param],samps)
    return param_df

# Define equation of within-host dynamics (TIVE Model)
def tive(time, x, trigger=0, beta=0, alpha=0,delta=0,dX=0,p=0,c=0,r=0,d=0):
    """ This defines the TIVE dynamical system using the parameters passed to 
    the function. The if/else blocks function as the indicator variable in the 
    equations."""
    # Pass a vector x containing initial conditions for t,i,v, and e
    t,i,v,e = x
    # Convert the time steps to an integer (used to extract value from external trigger indicator vector)
    time_int = int((np.round(time,2) * spd))
    dx = np.zeros(4)
    if (v >= v_0):
        dx[0] = - beta * v * t + alpha*t*(1-(t + i)/N) 
        dx[1] = (beta * v * t - delta * i - dX * i * e ) 
        dx[2] = p * i - c * v * e
        dx[3] = r*i - d*e
    else:
        dx[0] = alpha*t*(1-(t + i)/N)
        dx[1] = 0 
        dx[2] = 0 + trigger + v0_vect[time_int]
        dx[3] = r*i - d*e
    return dx   

def make_Sol_dataframe(param_df):
    """ Creates the dataframe that will hold the solutions of the TIVE equations
    
    param_df (dataframe): dataframe containing parameter combinations """
    # Create a list of T, I, V, E repeating for the number of samples (for multi-index)
    tive_letters = ["T","I","V","E"]
    tive_list = []
    samps = len(param_df)
    for i in range(samps):
        tive_list += tive_letters
    # create an array of ids, with each id repeating 4 times (to store T, I, V, E solutions for each particle)
    # This is for multi-index purposes
    id_array_temp = np.array(pd.Series(range(0,samps)).astype(str))
    id_array = np.repeat(id_array_temp,4)
    # For multi index, store ids, and repeating tive in a list
    arrays = [id_array, np.array(tive_list)]
    # Create a multi-indexed dataframe using the list for multi-indexed columns above
    # The Multi-index works as follows:
        # The first level of the multi-index is particle id
        # for each particle id there are 4 sub columns storing solutions to TIVE equation of corresponding particle
    tive_multi = pd.DataFrame(np.zeros((len(t1), 4*samps)), columns=arrays)
    # Add in a column to keep track of what times the equation is evaluating
    tive_multi.insert(0,"time",t1)
    # Label the levels of the multi indexed column
    tive_multi.columns = tive_multi.columns.rename("Particle ID", level=0)
    tive_multi.columns = tive_multi.columns.rename("TIVE Sols", level=1)
    return tive_multi



def simulation_solver(param_df):
    """Solve system for given parameters. Takes in a dataframe of parameter 
    values
    
    param_df (dataframe): dataframe containing parameter combinations """
    tive_multi = make_Sol_dataframe(param_df)
    # For each particle id, evaluate the tive model with the particle's parameters
    for col_id in param_df["id"]:
        row = param_df.loc[param_df["id"] == col_id].index.item()
        beta = param_df.loc[row,"beta"]
        alpha = param_df.loc[row,"alpha"]
        delta = param_df.loc[row,"delta"]
        dX = param_df.loc[row,"dX"]
        p = param_df.loc[row,"p"]
        c = param_df.loc[row,"c"]
        r = param_df.loc[row,"r"]
        d = param_df.loc[row,"d"]
        # create a vector storing the external triggering of infection from outside sources
        # host is exposed once a day to infection
        v0_vect = np.zeros(spd*days+1001)
        for i in range(spd*days+1001):
            if (i % spd == 0) and (i != 0):
                # exposure amount (percentage of initial viral load) is a parameter, wt
                v0_vect[i] = wt*v_0
            else:
                v0_vect[i] = 0
        # solve tive equation for given particle       
        sol = solve_ivp(tive, (0,days), init1, t_eval = t1, args=(0,beta,alpha,delta,dX,p,c,r,d)).y
        y = np.transpose(sol)
        # store solutions for T, I, V, E curves in respective column for given ID
        tive_multi[col_id,"T"] = y[:,0]
        tive_multi[col_id,"I"] = y[:,1]
        tive_multi[col_id,"V"] = y[:,2]
        tive_multi[col_id,"E"] = y[:,3]
    return tive_multi
    
def get_peaks(param_df):
    """ Make a multi-index dataframe storing x and y coords of peak # of 
    infectious cells (or viral load) for each parameter combination passed to 
    the function (as a dataframe). Here the "x coord" refers to time and the y 
    coord is the infectous cells/viral load. For values where a peak does not 
    occur, an na will be stored. The multi-index dataframe contains parameter 
    IDs as the main columns, each with sub-columns containing the time (x) and 
    viral load (y) values. 
    
    param_df (dataframe): dataframe containing parameter combinations"""
    # Generate solutions of tive equations passing the parameter combos 
    tive_multi = simulation_solver(param_df)
    # Create the multi indexed dataframe
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
    # Populate dataframe with nas, will be replaced with peak values
    # At the time when peak occurs, all others will remain as na
    xy_multi = pd.DataFrame(empty, columns=arrays)
    xy_multi.columns = xy_multi.columns.rename("Particle ID", level=0)
    xy_multi.columns = xy_multi.columns.rename("xy_sols", level=1)
    # Extract peaks for each ID
    for col_id in param_df["id"]:
        prev_step = tive_multi[col_id,"I"].shift(fill_value=0)
        next_step = tive_multi[col_id,"I"].shift(fill_value=0,periods = -1)
        current = tive_multi[col_id,"I"]
        # If the viral load for a given time is greater than the previous step
        # and the next time step, then it is a peak viral load
        peak = (current > prev_step) & (current > next_step) & (current > 100)
        # Extract the time which th epeak occurs
        x = pd.Series(tive_multi["time"][peak][1:])
        # Extract the 
        y = tive_multi[col_id,"I"][peak][1:]
        d = {'x':x, 'y':y}
        addition = pd.DataFrame(data = d)
        xy = pd.concat([xy.loc[:,["x","y"]], addition], ignore_index = True)
        xy_multi[col_id, "x"].iloc[x.index] =  x
        xy_multi[col_id, "y"].iloc[y.index] =  y
    return xy_multi


def converge_check(xdata,ydata):
    """ This function checks if a curve converges to a constant value 
    eventually. If it does, then it will label the curve with the "converges" 
    label. Otherwise, the function returns 0. 
    
    xdata (Series, 1-D array): times at which peak occurs 
    ydata (Series, 1-D array): infectous cells/viral load at times in xdata"""
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
        return "fast"
    else:
        return 0
    
def regression_check(xdata, ydata):
    """ This function checks if an exponential model or linear model fits the 
    peak curves better, or if the peaks remain constant.
    
    xdata (Series, 1-D array): times at which peak occurs 
    ydata (Series, 1-D array): infectous cells/viral load at times in xdata"""
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
        return "slow"
    # If exponential performs better than linear, label as exponential
    elif (exp_r > lin_r) and (exp_pval < 0.001):
        return "fast"
    # For remaining curves (linear performs better), if slope significant
    # label as linear
    elif lin_pval < 0.001:
        return "slow"
    # for curves where slope is insignificant, label as constant
    else:
        # c stands for constant
        return "slow"


def LabelMaker(ID, xy_multi):
    """ Creates a label for parameter combo ID
    
    ID (str): id of parameter combination to be labeled
    xy_multi (dataframe): multi-indexed dataframe containing coordinates of peaks 
    for all IDS"""
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
    

def SingleIteration(X,y,max_samps=10, k_neigh=5, n_estimates=100, RandState=0):
    """ Performs a sinle iteration of iterative smote. Splits the ground truth 
    data into training and testing set, fits a random forest model on 
    training data, calculates number of synthetic samples using the uncertainty 
    for predictions of test data.
    
    X (dataframe): dataframe of predictors
    y (Series, list): response
    max_samps (int): maximum number of synthetic samples generated around each 
    datapoint
    k_neigh (int): number of nearest neighbors used to generate synthetic data
    n_estimates (int): number of estimators for random forest
    RandState (int): random state of random forest
    
    returns:
        new_X (dataframe): synthetic samples
        score (float): score of random forest before generating synthetic samples
        len(X) (int): number of observations before generating synthetic samples
    """
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RandState)
    # define random forest and fit to training data
    RF = RandomForestClassifier(n_estimators=n_estimates, 
                                random_state=RandState)
    RF.fit(X_train,y_train)
    score = RF.score( X_test, y_test)
    # Get predicted class probabilities for testing data
    probs = RF.predict_proba(X_test)
    # Get the index of the class with the max probability per data point
    maxs = np.argmax(probs,axis=1)
    # Subset to only the max class probabilities
    max_arr=probs[range(len(probs)),maxs]
    
    # Calculate the number of samples to generate around each observation
    num_samps = np.round(max_samps*((1-max_arr)/(1-(1/np.unique(y_train).size)))).astype(int)
    # Generate synthetic data from convex combinations
    synth_data = gen_convex_combos(data=X_test,m=num_samps,n=k_neigh)
    # Dataframe of all new synthetic data
    new_X = pd.DataFrame(dict(zip(X.columns, synth_data.transpose())))
    return new_X, score, len(X)

def gen_convex_combos(data, m, n=5, RandState=0):
    """ Generates convex combinations for each datapoint.
    data (dataframe): dataframe of parameter combinations which synthetic data 
    will be generated around
    m (series, list): vector where the ith observation stores the number of 
    synthetic samples to generate around the observation of index i in data
    n (int): Number of nearest neighbors used to generate convex neighborhood
    RandState (int): Random state"""
    # Convert dataframe to an array 
    dataArray = data.values
    # Get number of observations and number of dimensions
    num_obs = dataArray.shape[0]
    num_dim = dataArray.shape[1]
    # Compute pairwise distances of all points in the dataframe
    # This will retrun an nxn symmetric array with 0s along the diagonal
    distances = cdist(dataArray, dataArray)
    # Array with columns containing the indices that would sort points 
    # to the data point whose index is the column
    nearestIndices = np.argsort(distances,axis=0)[1:n+1]
    # Create array to store synthetic data
    synthetic_data = np.empty((0,num_dim))
    # For each observation, find the convex hull of the n nearest neighbors
    for i in range(num_obs):
        neighbors = dataArray[nearestIndices[:,i]]
        convexHull = ConvexHull(neighbors)
        # For each entry, l, in the vector m, generate l points within 
        # the convex hull
        for j in range(m[i]):
            # generate random weights for each vertex of complex hull
            weights = np.random.rand(len(convexHull.vertices))
            # Make sum of weights equal to 1
            weights /= np.sum(weights)
            # coordainates of vertices
            vertices = convexHull.points[convexHull.vertices]
            # New point is weight dotted with vertices
            print("weights: ", weights)
            print("vertices: ", vertices)
            newpoint = np.dot(weights, vertices)
            # add point to synthetic data array
            synthetic_data = np.vstack([synthetic_data, newpoint])
    # Return the array containing all synthetic data
    return synthetic_data

def AditSmote(params_dict, iterations, n, max_samps, k_neigh, n_estimates=100, RandState=0):
    """ Performs Iterative Smote with a given number of iterations.
    
    params_dict (dict): Dictionary mapping parameter names to the value samples
    are centered around
    iterations (int): number of iterations to perform
    n (int): number of initial samples to draw
    max_samps (int): maximum number of smaples to generate around a point
    k_neigh (int): Number of nearest neighbors used to generate convex neighborhood
    n_estimates (int): number of estimators for random forest
    RandState (int): random state
    """
    # First Generate Initial (unlabeled) parameter combo samples
    parameter_df = initial_params(n, params_dict)
    # Iterate to efficiently draw samples
    iter_score = []
    num_points = []
    for k in range(iterations):
        xy_multi = get_peaks(parameter_df)
        # Label each curve
        labels = parameter_df["id"].apply(LabelMaker, args=(xy_multi,)).reset_index()   
        labels['index'] = labels['index'].astype(str)
        # Rename columns from default apply format to make more descriptive of data
        labels = labels.rename(columns={"index": "ID", "id" : "label"})
        # Generate synthetic data from convex combos
        synth_data, score, length = SingleIteration(parameter_df.iloc[:,1:],labels["label"], 
                                     max_samps=max_samps, k_neigh=k_neigh, 
                                     n_estimates=n_estimates, RandState=RandState)
        iter_score += [score]
        num_points += [length]
        # Add an ID column to the synthetic samples as the first column
        prev_ids = len(parameter_df)
        id_col = pd.Series(range(prev_ids,prev_ids+len(synth_data))).astype(str)
        synth_data.insert(0, "id", id_col)
        # Add synthetic parameter combos to original parameter combinations
        parameter_df = pd.concat([parameter_df, synth_data], ignore_index=True)
    # Label the final dataframe after k iterations
    xy_multi = get_peaks(parameter_df)
    labels = parameter_df["id"].apply(LabelMaker, args=(xy_multi,)).reset_index()   
    labels['index'] = labels['index'].astype(str)
    # Rename columns from default apply format to make more descriptive of data
    labels = labels.rename(columns={"index": "ID", "id" : "label"})
    parameter_df["label"] = labels["label"]
    # Fit Random Forest and get accuracy for final parameters
    X = parameter_df.iloc[:,1:9]
    y = parameter_df["label"]
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y)
    RF =  RandomForestClassifier()
    RF.fit(X_train, y_train)
    iter_score += [RF.score(X_test,y_test)]
    num_points += [len(parameter_df)]
    AlgoStats = pd.DataFrame()
    AlgoStats["score"] = iter_score
    AlgoStats["num_points"] = num_points
    return parameter_df, AlgoStats


Adit_df, AlgoStats = AditSmote(params_dict, 10, 50, 10, 10, n_estimates=100, RandState=0)

# writing the final parameter dataframe
compression_opts = dict(method='zip',
                        archive_name='Adit_df.csv') 
Adit_df.to_csv('Adit_df.zip',compression=compression_opts)

# writing the statistics dataframe
compression_opts2 = dict(method='zip',
                        archive_name='AlgoStats.csv')
AlgoStats.to_csv("AlgoStats.zip",compression=compression_opts2)

