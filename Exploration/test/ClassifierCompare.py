
"""
Created on Sat Apr  1 15:50:46 2023

@author: jeremynachison
"""

import LabelMaker as lm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
# All the classifiers to be evaluated
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# set seed
import random
random.seed(10)
# Read in the relevant data

# contains the coordinates of all peaks in each viral load curve
xy_multi = pd.read_csv('Data/dense/d_xy_multi.csv', header=[0,1])
# contains the data of all particles by particle id
param_df = pd.read_csv("Data/dense/d_param_df.csv",index_col=0)
# change parameter ids to string, for consistency with xy_multi and tive_multi
# this is to allow writing to feather, pickle , etc...
param_df["id"] = param_df["id"].astype(str)

# For each particle generate a label, reset index to use previous indexing as id column
labels = param_df["id"].apply(lm.LabelMaker).reset_index()
labels['index'] = labels['index'].astype(str)
# Rename columns from default apply format to make more descriptive of data
labels = labels.rename(columns={"index": "ID", "id" : "label"}) 


# predictors (the particles)
X = param_df.iloc[:,1:]
# response (the labels of each particle)
y = labels["label"]

# Test model on original data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y)
# Use SMOTE method to generate training data
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X, y)
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(
    over_X, over_y, test_size=0.2, stratify=over_y)

svmsample = SVMSMOTE()
svm_X, svm_y = svmsample.fit_resample(X, y)
svm_X_train, svm_X_test, svm_y_train, svm_y_test = train_test_split(
    svm_X, svm_y, test_size=0.2, stratify=svm_y)


classification_fcts = {#"KNN":KNeighborsClassifier(n_neighbors = 5), 
                      "Linear SVM":SVC(kernel='linear'), 
                      "Radial Basis SVM":SVC(kernel='rbf'),
                     #"Gaussian Process":GaussianProcessClassifier(),
                     "Decision Tree":DecisionTreeClassifier(),
                     "Random Forest":RandomForestClassifier(),
                     #"AdaBoost":AdaBoostClassifier(),
                     "Naive Bayes":GaussianNB(),
                     "QDA":QuadraticDiscriminantAnalysis()}

scores = {}
for method in classification_fcts:
    clf = classification_fcts[method]
    fitted_clf = clf.fit(X_train, y_train)
    clf_score = fitted_clf.score(X_test,y_test)
    scores[method] = [clf_score]

SMOTE_scores = {}
for method in classification_fcts:
    smoted = classification_fcts[method]
    smoted_clf = smoted.fit(over_X_train, over_y_train)
    smoted_score = smoted_clf.score(X_test,y_test)
    SMOTE_scores[method] = [smoted_score]
    
svmSMOTE_scores = {}
for method in classification_fcts:
    svm_smoted = classification_fcts[method]
    svm_smoted_clf = svm_smoted.fit(svm_X_train, svm_y_train)
    svm_smoted_score = svm_smoted_clf.score(X_test,y_test)
    svmSMOTE_scores[method] = [svm_smoted_score]


corrmat = X.corr()
