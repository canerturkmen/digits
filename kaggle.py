# -*- coding: utf-8 -*-
"""
@author: Caner
"""
#%%
import sys

import pandas as pd
from sklearn import decomposition, svm, metrics, ensemble
import numpy as np
import pylab as pl

sys.path.append("/Users/Caner/code/digits/")
from utils import *

X = pd.read_csv("/Users/Caner/code/digits/train.csv")


#%% -----------------------------
#  0. Feature Extraction
## -----------------------------

mx = array(X)
n_data = 10000 # number of data points to use in feature extraction

labels = mx[:n_data,0]
data = mx[:n_data,1:]

# Fit the NMF to extract "topics" from pixel data
# nmf = decomposition.NMF(n_components=30, max_iter=400).fit(data)

nmf = decomposition.KernelPCA(n_components=20, max_iter=200).fit(data)


#%% Evaluate feature extraction

demo_features(nmf, data, labels)


#%% --------------------------------
#  I. Predictive Models
## --------------------------------

# Transform the entire data in order by NMF

d_t = nmf.transform(mx[:,1:])
target = mx[:,0]

# get the training data set
train_d = d_t[:30000]
train_t = target[:30000]

# get the test data set
test_d = d_t[30001:]
test_t = target[30001:]


#%% Random forest classifier
rfclassifier = ensemble.RandomForestClassifier(n_estimators=300)
rfclassifier.fit(train_d, train_t)

report_results(test_d, test_t, rfclassifier)
# predict(rfclassifier, nmf)


#%% SVM classifier
model = svm.SVC(gamma=0.002)
model.fit(train_d, train_t)




