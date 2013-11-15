# -*- coding: utf-8 -*-
"""
@author: Caner
"""

import pandas as pd
from sklearn import decomposition, svm, metrics, ensemble
import numpy as np
import pylab as pl

X = pd.read_csv("/Users/Caner/code/digits/train.csv")

def show_ix(ix):
    """
    Display the digit with index ix
    """
    tup = array(X)[ix] # the corresponding tuple retrieved
    print "The tag is: %s" % tup[0]
    # print tup
    pl.imshow(reshape(tup[1:], (28,28)), cmap=pl.cm.gray_r)

mx = array(X)

labels = mx[:1000,0]
data = mx[:,1:]

# Fit the NMF to extract "topics" from pixel data

nmf = decomposition.NMF(n_components=10, max_iter=100).fit(data[:1000])

# get the picture for top 1000 

top100_nmf = nmf.transform(data[:1000])

df = pd.DataFrame(c_[labels, top100_nmf])

pl.imshow(array(df.groupby(0,axis=0).sum()), interpolation='none')


## -------------------------
#  I. SVM predictor
## -------------------------

# Transform the entire data in order by NMF

d_t = nmf.transform(data)
target = mx[:,0]

# get the training data set
train_d = d_t[:30000]
train_t = target[:30000]

# fit the model
model = svm.SVC(gamma=0.002)
model.fit(train_d, train_t)

# get the test data set
test_d = d_t[30001:]
test_t = target[30001:]

# predict
prediction = model.predict(test_d)

# report metrics
print metrics.confusion_matrix(test_t, prediction)
print metrics.classification_report(test_t, prediction)

## ---------------------------
# Ib. Random Forest Predictor
## ---------------------------

d_t = nmf.transform(data)
target = mx[:,0]

# get the training data set
train_d = d_t[:30000]
train_t = target[:30000]

# get the test data set
test_d = d_t[30001:]
test_t = target[30001:]

rfclassifier = ensemble.RandomForestClassifier(n_estimators=100)
rfclassifier.fit(train_d, train_t)

prediction_rf = rfclassifier.predict(test_d)

print metrics.confusion_matrix(test_t, prediction_rf)
print metrics.classification_report(test_t, prediction_rf)


## ---------------------------
# II. Kaggle Scores Exporter
## ---------------------------

test = pd.read_csv("/Users/Caner/Downloads/test.csv")

test_a = array(test)

res = rfclassifier.predict(nmf.transform(test_a))
res_df = pd.DataFrame({"ImageId": range(1,28001), "Label": res})

res_df.to_csv("/Users/Caner/Desktop/res_rf.csv", sep=",", fmt="%d", index=False)

