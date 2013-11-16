# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:01:11 2013

@author: Caner Turkmen
"""

import pandas as pd
from numpy import *
import pylab as pl
from sklearn import metrics

def predict(predictor, reductor, **kwargs):
    """
    Predict results and write them to a file,
    
    :param predictor: any sk-learn model object that implements predict()
    :param reductor: any sk-learn model object for dimensionality reduction that implements transform()
    :param file: keyword argument, path of file to write to e.g. "/Users/Caner/Desktop/res_rf.csv"
    :param testfile: keyword argument, path of the file where the test data is
    
    :returns: None
    """
    
    fileout = "/Users/Caner/Desktop/res_rf.csv" if kwargs.get("file") is None else kwargs["file"]
    filetest = "/Users/Caner/Downloads/test.csv" if kwargs.get("testfile") is None else kwargs["testfile"]
    
    # read test file into numpy.ndarray   
    test_a = array(pd.read_csv(filetest))
    
    res = predictor.predict(reductor.transform(test_a))
    res_df = pd.DataFrame({"ImageId": range(1,28001), "Label": res})
    
    res_df.to_csv(fileout, sep=",", fmt="%d", index=False)
    
def show_ix(ix, X):
    """
    Display the digit with index ix
    
    :param ix: the index of digit within data to draw
    :param X: the data to retrieve digit from
    
    :returns: None
    """
    tup = array(X)[ix] # the corresponding tuple retrieved
    print "The tag is: %s" % tup[0]
    # print tup
    pl.imshow(reshape(tup[1:], (28,28)), cmap=pl.cm.gray_r)
    

def report_results(test_d, test_t, predictor):
    """
    fits a model to test partition of data and reports on the results

    :param test_d: test dataset (partition)
    :param test_t: test targets
    :param predictor: a sk-learn object that implements predict()

    :returns: accuracy score of the given scenario    
    """
    prediction_rf = predictor.predict(test_d)

    print metrics.confusion_matrix(test_t, prediction_rf)
    print metrics.classification_report(test_t, prediction_rf)

    accu = metrics.accuracy_score(test_t, prediction_rf)
    print accu
    return accu
    
def demo_features(reductor, data, targets):
    """
    demonstrates calculated / transformed features over predetermined targets
    
    :param reductor: a sklearn object for feature extraction that implements the transform() method
    :param data: the training data to transform
    :param targets: the targets array to aggregate over
    
    """
    reduced = reductor.transform(data)
    df = pd.DataFrame(c_[targets, reduced])
    features = array(df.groupby(0,axis=0).sum())
    pl.imshow(features, interpolation='none')
    print sum(metrics.pairwise.euclidean_distances(features))