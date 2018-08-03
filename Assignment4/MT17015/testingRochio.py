#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:17:50 2018

@author: himanshuaggarwal
"""

# Testing for Rochio

import json
import collections
import nltk
import string
import operator, math 
from nltk import word_tokenize
import re
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, load_npz
import operator


# =============================================================================
# Testing Rochio
# =============================================================================

def testRochio(testData, mu):
    
    res = []
    
    for d in testData:
        dist = collections.defaultdict(float)
#        print d[0,-1]
        for i in range(mu.shape[0]):
            dist[i] = euclidean_distances(mu[i][0,:-1], d[0,:-1])
#            print i, dist[i]
        res.append(min(dist.iteritems(), key=operator.itemgetter(1))[0])
#        print "max", res
                
    return res
# =============================================================================
# Function Calling
# =============================================================================

for i in [0.1, 0.2, 0.5]:    
    testData = load_npz('data/testdata_'+str(i)+'.npz')
    mu = load_npz('data/mu_'+str(i)+'.npz')
    prediction = testRochio(testData, mu)
    groundTruth = testData[:,-1].todense().flatten().tolist()[0]
    print accuracy_score(groundTruth, prediction)
