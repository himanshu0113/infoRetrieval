#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:17:50 2018

@author: himanshuaggarwal
"""

# Testing for KNN

import json
import collections
import nltk
import string
import operator, math 
from nltk import word_tokenize
import re
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, load_npz
import operator


# =============================================================================
# Testing KNN
# =============================================================================

def testKNN(testData, trainData, K):
    
    res = []
    
    for d in testData:
        dist = collections.defaultdict(float)
#        print d[0,-1]
        for i in range(trainData.shape[0]):
            dist[i] = cosine_similarity(trainData[i][0,:-1], d[0,:-1])
#            print i, dist[i]
        neighbours = [i[0] for i in sorted(dist.iteritems(), key=operator.itemgetter(1), reverse=True)[:K]]
        list_of_labels = [trainData[i][0,-1] for i in neighbours]
        res.append(collections.Counter(list_of_labels).most_common()[0][0])
#        print "max", res
                
    return res
# =============================================================================
# Function Calling
# =============================================================================

for i in [0.5]:
    
    K = 3
    
    trainData = load_npz('data/traindata_'+str(i)+'.npz')
    testData = load_npz('data/testdata_'+str(i)+'.npz')

    prediction = testKNN(testData[:200], trainData, K)
    groundTruth = testData[:200,-1].todense().flatten().tolist()[0]
    print accuracy_score(groundTruth, prediction)
