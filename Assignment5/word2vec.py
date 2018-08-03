#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 01:01:20 2018

@author: himanshuaggarwal
"""

import gensim
import logging
import json
import collections
from scipy.sparse import csr_matrix, save_npz
import numpy as np

path = "/Users/himanshuaggarwal/Git Repositories/Data/GoogleNews-vectors-negative300.bin"

# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, limit=50000)  

docWords = collections.defaultdict(str, json.load(open('data/docWords.json')))

vectors = []
temp = []
for doc in docWords:
    temp = []
    for w in doc: 
        if w in model.vocab:
            temp.extend(model[w])
    vectors.append(temp)

b = np.zeros([len(vectors),len(max(vectors,key = lambda x: len(x)))])
for i,j in enumerate(vectors):
    b[i][0:len(j)] = j
    

#vec = np.matrix(vectors)
#
#save_npz('data/w2v.npz', vectors)

