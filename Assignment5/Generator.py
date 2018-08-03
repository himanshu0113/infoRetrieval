#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 01:36:05 2018

@author: himanshuaggarwal
"""

# =============================================================================
# CS508: Information Retrieval 
# Assignment 5
# Submitted By: Himanshu aggarwal (MT17015)
# 
# =============================================================================

import os
from nltk import word_tokenize
import nltk
import string
import re
import collections
from random import shuffle
from scipy.sparse import csr_matrix, save_npz
import json

stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)
wnl = nltk.WordNetLemmatizer()

path = "/Users/himanshuaggarwal/Git Repositories/Data/20_newsgroup_subset"

allWords = []

termID = {}
docTerms = collections.defaultdict(list)

# =============================================================================
# Loading file names and classes
# =============================================================================

numDocs = collections.defaultdict(int)
classMap = collections.defaultdict(int)

def loadFiles():
    classes = []
    docs = []
    i = 0
    
    for root, dirs, files in os.walk(path, topdown=False):
        numDocs[root] = len(files)
        i+=1
        for fname in files:
            docs.append(os.path.join(root, fname))
            
        for name in dirs:
            classes.append(name)
    
    shuffle(docs)
    
    i = 0
    for c in classes:
        classMap[c] = i
        i+=1
    
    return classes, docs

# =============================================================================
# Extract terms from documents
# =============================================================================
    
def extractVocab(Docs):
    
    for name in Docs:
        f = open(name)
        raw = f.read()
        raw = raw.translate(None, string.punctuation)
        new_raw = filter(lambda x: x in printable, raw)
        new_raw = re.sub(r'[^\w]', ' ', new_raw)
        new_raw = re.sub(r'\w*\d+\w*', ' ', new_raw)
        try:
            tokens = word_tokenize(new_raw)
            words = [w.lower() for w in tokens if w.lower() not in stop_words]
            wnl = nltk.WordNetLemmatizer()
            lem_words = [wnl.lemmatize(t) for t in words]
            lem_words_set = list(set(lem_words))
            docTerms[name] = lem_words_set
            allWords.extend(lem_words_set)
#            V.extend(lem_words)
        except:
            print "tokenizarion error"
            pass
       
    W = list(set(allWords))
    i = 0
    for term in W:
        termID[term] = i
        i+=1

    return W

# =============================================================================
# Count Documents in class
# =============================================================================
    
def countDocsInClass(c):
    return numDocs[os.path.join(path, c)]

# =============================================================================
# Create Vector
# =============================================================================

def createVector(doc, voc):
    d = [0] * len(voc)
    
#    for k,v in invertedIndex.items():
#        if doc in v.keys():
#            d[termID[k]] = invertedIndex[k][doc]
    
    for t in docTerms[doc]:
        d[termID[t]] = 1
            
    return d

# =============================================================================
# Create all the samples
# =============================================================================
    
def generateVectors(C,D):
    vectors = []
    
    vocab = extractVocab(D)
    
#    for c in range(len(C)):    
#        for i in range(len(D)):
#            if C[c] in D[i].split("/"):
#                d = createVector(D[i], invertedIndex)
#                d.append(c)
#                vectors.append(d)
    
    for d in D:
        vec = createVector(d, vocab)
        vec.append(classMap[d.split('/')[-2]])
        vectors.append(vec)
                
    return vectors

# =============================================================================
# Function calling
# =============================================================================

C, D = loadFiles()
document_vectors = csr_matrix(generateVectors(C,D))
#save_npz('data/vectors.npz', document_vectors)

with open('data/docWords.json', 'w') as d:
        json.dump(docTerms, d)

