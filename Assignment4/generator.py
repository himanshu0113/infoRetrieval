# =============================================================================
# CS508: Information Retrieval 
# Assignment 4
# Submitted By: Himanshu aggarwal (MT17015)
# 
# =============================================================================

import os
from nltk import word_tokenize
import nltk
import string
import re
import collections
import json
from random import shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
import math
import numpy as np
from operator import add
from scipy.sparse import csr_matrix, save_npz

stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)
wnl = nltk.WordNetLemmatizer()

path = "/Users/himanshuaggarwal/Git Repositories/Data/20_newsgroup_subset"

# =============================================================================
# TFIDF
# =============================================================================

all_docs = []

#for storing inverted index
inverted_index = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
tf_value = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
idf_value = collections.defaultdict(float)
termID = {}
docTerms = collections.defaultdict(list)

sorted_terms = []

#Frequency
def freq(term, doc):
    return doc.count(term)

#Term-Frequency
def tf(term, document):
    f = freq(term,document)
    if f>0:
        return math.log(f) + 1
    else:
        return float(0)

#Document Frequency
def df(term, docs):
    count = 0
    for doc in docs:
        if freq(term, doc)>0:
            count+=1
    return count
    
#Inverse Document-Frequency
def idf(term, docs):
    return math.log(len(docs)/float(len(tf_value[term].keys()))) + 1
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
            all_docs.append(lem_words)
            lem_words_set = list(set(lem_words))
            docTerms[name] = lem_words_set
            for lw in lem_words_set:
                tf_value[lw][name] = tf(lw, lem_words)
#            V.extend(lem_words)
        except:
            print "tokenizarion error"
            pass
       
    i = 0
    for term in tf_value.keys():
        idf_value[term] = idf(term, all_docs)
        termID[term] = i
        i+=1
        for k in tf_value[term].keys():
            inverted_index[term][k] = tf_value[term][k] * idf_value[term]
            
#    storing varience
#    termVar = collections.defaultdict(float)
#    for t in inverted_index.keys():
#        termVar[t] = np.var(inverted_index[t].values())
        
#    sorted_terms = sorted(termVar, key=termVar.get)
#    return sorted_terms[int(len(sorted_terms)*0.5):]
    return inverted_index

# =============================================================================
# Count Documents in class
# =============================================================================
    
def countDocsInClass(c):
    return numDocs[os.path.join(path, c)]

# =============================================================================
# Create Vector
# =============================================================================

def createVector(doc, invertedIndex):
    d = [0] * len(invertedIndex)
    
#    for k,v in invertedIndex.items():
#        if doc in v.keys():
#            d[termID[k]] = invertedIndex[k][doc]
    
    for t in docTerms[doc]:
        d[termID[t]] = invertedIndex[t][doc]
            
    return d

# =============================================================================
# Create all the samples
# =============================================================================
    
def generateVectors(C,D):
    vectors = []
    
    invertedIndex = extractVocab(D)
    
#    for c in range(len(C)):    
#        for i in range(len(D)):
#            if C[c] in D[i].split("/"):
#                d = createVector(D[i], invertedIndex)
#                d.append(c)
#                vectors.append(d)
    
    for d in D:
        vec = createVector(d, invertedIndex)
        vec.append(classMap[d.split('/')[-2]])
        vectors.append(vec)
                
    return vectors

# =============================================================================
# Train Rochio
# =============================================================================

def trainRochio(C, data):
    temp = collections.defaultdict(list)
    mu = []
    
    for d in data:
        temp[d[-1]].append(d)
        
    mu = csr_matrix(temp[0]).mean(axis=0)
    for c in range(1, len(C)):
        mu = np.concatenate((mu, csr_matrix(temp[c]).mean(axis=0)))
        
        
    
        
#    mu = collections.defaultdict(list)
#    count = collections.defaultdict(lambda: 0)
#    
#    for c in C:
#        mu[classMap[c]] = [0]*len(data[0])
#     
#    for d in data:
#        mu[d[-1]] = map(add, mu[int(d[-1])], d)
#        count[d[-1]]+=1
#        
#    for c in range(len(C)):
#        temp = []
#        for x in mu[c]:
#            temp.append(x/count[c])
#        mu[c] = temp
#        mu[c][:] = [(float(x)/count[c]) for x in mu[c]]
        
    return csr_matrix(mu)

# =============================================================================
# Function calling
# =============================================================================

C, D = loadFiles()
document_vectors = generateVectors(C,D)

for i in [0.1, 0.2, 0.5]:
    trainData, testData = train_test_split(document_vectors, test_size=i, random_state=123)
    mu = trainRochio(C, trainData)
    trainData = csr_matrix(trainData)
    testData = csr_matrix(testData)
    save_npz('data/mu_'+str(i)+'.npz', mu)
    save_npz('data/testdata_'+str(i)+'.npz', testData)
    save_npz('data/traindata_'+str(i)+'.npz', trainData)

