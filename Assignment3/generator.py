# =============================================================================
# CS508: Information Retrieval 
# Assignment 3
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
            for lw in lem_words_set:
                tf_value[lw][name] = tf(lw, lem_words)
#            V.extend(lem_words)
        except:
            print "tokenizarion error"
            pass
        
    for term in tf_value.keys():
        idf_value[term] = idf(term, all_docs)
        for k in tf_value[term].keys():
            inverted_index[term][k] = tf_value[term][k] * idf_value[term]
            
#    storing varience
    termVar = collections.defaultdict(float)
    for t in inverted_index.keys():
        termVar[t] = np.var(inverted_index[t].values())
        
    sorted_terms = sorted(termVar, key=termVar.get)
    return sorted_terms[int(len(sorted_terms)*0.5):]

# =============================================================================
# Count Documents in class
# =============================================================================
    
def countDocsInClass(c):
    return numDocs[os.path.join(path, c)]

# =============================================================================
# Concatinate Text in class c
# =============================================================================
    
def concatinateText(c):
    text = ""
    
    for i in range(len(D)):
        if c in D[i].split("/"):
            raw = open(D[i]).read()
            raw = raw.translate(None, string.punctuation)
            raw = filter(lambda x: x in printable, raw)
            text+=" "+raw
            
    return text.split()
    
    
# =============================================================================
# NB Training Function
# =============================================================================

def trainNB(C, D):
    prior = collections.defaultdict(float)
    
    T = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    condProb = collections.defaultdict(lambda: collections.defaultdict(float))
    sumCount = 0
    
    V = extractVocab(D)
    N = len(D)
    
    for c in C:
        sumCount = 0
        Nc = countDocsInClass(c)
        prior[c] = float(Nc)/N
        textc = concatinateText(c)
        textCount = Counter(textc)
        
        for t in V:
            T[c][t] = textCount[t]
            sumCount+=T[c][t] + 1
        
        for t in V:
            condProb[t][c] = float(T[c][t] + 1)/sumCount
            
    return V, prior, condProb
    

# =============================================================================
# Function calling
# =============================================================================

C, D = loadFiles()

for i in [0.1, 0.2, 0.3, 0.5]:
    trainData, testData = train_test_split(D, test_size=i, random_state=123)
    V, prior, condProb = trainNB(C, trainData)
    dataToWrite = [C, testData, V, prior, condProb]
    with open('training_'+str(i)+'.json', 'w') as d:
        json.dump(dataToWrite, d)