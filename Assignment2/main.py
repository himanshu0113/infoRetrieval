#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:52:09 2018

@author: himanshuaggarwal
"""

import json
import collections
import nltk
import string
import operator, math
from numpy import linalg as LA
from autocorrect import spell
from word2number import w2n
from num2words import num2words as n2w


inverted_index = collections.defaultdict(lambda: collections.defaultdict(lambda: 0), json.load(open('inverted_index.json')))
docID = collections.defaultdict(str, json.load(open('docIDs.json')))
idf_value = collections.defaultdict(float, json.load(open('idf.json')))


#loading cache
try:
    cache_store = json.load(open('cache.json'))
    cache = collections.defaultdict(list, cache_store[0])
    cache_count = cache_store[1]
except:
    cache = collections.defaultdict(list)
    cache_count = 0



stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)


#------------------------------------------------------------------------------

#Save Cache
def saveCache():
    print "Saving cache"
    cache_store = [cache, cache_count]
    with open('cache.json', 'w') as c:
        json.dump(cache_store, c)

#Print result
def printResult(rank):
    print "There are ", len(rank), " resuts."
    d = raw_input('Press \'D\' to display top 5 results\n')
    if d.lower() == 'd':
        print remap_dict(rank[:5])
#    print rank[:5]
    
#Adding query results to cache
def addToCache(query, scoring, result):
    cache[cache_count] = [query, scoring, result]
    global cache_count
    cache_count = (cache_count+1)%20

def getFromCache(query, scoring):
    for cac in cache:
        if cache[cac][0] == query:
            if cache[cac][1] == scoring:
                return cache[cac][2]
            
    return []

# Remapping file names in dictionary
def remap_dict(l):
    return [docID[x] for x in l]


#Query Preprocess
def queryPreprocess(q):
    q = q.translate(None, string.punctuation)
    q = q.lower().split()
#    q = [spell(w) for w in q if not w.isdigit() else w]       #spelling correction
    
    temp = []
    for w in q:
        if w.isdigit():
            temp.extend(n2w(int(w)).lower().split())
            temp.append(w)
        else:
            try:
                temp.append(str(w2n.word_to_num(w)))
                
            except:
                pass
            w = spell(w)
            temp.append(w)
    
#    q.extend(temp)
    q = temp
        
    q = [w.lower() for w in q if w.lower() not in stop_words]
    wnl = nltk.WordNetLemmatizer()
    q = [wnl.lemmatize(t) for t in q]
#    print "bye"
    return q
#Query input and preprocessing
def getQuery():
    print 'Query time'
    query = raw_input()
#    query = str(query)
    return query
    
    
#Ranking results according to TF-IDF score
def tfidfRank():
#    print "tfidf"
    query = getQuery()
    res =getFromCache(query, "tfidf")
    if res != []:
        print "FROM CACHE"
        printResult(res)
        return
    
    q = queryPreprocess(query)
    result = []
    rank = []
    for w in q:
        result.append(inverted_index[w])
        
    docs = collections.defaultdict(float)
    for r in result:
        for k in r.keys():
            docs[k] = 0
    
    for doc in docs:
        for r in result:
            if doc in r:
                docs[doc] += r[doc]
                
    sorted_docs = sorted(docs.items(), key=operator.itemgetter(1), reverse=True)
    rank = [x for x,_ in sorted_docs]
    printResult(rank)
    addToCache(query, "tfidf", rank)
    
    
#dot product
def dotProd(a, b):
    if len(a)!=len(b):
        return 0
    return float(sum(i[0]*i[1] for i in zip(a,b)))

def normVec(a):
    return float(LA.norm(a))

# Query Vector formation
def queryVector(q):
    qTf = []
    qidf = []
    for w in q:
        qTf.append(math.log(q.count(w)) + 1)
        qidf.append(idf_value[w])
    
    tfidf = [a*b for a,b in zip(qTf, qidf)]
    return tfidf
    
#Cosine similarity
def cosine(a, b):
    try:
        cos = dotProd(a, b) / (normVec(a) * normVec(b))
    except ZeroDivisionError:
        cos = 0
        
    return cos

#Ranking results according to Cosine similaritys    
def cosineRank():
    query = getQuery()
    res =getFromCache(query, "cosine")
    if res != []:
        print "FROM CACHE"
        printResult(res)
        return
    
    q = queryPreprocess(query)
#    print "cosine"
    q_vector = queryVector(q)
    
    result = []
    for w in q:
        result.append(inverted_index[w])
    
    
    docs_vector = collections.defaultdict(list)
    for r in result:
        for k in r.keys():
            docs_vector[k] = []
            
    for doc in docs_vector:
        for r in result:
            if doc in r:
                docs_vector[doc].append(r[doc])
            else:
                docs_vector[doc].append(float(0))
    
    rank = []
    for doc in docs_vector:
        rank.append([doc, cosine(q_vector, docs_vector[doc])])
    
    rank.sort(key=lambda x: x[1], reverse=True)
    rank = [x for x,_ in rank]
    printResult(rank)
    addToCache(query, "cosine", rank)

#------------------------------------------------------------------------------

# Mapping functions
options = {0:tfidfRank,
           1:cosineRank,}
    

# Writing menu driven program
while True:
    try:
        opt = input("1. Query (TF-IDF) \n2. Query (COSINE)\n3. Exit\n")
        if opt == 3:
            saveCache()
            break
        # Calling function
        options[opt-1]()
    except:
        print 'Invalid option. Choose again.'
        pass
        
print 'Thank you'