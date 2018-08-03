# =============================================================================
# CS508: Information Retrieval 
# Assignment 2
# Submitted By: Himanshu aggarwal (MT17015)
# Start Date: 07/03/18
# End Date: 
# 
# =============================================================================

import os
from nltk import word_tokenize
import nltk
import string
import re
import collections
import math
import json


stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)

all_docs = []

#for storing inverted index
inverted_index = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
tf_value = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
idf_value = collections.defaultdict(float)

sum = 0
total_length = 0
dirno = -1
fno = -1
d = []

#------------------------------------------------------------------------------

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
#    return math.log(len(docs)/df(term, docs)) + 1

#------------------------------------------------------------------------------


docID = collections.defaultdict(str)

for root, dirs, files in os.walk("/Users/himanshuaggarwal/Git Repositories/Data/stories", topdown=False):
    for name in dirs:
        d.append(name)

d.append('')    #for root di

wnl = nltk.WordNetLemmatizer()

for root, _, files in os.walk("/Users/himanshuaggarwal/Git Repositories/Data/stories", topdown=False):
    files = [f for f in files if f != 'index.html' and f[0] != '.']
    dirno += 1
    fno = 0
    for name in files:
        f = open(os.path.join(root, name))
        raw = f.read()
        raw = raw.translate(None, string.punctuation)
        fno += 1
        new_raw = filter(lambda x: x in printable, raw)
#        new_raw = re.sub(r'cmu', ' ', new_raw)
        new_raw = re.sub(r'[^\w]', ' ', new_raw)
#        new_raw = re.sub(r'\w*\d+\w*', ' ', new_raw)
        
        try:
            tokens = word_tokenize(new_raw)
            words = [w.lower() for w in tokens if w.lower() not in stop_words]
            words = [w for w in words if w not in punc]
#            words = [n2w(w) if w.isdigit() else w for w in words]
#            words = list(set(words))
            lem_words = [wnl.lemmatize(t) for t in words]
            all_docs.append(lem_words)
            lem_words_set = list(set(lem_words))
            for lw in lem_words_set:
                docID[str(dirno)+str(fno)] = d[dirno]+name
                tf_value[lw][(str(dirno)+str(fno))] = tf(lw, lem_words)
#                print lw, d[dirno]+'_'+name
#                inverted_index[lw].append(d[dirno]+'_'+name)
#            print 'done'
        except:
            print "tokenizarion error"
            pass
    
print "tf done"

#------------------------------------------------------------------------------

#Finding terms in title of documents
titles = collections.defaultdict(list)
titledata = open("/Users/himanshuaggarwal/Git Repositories/Data/stories/index.html")
for i in range(29):
    line = titledata.readline()

for i in range(452):
    line = titledata.readline()
    titleTerms = line.split(">")[-1].split()
    titleTerms = [w.lower() for w in titleTerms if w.lower() not in stop_words]
    titleTerms = [wnl.lemmatize(t) for t in titleTerms]
    titles[line.split(">")[3].split("<")[0]].extend(titleTerms)
        

def termInTitle(term, docKey):
    if docID[docKey] in titles:
        if term in titles[docID[docKey]]:
            return True
        
    return False
            

#------------------------------------------------------------------------------

# IDF and cosine
extraWeight = 1

for term in tf_value.keys():
    idf_value[term] = idf(term, all_docs)
#    print term
    for k in tf_value[term].keys():
        if termInTitle(term, k):
            extraWeight = 3
        else:
            extraWeight = 1
        inverted_index[term][k] = (tf_value[term][k] * idf_value[term]) * extraWeight
    
print "idf, tfidf done"


#------------------------------------------------------------------------------
with open('inverted_index.json', 'w') as ii:
    json.dump(inverted_index, ii)
    

with open('docIDs.json', 'w') as d:
    json.dump(docID, d)


with open('idf.json', 'w') as idf:
    json.dump(idf_value, idf)
    


    
#------------------------------------------



