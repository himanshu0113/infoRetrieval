# IR HW1
# Submitted by: Himanshu Aggarwal, MT17015

import os
from nltk import word_tokenize
import nltk
import string
import re
import collections
import json


stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)

complete_words = []

#for storing inverted index
inverted_index = collections.defaultdict(list)

sum = 0
total_length = 0
dirno = -1
fno = -1
d = []

docID = collections.defaultdict(str)

for root, dirs, files in os.walk("/Users/himanshuaggarwal/PycharmProjects/IR_HW/20_newsgroups", topdown=False):
    for name in dirs:
        d.append(name) 




for root, _, files in os.walk("/Users/himanshuaggarwal/PycharmProjects/IR_HW/20_newsgroups", topdown=False):
    files = [f for f in files if not f[0] == '.']
    dirno += 1
    fno = 0
    for name in files:
        f = open(os.path.join(root, name))
        raw = f.read()
        fno += 1
        new_raw = filter(lambda x: x in printable, raw)
        new_raw = re.sub(r'cmu', ' ', new_raw)
        new_raw = re.sub(r'[^\w]', ' ', new_raw)
        new_raw = re.sub(r'\w*\d+\w*', ' ', new_raw)
        
        try:
            tokens = word_tokenize(new_raw)
            words = [w.lower() for w in tokens if w.lower() not in stop_words]
            words = list(set(words))
            wnl = nltk.WordNetLemmatizer()
            lem_words = [wnl.lemmatize(t) for t in words]
            lem_words = list(set(lem_words))
            for lw in lem_words:
                docID[str(dirno)+str(fno)] = d[dirno]+'_'+name
                inverted_index[lw].append(str(dirno)+str(fno))
#                print lw, d[dirno]+'_'+name
#                inverted_index[lw].append(d[dirno]+'_'+name)
#            complete_words.append(lem_words)
        except:
            print "tokenizarion error"
            pass
    

for k in inverted_index.keys():
    inverted_index[k] = sorted(list(set(inverted_index[k])))
    

with open('inverted_index.json', 'w') as ii:
    json.dump(inverted_index, ii)

    
# Skip dict
skips = 25
skip_dict = collections.defaultdict(list)
for k in inverted_index.keys():
    l = len(inverted_index[k])
    for i in range(l):
        if i%skips ==0 and l>(i+skips):
            skip_dict[k].append(1)
        else:
            skip_dict[k].append(0)

with open('docIDs.json', 'w') as d:
    json.dump(docID, d)
    
with open('skip_dict.json', 'w') as sd:
    json.dump(skip_dict, sd)
    
# =============================================================================
#     
# try:
#     f = open("/Users/himanshuaggarwal/PycharmProjects/IR_HW/20_newsgroups/alt.atheism/51060")
#     raw = f.read()
# except (OSError, IOError) as e:
#     print "Error Encountered: ", e
# 
# import string
# 
# # removing non-printable words from the raw text
# printable = set(string.printable)
# raw = filter(lambda x: x in printable, raw)
# 
# try:
#     print "in"
#     tokens = word_tokenize(raw)
# except:
#     pass
# 
# 
# #words = [w.lower() for w in tokens if w not in (stop_words and punc)]  #more words
# words = [w.lower() for w in tokens if w not in stop_words]  #less words
# words = [w for w in words if (w not in punc and not w.isdigit() and w not in stop_words)]
# # =============================================================================
# # s = unicode(words[0]).encode('utf8')
# # print s
# # =============================================================================
# # =============================================================================
# # 
# # porter = nltk.PorterStemmer()
# # lancaster = nltk.LancasterStemmer()
# # 
# # x = [porter.stem(t) for t in words]
# # y = [lancaster.stem(t) for t in words]
# # =============================================================================
# 
# wnl = nltk.WordNetLemmatizer()
# lem_words = [wnl.lemmatize(t) for t in words]
# lem_words = list(set(lem_words))
# 
# =============================================================================
#len_new = [w for w in lem_words if (w not in punc and not w.isdigit() and w not in stop_words)]
    



