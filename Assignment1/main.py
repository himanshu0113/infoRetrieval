#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:41:59 2018

@author: himanshuaggarwal
"""
from wordcloud import WordCloud
import time
import json
import collections


inverted_index = collections.defaultdict(list, json.load(open('inverted_index.json')))
docID = collections.defaultdict(str, json.load(open('docIDs.json')))
skip_dict = collections.defaultdict(list, json.load(open('skip_dict.json')))

# Function for Calculating index
def call_ii():
    print 'Option 1'
    
1
# Function for Word Cloud
def make_word_cloud():
    print 'Word Cloud'
    frq = { key:len(value) for key, value in inverted_index.items()}
#    text = ' '.join(text)
    wordcloud = WordCloud(width=900,height=500).generate_from_frequencies(frq)
    image = wordcloud.to_image()
    image.show()
    image.save('word_cloud1.png')
 
    
# Check Skip
def has_skip(x, i):
    return skip_dict[x][i]


# Skip Pointers
def skip_pointer_and(x, y):
    print 'Skip Pointers Algorithm'
    xp = inverted_index[x]
    yp = inverted_index[y]
    
    result = set()
    
    i, j = 0, 0
    
    while i<len(xp) and j<len(yp):
        if xp[i] == yp[j]:
            result.add(xp[i])
            i +=1
            j +=1
        else:
            if xp[i]<yp[j]:
                if has_skip(x,i) and xp[i+5]<yp[j]:
                    while has_skip(x,i) and xp[i+5]<yp[j]:
                        i += 5
                else:
                    i +=1
            else:
                if yp[j]<xp[i]:
                    if has_skip(y,j) and yp[j+5]<xp[i]:
                        while has_skip(y,j) and yp[j+5]<xp[i]:
                            j += 5
                    else:
                        j +=1
    
    return list(result)


def set_and(x, y):
    print 'Normal AND operation'
    xp = inverted_index[x]
    yp = inverted_index[y]
    
    result = set()
    
    i, j = 0, 0
    
    while i<len(xp) and j<len(yp):
        if xp[i] == yp[j]:
            result.add(xp[i])
            i +=1
            j +=1
        else:
            if xp[i]<yp[j]:
                i +=1
            else:
                j +=1
    
    return list(result)


def set_or(x, y):
    print 'Normal OR operation'
    xp = inverted_index[x]
    yp = inverted_index[y]
    
    result = set(xp)
    
    j = 0
    
    while j<len(yp):
        result.add(yp[j])
        j += 1
    
    return list(result)

# MApping for query operators
op = {'ands': lambda x, y: skip_pointer_and(x, y),
      'andn': lambda x, y: set_and(x, y),
      'and': lambda x, y: list(set(inverted_index[x]).intersection(set(inverted_index[y]))),
      'or': lambda x, y: list(set(inverted_index[x]).union(set(inverted_index[y]))),
      'orn': lambda x, y: set_or(x, y),
      'and not': lambda x, y: list(set(inverted_index[x]).difference(set(inverted_index[y]))),
      'or not': lambda x, y: list(set([wd for l in inverted_index.keys() for wd in inverted_index[l]]).difference(set(inverted_index[y])).union(set(op['and'](x,y)))),}

# Remapping file names in dictionary
def remap_dict(l):
    return [docID[x] for x in l]
    

# Function for Query
def query():
    print 'Query time'
    q = raw_input("\n")
    q = q.lower().split()
    x, y = q[0], q[-1]
    if len(q) == 3:
        if q[1] in op:
            start_time = time.time()
            result = op[q[1]](x, y)
            print 'There are ', len(result), ' results.'
            print("--- %s seconds ---" % (time.time() - start_time))
            d = raw_input('Press \'D\' to display all\n')
            if d.lower() == 'd':
                print remap_dict(result)
        else:
            print 'Invalid Operator'
    elif len(q) == 4:
        if (q[1]+' '+q[2]) in op:
            start_time = time.time()
            result = op[q[1]+' '+q[2]](x,y)
            print 'There are ', len(result), ' results'
            print("--- %s seconds ---" % (time.time() - start_time))
            d = raw_input('Press \'D\' to display all\n')
            if d.lower() == 'd':
                print remap_dict(result)
        else:
            print 'Invalid Operator'
    else:
        print 'Invalid query'

#    print x, op, y
    

# Mapping functions
options = {0:call_ii,
           1:make_word_cloud,
           2:query,}
    

# Writing menu driven program
while True:
    try:
        opt = input("1. Calculate Inverted Index\n2. Show Word Cloud\n3. Enter Query\n4. Exit\n")
        if opt == 4:
            break
        # Calling function
        options[opt-1]()
    except:
        print 'Invalid option. Choose again.'
        pass
        
print 'Thank you'
