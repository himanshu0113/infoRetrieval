#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 02:04:38 2018

@author: himanshuaggarwal
"""
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from sklearn import metrics

# =============================================================================
# Distance
# =============================================================================

def dist(a, b):
    return norm(a - b)

# =============================================================================
# Kmeans
# =============================================================================

def kmeans(X, k):

    #random centroids
    m, n= np.shape(X)
    #print m, n
    C = np.mat(np.zeros((k, n)))
    label = np.mat(np.zeros((m, 2)))

    C[:, range(0,n)] = X[np.random.choice(m, k, replace = False), :].todense()
    C = csr_matrix(C)
    #print C

    # copy of initial clusteres assigned
    cluster_update = True
    n_iter = 0
    # copy of initial clusteres assigned
    objective = []

    # Running until cluster updation stops
    while(cluster_update):
        cluster_update = False

        for i in range(m):
            min_dist = np.inf
            min_index = -1

            for j in range(k):
                dist_ij = dist(X[i,:], C[j,:])
                if dist_ij<min_dist:
                    min_dist = dist_ij
                    min_index = j
                    #print min_index

            if label[i,0] != min_index:
                cluster_update = True

            # **2 is for normalizing the distance
            label[i, :] = min_index, min_dist**2

        for i in range(k):
            points = X[np.nonzero(label[:, 0].A == i)[0]]
            C[i, :] = np.mean(points, axis=0)
            
        # value of objective function
        s = 0
        for i in range(m):
            s += label[i, 1]

        objective.append(s)

        n_iter = n_iter + 1
        
        if(n_iter==5): 
            break

    return label, n_iter, objective


# =============================================================================
# Main
# =============================================================================

K = 5

vectors = load_npz('/Users/himanshuaggarwal/Git Repositories/IR/Assignment5/data/vectors.npz')
#vectors2 = load_npz(open('data/w2v.npz'))

#choosing vectors
#    storing varience
#termVar = collections.defaultdict(float)
#for t in range(vectors.shape[0]):
#    termVar[t] = np.var(vectors[t, :-1].todense())
#    
#sorted_terms = sorted(termVar, key=termVar.get)
#
#impPos = sorted_terms[int(len(sorted_terms)*0.5):]


#label, iters, objective = kmeans(vectors[:,:-1], K)
label, iters, objective = kmeans(vectors, K)

#ari = metrics.adjusted_rand_score(vectors[:,-1], label)
#ami = metrics.adjusted_mutual_info_score(vectors[:,-1], label)  

print objective

