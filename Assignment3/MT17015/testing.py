# =============================================================================
# CS508: Information Retrieval 
# Assignment 3
# Submitted By: Himanshu aggarwal (MT17015)
# 
# =============================================================================


import json
import collections
import nltk
import string
import operator, math 
from nltk import word_tokenize
import re
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

stop_words = set(nltk.corpus.stopwords.words('english'))
punc = set(string.punctuation)
printable = set(string.printable)
wnl = nltk.WordNetLemmatizer()


# =============================================================================
# Extract Tokens from document which exist in Vocabulary
# =============================================================================

def extractTokens(V, d):
    f = open(d)
    raw = f.read()
    raw = raw.translate(None, string.punctuation)
    new_raw = filter(lambda x: x in printable, raw)
    new_raw = re.sub(r'[^\w]', ' ', new_raw)
    new_raw = re.sub(r'\w*\d+\w*', ' ', new_raw)
    try:
        tokens = word_tokenize(new_raw)
        words = [w.lower() for w in tokens if w.lower() not in stop_words]
        words = list(set(words))
        wnl = nltk.WordNetLemmatizer()
        lem_words = [wnl.lemmatize(t) for t in words]
        lem_words = list(set(lem_words))
        return [w for w in lem_words if w in V]
    except:
        print "tokenizarion error"
        pass

# =============================================================================
# Test NB
# =============================================================================

def testNB(C, V, prior, condProb, d):
    score = collections.defaultdict(float)
    
    W = extractTokens(V, d)
    
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condProb[t][c])
    
    return max(score.iteritems(), key=operator.itemgetter(1))[0]


# =============================================================================
# Function Calling
# =============================================================================

#, 0.2, 0.3, 0.5
for i in [0.1]:
    predicted = []
    groundTruth = []
    
    with open('old/training_'+str(i)+'.json', 'r') as f:
        readFromFile = json.load(f)
        C, testData, V, prior, condProb = readFromFile
    
#    i =0
    for d in testData:
        predicted.append(testNB(C, V, prior, condProb, d))
        groundTruth.append(d.split("/")[-2])
#        print predicted[i], groundTruth[i]
#        i+=1
        
    
    confmat = confusion_matrix(groundTruth, predicted)
    print confmat
    print accuracy_score(groundTruth, predicted)
    
    plt.matshow(confmat)
    plt.colorbar()
    plt.show()
#    plt.save('confusionMatrix_'+str(i)+'.png', format='png')
    