#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import word2vec
from random import shuffle

words = open('vocabulary').read().split('\n')
data = open('data').read().split('\n')
shuffle(data)
comments = []
scores = []
for line in data:
    if line:
        cols = line.split()
        dic = {}
        for c in cols[1:]:
            p = c.find(':')
            word = int(c[:p])
            freq = int(c[p + 1:])
            if word >= 100:
                dic[word] = freq
        scores.append(int(int(cols[0]) >= 7))
        comments.append(dic)
scores = np.array(scores)

model = word2vec.load('original.bin')


def getFeature(comment):
    s = np.array([model[words[word]] * freq for word, freq in comment.iteritems() if words[word] in model]).sum(0)
    return s / sum([freq for word, freq in comment.iteritems() if words[word] in model])


features = np.array([getFeature(c) for c in comments])


def nBayesClassifier(traindata, trainlabel, testdata, testlabel, threshold):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(traindata, trainlabel)
    ypred = (clf.predict_proba(testdata)[:, 1] > threshold).astype(int)
    print ypred[:20]
    print testlabel[:20]
    accuracy = float(np.sum(ypred == testlabel)) / len(testdata)
    print 'pred=0 test=1', float(np.sum(ypred < testlabel)) / len(testdata)
    print 'pred=1 test=0', float(np.sum(ypred > testlabel)) / len(testdata)
    return ypred, accuracy


split = len(features) // 5
testdata = features[:split]
testlabel = scores[:split]
traindata = features[split:]
trainlabel = scores[split:]

print nBayesClassifier(traindata, trainlabel, testdata, testlabel, 0.5)[1]
