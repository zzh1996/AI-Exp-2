#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import word2vec
from random import shuffle
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing

print 'Loading data...'
words = open('vocabulary').read().split('\n')
data = open('data').read().split('\n')

print 'Parsing data...'
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
            # if word >= 100:
            dic[word] = freq
        scores.append(int(int(cols[0]) >= 7))
        comments.append(dic)
scores = np.array(scores)

print 'Loading model...'
model = word2vec.load('original.bin')


def getFeature(comment):
    s = np.array([model[words[word]] * freq for word, freq in comment.iteritems() if words[word] in model]).sum(0)
    return s / sum([freq for word, freq in comment.iteritems() if words[word] in model])


print 'Extracting features...'
features = np.array([getFeature(c) for c in comments])

print 'Generating train and test data'


def gendata(i):
    testdata = features.reshape(-1, 5, 200).transpose(1, 0, 2)[i]
    testlabel = scores.reshape(-1, 5).transpose(1, 0)[i]
    traindata = np.concatenate([features.reshape(-1, 5, 200).transpose(1, 0, 2)[:i].reshape(-1, 200),
                                features.reshape(-1, 5, 200).transpose(1, 0, 2)[i + 1:].reshape(-1, 200)])
    trainlabel = np.concatenate([scores.reshape(-1, 5).transpose(1, 0)[:i].reshape(-1),
                                 scores.reshape(-1, 5).transpose(1, 0)[i + 1:].reshape(-1)])
    return traindata, trainlabel, testdata, testlabel


traindata, trainlabel, testdata, testlabel = zip(*[gendata(i) for i in range(5)])


def nBayesClassifier(traindata, trainlabel, testdata, testlabel, threshold):
    clf = GaussianNB()
    clf.fit(traindata, trainlabel)
    ypred = (clf.predict_proba(testdata)[:, 1] > threshold).astype(int)
    accuracy = float(np.sum(ypred == testlabel)) / len(testdata)
    # print 'pred=0 test=1', float(np.sum(ypred < testlabel)) / len(testdata)
    # print 'pred=1 test=0', float(np.sum(ypred > testlabel)) / len(testdata)
    return ypred, accuracy


print 'nBayesClassifier...'
acc = [[nBayesClassifier(traindata[i], trainlabel[i], testdata[i], testlabel[i], threshold)[1] for i in range(5)]
       for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]]
print np.array(acc)


def lsClassifier(traindata, trainlabel, testdata, testlabel, lambda_):
    reg = linear_model.Ridge(lambda_)
    reg.fit(traindata, trainlabel)
    ypred = np.where(reg.predict(testdata) >= 0.5, 1, 0)
    accuracy = float(np.sum(ypred == testlabel)) / len(testdata)
    # print 'pred=0 test=1', float(np.sum(ypred < testlabel)) / len(testdata)
    # print 'pred=1 test=0', float(np.sum(ypred > testlabel)) / len(testdata)
    return ypred, accuracy


print 'lsClassifier...'
acc = [[lsClassifier(traindata[i], trainlabel[i], testdata[i], testlabel[i], lambda_)[1] for i in range(5)]
       for lambda_ in [1e-4, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 5000, 10000]]
print np.array(acc)


def softsvm(traindata, trainlabel, testdata, testlabel, sigma, C):
    print 'training', sigma, C
    clf = svm.SVC(C=C, kernel='rbf', gamma=1.0 / sigma ** 2, max_iter=5000)
    clf.fit(traindata, trainlabel)
    print 'validating', sigma, C
    ypred = clf.predict(testdata)
    # print ypred[:20]
    # print testlabel[:20]
    accuracy = float(np.sum(ypred == testlabel)) / len(testdata)
    # print accuracy
    # print 'pred=0 test=1', float(np.sum(ypred < testlabel)) / len(testdata)
    # print 'pred=1 test=0', float(np.sum(ypred > testlabel)) / len(testdata)
    return ypred, accuracy


print 'Calculating d...'
# d=((traindata[:,np.newaxis,:]-traindata)**2).sum()/len(traindata)**2
# ^^^ out of memory

# s=0
# straindata=traindata.copy()
# np.random.shuffle(straindata)
# for i,d in enumerate(straindata):
#     s+=((traindata-d)**2).sum()
#     print i,s/(i+1)/len(traindata)
# d=s/len(traindata)**2
# print 'd=',d
# ^^^ too slow
d = 0.0677

print 'softsvm...'
acc = [[softsvm(traindata[i], trainlabel[i], testdata[i], testlabel[i], sigma, C)[1] for i in range(5)]
       for sigma, C in itertools.product([0.01 * d, 0.1 * d, d, 10 * d, 100 * d], [1, 10, 100, 1000])]
# acc = [[softsvm(traindata[i], trainlabel[i], testdata[i], testlabel[i], sigma, C)[1] for i in range(1)]
#        for sigma, C in itertools.product([0.1 * d], [1])]
print np.array(acc)
