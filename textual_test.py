#!/usr/bin/python
# -*- coding: utf-8 -*-
from experiment_numpy import Experiment
from hcac_numpy import Dataset
from sklearn import datasets
import numpy as np
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import string
import sys
from hcac_ml import ML
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import csv

# from stop_words import get_stop_words
from nltk.corpus import stopwords

def remove_punctuation(s):
    s = s.translate(None, string.punctuation)
    return s
    pass

def set_lower(s):
    s = s.lower()
    return s
    pass

def pre_processing(data):
    a,b = data.shape
    for i in range(a):
        data[i][0] = set_lower(data[i][0])
        data[i][0] = remove_punctuation(data[i][0])
    return data

# f = open("datasets/solo2.csv", "r")
f = open("datasets/Dilma.txt", "r")
data = []
target = []
first = True
cont = 300
i = 0
for line in f:
    data.append([str(line.split("\t")[0])])
    target.append(int(line.split("\t")[1]))
# print data
data = np.array(data)
target = np.array(target)


d = pre_processing(data)

filtered_sentence = []
fs = []
word_tokens = []
stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer("portuguese")
for i in range(d.shape[0]):
    word_tokens.append(word_tokenize(d[i][0]))

    # fs = [stemmer.stem(w.decode('utf-8')) for w in word_tokens[i] if not w in stop_words]
    fs = [w for w in word_tokens[i] if not w in stop_words]
    if(len(fs) > 0):
        for i in xrange(len(fs)):
            filtered_sentence.append(fs[i])
filtered_sentence = list(set(filtered_sentence))
# print len(filtered_sentence), "number of words"
# print filtered_sentence
number_of_words =  len(filtered_sentence)

vector = []
for i in range(data.shape[0]):
    v = np.zeros(number_of_words)
    for word in word_tokens[i]:
        if (word in filtered_sentence):
            v[filtered_sentence.index(word)] +=  1
    vector.append(v)
vec = np.array(vector)

data = vec
dataset = Dataset(data,target)

# print dataset.data

h = ML(int(data.shape[0]*0.5),dataset.data.shape[0], "cosine", 5, dataset.target)
h.fit(dataset)
print h.f_score()[0]
h.getInstaceConstraintsFromCluster()
X = dataset.data
A = np.identity(len(dataset.data[0]))

slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(10):
    print "slack used: ", slack[i]
    maha = h.MITML(X, A, slack[i])
    MM = distance.pdist(dataset.data, 'mahalanobis',VI=maha)
    cluster = linkage(MM, method='average',metric='cosine')
    h.cluster = cluster
    h.result_cluster = cluster
    print h.f_score()


    print "------"

# print "e"
