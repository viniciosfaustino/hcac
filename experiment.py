#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import numpy as np

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from hcac import HCAC
import matplotlib.pyplot as plt
import time
import math

class Experiment(HCAC):

    def __init__(self, constraints, n, dist_func, pool_size, target):
        self.constraints = constraints      #number of user interventions
        self.number_of_elements = n        #number of elements in the dataset
        self.dist_func = dist_func          #distance function to be used
        self.pool_size = pool_size          #size of the pool
        self.confidence = None              #the confidence of the merges
        self.cluster = None                 #the structure used to keep the possible merges
        self.result_cluster = None          #the structure used to store all the merges made with the dataset
        self.distance = None                #distance matrix used
        self.elements_in_cluster = None     #number of elements in each cluster
        self.target = target                #contains the labels of the examples


    def get_entropy(self, a, b):
        #this function gets two cluster a calculates the entropy of the merge of them
        #it returns a float number with the entropy
        a = int(a)
        b = int(b)
        target = []
        for t in self.target:
            target.append(t)
        self.number_of_classes = len(set(target))
        self.number_of_elements_per_class = np.zeros(self.number_of_classes)
        if (a >= self.number_of_elements):
            self.get_class(a)
        else:
            class_index = int(self.target[a])
            self.number_of_elements_per_class[class_index] += 1

        if(b >= self.number_of_elements):
            self.get_class(b)
        else:
            class_index = int(self.target[b])
            self.number_of_elements_per_class[class_index]  += 1

        total_elements = np.sum(self.number_of_elements_per_class)
        total_elements = int(total_elements)
        ent = 0
        total = total_elements + 1
        for i in range(0, self.number_of_classes):
            if(self.number_of_elements_per_class[i] > 0.0):
                p = self.number_of_elements_per_class[i]+1 / (float(total_elements)+self.number_of_classes)
                ent = ent - (p * math.log(p, 2))

        return ent

    def get_class_recursive(self, x):
        #this recursive function will get all the examples class from the target
        x = int(x)
        if (x >= self.number_of_elements):
            pos = x - self.number_of_elements
            a = self.result_cluster[pos][0]
            b = self.result_cluster[pos][1]
            self.get_class(a)
            self.get_class(b)
        else:
            class_index = int(self.target[x])
            self.number_of_elements_per_class[class_index]  += 1
        return

    def get_class(self, x):
        x = int(x)
        stck = []
        stck.insert(0,x)
        while stck:
            x = int(stck.pop())
            if (x >= self.number_of_elements):
                pos = x - self.number_of_elements
                stck.insert(0, self.result_cluster[pos][1])
                stck.insert(0, self.result_cluster[pos][0])
            else:
                class_index = int(self.target[x])
                self.number_of_elements_per_class[class_index] +=1

    #////////////////#
    def f_score(self):
        #this function calculates the f_score of the resultant cluster after the clustering process
        #it returns a tuple that contains the average fscore and a list with each class fscore
        r = 0
        last = self.cluster[-1,:]
        x = int(last[0])
        y = int(last[1])

        last_cluster = int(max(x,y))
        number_of_classes = len(set(self.target))
        k = np.zeros(number_of_classes)
        for i in range(self.number_of_elements):
            k[int(self.target[i])] = k[int(self.target[i])] + 1
        score = 0.0
        f = np.copy(np.zeros(number_of_classes))
        for j in range(0, last_cluster+1):
            self.number_of_elements_per_class = np.zeros(number_of_classes)
            pc = np.where(self.cluster[:,0:2] == j)
            cluster_size = self.cluster[pc[0][0]][3]
            self.get_class(int(j))
            for i in xrange(number_of_classes):
                n = self.number_of_elements_per_class[i]
                if(k[i]>0):
                    r = n/k[i]
                else:
                    r = 0
                p = n/float(cluster_size)

                if ((p + r) > 0):
                    value = 2.0*((p*r)/(p+r))
                else:
                    value = 0.0
                if value > f[i]:
                    f[i] = value


        self.number_of_elements_per_class = np.zeros(number_of_classes)
        self.get_class(x)
        self.get_class(y)
        for i in range(0,number_of_classes):
            score = score + (self.number_of_elements_per_class[i]/float(self.number_of_elements)) * float(f[i])
        return (score, f)
