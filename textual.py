#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from hcac_ml import ML
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage

from experiment import Experiment
from hcac import Dataset
from preprocessing import split_data_target
import collections

if __name__ == '__main__':
    name = "brasui"
    file_path = str(sys.argv[2])
    file = open(file_path+"/"+name+"_results.txt", "w+")
    file2 = open(file_path+"/"+name+"_results.csv", "w+")
    input = sys.argv[1]

    data = np.loadtxt(input)
    data, target = split_data_target(data)
    dataset = Dataset(data,target)
    print collections.Counter(dataset.target)

    h = ML(int(dataset.data.shape[0]*0.3),dataset.data.shape[0], "cosine", 5, dataset.target)
    print "starting semi-supervised clustering"
    h.fit(dataset)
    print "semi-supervised clustering finished"
    print h.f_score()
    print "getting instances constraints from cluster"
    h.getInstaceConstraintsFromCluster()
    print "done with constraints"
    X = dataset.data
    A = np.identity(len(dataset.data[0]))

    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(10):
        print "slack used: ", slack[i]
        file.write("slack used: "+ str(slack[i])+"\n")

        print "initializing itml"
        maha = h.MITML(X, A, slack[i])
        print "recalculating distances"
        MM = distance.pdist(dataset.data, 'mahalanobis',VI=maha)
        print "done with distances"
        print "doing unsupervised clustering"
        cluster = linkage(MM, method='average',metric='euclidean')
        h.cluster = cluster
        print "done with unsupervised clustering"
        h.result_cluster = cluster
        print h.f_score()
        file.write(str(h.f_score())+"\n")
        file2.write(str(slack[i])+","+str(h.f_score()[0])+"\n")
        file.write("----------\n")


        print "------"
#to run the program, execute:
# $python tests_ml.py <path to data> <path to save results>
