#!/usr/bin/python
# -*- coding: utf-8 -*-
from experiment import Experiment
from hcac import Dataset
from sklearn import datasets
from preprocessing import split_data_target
import numpy as np
import sys
import pickle

import time
#to run the program, execute:
# $python tests.py <path to save the results>

#this file is a structure to use for testing datasets

def run_test(dataset_name, dataset, p_intervention, distance_function='euclidean',method=None):
    file_path = str(sys.argv[1])
    if file_path[-1] == '/':
        file_path = file_path[0:-1]
    f = open(str(file_path+"/"+dataset_name+str(p_intervention)+".txt"), "w+")
    saida = open(str(file_path+"/results_"+dataset_name+".txt"), "a")
    score_file = open(str(file_path+"/fscore_"+dataset_name+".txt"), "a")
    start_time = time.time()
    n_intervention = int(dataset.data.shape[0]/100.0 * float(p_intervention)) - 1
    h = Experiment(n_intervention,dataset.data.shape[0], distance_function, 5, dataset.target)
    h.fit(dataset)
    for i in range(h.cluster.shape[0]):
        f.write(str(h.cluster[i]))
        f.write("\n")
    score = h.f_score()
    print("fscore", score)
    f.close()
    saida.write(str("Percentage of intervention: "+str(p_intervention))+'\n')
    saida.write(str("F-Score: "+str(score[0]))+'\n')
    print(str(p_intervention)+" "+str(score[0])+"\n")
    score_file.write(str(p_intervention)+" "+str(score[0])+"\n")
    saida.write(str("F-Score per class: "+str(score[1]))+'\n')
    elapsed_time = time.time() - start_time
    elapsed_time = round(elapsed_time, 2)
    saida.write("Elapsed time: "+str(elapsed_time)+" s")
    saida.write("\n---------------------------------\n")
    score_file.close()


def run_full_test(dataset_name, dataset, distance_function=None):
    percentage = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for p in percentage:
        run_test(dataset_name, dataset, p, distance_function)

if __name__ == '__main__':
    embeddings = ["skip_s50", "skip_s100", "cbow_s50", "cbow_s100"]
    methods = ["std", "no_stopwords"]
    dataset =

    datasets = ["eleicao"]

    for embedding in embeddings:
        for method in methods:
            file_handler = open("datasets/"+embedding+"_"+method+"_"+dataset+".data")
    file_handler = open("datasets/skip_s100/avg_tweet2.data", "rb")
    # dataset = np.loadtxt("datasets/avg_tweet2.data")
    dataset = np.array(pickle.load(file_handler))
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    run_full_test("avg_tweet", dataset, "euclidean")

    file_handler = open("datasets/skip_s100/clean_avg_tweet2.data", "rb")
    # dataset = np.loadtxt("datasets/avg_tweet2.data")
    dataset = np.array(pickle.load(file_handler))
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    run_full_test("clean_avg_tweet", dataset, "euclidean")
#to run the program, execute:
# $python tests.py <path to save the results>
