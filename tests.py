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
        file_path = file_path[0,-1]
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
    run_test(dataset_name, dataset, 0, distance_function)
    run_test(dataset_name, dataset, 5, distance_function)
    run_test(dataset_name, dataset, 10, distance_function)
    run_test(dataset_name, dataset, 20, distance_function)
    run_test(dataset_name, dataset, 30, distance_function)
    run_test(dataset_name, dataset, 40, distance_function)
    run_test(dataset_name, dataset, 50, distance_function)
    run_test(dataset_name, dataset, 60, distance_function)
    run_test(dataset_name, dataset, 70, distance_function)
    run_test(dataset_name, dataset, 80, distance_function)
    run_test(dataset_name, dataset, 90, distance_function)
    run_test(dataset_name, dataset, 100, distance_function)


if __name__ == '__main__':
    # dataset = datasets.load_iris()
    # # print dataset.target.shape
    # run_full_test("iris", dataset, "euclidean")
    #
    # data = np.genfromtxt("datasets/ecoli2.data", delimiter=",")
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.genfromtxt("datasets/ecoli2.data", delimiter=",", dtype="str")
    # np.place(target, target=="cp", 0)
    # np.place(target, target=="im", 1)
    # np.place(target, target=="pp", 2)
    # np.place(target, target=="imU", 3)
    # np.place(target, target=="om", 4)
    # np.place(target, target=="omL", 5)
    # np.place(target, target=="imL", 6)
    # np.place(target, target=="imS", 7)
    # target = np.delete(target, np.s_[0:7], axis=1)
    # tar = []
    # for i in range(target.shape[0]):
    #     tar.append(int(target[i]))
    # target = np.array(tar)
    # dataset = Dataset(data, target)
    # run_full_test("ecoli2",dataset, "euclidean")
    #
    # data = np.genfromtxt("datasets/breast-cancer-wisconsin2.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.divide(target, 4)
    # dataset = Dataset(data, target)
    # run_full_test("bc",dataset, "euclidean")
    #
    # data = np.genfromtxt("datasets/ctg_norm.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.subtract(target, 1)
    # dataset = Dataset(data, target)
    # run_full_test("ctg",dataset, "euclidean")
    #
    # dataset = np.loadtxt("datasets/brasui.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("brasui", dataset, "cosine")
    #
    # dataset = np.loadtxt("datasets/dilma_balanceado.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("dilma_balanceado", dataset, "cosine")
    # #
    # dataset = np.loadtxt("datasets/dilma1.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("dilmas", dataset, "cosine")
    # # #
    # dataset = np.loadtxt("datasets/solo.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("solo", dataset, "cosine")
    #
    # dataset = np.loadtxt("datasets/solo2.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("solo2", dataset, "cosine")
    #
    # dataset = np.loadtxt("datasets/brasui2.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_full_test("brasui2", dataset, "cosine")

    file_handler = open("datasets/edilma.data", "rb")
    # dataset = np.loadtxt("datasets/edilma.data")
    dataset = pickle.load(file_handler)
    data,target = split_data_target(dataset)
    print(data.shape)
    for i in range(data.shape[0]):
        print(len(data[i][0]),"\n")
    # for i in range(data.shape[0]):
    #     print(data[i].shape)
    dataset = Dataset(data,target)
    run_full_test("esolo", dataset, "euclidean")
#to run the program, execute:
# $python tests.py <path to save the results>
