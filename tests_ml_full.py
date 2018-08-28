#!/usr/bin/python
# -*- coding: utf-8 -*-
from experiment import Experiment
from hcac import Dataset
from sklearn import datasets
from preprocessing import split_data_target
from tests_ml import run_test
import numpy as np
import sys
import collections

import time
#to run the program, execute:
# $python tests_ml_full.py <path to save the results>
if __name__ == '__main__':
    # dataset = datasets.load_iris()
    # # print dataset.target.shape
    # run_test(dataset, "iris", sys.argv[1], "euclidean")
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
    # run_test(dataset,"ecoli2",sys.argv[1], "euclidean")
    #
    # data = np.genfromtxt("datasets/breast-cancer-wisconsin2.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.divide(target, 4)
    # dataset = Dataset(data, target)
    # run_test(dataset, "bc", sys.argv[1], "euclidean")
    #
    # data = np.genfromtxt("datasets/ctg_norm.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.subtract(target, 1)
    # dataset = Dataset(data, target)
    # run_test(dataset,"ctg", sys.argv[1], "euclidean")
    #
    # dataset = np.loadtxt("datasets/brasui.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_test(dataset,"brasui", sys.argv[1], "cosine")

    # dataset = np.loadtxt("datasets/dilma_balanceado.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_test(dataset,"dilma_balanceado", sys.argv[1], "cosine")
    #
    dataset = np.loadtxt("datasets/dilma1.data")
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    print collections.Counter(dataset.target)
    run_test(dataset,"dilma", sys.argv[1], "cosine")

    # dataset = np.loadtxt("datasets/solo.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_test(dataset,"solo", sys.argv[1], "cosine")

    # dataset = np.loadtxt("datasets/solo2.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # print target
    # run_test(dataset,"solo2", sys.argv[1], "cosine")

    # dataset = np.loadtxt("datasets/brasui2.data")
    # data,target = split_data_target(dataset)
    # dataset = Dataset(data,target)
    # run_test(dataset,"brasui2", sys.argv[1], "cosine")

#to run the program, execute:
# $python tests_ml_full.py <path to save the results>
