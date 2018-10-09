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
# $python main_ml.py <path to save the results>
if __name__ == '__main__':

    #the code bellow runs the hcac-ml algorithm to the dataset in the datasets folder.
    #The datasets are organized with a bag of words (bow) structure in numpy, with the label in the last column     

    dataset = np.loadtxt("datasets/dilma2.data")
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    run_test(dataset,"dilma2", sys.argv[1], "cosine")

    dataset = np.loadtxt("datasets/solo2.data")
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    run_test(dataset,"solo2", sys.argv[1], "cosine")

    dataset = np.loadtxt("datasets/brasui2.data")
    data,target = split_data_target(dataset)
    dataset = Dataset(data,target)
    run_test(dataset,"brasui2", sys.argv[1], "cosine")

#to run the program, execute:
# $python main_ml.py <path to save the results>
