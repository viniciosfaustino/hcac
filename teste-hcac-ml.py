#! /usr/bin/python

from sklearn import datasets
from hcac import Dataset
from hcac_ml import ML
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import sys
import pickle


def run_test(dataset, name):
    print(name)
    file_path = str(sys.argv[1])
    file = open(file_path+"/"+name+"_results.txt", "w+")
    file2 = open(file_path+"/"+name+"_results.csv", "w+")
    n = dataset.data.shape[0]
    h = ML(int(n*0.3),dataset.data.shape[0], "euclidean", 5, dataset.target)
    A = np.identity(len(dataset.data[0]))
    h.fit(dataset) #running semi supervised clustering
    print(h.f_score())
    h.getInstaceConstraintsFromCluster()
    X = dataset.data
    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(10):
        file.write("slack used: "+ str(slack[i])+"\n")
        maha = h.MITML(X, A, slack[i])
        MM = distance.pdist(dataset.data, 'mahalanobis',VI=maha)
        cluster = linkage(MM, method='average',metric='euclidean')
        h.cluster = cluster
        h.result_cluster = cluster
        file.write(str(h.f_score()[0])+"\n")
        file2.write(str(slack[i])+","+str(h.f_score()[0])+"\n")
        file.write("----------\n")




if __name__ == '__main__':
    # dataset = datasets.load_iris()
    # run_test(dataset, "iris")
    # data = np.genfromtxt("datasets/breast-cancer-wisconsin2.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.divide(target, 4)
    # dataset = Dataset(data,target)
    # run_test(dataset, "breast_cancer")
    #
    # data = np.genfromtxt("datasets/ctg_norm.data", delimiter=",")
    # target = np.array(data[:,-1], dtype=int)
    # data = np.delete(data, np.s_[-1], axis=1)
    # target = np.subtract(target, 1)
    # dataset = Dataset(data, target)
    # run_test(dataset, "ctg")

    # data = np.genfromtxt("datasets/ecoli2.data", delimiter=",")
    # data = np.delete(data, np.s_[-1], axis=1)
    # print(data)
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
    # run_test(dataset, "ecoli2")

    file_handler = open("datasets/edilma.data", "rb")
    # dataset = np.loadtxt("datasets/edilma.data")
    dataset = np.array(pickle.load(file_handler))
    data,target = split_data_target(dataset)
    print(data.shape)
    # for i in range(data.shape[0]):
    #     print(len(data[i][0]),"\n")
    # for i in range(data.shape[0]):
    #     print(data[i].shape)
    dataset = Dataset(data,target)
    run_test(dataset, "esolo")
    # run_full_test("esolo", dataset, "euclidean")
