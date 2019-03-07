#! /usr/bin/python

from sklearn import datasets
from hcac import Dataset
from hcac_ml import ML
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import sys
import os
import pickle
from preprocessing import split_data_target


def run_test(path_to_save, dataset, name):
    print(name)
    file = open(path_to_save+"/"+name+"_results.txt", "w+")
    file2 = open(path_to_save+"/"+name+"_results.csv", "w+")
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
    path_to_save = str(sys.argv[1])
    if path_to_save[-1] == '/':
        path_to_save = path_to_save[0:-1]

    embeddings = ["skip_s50", "skip_s100"]
    methods = ["std", "no_stopwords"]
    datasets = ["eleicao","dilma"]
    for d in datasets:
        try:
            os.mkdir(os.path.join(path_to_save,d))
        except:
            pass

        for embedding in embeddings:
            try:
                os.mkdir(os.path.join(path_to_save, d, embedding))
            except:
                pass

            for method in methods:
                try:
                    os.mkdir(os.path.join(path_to_save, d, embedding, method))
                except:
                    pass

                file_handler = open("datasets/"+embedding+"_"+method+"_"+d+".data", "rb")
                dataset = np.array(pickle.load(file_handler))
                data,target = split_data_target(dataset)
                dataset = Dataset(data,target)
                new_path = os.path.join(path_to_save, d, embedding, method)
                run_test(new_path, dataset, d)
                #
                #
                # file_handler = open("datasets/skip_s100/clean_avg_tweet2.data", "rb")
                # # dataset = np.loadtxt("datasets/eleicao_bow.data")
                # dataset = np.array(pickle.load(file_handler))
                # data,target = split_data_target(dataset)
                # print(data.shape)
                # dataset = Dataset(data,target)
                # run_test(dataset, "clean_avg_tweet2")
                # # run_full_test("esolo", dataset, "euclidean")
