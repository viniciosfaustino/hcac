import numpy as np
import pandas as pd
from hcacModule import HCAC
from datasetModule import Dataset
from clusterModule import Cluster
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':
    iris = load_iris()
    dataset = Dataset(iris.data, "iris", iris.target)
    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # user_interventions = [0]
    pool_size = 5
    print(dataset.name)
    for ui in user_interventions:
        hcac = HCAC(dataset, pool_size, int(ui*dataset.size)-2)
        hcac.do_clustering()
        # print("maoe-=-----------")
        print(hcac.get_fscore())
        print()

    bc = load_breast_cancer()
    dataset = Dataset(bc.data, "bc", bc.target)
    pool_size = 5
    print(dataset.name)
    for ui in user_interventions:
        hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
        hcac.do_clustering()
        # print("maoe-=-----------")
        print(hcac.get_fscore())
        print()