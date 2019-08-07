from hcacModule import HCAC
from mlModule import ML
from datasetModule import Dataset
from sklearn.datasets import load_iris
from evaluationModule import get_fscore

if __name__ == '__main__':
    iris = load_iris()
    dataset = Dataset(iris.data, "iris", iris.target)
    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # user_interventions = [1]
    pool_size = 5
    print(dataset.name)
    # slack = 0.5
    for ui in user_interventions:
        for s in slack:
            ml = ML(dataset, pool_size, int(ui*dataset.size)-2, s)
            ml.do_clustering()
            # print(ml.cluster.entries)
            hcac = HCAC(dataset, pool_size, int(ui*dataset.size)-2)
            hcac.do_clustering()
            print("hcac - ml:",get_fscore(hcac) - get_fscore(ml))

        print()
