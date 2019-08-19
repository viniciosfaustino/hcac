import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from scipy.cluster.hierarchy import linkage

from hcacModule import HCAC
from mlModule import ML
from datasetModule import Dataset

from evaluationModule import get_fscore

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    iris = load_iris()
    dataset = Dataset(iris.data, "iris", iris.target)

    bc = load_breast_cancer()
    dataset = Dataset(bc.data, "bc", bc.target)
    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # user_interventions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # user_interventions = [0.5]
    pool_size = 5
    no_intervention = HCAC(dataset, pool_size, 0)
    no_intervention.do_clustering()
    print(dataset.name)
    # slack = [0.6]

    for s in slack:
        score_h = []
        score_m = []
        print("slack used:", s)
        for ui in user_interventions:
            # print("intervention: ", ui)
            ml = ML(dataset, pool_size, int(ui * dataset.size) - 2, s)
            ml.do_clustering()
            # print("ml:", get_fscore(ml))
            score_m.append(get_fscore(ml))

            hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
            hcac.do_clustering()
            score_h.append(get_fscore(hcac))

        plt.plot(user_interventions, score_h, label="hcac", color="r")
        plt.plot(user_interventions, score_m, label="ml", color="b")
        plt.xlabel("user intervention")
        plt.ylabel("fscore")
        plt.legend()
        # plt.show()
        plt.savefig(dataset.name+"_"+str(s)+".png")
        plt.clf()

        # print("HCAC: ", get_fscore(hcac))
        #     # print("ml", get_fscore(ml))
        #     #
        #     # # print("hcac - ml:",get_fscore(hcac) - get_fscore(ml))
        # print()
