import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from scipy.cluster.hierarchy import linkage

from hcacModule import HCAC
from mlModule import ML
from datasetModule import Dataset

from evaluationModule import get_fscore

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    datasets = ["iris", "new_wine2", "new_ecoli2", "new_ionosphere", "new_habberman", "new_segmentation2"]
    for d in datasets:
        path = os.path.join("datasets",d+".data")
        try:
            os.stat(os.path.join("results", d))
        except:
            os.mkdir(os.path.join("results", d))
        df = pd.read_csv(path)
        data = df.values[:, :-1]
        labels = df.values[:, -1].astype(int)
        dataset = Dataset(data, d, labels)


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
            print("slack:",s)
            score_h = []
            score_m = []
            for ui in user_interventions:
                print("intervention: ", ui, end="\r")
                ml = ML(dataset, pool_size, int(ui * dataset.size) - 2, s)
                ml.do_clustering()
                # print("ml:", get_fscore(ml))
                score_m.append(get_fscore(ml))

                hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
                hcac.do_clustering()
                score_h.append(get_fscore(hcac))
            print()
            plt.plot(user_interventions, score_h, label="hcac", color="r")
            plt.plot(user_interventions, score_m, label="ml", color="b")
            plt.xlabel("User Intervention")
            plt.ylabel("FScore")
            plt.title(dataset.name+": FScore x User intervention")
            plt.legend()
            path = os.path.join("results",dataset.name,dataset.name+"_"+str(s)+".png")
            plt.savefig(path)
            plt.clf()
