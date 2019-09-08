import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from hcacModule import HCAC
from mlModule import ML
from datasetModule import Dataset

from evaluationModule import get_fscore
from utils import *

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    datasets = ["new_iris", "new_wine2", "new_ecoli2", "new_ionosphere", "new_haberman", "new_segmentation2"]
    # datasets = ["new_wine2", "new_ecoli2", "new_ionosphere", "new_haberman", "new_segmentation2"]

    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # fh = open("datasets/bow_no_stopwords_eleicao.data", "rb")
    # data, label = split_data_label(pickle.load(fh))
    dataset = np.loadtxt("datasets/bow_no_stopwords_eleicao.data")
    data, label = split_data_label(dataset)
    dataset = Dataset(data, "bow_eleicao", label)
    pool_size = 5
    print(dataset.name)
    if not os.path.exists(os.path.join("results", dataset.name)):
        os.mkdir(os.path.join("results", dataset.name))
    for s in slack:
        score_h = []
        score_m = []
        print("slack: ", s)
        for ui in user_interventions:
            print("intervention: ", ui, end="\r")

            ml = ML(dataset, pool_size, int(ui * dataset.size) - 2, s)
            ml.do_clustering()
            score_m.append(get_fscore(ml))
            hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
            hcac.do_clustering()
            score_h.append(get_fscore(hcac))

        print()

        plt.plot(user_interventions, score_h, label="hcac", color="r")
        plt.plot(user_interventions, score_m, label="ml", color="b")
        plt.xlabel("User Intervention")
        plt.ylabel("FScore")
        plt.title(dataset.name + ": FScore x User intervention")
        plt.legend()
        path = os.path.join("results", dataset.name, dataset.name + "_" + str(s) + ".png")
        plt.savefig(path)
        plt.clf()




    # for d in datasets:
    #     path = os.path.join("datasets", d + ".data")
    #     file_handler = open(path, "rb")
    #     dataset = pickle.load(file_handler)
    #     df = pd.read_csv(path)
    #     data = df.values[:, :-1]
    #     labels = df.values[:, -1].astype(int)
    #     dataset = Dataset(data, d.split('_')[1], labels)
    #
    #     if not os.path.exists(os.path.join("results", d.split('_')[1])):
    #         os.mkdir(os.path.join("results", d.split('_')[1]))
    #
    #     user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    #     slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    #
    #     pool_size = 5
    #     print(dataset.name)
    #
    #     for s in slack:
    #         print("slack: ", s)
    #         score_h = []
    #         score_m = []
    #         for ui in user_interventions:
    #             print("intervention: ", ui, end="\r")
    #
    #             ml = ML(dataset, pool_size, int(ui * dataset.size) - 2, s)
    #             ml.do_clustering()
    #             score_m.append(get_fscore(ml))
    #             hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
    #             hcac.do_clustering()
    #             score_h.append(get_fscore(hcac))
    #
    #         print()
    #
    #         plt.plot(user_interventions, score_h, label="hcac", color="r")
    #         plt.plot(user_interventions, score_m, label="ml", color="b")
    #         plt.xlabel("User Intervention")
    #         plt.ylabel("FScore")
    #         plt.title(dataset.name + ": FScore x User intervention")
    #         plt.legend()
    #         path = os.path.join("results", dataset.name, dataset.name + "_" + str(s) + ".png")
    #         plt.savefig(path)
    #         plt.clf()
