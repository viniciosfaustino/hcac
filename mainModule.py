import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from hcacModule import HCAC
from mlModule import ML
from datasetModule import Dataset

from evaluationModule import get_fscore
from utils import split_data_label


def save_results(intervention, score, dataset_name, algorithm):
    df = pd.DataFrame({"intervention": intervention, "fscore": score})
    path = os.path.join("results", dataset_name, dataset_name + "_"+algorithm+".csv")
    df.to_csv(path, index=None)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # datasets = ["new_iris", "new_wine2", "new_ecoli2", "new_ionosphere", "new_haberman", "new_segmentation2"]
    # datasets = ["new_wine2", "new_ecoli2", "new_ionosphere", "new_haberman", "new_segmentation2"]

    datasets = ["brasui.data", "solo.data", "eleicao.data", "dilma.data", "tweets.data"]

    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    slack = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

    pool_size = 5
    for d in datasets:
        with open(os.path.join("datasets",d), "rb") as file_handler:
            dataset = pickle.load(file_handler)
        data, labels = split_data_label(dataset)
        dataset = Dataset(data, d, _label=labels)
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

            save_results(user_interventions, score_h, dataset.name, "hcac")
            save_results(user_interventions, score_m, dataset.name, "hcac-ml")




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
