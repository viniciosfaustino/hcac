import numpy as np
from datasetModule import Dataset


class Cluster():
    def __init__(self, _number_of_elements:int):
        self.entries = []
        self.cluster_size = np.ones(_number_of_elements)
        self.classes_per_cluster = None
        self.max_entries = _number_of_elements

    def start_class_counter(self, number_of_classes: int, label: list):
        self.classes_per_cluster = [np.zeros(number_of_classes) for j in range(self.max_entries)]
        self.classes_per_cluster = np.array(self.classes_per_cluster)
        # for i in range(self.max_entries):
        #     self.classes_per_cluster[i][label[i]] += 1

    def add_entry(self, index: tuple, distance: float, number_of_elements: int):
        self.entries.append([index[0], index[1], distance, number_of_elements])



    def get_class_from_cluster(self):
        return int

    def get_new_entry_size(self, index: tuple) -> int:
        num = self.cluster_size[index[0]] + self.cluster_size[index[1]]
        self.cluster_size[index[0]] = num
        self.cluster_size = np.delete(self.cluster_size, index[1])
        return num
