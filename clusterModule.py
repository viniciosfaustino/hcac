import numpy as np
from datasetModule import Dataset


class Cluster():
    def __init__(self, _number_of_elements:int):
        self.entries = []
        self.cluster_size = np.ones(_number_of_elements)
        self.classes_per_cluster = None
        self.max_entries = _number_of_elements - 1

    def start_class_counter(self, number_of_classes: int):
        self.classes_per_cluster = [np.zeros(number_of_classes) for j in range(self.max_entries)]

    def add_entry(self, index_a: int, index_b:int, distance: float, number_of_elements: int):
        self.entries.append([index_a, index_b, distance, number_of_elements])

    def update_class_counter(self, pos: int, index: tuple, label: list[int]):
        for i in range(2):
            if index[i] <= self.max_entries:
                self.classes_per_cluster[pos][label[index[i]]] += 1
            else:
                self.classes_per_cluster[pos] += self.classes_per_cluster[index[i] - self.max_entries + 1]

    def get_class_from_cluster(self):
        return int

    def get_new_entry_size(self, index: tuple) -> int:
        num = self.cluster_size[index[0]] + self.cluster_size[index[1]]
        self.cluster_size[index[0]] = num
        self.cluster_size = np.delete(self.cluster_size, index[1])
        return num
