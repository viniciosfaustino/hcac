import numpy as np
from datasetModule import Dataset


class Cluster():
    def __init__(self, _number_of_elements: int):
        self.entries = []
        self.cluster_size = np.ones(_number_of_elements)
        self.classes_per_cluster = None
        self.max_entries = _number_of_elements

    def init_class_counter(self, number_of_classes: int, label: list = None):
        self.classes_per_cluster = [np.zeros(number_of_classes) for j in range(self.max_entries)]
        self.classes_per_cluster = np.array(self.classes_per_cluster)
        # for i in range(self.max_entries):
        #     self.classes_per_cluster[i][label[i]] += 1

    def add_entry(self, pair: tuple, distance: float, number_of_elements: int) -> None:
        self.entries.append([pair[0], pair[1], distance, number_of_elements])

    def get_new_entry_size(self, pair: tuple) -> int:
        num = self.cluster_size[pair[0]] + self.cluster_size[pair[1]]
        self.cluster_size[pair[0]] = num
        self.cluster_size = np.delete(self.cluster_size, pair[1])
        return num

    def get_class_counter_from_cluster(self, x: int, label: list) -> None:
        x = int(x)
        stck = [x]
        while stck:
            x = int(stck.pop())
            if x >= self.max_entries:
                pos = x - self.max_entries
                stck.insert(0, self.entries[pos][1])
                stck.insert(0, self.entries[pos][0])
            else:
                class_index = int(label[x])
                self.classes_per_cluster[x][class_index] += 1
