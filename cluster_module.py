import numpy as np
from dataset_module import Dataset


class Cluster():
    def __init__(self, _number_of_elements:int):
        self.entries = []
        self.cluster_size = np.ones(_number_of_elements).tolist()

    def add_entry(self, index_a: int, index_b:int, distance: float, number_of_elements: int):
        self.entries.append([index_a, index_b, distance, number_of_elements])

    def get_fscore(self, dataset:Dataset):
        if dataset.label is None:
            raise Exception("The dataset has no label")
        else:
            pass

    def get_class_from_cluster(self):
        return int

    def get_new_entry_size(self, index: tuple):
        num = self.cluster_size[index[0]] + self.cluster_size[index[1]]
        self.