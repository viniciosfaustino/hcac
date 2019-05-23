from dataset_module import Dataset

class Cluster():
    def __init__(self, _number_of_elements:int):
        self.number_of_elements = _number_of_elements
        self.entries = []

    def get_fscore(self, dataset:Dataset):
        if dataset.label is None:
            raise Exception("The dataset has no label")
        else:
            pass

    def get_class_from_cluster(self):
        return int