import numpy as np


class Dataset():
    def __init__(self, _data: np.ndarray, _name: str, _label: list = None):
        self.data = _data
        self.label = _label
        self.name = _name
        self.size = self.data.shape[0]
        if self.label is not None:
            self.number_of_classes = len(self.label)

