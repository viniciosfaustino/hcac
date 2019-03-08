import numpy as np


#this function get a np array and split its data and target in different arrays
def split_data_target(data):
    target = np.array(data[:,-1], dtype=int)
    data = np.delete(data, np.s_[-1], axis=1)
    data = np.array(data, dtype=float)
    print(data)
    return data, target

#this function normalizes the values of the data array
def normalize_data(data):
    for i in xrange(data.shape[0]):
        if np.linalg.norm(data[i]) > 0:
            data[i] = data[i]/np.linalg.norm(data[i])
    return data


class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target
