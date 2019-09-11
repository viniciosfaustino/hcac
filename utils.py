import numpy as np

def split_data_label(data):
    # print(type(data))
    label = np.array(data[:, -1], dtype=int)
    data = np.delete(data, np.s_[-1], axis=1)
    data = np.array(data, dtype=float)
    return data, label
