from hcacModule import HCAC
import numpy as np


def get_fscore(hcac: HCAC):
    f_max = [0 for i in range(hcac.dataset.number_of_classes)]
    for i in range(2*hcac.dataset.size-1):
        c_size = np.sum(hcac.cluster.classes_per_cluster[i])
        for j in range(hcac.dataset.number_of_classes):
            f = 0
            k_size = hcac.cluster.classes_per_cluster[2*hcac.dataset.size - 2][j]
            n = hcac.cluster.classes_per_cluster[i][j]
            if k_size > 0 and c_size > 0:
                r = n / k_size
                p = n / c_size
                if r + p > 0:
                    f = (2 * r * p) / (r + p)
            else:
                f = 0
            if f > f_max[j]:
                f_max[j] = f
    score = 0
    for i in range(hcac.dataset.number_of_classes):
        score += f_max[i] * hcac.cluster.classes_per_cluster[2*hcac.dataset.size - 2][i] / hcac.dataset.size
    return score
