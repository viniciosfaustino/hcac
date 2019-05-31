import numpy as np
from datasetModule import Dataset
from clusterModule import Cluster
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.cluster.hierarchy import linkage
from math import log


class HCAC():
    def __init__(self, _pool_size: int, _max_user_intervention: int, _dataset: Dataset,
                 _distance_function: str = "euclidean", _linkage_method: str = 'average',
                 _is_validation: bool = True):

        self.pool_size = _pool_size
        self.max_user_intervention = _max_user_intervention
        self.intervention_counter = 0
        self.dataset = _dataset
        self.distance_function = _distance_function
        self.linkage_method = _linkage_method
        self.is_validation = _is_validation

        self.cluster = Cluster(self.dataset.size)
        if self.is_validation:
            self.cluster.start_class_counter(self.dataset.number_of_classes)

        self.confidence_array = []
        self.confidence_array = self.get_confidence_array()
        self.threshold = self.get_threshold()
        self.distance_matrix = None
        self.set_distance_matrix()
        self.alias = np.arange(self.dataset.size)

    def set_distance_matrix(self):
        if self.distance_function == 'cosine':
            distance = cosine_distances(self.dataset.data)
        else:
            distance = euclidean_distances(self.dataset.data)

        np.fill_diagonal(distance, np.inf)
        self.distance_matrix = distance

    def get_confidence_array(self) -> list[float]:
        cluster = linkage(self.distance_matrix, method=self.linkage_method, metric=self.distance_function)
        merge_distances = cluster[:, 2]
        confidence = sorted(merge_distances)
        return confidence

    def get_threshold(self) -> float:
        if self.max_user_intervention > 2:
            return self.confidence_array[self.max_user_intervention-2]
        else:
            return self.confidence_array[0]

    def do_clustering(self):
        count = self.dataset.size
        while self.distance_matrix.shape[0] > 2:
            index = self.get_pair_to_merge()
            number_of_elements = self.cluster.get_new_entry_size(index)
            cluster_number = (self.alias[index[0]], self.alias[index[1]])

            self.cluster.add_entry(cluster_number[0], cluster_number[1], self.distance_matrix[index], number_of_elements)
            self.cluster.update_class_counter(count-self.dataset.size, index)

            self.alias[index[0]] = count
            self.alias = np.delete(self.alias, index[1])
            count += 1

            self.update_distance_matrix(index[0], index[1])
        self.cluster.add_entry(0, 1, self.distance_matrix[0][1], self.dataset.size)

    def get_pair_to_merge(self) -> tuple:
        index = np.argmin(self.distance_matrix)
        min_dist_index = np.unravel_index(index, self.distance_matrix.shape)
        merge_confidence = self.get_merge_confidence(min_dist_index)
        if merge_confidence < self.threshold and self.intervention_counter < self.max_user_intervention:
            pool = self.create_pool(min_dist_index)
            if self.is_validation:
                min_dist_index = self.select_merge(pool)
            else:
                #aqui vai uma função de exibir para o usuário quais o pool
                pass

        return min(min_dist_index), max(min_dist_index)

    def get_merge_confidence(self, index: tuple) -> float:
        min_dist = self.distance_matrix[index]
        row = np.delete(self.distance_matrix[index[0]], index[0])
        aux_dist = [np.amin(row)]
        row = np.delete(self.distance_matrix[index[1]], index[1])
        aux_dist.append(np.amin(row))
        return min(row) - min_dist

    def update_distance_matrix(self, index_a: int, index_b: int):
        for i in range(self.distance_matrix.shape[0]):
            self.distance_matrix[i][index_a] = (self.distance_matrix[i][index_a] + self.distance_matrix[i][index_b])/2.0
            self.distance_matrix[index_a][i] = self.distance_matrix[i][index_a]
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=1)

    def create_pool(self, index: tuple):
        row_a = np.array(self.distance_matrix[index[0]], copy=True)
        row_a[index[0]] = np.inf
        row_b = np.array(self.distance_matrix[index[1]], copy=True)
        row_b[index[1]] = np.inf

        row = np.append(row_b, [row_a])
        sorted_index = np.argsort(row)

        pool = []
        size = min(row_a.shape[0], self.pool_size)

        for i in range(size):
            if (sorted_index[i] - sorted_index.shape[0]) >= 0:
                pool.append([index[1], sorted_index[i] - sorted_index.shape[0]])
            else:
                pool.append([index[0], sorted_index[i]])

        return pool

    def get_entropy(self, index: tuple):
        elements_per_class = self.cluster.classes_per_cluster[index[0]] + self.cluster.classes_per_cluster[index[1]]
        total_elements = np.sum(elements_per_class)
        base = self.dataset.number_of_classes
        e = 0
        for i in range(base):
            p = elements_per_class[i]/total_elements
            e -= p * log(p, base)
        return e

    def select_merge(self, pool: list) -> int:
        entropy = [self.get_entropy(pool[i]) for i in range(len(pool))]

        return entropy.index(min(entropy))

    def get_fscore(self):
        if not self.is_validation:
            raise Exception("The dataset has no label")
        else:
            f_max = [0 for i in range(self.dataset.number_of_classes)]
            for i in range(self.dataset.size):
                c_size = np.sum(self.cluster.classes_per_cluster[i])
                for j in range(self.dataset.number_of_classes):
                    k_size = np.sum(self.cluster.classes_per_cluster[:,j])
                    n = self.cluster.classes_per_cluster[i][j]
                    r = n/k_size
                    p = n/c_size
                    if r + p > 0:
                        f = (2*r*p)/(r+p)
                    else:
                        f = 0
                    if f > f_max[j]:
                        f_max[j] = f
            score = 0
            for i in range(self.dataset.number_of_classes):
                score += f_max[i]*np.sum(self.cluster.classes_per_cluster[:,i])/self.dataset.size
            return score


