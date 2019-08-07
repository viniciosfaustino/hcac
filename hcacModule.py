import numpy as np
from datasetModule import Dataset
from clusterModule import Cluster
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.cluster.hierarchy import linkage
from math import log


class HCAC:
    def __init__(self, _dataset: Dataset, _pool_size: int, _max_user_intervention: int,
                 _distance_function: str = "euclidean", _linkage_method: str = 'weighted',
                 _is_validation: bool = True):

        self.pool_size = _pool_size
        self.max_user_intervention = _max_user_intervention
        self.intervention_counter = 0
        self.dataset = _dataset
        self.distance_function = _distance_function
        self.distance_matrix = None
        self.set_distance_matrix()
        self.linkage_method = _linkage_method
        self.is_validation = _is_validation

        self.cluster = Cluster(self.dataset.size)
        if self.is_validation:
            self.cluster.init_class_counter(self.dataset.number_of_classes, self.dataset.label)

        self.confidence_array = []
        self.confidence_array = self.get_confidence_array()
        self.threshold = self.get_threshold()

        self.alias = np.arange(self.dataset.size) #current cluster id

        self.cluster_similarity = {}
        self.cluster_dissimilarity = {}
        self.alias = np.arange(self.dataset.size)
        self.current_id = [i for i in range(self.dataset.size)]

    def set_distance_matrix(self):
        if self.distance_function == 'cosine':
            distance = cosine_distances(self.dataset.data)
        else:
            distance = euclidean_distances(self.dataset.data)
        np.fill_diagonal(distance, np.inf)
        self.distance_matrix = distance

    def get_confidence_array(self) -> list:

        dist = np.copy(self.distance_matrix)
        confidence = []
        while dist.shape[0] > 2:
            flat_index = np.argmin(dist)
            index = np.unravel_index(flat_index, dist.shape)
            confidence.append(self.get_merge_confidence(index, dist))
            for i in range(dist.shape[0]):
                dist[i][index[0]] = (dist[i][index[0]] + dist[i][index[1]]) / 2.0
                dist[index[0]][i] = dist[i][index[0]]
            dist = np.delete(dist, index[1], axis=0)
            dist = np.delete(dist, index[1], axis=1)

        return sorted(confidence)

    def get_threshold(self) -> float:
        if self.max_user_intervention > 1:
            return self.confidence_array[self.max_user_intervention-1]
        else:
            return self.confidence_array[0]

    def do_clustering(self):
        count = self.dataset.size
        while self.distance_matrix.shape[0] > 2:
            index = self.get_pair_to_merge()
            number_of_elements = self.cluster.get_new_entry_size(index)
            cluster_number = (self.alias[index[0]], self.alias[index[1]])

            self.cluster.add_entry(cluster_number, self.distance_matrix[index], number_of_elements)
            # alt_index = self.alias[index[0]], self.alias[index[1]]

            self.update_class_counter(count - self.dataset.size, index)
            self.current_id = np.delete(self.current_id, index[1])
            # print("a",self.alias[index[0]])
            self.alias[index[0]] = count
            # print("b", self.alias[index[0]])
            self.alias = np.delete(self.alias, index[1])
            # print(count, self.alias)
            count += 1

            self.update_distance_matrix(index)

        cluster_number = (self.alias[0], self.alias[1])
        self.cluster.add_entry(cluster_number, self.distance_matrix[0][1], self.dataset.size)
        index = (0, 1)
        self.update_class_counter(count - self.dataset.size, index)


    def get_pair_to_merge(self) -> tuple:
        index = np.argmin(self.distance_matrix)
        min_dist_index = np.unravel_index(index, self.distance_matrix.shape)
        merge_confidence = self.get_merge_confidence(min_dist_index)
        if merge_confidence < self.threshold and self.intervention_counter < self.max_user_intervention:
            pool = self.create_pool(min_dist_index)
            if self.is_validation:
                pool_index = self.select_merge(pool)
            else:
                #aqui vai uma função de exibir para o usuário quais o pool
                pass
            min_dist_index = pool[pool_index]

            self.intervention_counter += 1
        # print(self.distance_matrix)

        return min(min_dist_index), max(min_dist_index)

    def get_merge_confidence(self, index: tuple, distance_matrix=None) -> float:
        replace = False
        if distance_matrix is None:
            distance_matrix = np.copy(self.distance_matrix)
            replace = True
        min_dist = distance_matrix[index]
        row = np.copy(distance_matrix[index[0]])
        row[index[1]] = np.inf

        aux_dist = [np.amin(row)]

        row = np.copy(distance_matrix[index[1]])
        row[index[0]] = np.inf

        aux_dist.append(np.amin(row))

        if replace:
            np.fill_diagonal(distance_matrix, np.inf)
            self.distance_matrix = distance_matrix

        return min(aux_dist) - min_dist

    def update_distance_matrix(self, index: tuple):
        for i in range(self.distance_matrix.shape[0]):
            self.distance_matrix[i][index[0]] = (self.distance_matrix[i][index[0]] + self.distance_matrix[i][index[1]])/2.0
            self.distance_matrix[index[0]][i] = self.distance_matrix[i][index[0]]
        self.distance_matrix = np.delete(self.distance_matrix, index[1], axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, index[1], axis=1)
        np.fill_diagonal(self.distance_matrix, np.inf)

    def create_pool(self, index: tuple):
        row_a = np.array(self.distance_matrix[index[0]], copy=True)
        row_a[index[1]] = np.inf
        row_b = np.array(self.distance_matrix[index[1]], copy=True)
        row_b[index[0]] = np.inf

        row = np.append(row_b, [row_a])
        sorted_index = np.argsort(row)

        pool = []
        size = min(row_a.shape[0], self.pool_size)
        row_size = sorted_index.shape[0]
        for i in range(size):
            if sorted_index[i] - row_size/2 >= 0:
                a = int(sorted_index[i] - row_size/2)
                pool.append((min(index[1], a), max(index[1], a)))
            else:
                a = sorted_index[i]
                pool.append((min(index[0], a), max(index[0], a)))
        pool.append(index)
        return sorted(pool)

    def get_entropy(self, index: tuple):
        elements_per_class = self.cluster.classes_per_cluster[index[0]] + self.cluster.classes_per_cluster[index[1]]

        if np.all(elements_per_class == 0):
            pos = self.current_id[index[0]], self.current_id[index[1]]
            elements_per_class[self.dataset.label[pos[0]]] += 1
            elements_per_class[self.dataset.label[pos[1]]] += 1
        total_elements = np.sum(elements_per_class)
        base = self.dataset.number_of_classes
        e = 0
        for i in range(base):
            p = elements_per_class[i]/total_elements
            if p > 0:
                e -= p * log(p, base)
        return e

    def select_merge(self, pool: list) -> int:
        entropy = [self.get_entropy(pool[i]) for i in range(len(pool))]
        s = entropy.index(min(entropy))
        selected = pool[s]

        self.set_cluster_similarity(pool, selected)
        self.set_cluster_dissimilarity(pool, selected)

        return entropy.index(min(entropy))

    def update_class_counter(self, pos: int, index: tuple):
        for i in range(2):
            if self.alias[index[i]] < self.cluster.max_entries:
                j = self.current_id[index[i]]
                self.cluster.classes_per_cluster[pos][self.dataset.label[j]] += 1
            else:
                self.cluster.classes_per_cluster[pos] += self.cluster.classes_per_cluster[self.alias[index[i]] - self.dataset.size]

    def set_cluster_similarity(self, pool, selected):
        dist = [self.distance_matrix[pair] for pair in pool]
        min_dist = min(dist)
        alias = self.alias[selected[0]], self.alias[selected[1]]
        self.cluster_similarity[alias] = min_dist

    def set_cluster_dissimilarity(self, pool, selected):
        dist = [self.distance_matrix[pair] for pair in pool]
        max_dist = max(dist)
        for pair in pool:
            if pair != selected:
                alias = self.alias[pair[0]], self.alias[pair[1]]
                self.cluster_dissimilarity[alias] = max_dist

    def get_output(self):
        return self.cluster, self.cluster_similarity, self.cluster_dissimilarity
