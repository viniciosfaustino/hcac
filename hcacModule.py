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

    def set_distance_matrix(self) -> None:
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

    def do_clustering(self) -> None:
        count = self.dataset.size
        
        while self.distance_matrix.shape[0] > 2:
            index = self.get_pair_to_merge()
            
            number_of_elements = self.cluster.get_new_entry_size(index)

            # the index in distance matrix only decreases, but in the cluster the index only increases
            # so there is a mapping to matrix index to cluster valid index
            a = min(self.alias[index[0]], self.alias[index[1]])
            b = max(self.alias[index[0]], self.alias[index[1]])
            alias_index = a,b

            self.cluster.add_entry(alias_index, self.distance_matrix[index], number_of_elements)

            # when its runnig a validation dataset, the number of real classes in each cluster is updated
            # after mergig two clusters
            if self.is_validation:
                self.update_class_counter(count - self.dataset.size, index)

            self.current_id = np.delete(self.current_id, index[1])

            self.alias[index[0]] = count
            self.alias = np.delete(self.alias, index[1])
            
            count += 1

            self.update_distance_matrix(index)

        alias_index = (self.alias[0], self.alias[1])
        self.cluster.add_entry(alias_index, self.distance_matrix[0][1], self.dataset.size)
        index = (0, 1)
        self.update_class_counter(count - self.dataset.size, index)

    def get_pair_to_merge(self) -> tuple:
        """use the merge confidence to decide if there will be a pool, otherwise will use the pair of cluster with the
           minimal distance

        """
        index = np.argmin(self.distance_matrix)
        min_dist_index = np.unravel_index(index, self.distance_matrix.shape)

        merge_confidence = self.get_merge_confidence(min_dist_index)
        if merge_confidence < self.threshold and self.intervention_counter < self.max_user_intervention:
            pool = self.create_pool(min_dist_index)
            if self.is_validation:
                pool_index = self.select_merge(pool)
            else:
                # aqui vai uma função de exibir para o usuário quais as opcoes de merge
                pass
            min_dist_index = pool[pool_index]

            self.intervention_counter += 1

        return min(min_dist_index), max(min_dist_index)

    def get_merge_confidence(self, pair: tuple, distance_matrix=None) -> float:
        # Provide a distance matrix is optional, in this case, when distance matrix is None,
        # self.distance_matrix will be used instead

        if distance_matrix is None:
            distance_matrix = np.copy(self.distance_matrix)

        min_dist = distance_matrix[pair]
        row = np.copy(distance_matrix[pair[0]])
        row[pair[1]] = np.inf

        aux_dist = [np.amin(row)]

        row = np.copy(distance_matrix[pair[1]])
        row[pair[0]] = np.inf

        aux_dist.append(np.amin(row))

        return min(aux_dist) - min_dist

    def update_distance_matrix(self, index: tuple):
        for i in range(self.distance_matrix.shape[0]):
            self.distance_matrix[i][index[0]] = (self.distance_matrix[i][index[0]]
                                                 + self.distance_matrix[i][index[1]])/2.0
            self.distance_matrix[index[0]][i] = self.distance_matrix[i][index[0]]
        self.distance_matrix = np.delete(self.distance_matrix, index[1], axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, index[1], axis=1)
        np.fill_diagonal(self.distance_matrix, np.inf)

    def create_pool(self, pair: tuple) -> list:
        row_a = np.array(self.distance_matrix[pair[0]], copy=True)
        row_a[pair[1]] = np.inf
        row_b = np.array(self.distance_matrix[pair[1]], copy=True)
        row_b[pair[0]] = np.inf

        # unifying the distances of the clusters in pair, and getting the indexes sorted

        row = np.append(row_b, [row_a])
        sorted_index = np.argsort(row)

        pool = []
        size = min(row_a.shape[0], self.pool_size)
        row_size = sorted_index.shape[0]

        # creating the pool with the possible merges by locating which is from the first and second clusters in pair
        for i in range(size):
            if sorted_index[i] - row_size/2 >= 0:
                a = int(sorted_index[i] - row_size/2)
                pool.append((min(pair[1], a), max(pair[1], a)))
            else:
                a = sorted_index[i]
                pool.append((min(pair[0], a), max(pair[0], a)))
        pool.append(pair)
        return sorted(pool)

    def get_entropy(self, pair: tuple):
        # getting the sum of elements in each class
        elements_per_class = self.cluster.classes_per_cluster[pair[0]] + self.cluster.classes_per_cluster[pair[1]]

        # if the counter wasn't initialized, it will added 1 in the class that the example belongs
        if np.all(elements_per_class == 0):
            pos = self.current_id[pair[0]], self.current_id[pair[1]]
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
        # select the merge iwth the lowest entropy and sets the cluster similarity and dissimilarity given the pool
        entropy = [self.get_entropy(pool[i]) for i in range(len(pool))]
        s = entropy.index(min(entropy))
        selected = pool[s]

        self.set_cluster_similarity(pool, selected)
        self.set_cluster_dissimilarity(pool, selected)

        return entropy.index(min(entropy))

    def update_class_counter(self, pos: int, pair: tuple):
        for i in range(2):
            if self.alias[pair[i]] < self.cluster.max_entries:
                j = self.current_id[pair[i]]
                self.cluster.classes_per_cluster[pos][self.dataset.label[j]] += 1
            else:
                self.cluster.classes_per_cluster[pos] += self.cluster.classes_per_cluster[self.alias[pair[i]]
                                                                                          - self.dataset.size]

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
