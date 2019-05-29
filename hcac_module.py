import numpy as np
from dataset_module import Dataset
from cluster_module import Cluster
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.cluster.hierarchy import linkage


class HCAC():
    def __init__(self, _pool_size: int, _max_user_intervention: int, _dataset: Dataset,
                 _distance_function: str = "euclidean", _linkage_method: str = 'average',
                 _is_validation: bool = True):

        self.pool_size = _pool_size
        self.max_user_intervention = _max_user_intervention
        self.dataset = _dataset
        self.distance_function = _distance_function
        self.linkage_method = _linkage_method
        self.is_validation = _is_validation

        self.cluster = Cluster(self.dataset.size)

        self.confidence_array = []
        self.confidence_array = self.get_confidence_array()
        self.threshold = self.get_threshold()
        self.distance_matrix = None
        self.set_distance_matrix()


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
        while self.distance_matrix.shape[0] > 2:
            index_merge = self.get_pair_to_merge()
            number_of_elements = self.cluster.get_entry_size(index_merge)
            self.cluster.add_entry(index_merge[0], index_merge[1], self.distance_matrix[index_merge], 0)
        pass


    def get_pair_to_merge(self) -> tuple:
        min_dist_index = np.argmin(self.distance_matrix)
        min_dist_index = np.unravel_index(min_dist_index, self.distance_matrix.shape)

        if self.get_merge_confidence(min_dist_index) < self.threshold:
            pass
        else:
            pass
        return min_dist_index

    def get_merge_confidence(self, index: tuple) -> float:
        min_dist = self.distance_matrix[index]
        aux_vec = np.delete(self.distance_matrix[index[0]], index[0])
        aux_dist = [np.amin(aux_vec)]
        aux_vec = np.delete(self.distance_matrix[index[1]], index[1])
        aux_dist.append(np.amin(aux_vec))
        return min(aux_vec) - min_dist


    def update_distance_matrix(self, index_a: int, index_b: int):
        for i in range(self.distance_matrix.shape[0]):
            self.distance_matrix[i][index_a] = (self.distance_matrix[i][index_a] + self.distance_matrix[i][index_b])/2.0
            self.distance_matrix[index_a][i] = self.distance_matrix[i][index_a]
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=1)

    def create_pool(self, index:int):
        return list

    def get_entropy(self):
        return list

    def select_merge(self, pool:list):
        return int

