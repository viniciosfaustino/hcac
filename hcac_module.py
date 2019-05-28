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
        self.confidence_array = self.get_confidence()
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

    def get_confidence(self) -> list[float]:
        cluster = linkage(self.distance_matrix, method=self.linkage_method, metric=self.distance_function)
        merge_distances = cluster[:, 2]
        confidence = sorted(merge_distances)
        return confidence

    def update_distance_matrix(self, index_a: int, index_b: int):
        for i in range(self.distance_matrix.shape[0]):
            self.distance_matrix[i][index_a] = (self.distance_matrix[i][index_a] + self.distance_matrix[i][index_b])/2.0
            self.distance_matrix[index_a][i] = self.distance_matrix[i][index_a]
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, index_b, axis=1)

    def get_threshold(self) -> float:
        if self.max_user_intervention > 2:
            return self.confidence_array[self.max_user_intervention-2]
        else:
            return self.confidence_array[0]

    def do_clustering(self):
        pass


    def create_pool(self, index:int):
        return list

    def get_entropy(self):
        return list

    def select_merge(self, pool:list):
        return int

