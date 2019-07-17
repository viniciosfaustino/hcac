from hcacModule import HCAC
from datasetModule import Dataset
from clusterModule import Cluster
from mitmlModule import MITML
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import numpy as np


class ML:

    def __init__(self, _dataset: Dataset, _pool_size: int, _max_user_intervention: int,
                 _distance_function: str = "euclidean", _linkage_method: str = 'weighted',
                 _is_validation: bool = True):
        self.pool_size = _pool_size
        self.max_user_intervention = _max_user_intervention
        self.dataset = _dataset
        self.distance_function = _distance_function
        self.distance_matrix = np.zeros(self.dataset.size**2).reshape(self.dataset.size, self.dataset.size)
        self.linkage_method = _linkage_method

        self.is_validation = _is_validation

        self.cluster = Cluster(self.dataset.size)
        if self.is_validation:
            self.cluster.init_class_counter(self.dataset.number_of_classes, self.dataset.label)

        self.instance_similarity = {}
        self.instance_dissimilarity = {}

    def do_clustering(self):
        hcac = HCAC(self.dataset, self.pool_size, self.max_user_intervention)
        hcac.do_clustering()

        self.get_instance_constraints_from_cluster(hcac.cluster, hcac.cluster_similarity, hcac.cluster_dissimilarity)

        mitml = MITML()
        mahalanobis = mitml.run()
        mahalanobis_distance_matrix = distance.pdist(self.data, 'mahalanobis', VI=mahalanobis)
        mahalanobis_hierarchy = linkage(mahalanobis_distance_matrix, method = self.linkage_method, metric = self.distance_function)

        self.cluster.entries = mahalanobis_hierarchy
        self.cluster.get_class_counter_from_cluster()

    def get_instance_constraints_from_cluster(self, cluster: list):
        pass



