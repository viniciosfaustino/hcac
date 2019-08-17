from hcacModule import HCAC
from datasetModule import Dataset
from clusterModule import Cluster
from mitmlModule import MITML
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
import numpy as np


class ML:

    def __init__(self, _dataset: Dataset, _pool_size: int, _max_user_intervention: int, _slack: float,
                 _distance_function: str = "euclidean", _linkage_method: str = 'weighted',
                 _is_validation: bool = True):
        self.dataset = _dataset
        self.pool_size = _pool_size
        self.max_user_intervention = _max_user_intervention
        self.slack = _slack

        self.distance_function = _distance_function
        self.distance_matrix = np.zeros(self.dataset.size**2).reshape(self.dataset.size, -1)
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
        self.hcac = hcac
        self.cluster.entries = hcac.cluster.entries

        self.get_all_instance_constraints_from_cluster(hcac.cluster_similarity, hcac.cluster_dissimilarity)

        identity = np.identity(self.dataset.data.shape[1], dtype=float)
        mitml = MITML(self.slack, 1)

        mahalanobis = mitml.run(self.dataset.data, identity, self.instance_similarity, self.instance_dissimilarity)
        # print(mahalanobis)
        mahalanobis_distance_matrix = distance.pdist(self.dataset.data, 'mahalanobis', VI=mahalanobis)
        # print(mahalanobis_distance_matrix)

        mahalanobis_hierarchy = linkage(mahalanobis_distance_matrix, method=self.linkage_method,
                                        metric=self.distance_function)

        # print(mahalanobis_hierarchy)

        self.cluster.entries = mahalanobis_hierarchy
        for i in range(self.dataset.size, 2*self.dataset.size-1):
            self.cluster.get_class_counter_from_cluster(i, self.dataset.label)
        # print(self.cluster.classes_per_cluster)
        print()

    def get_all_instance_constraints_from_cluster(self, cluster_similarity, cluster_dissimilarity):
        self.instance_similarity = self.get_instance_constraints_from_cluster(cluster_similarity)
        self.instance_dissimilarity = self.get_instance_constraints_from_cluster(cluster_dissimilarity)

    def get_instance_constraints_from_cluster(self, constraints: dict) -> dict:
        instances = {}
        for pair in constraints.keys():
            a, b = pair
            a_instances = self.get_cluster_instances(a)
            b_instances = self.get_cluster_instances(b)
            for i in a_instances:
                for j in b_instances:
                    if i != j:
                        instances[(i, j)] = constraints[pair]

        # print(instances)
        return instances

    def get_cluster_instances(self, x: int) -> list:
        instances = []
        stck = [x]
        while stck:
            x = int(stck.pop())
            if x >= self.dataset.size:
                pos = x - self.dataset.size
                stck.insert(0, self.cluster.entries[pos][1])
                stck.insert(0, self.cluster.entries[pos][0])
            else:
                instances.append(x)

        return instances

    def get_output(self) -> tuple:
        return self.cluster, self.instance_similarity, self.instance_dissimilarity
