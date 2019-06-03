import numpy as np
from hcacModule import HCAC
from datasetModule import Dataset
from sklearn.metrics.pairwise import euclidean_distances

syntetic_data = [[1, 2], [2, 2], [2, 6], [4, 4], [4, 5], [6, 3], [6, 4]]
syntetic_data = [[1, 1], [1, 3], [1, 7], [1, 9], [1, 10]]
syntetic_data = np.array(syntetic_data)
print(syntetic_data)
label = [0, 0, 1, 2, 2]
exp_dist = euclidean_distances(syntetic_data)
np.fill_diagonal(exp_dist, np.inf)
exp_conf = [1.0, 2.0, 2.5]
pool = [(2, 3), (2, 4), (3, 4)]

dataset = Dataset(syntetic_data, "syntetic", label)
user_intervention = 2
pool_size = 2
hcac = HCAC(dataset, pool_size, user_intervention)


def test_set_distace_matrix():
    print(syntetic_data)
    assert np.array_equal(exp_dist, hcac.distance_matrix)


def test_get_confidence_array():
    assert np.array_equal(hcac.confidence_array, exp_conf)


def test_get_threshold():
    assert hcac.get_threshold() == exp_conf[user_intervention-1]


def test_get_entropy():
    index = (3, 4)
    assert hcac.get_entropy(index) == 0.0
    index = (1, 2)
    print(hcac.get_entropy(index))
    print(hcac.macaco)
    assert hcac.get_entropy(index) != 0


def test_create_pool():
    assert hcac.create_pool(index=(3, 4)) == pool


def test_select_merge():
    assert hcac.select_merge(pool) == pool.index((3, 4))


def test_update_distance_matrix():
    new_dist = [np.inf, 2., 6., 8.5,
                2., np.inf, 4., 6.5,
                6., 4., np.inf, 2.5,
                8.5, 6.5, 2.5, np.inf]
    new_dist = np.array(new_dist).reshape(4,4)
    index = (3, 4)
    hcac.update_distance_matrix(index)
    assert np.array_equal(hcac.distance_matrix, new_dist)