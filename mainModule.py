from hcacModule import HCAC
from datasetModule import Dataset
from sklearn.datasets import load_iris
<<<<<<< HEAD
from evaluationModule import get_fscore
=======
from sklearn.datasets import load_breast_cancer
>>>>>>> 3d83c3f591515468f165edacbd1f4fd74a0440b8

if __name__ == '__main__':
    iris = load_iris()
    dataset = Dataset(iris.data, "iris", iris.target)
<<<<<<< HEAD
    user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # user_interventions = [1]
=======
    #user_interventions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    user_interventions = [0]
>>>>>>> 3d83c3f591515468f165edacbd1f4fd74a0440b8
    pool_size = 5
    print(dataset.name)
    for ui in user_interventions:
        hcac = HCAC(dataset, pool_size, int(ui*dataset.size)-2)
        hcac.do_clustering()
<<<<<<< HEAD
        print(get_fscore(hcac))
=======
        # print("maoe-=-----------")
        print(100*ui, "%: ",hcac.get_fscore())
    print()

    bc = load_breast_cancer()
    dataset = Dataset(bc.data, "bc", bc.target)
    print(dataset.name)
    for ui in user_interventions:
        hcac = HCAC(dataset, pool_size, int(ui * dataset.size) - 2)
        hcac.do_clustering()
        # print("maoe-=-----------")
        print(100 * ui, "%: ", hcac.get_fscore())
    print()
>>>>>>> 3d83c3f591515468f165edacbd1f4fd74a0440b8
