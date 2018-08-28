from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot(dataset_name, path):
    # percent = [1,5,10,20,30,40,50,60,70,80,90,100]
    slack = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    for s in slack:
        # bc_2.0_cluster.txt
        file = open(path+dataset_name+"_"+str(s)+"_cluster.txt", "r")
        cluster = []
        print "reading cluster"
        for line in file:
            line = line[1:-2]
            nums = [float(n) for n in line.split()]
            cluster.append(nums)
        print "generating dendrogram"
        plt.figure(figsize=(20,10))
        plt.xlabel("Clusters")
        plt.ylabel("Distance")
        plt.title("Dendrogram of resultant cluster using HCAC with "+str(s/10)+" slack")
        dn = dendrogram(cluster)
        plt.savefig("/home/vinicios/Documents/graphs_ml/dendrogram_"+dataset_name+"_"+str(s)+".jpg")
        plt.close()
        file.close()

if __name__ == '__main__':
    # plot("bc",sys.argv[1])
    # plot("brasui",sys.argv[1])
    # plot("ctg",sys.argv[1])
    plot("dilma",sys.argv[1])
    # plot("dilma_balanceado",sys.argv[1])
    # plot("ecoli2",sys.argv[1])
    # plot("iris",sys.argv[1])
    # plot("solo",sys.argv[1])
    # plot("solo2",sys.argv[1])
    # plot("brasui2",sys.argv[1])
