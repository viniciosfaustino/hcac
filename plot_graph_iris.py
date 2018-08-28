from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import sys
#to run the program, execute:
# $python plot.py <path to save the results>
ml = {}
ml["bc"] = [0.788]
ml["brasui"] = [0.648]
ml["iris"] = [0.817]
ml["ctg"] = [0.806]
ml["ecoli2"] = [0.807]
ml["dilmas"] = [0.776]
ml["dilma_balanceado"] = [0.672]
ml["solo"] = [0.649]
ml["solo2"] = [0.670]
ml["brasui2"] = [0.725]

def plot(dataset_name, path):
    file = open(path+"fscore_"+dataset_name+".txt", "r")
    fscore = []
    percent = []
    for line in file:
        nums = [float(n) for n in line.split(" ")]
        percent.append(nums[0])
        fscore.append(nums[1])
    # print fscore
    # print ml[dataset_name]
    plt.ylabel("FScore")
    plt.xlabel("Percentage")
    plt.title("FScore para HCAC e HCAC-ML")
    h = np.array([ml[dataset_name][0] for i in xrange(len(fscore))])
    plt.plot(percent, fscore, label="HCAC")
    plt.plot(percent,h, label="HCAC-ML")
    plt.legend()
    # plt.hlines(ml[dataset_name],0,100)
    plt.savefig("/home/vinicios/Desktop/ayea/fscore_"+dataset_name+".jpg")
    plt.close()

#to run the program, execute:
# $python plot.py <path to save the results>

if __name__ == '__main__':
    plot("bc",sys.argv[1])
    plot("brasui",sys.argv[1])
    plot("ctg",sys.argv[1])
    plot("dilmas",sys.argv[1])
    plot("dilma_balanceado",sys.argv[1])
    plot("ecoli2",sys.argv[1])
    plot("iris",sys.argv[1])
    plot("solo",sys.argv[1])
    plot("solo2",sys.argv[1])
#to run the program, execute:
# $python plot.py <path to save the results>
