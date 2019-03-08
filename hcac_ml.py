#! /usr/bin/python
from __future__ import division
from hcac import HCAC
from experiment import Experiment
import numpy as np


#sort the two variables to the smaller to greater
def swap(x,y):
    if x > y:
        aux = x
        x = y
        y = aux
    return x,y

class ML(Experiment):
    instance_similarity_dict = {}
    instance_dissimilarity_dict = {}

    cluster_similarity = []
    cluster_similarity_dict = {}
    cluster_similarity_list = []

    cluster_dissimilarity = []
    cluster_dissimilarity_dict = {}
    cluster_dissimilarity_list = []

    instances = []
    instancesA = []
    instancesB = []

    upper_cluster = {}
    lower_cluster = {}
    upper_instance = {}
    lowerInstance = {}


    def MITML(self, X, A0, slack):
        n = 1
        A = np.matrix(A0)
        e = {}
        lamb = {}
        cont = 0
        for pair in self.instance_dissimilarity_dict:
            if self.instance_dissimilarity_dict[pair] == 0.0:
                e[pair] = 0.01
            else:
                e[pair] = float(self.instance_dissimilarity_dict[pair])
            lamb[pair] = 0.0
        for pair in self.instance_similarity_dict:
            if self.instance_similarity_dict[pair] == 0.0:
                e[pair] = 0.001
            else:
                e[pair] = float(self.instance_similarity_dict[pair])
            lamb[pair] = 0.0

        cont = 0
        converge = True
        maxIteractions = 0
        while converge:
            for pair in self.instance_similarity_dict:
                cont +=1
                delta = -1
                i,j = pair
                row = np.matrix(np.subtract(X[i],X[j]))
                p = row*A*row.T
                if(p == 0.00):
                    p = 0.001
                arg1 =  lamb[pair]
                part1 = delta/2
                part2 = np.divide(1,p)
                part3 = slack/e[pair]
                arg2 = np.multiply(part1, np.subtract(part2,part3))
                alpha =  min(arg1, arg2)

                beta = delta*alpha / (1 - delta*alpha*p)

                e[pair] = slack * e[pair] / (slack + delta * alpha * e[pair])
                lamb[pair] = lamb[pair] - alpha
                part1 = np.dot(A,row.transpose())
                part2 = np.dot(row, A)
                part3 = np.dot(part1, part2)
                A = np.add(A, np.multiply(beta, part3))
            cont = 0
            for pair in self.instance_dissimilarity_dict:
                cont +=1
                delta = 1
                i,j = pair
                row = np.matrix(np.subtract(X[i],X[j]))
                p = float(row*A*row.T)
                if(p == 0.00):
                    p = 0.001
                arg1 = float(lamb[pair])
                part1 = delta/2
                part2 = np.divide(1,p)
                part3 = slack/float(e[pair])
                arg2 = np.multiply(part1, np.subtract(part2,part3))
                alpha =  min(arg1, arg2)
                beta = delta*alpha / (1 - delta*alpha*p)

                e[pair] = slack * e[pair] / (slack + delta * alpha * e[pair])
                lamb[pair] = lamb[pair] - alpha
                part1 = np.dot(A,row.transpose())
                part2 = np.dot(row, A)
                part3 = np.dot(part1, part2)
                A = np.add(A, np.multiply(beta, part3))
            if cont < maxIteractions:
                cont +=1
            else:
                converge = False
        return A


    def set_cluster_similarity(self, abs_pool, minEntropy, dist):
        x,y = abs_pool[minEntropy]

        p = swap(x,y)
        if dist == 0.0:
            dist = 0.001
        self.cluster_similarity_dict[p] = dist
        self.upper_cluster[p] = dist
        self.cluster_similarity_list.append(p)

    def set_cluster_dissimilarity(self, entropy, abs_pool, rel_pool):
        minEntropy = np.argmin(entropy[:,1])
        dist = self.distance[rel_pool[minEntropy]]
        if dist == 0.0:
            dist = 0.001
        for i in range(0, minEntropy):
            x,y = abs_pool[i]
            p = swap(x,y)
            self.lower_cluster[p] = dist
            self.cluster_dissimilarity_dict[p] = dist


    def get_absolute_index(self,x):
        n = self.number_of_elements
        if (x >= self.number_of_elements):
            a = self.result_cluster[x - n][0]
            b = self.result_cluster[x - n][1]
            self.get_absolute_index(a)
            self.get_absolute_index(b)
        else:
            self.instances.append(x)

    def getInstances(self, x, y):
        self.get_absolute_index(x)
        self.instancesA = np.copy(self.instances)
        self.instances = []
        self.get_absolute_index(y)
        self.instancesB = np.copy(self.instances)
        self.instances = []

    def getConstraintsFromCluster(self, clusterConstraints, b):
        #b means the bound to use, 0 is upper and 1 is lower

        boundDict = {}
        instanceDict = {}
        n = self.number_of_elements
        for pair in clusterConstraints:
            dist = clusterConstraints[pair]
            x,y = pair
            pair = swap(x,y)

            self.getInstances(x,y)
            for ia in self.instancesA:
                for ib in self.instancesB:
                    ia,ib = swap(ia,ib)
                    instanceDict[(ia,ib)] = dist
                    if(b == 0):
                        self.upper_instance[(ia,ib)] = dist
                    else:
                        self.lowerInstance[(ia,ib)] = dist
        return instanceDict

    def get_instace_constraints_from_cluster(self):
        self.instance_similarity_dict = dict(self.getConstraintsFromCluster(self.cluster_similarity_dict, 0))
        self.instance_dissimilarity_dict = dict(self.getConstraintsFromCluster(self.cluster_dissimilarity_dict, 1))

    def query_pool(self, a, b, current_cluster):
        #current_cluster is a list that keeps the correspondent cluster label of the position.
        #in the distance matrix we have q elements, where q = n-k, n = number of elements, k = number of previous merges
        #the position 1..q doesnt correspond directly to the label 1..q. Actually, the position 1..q referes to the current_cluster[1..q]
        dc = np.copy(self.distance)
        pos = current_cluster[a], current_cluster[b]
        dc[a][b] = np.inf #set infinity distance to the diagonals
        dc[b][a] = np.inf
        abs_pool = []
        rel_pool = []
        abs_pool.append(pos)    #abs_pool keeps the current position of the cluster
        rel_pool.append((a,b))  #rel_pool keeps the 1..q value of the label of the cluster
        i = 1
        while(i < self.pool_size and dc.shape[0] > self.pool_size): #repeat until the size of the pool reachs its maximum and while have enough pairs to choose
            posa = np.argmin(dc[a])
            posb = np.argmin(dc[b])
            if dc[a][posa] < dc[b][posb]:
                abs_pos = current_cluster[a], current_cluster[posa]
                rel_pos = a, posa
                dc[a][posa] = np.inf
                dc[posa][a] = np.inf
            else:
                abs_pos = current_cluster[b], current_cluster[posb]
                rel_pos = b, posb
                dc[b][posb] = np.inf
                dc[posb][b] = np.inf
            rel_pool.append(rel_pos)
            abs_pool.append(abs_pos)
            i += 1

        entropy = []
        for i in range(len(abs_pool)):
            entropy.append((rel_pool[i],self.get_entropy(abs_pool[i][0],abs_pool[i][1])))

        entropy = np.array(entropy)

        self.set_cluster_dissimilarity(entropy, abs_pool, rel_pool) #insert the dissimilarity clusters
        k = np.argsort(entropy[:,1])
        entropy = entropy[k]
        minEntropy = np.argmin(entropy[:,1])
        self.set_cluster_similarity(abs_pool, minEntropy, self.distance[a][b]) #inser the similarity clusters

        if entropy[0][1] == 0.0:
            choice = a,b
        else:
            choice = entropy[0,0]

        return choice
