#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import time
import math

np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True, linewidth=10000, precision=3)

def normalize_data(data):
    for i in xrange(data.shape[0]):
        if np.linalg.norm(data[i]) > 0:
            data[i] = data[i]/np.linalg.norm(data[i])
    return data

class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target


class HCAC:

    def __init__(self, constraints, n, dist_func, pool_size):
        self.constraints = constraints      #number of user interventions
        self.number_of_elements = n        #number of elements in the dataset
        self.dist_func = dist_func          #distance function to be used
        self.pool_size = pool_size          #size of the pool
        self.confidence = None              #the confidence of the merges
        self.cluster = None                 #the structure used to keep the possible merges
        self.result_cluster = None          #the structure used to store all the merges made with the dataset
        self.distance = None                #distance matrix used
        self.elements_in_cluster = None
    #////////////////////////////////#
    #this function gets a ndarray of the data and returns a distance matrix using a previous informed distance method
    def distance_function(self, data):
        if (self.dist_func == "cosine"):
            dist = cosine_distances(data)
        else:
            dist = euclidean_distances(data) #by default, if it gets a unknown distance function, returns the euclidean distances
            if (self.dist_func != "euclidean" and self.dist_func != None):
                print "Distance function '", self.dist_func, "' is invalid, using euclidean instead."

        np.fill_diagonal(dist, np.inf)        #filling the diagonal with "infinity values" to avoid mistakes with the smallest distance
        return dist

    #/////////////////////////#
    def split_cluster(self, x):
        i = np.where(self.cluster[:,0] == x)    #get all the occurences of cluster x in the first column
        j = np.where(self.cluster[:,1] == x)    #does the same to the second
        k = np.array(list(i[0]) + list(j[0]))   #merges all te occurences

        cluster_x = self.cluster[k]             #get all the clustering possibilities with cluster x
        k = np.argsort(cluster_x[:,2])      #sorting by the distance
        cluster_x = cluster_x[k]
        return cluster_x

    #///////////////////////////////#

    #this function recalculate the distances from the new cluster from the previous
    def new_distances(self, x, y, distance=None):
        if distance is None:
            distance = self.distance
        for i in range(distance.shape[0]):
            if x != i and y != i:
                distance[x][i] = float(self.elements_in_cluster[x]*distance[x][i] + self.elements_in_cluster[y]*distance[y][i])/float(self.elements_in_cluster[x] + self.elements_in_cluster[y])
                distance[i][x] = distance[x][i]
        np.fill_diagonal(distance, np.inf)
        distance = np.delete(distance, y, 0)
        distance = np.delete(distance, y, 1)
        return distance
        pass
    #///////////////////////#
    def get_confidence(self):
        self.confidence = []
        distance_copy = np.copy(self.distance)
        while (distance_copy.shape[0] > 2):
            pos = np.argmin(distance_copy)
            pos = np.unravel_index(pos, (distance_copy.shape))
            min_dist = distance_copy[pos]
            distance_copy[pos[0]][pos[1]] = np.inf
            distance_copy[pos[1]][pos[0]] = np.inf
            x,y = pos
            px2 = np.argmin(distance_copy[x])
            py2 = np.argmin(distance_copy[y])
            if(distance_copy[x][px2] < distance_copy[y][py2]):
                min_dist_2 = distance_copy[x][px2]
            else:
                min_dist_2 = distance_copy[y][py2]
            self.confidence.append(min_dist_2 - min_dist)
            distance_copy[pos[0]][pos[1]] = min_dist
            distance_copy[pos[1]][pos[0]] = min_dist
            self.elements_in_cluster[x] += self.elements_in_cluster[y]
            distance_copy = self.new_distances(x,y, distance_copy)
        self.confidence.sort()
        self.elements_in_cluster = [1]*self.distance.shape[0]

    #///////////////////#
    def query_pool(self, a, b, current_cluster):
        dc = np.copy(self.distance)
        pos = current_cluster[a], current_cluster[b]
        dc[a][b] = np.inf
        dc[b][a] = np.inf
        abs_pool = []
        rel_pool = []
        abs_pool.append(pos)
        rel_pool.append((a,b))
        i = 1
        while(i < self.pool_size and dc.shape[0] > self.pool_size):
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
            i +=1

        entropy = []
        for i in range(len(abs_pool)):
            entropy.append((rel_pool[i],self.get_entropy(abs_pool[i][0],abs_pool[i][1])))

        entropy = np.array(entropy)
        k = np.argsort(entropy[:,1])
        entropy = entropy[k]
        if entropy[0][1] == 0.0:
            choice = a,b
        else:
            choice = entropy[0,0]

        return choice
    #/////////////////////#
    def fit(self, dataset):
        self.distance = self.distance_function(dataset.data)
        np.fill_diagonal(self.distance, np.inf)
        self.elements_in_cluster = [1]*dataset.data.shape[0]
        self.get_confidence()
        threshold = self.confidence[self.constraints-2]
        current_index = self.distance.shape[0]
        current_cluster = np.arange(self.distance.shape[0])
        self.result_cluster = []
        used_interactions = 0
        while(self.distance.shape[0] > 2 ):
            pos = np.argmin(self.distance)
            pos = np.unravel_index(pos, (self.distance.shape))
            min_dist = self.distance[pos]
            x,y = pos[0],pos[1]
            x,y = min(pos), max(pos)
            self.distance[x][y] = np.inf
            self.distance[y][x] = np.inf

            px2 = np.argmin(self.distance[x])
            py2 = np.argmin(self.distance[y])
            if(self.distance[x][px2] < self.distance[y][py2]):
                min_dist_2 = self.distance[x][px2]
            else:
                min_dist_2 = self.distance[y][py2]

            self.distance[x][y] = min_dist
            self.distance[y][x] = min_dist
            if (min_dist_2 - min_dist) <= threshold and used_interactions < self.constraints:
                used_interactions += 1
                x,y = self.query_pool(pos[0],pos[1], current_cluster)

            self.elements_in_cluster[x] += self.elements_in_cluster[y]
            count = self.elements_in_cluster[x]

            a = current_cluster[x]
            b = current_cluster[y]

            current_cluster[x] = current_index
            current_index +=1
            current_cluster = np.delete(current_cluster, y)

            self.result_cluster.append([a, b, min_dist, count])

            self.distance = self.new_distances(x,y, self.distance)
            self.elements_in_cluster.pop(y)
        self.result_cluster.append([current_cluster[0], current_cluster[1], self.distance[0][1], self.number_of_elements])
        self.cluster = np.array(self.result_cluster)
