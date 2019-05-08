# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:18:34 2018

@author: Leora Betesh
"""
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

class DBscanClassifier():
    
    def __init__(self,epsilon,min_points):
        self.cluster_num=0 
        self.epsilon=epsilon 
        self.min_points=min_points 
        
    def create_clusters(self,data):
        self.data=data
        self.cluster_labels = [0]*len(data)
        self.neighbors = {}
        self.clusters = []
        for point_idx in range(len(self.data)):
            if self.cluster_labels[point_idx] == 0:
                self.neighbors[point_idx] = self.region_query(point_idx)
                if len(self.neighbors[point_idx]) >= self.min_points:
                    self.cluster_num += 1
                    new_cluster = self.expand_cluster(point_idx)
                    self.clusters.append(new_cluster)
                else:
                    self.cluster_labels[point_idx] = -1
        
        
    def region_query(self,point_num):
        neighbors = []
        for other_point in [x for x in range(0, len(self.data)) if x != point_num]:
            if np.linalg.norm(self.data[point_num] - self.data[other_point]) < self.epsilon:
                neighbors.append(other_point)          
        return neighbors

    def expand_cluster(self,core_point_idx):
        cluster = [core_point_idx]
        for neighbor_i in self.neighbors[core_point_idx]:
            if self.cluster_labels[neighbor_i] == -1:
                self.cluster_labels[neighbor_i] = self.cluster_num
                cluster.append(neighbor_i)
            elif self.cluster_labels[neighbor_i] == 0:
                self.cluster_labels[neighbor_i] = self.cluster_num
                self.neighbors[neighbor_i] = self.region_query(neighbor_i)
                if len(self.neighbors[neighbor_i]) >= self.min_points:
                    expanded_cluster = self.expand_cluster(neighbor_i)
                    cluster = cluster + expanded_cluster
                else:
                    cluster.append(neighbor_i)
        return cluster       

    def show_clusters(self): 
        ax = plt.axes(projection='3d')       
        ax.scatter3D(self.data[:,0],self.data[:,1],self.data[:,2],s=50,c = self.cluster_labels, cmap=cmap_bold)  
        plt.show()

    def compare_with_sklearn(self, eps, min_points):
        skl_classifier = DBSCAN(eps=eps, min_samples=min_points).fit(self.data)
        skl_labels = skl_classifier.labels_
    
        for i in range(0, len(skl_labels)):
            if not skl_labels[i] == -1:
                skl_labels[i] += 1 
                
            if not skl_labels[i] == self.cluster_labels[i]:
                print ("Data Point {0} does not match.  SKlearn: {1}, My dbscan: {2}".format(i, skl_labels[i], self.cluster_labels[i])) 
     

def main():

    classifier = DBscanClassifier(epsilon=.6,min_points=3)
    classifier.create_clusters(load_iris().data)
    classifier.show_clusters()
    classifier.compare_with_sklearn(eps=.6,min_points=3)
    
if __name__ == '__main__':
    main()