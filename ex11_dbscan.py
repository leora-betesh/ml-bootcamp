# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:18:34 2018

@author: Leora Betesh
"""
from sklearn.datasets import load_iris, make_circles, make_moons
import time
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
np.random.seed(0)
                            
class DBscanClassifier():
    
    def __init__(self,epsilon,min_points):
        self.cluster_num=0 
        self.epsilon=epsilon 
        self.min_points=min_points 
        
    def fit(self,data):
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
        plt.title("3 features plotted as clusters", size=18)
        plt.show()

    def compare_with_sklearn(self, eps, min_points):
        skl_classifier = DBSCAN(eps=eps, min_samples=min_points).fit(self.data)
        skl_labels = skl_classifier.labels_
    
        for i in range(0, len(skl_labels)):
            if not skl_labels[i] == -1:
                skl_labels[i] += 1 
                
            if not skl_labels[i] == self.cluster_labels[i]:
                print ("Data Point {0} does not match.  SKlearn: {1}, My dbscan: {2}".format(i, skl_labels[i], self.cluster_labels[i])) 
     

def dbscan_iris_data():
    classifier = DBscanClassifier(epsilon=.6,min_points=3)
    classifier.fit(load_iris().data)
    classifier.show_clusters()
    classifier.compare_with_sklearn(eps=.6,min_points=3)

def dbscan_round_data():
    noisy_circles = make_circles(n_samples=1500, factor=.5, noise=.05)
    noisy_moons = make_moons(n_samples=1500, noise=.05)
    
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1
    
    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2, 'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}) ]
    
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)    
        dbscan_sklearn = cluster.DBSCAN(eps=.3)
        dbscan_mine = DBscanClassifier(epsilon=.3,min_points=3)
        
        clustering_algorithms = (
            ('dbscan_sklearn', dbscan_sklearn),
            ('dbscan_mine', dbscan_mine)
        )
    
        for name, algorithm in clustering_algorithms:
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            
            if name == 'dbscan_mine':
                y_pred = algorithm.cluster_labels
            else:
                y_pred = algorithm.labels_.astype(np.int)
    
            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
    
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')
            plot_num += 1
    
    plt.show()
    
def main():
    
    # Run DBSCAN on iris dataset as per instructions
    dbscan_iris_data()
    
    # I didn't think the iris dataset was a good match for feeling out the dbscan algorithm 
    # so I ran it on two randomly generated sets.
    dbscan_round_data()

    
if __name__ == '__main__':
    main()