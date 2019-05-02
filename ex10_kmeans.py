# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:15:31 2019

@author: Leora Betesh
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import random
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
iris_data = load_iris().data
iris_labels = load_iris().target
     
class KMeansClassifier():
    def __init__(self, X, K, max_iter=300):
        self.X = X
        self.k = K
        self.max_iter = max_iter
        self.centroids = []
        self.clusters = []
                
    def cluster(self):
        self.initialize_centroids()
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.cluster_points() 
            self.recalc_centroids()
            if self.is_converged(old_centroids):
                print("-------------Converged----------------")
                return 1
            
            print("iteration",i,"-------------------------")
            for i in range(self.k):
                print("in cluster {0} there are {1} points".format(i+1, sum(1 for point in self.clusters[i])))         
            
            self.validate()
            
            
    
    def initialize_centroids(self):
        #randomly choose k centroids
        self.centroids = random.sample(list(self.X), self.k)
        
        #Use the first k datapoints in our datasset as starting centroids
#        for i in range(self.k):
#            self.centroids.append(self.X[i])
    
    #Assign each data point to a cluster by measuring the distance to each centroid.
    def cluster_points(self):
        #2a For each data point, compute the euclidian distance from all the centroids
        #and assign the cluster based on the minimal distance to all the centroids.
        self.clusters = []
        
        for i in range(self.k):
            self.clusters.append([])

        for instance in self.X:
            distances = [self.calc_distance(instance,self.centroids[i]) for i in range(self.k)]
            closest_cluster = distances.index(min(distances))
            self.clusters[closest_cluster].append(instance)
            
    #Adjust the centroid of each cluster to be the mean point of all the data points in the cluster 
    def recalc_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis = 0).tolist()   
    
    #Very simplistic for now: if last centroids aren't same as new ones we haven't converged yet
    def is_converged(self,old_centroids):
        return np.all(np.equal(np.asarray(self.centroids), np.asarray(old_centroids)))
    
    def calc_distance(self,a,b):
        return np.sqrt(((np.array(a)-np.array(b))**2).sum(axis=0))
    
    #Compare the distribution of the original classes of the iris data with the cluster groups
    def validate(self):
        total_accuracy = 0
        for i in range(self.k):
            cluster_classes = []
            for datapoint in self.clusters[i]:
                origIndex = np.where((iris_data==datapoint).all(axis=1))[0][0]
                cluster_classes.append(iris_labels[origIndex])
            majority_class = max(set(cluster_classes), key=cluster_classes.count)
            accuracy = round(cluster_classes.count(majority_class) / float(len(cluster_classes)) * 100)
            #print("accuracy for class",i,accuracy,"%")
            total_accuracy = total_accuracy + accuracy
            
        print("overall accuracy for this iteration",total_accuracy/3)

    def plot_orig_classes(self):
        # we only take the first two features.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        
        X = self.X[:, :3]
        y = iris_labels    

        #plt.scatter(X[:, 0], X[:, 1], X[:,2], c=y,cmap=cmap_bold,s=15)
        
        img = ax.scatter(X[:, 0], X[:, 1], X[:,2], c=y, cmap=cmap_bold)
        fig.colorbar(img)

        plt.show()
        
    def plot_clusters(self):
        # we only take the first three features.
        fig = plt.figure()
        
        newarr = []
        
        for i in range(self.k):
            row = self.clusters[i]
            for j in range(len(row)):
                row[j] = np.append(row[j],i).tolist()
                newarr.append(row[j])
            
        ax = fig.add_subplot(111, projection='3d')
        X = np.asarray(newarr)[:,0:3]
        y = np.asarray(newarr)[:,-1]
        #X2 = np.asarray(self.centroids)[:,0:3]
        
        ax.scatter(X[:, 0], X[:, 1], X[:,2], c=y, cmap=cmap_bold)
        #ax.scatter(X2[0], X2[1], X2[2], 'k*',s=50)
        plt.show()
        
def main():
    classifier = KMeansClassifier(iris_data,3)
    classifier.cluster()
    
    #For fun, try to plot the classes versus the clusters
    classifier.plot_orig_classes()
    classifier.plot_clusters()
    
if __name__ == '__main__':
    main()

    


