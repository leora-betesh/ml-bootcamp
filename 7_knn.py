# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:51:37 2019

@author: Leora Betesh
"""    
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Added this hoping to make a beautiful plot of the decision boundaries but couldn't get it working on my classifier.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                                    
class DataSet():
    def __init__(self, data, labels):
        assert len(data)==len(labels) 
        self.data=data
        self.labels=labels
        
    def train_test_split_lb(self, test_percentage):
        arr_rand = np.random.rand(self.data.shape[0])
        split = arr_rand < np.percentile(arr_rand, test_percentage*100)    
        return self.data[split], self.labels[split], self.data[~split], self.labels[~split]
    
    def plot_points(self):
        # we only take the first two features.
        X = self.data[:, :2]
        y = self.labels    
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1],c=y,cmap=cmap_bold,s=15)
        plt.show()

class KNN():
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train
        
    #Use the Euclidean norm to calculate distance of one data point to another    
    def calc_distance(self,data1,data2):
        return np.linalg.norm(data1-data2)
    
    #search the entire train dataset for the k nearest neighbours for an individual test instance
    def calc_KNN(self, test_item):
        distances = []
        for i in range(self.X_train.shape[0]):
            distances.append(self.calc_distance(test_item,self.X_train[i]))
    
        return np.argpartition(np.asarray(distances),self.k)[:self.k]  
    
    #predict the class of the incoming point.  Call the calc_KNN method to get the nearest neighbors
    def predict_class(self,test_item):
        knn_indices = self.calc_KNN(test_item)
        votes = []
        for i in range(len(knn_indices)):
            votes.append(self.Y_train[knn_indices[i]])
        
        return np.argmax(np.bincount(votes))
    
    def getAccuracy(self, predictions, Y_test):
        correct = 0
        for i in range(len(Y_test)):
            if Y_test[i] == predictions[i]:
                correct += 1
        
        return (correct/float(len(Y_test))) * 100.0
            

def main():
    iris_data = DataSet(load_iris().data,load_iris().target)
    X_train, Y_train, X_test, Y_test = iris_data.train_test_split_lb(.2)         
    iris_data.plot_points()

    #Create knn objects for different values of k and see which value of k gave best predictions
    for k in (range(3,25,2)):
        predictions = []
        
        classifier=KNN(k,X_train,Y_train)
        for i in range(len(X_test)):
            predictions.append(classifier.predict_class(X_test[i]))
        
        print("For K value of ",classifier.k,"Accuracy was",classifier.getAccuracy(predictions,Y_test),"%")

if __name__ == '__main__':
    main()
    
