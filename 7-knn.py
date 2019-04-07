# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:51:37 2019

@author: Leora Betesh
"""    
from sklearn.datasets import load_iris
import numpy as np

class DataSet():
    def __init__(self, data,labels):
        assert len(data)==len(labels) 
        self.data=data
        self.labels=labels
        
    def train_test_split_lb(self, test_percentage):
        arr_rand = np.random.rand(self.data.shape[0])
        split = arr_rand < np.percentile(arr_rand, test_percentage*100)    
        return self.data[split], self.labels[split], self.data[~split], self.labels[~split]

class KNN():
    def __init__(self, k):
        self.k = k

    def calc_distance(self,data1,data2):
        return np.linalg.norm(data1-data2)
        #return np.mean((data1 - data2)**2)
    
    #search the entire train dataset for the k nearest neighbours for an individual test instance
    def calc_KNN(self, X_train, test_item):
        distances = []
        for i in range(X_train.shape[0]):
            distances.append(self.calc_distance(test_item,X_train[i]))
    
        return np.argpartition(np.asarray(distances),k)[:k]  
    
    def predict_class(self,knn_indices,Y_train):
        votes = []
        for i in range(len(knn_indices)):
            votes.append(Y_train[knn_indices[i]])
        
        return np.argmax(np.bincount(votes))
    
    def getAccuracy(self,Y_test, predictions):
        correct = 0
        for i in range(len(Y_test)):
            if Y_test[i] == predictions[i]:
                correct += 1
        
        return (correct/float(len(Y_test))) * 100.0
        

    
iris_data = DataSet(load_iris().data,load_iris().target)
X_train, Y_train, X_test, Y_test = iris_data.train_test_split_lb(0.2) 

for k in (range(1,30,2)):
    predictions = []
    knn=KNN(k)
    for i in range(len(X_test)):
        knn_indices = knn.calc_KNN(X_train, X_test[i])
        predictions.append(knn.predict_class(knn_indices,Y_train))

    print("For K value of ",knn.k,"Accuracy was",knn.getAccuracy(Y_test, predictions),"%")

#6. Main function: write a main function that contains everything and calls all the
#functions that we have written.
	
#main()
    