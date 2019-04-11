# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:44:05 2019

@author: Leora Betesh

"""
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split

class DataSet():
    def __init__(self,X_train,Y_train):
        assert len(X_train)==len(Y_train) 
        self.X_train = X_train
        self.Y_train = Y_train
        # Save the data for class 0 separately from class 1 for probablities calculations
        self.X_train0 = self.X_train.loc[(self.Y_train==0),:]
        self.X_train1 = self.X_train.loc[(self.Y_train==1),:]        

# This class inherits the DataSet class        
class NaiveBayesClassifier(DataSet):
    def __init__(self, X_train,Y_train):
        DataSet.__init__(self, X_train,Y_train)

    # Train the model        
    def fit(self):
        # Overall probabilities of belonging to either class aka P(C) 
        self.overall_prob_class0 = len(self.X_train0) / len(self.X_train)
        self.overall_prob_class1 = len(self.X_train1) / len(self.X_train)
        
        # Mean and STDev values per class, per feature, stored in arrays
        self.mean0 = self.X_train0.mean(axis=0)
        self.std0 = self.X_train0.std(axis=0)
        self.mean1 = self.X_train1.mean(axis=0)
        self.std1 = self.X_train1.std(axis=0)
    
    def gaussian_probability_per_attr(self,x,mean,std):
        return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-(x - mean)**2 / (2*std**2))
    
    def makePrediction(self,X_test):
        predictions = []
        for rowNum, row in enumerate(X_test.itertuples(index=False)):
            class0prob = 1
            class1prob = 1
            for attrNum, attrVal in enumerate(row):
                class0prob = class0prob * self.gaussian_probability_per_attr(attrVal,self.mean0[attrNum], self.std0[attrNum])
                class1prob = class1prob * self.gaussian_probability_per_attr(attrVal,self.mean1[attrNum], self.std1[attrNum])
            class0prob = class0prob * self.overall_prob_class0
            class1prob = class1prob * self.overall_prob_class1
    
            if class0prob > class1prob:
                predictions.append(0)
            else:
                predictions.append(1)
                
        return predictions

    def getAccuracy(self,predictions,Y_test):
        correct = Y_test.values == predictions
        return (np.sum(correct)/float(len(Y_test))) * 100.0
    
def main():
    pidd = pd.read_csv('pima-indians-diabetes.csv', sep=",",header=None)
    X_train, X_test, Y_train, Y_test = train_test_split(pidd[pidd.columns[:-1]], pidd[pidd.columns[8]], test_size=0.2,random_state=42)  
    nbc = NaiveBayesClassifier(X_train,Y_train)
    nbc.fit()
    predictions = nbc.makePrediction(X_test)
    print("Accuracy was", nbc.getAccuracy(predictions,Y_test),"%")

if __name__ == '__main__':
    main()
