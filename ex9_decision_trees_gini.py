# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:05:57 2019

@author: Leora Betesh

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataSet():
    def __init__(self,train):
        self.train = train

class Node():
    def __init__(self,split_feature,split_value,left_child,right_child):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        
    def display(self):
        print("feature", self.split_feature,"value", self.split_value)
        self.left_child.display()
        self.right_child.display()
        
    def predict(self,row):
        if row[self.split_feature] > self.split_value:
            return self.left_child.predict(row)
        return self.right_child.predict(row)

class Leaf():
    def __init__(self,classValue):
        #Either M or B
        self.value = classValue
    
    def display(self):
        print("class",self.value)
        
    def predict(self,row):
        return self.value

class BinaryTree(DataSet):
    def __init__(self):
        self.root = None
        
    def fit(self, train):
        DataSet.__init__(self, train)
        rootNode = self.build_tree(train)
        self.set_root(rootNode)
        #Commenting out because I am not satisfied with my tree display implementation
        #self.print_tree()
    
    def set_root(self,node):
        if self.root == None:
            self.root = node

    def build_tree(self,rows):
        if self.calc_gini(rows) ==0:
            leafValue = rows.iloc[0,1]
            return Leaf(leafValue)
        
        best_gain, best_split_feature, best_split_value = self.find_best_split(rows)
        left_rows, right_rows = self.data_split(rows, best_split_feature, best_split_value)                    
        left_child = self.build_tree(left_rows)
        right_child = self.build_tree(right_rows)
        
        return Node(best_split_feature, best_split_value,left_child,right_child)
    
    def print_tree(self):
        self.root.display()
        
    def make_predictions(self, test):
        test_labels = []
  
        for rowNum, row_data in enumerate(test.itertuples(index=False)):
            row_label = self.root.predict(row_data)
            test_labels.append(row_label)
        
        return test_labels
        
    def get_accuracy(self,predictions,Y_test):
        correct = np.asarray(Y_test) == np.asarray(predictions)
        print("Accuracy of predictions", (np.sum(correct)/float(len(Y_test))) * 100.0)
    
    def class_counts(self,classes):
        counts = {}  # a dictionary of label -> count.
        for label in classes:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
      
    def data_split(self,dataset, feature_idx, split_val):
        leftTree, rightTree = [], []
        leftTree = dataset[dataset[feature_idx] > split_val]
        rightTree = dataset[dataset[feature_idx] <= split_val]     
        return pd.DataFrame(leftTree), pd.DataFrame(rightTree)
     
    # The calc_gini impurity can be computed by summing the probability of an item being chosen
    # times the probability of a mistake in categorizing that item.
    def calc_gini(self,rows):    
        counts = self.class_counts(rows.iloc[:,1])
        probabilities =  [x / float(len(rows)) for x in counts.values()]      
        return 1 - sum([p**2 for p in probabilities])
    
    #The uncertainty of the starting node, minus the weighted impurity of two child nodes
    def info_gain(self,left, right, current_uncertainty):
        p = (float(len(left)) / (len(left) + len(right)))
        return current_uncertainty - p * self.calc_gini(left) - (1 - p) * self.calc_gini(right)
    
    def find_best_split(self,rows):
        best_gain = 0  
        best_split_feature = 0
        best_split_value = 0
        current_uncertainty = self.calc_gini(rows)
        n_features = rows.shape[1] 
        
        for col in range(2,n_features):
            values = rows[col].unique()        
            for val in values:
                left_rows, right_rows = self.data_split(rows, col, val)
                gain = self.info_gain(left_rows, right_rows, current_uncertainty)
                if gain > best_gain:
                    best_gain, best_split_feature, best_split_value = gain, col, val
        print("best_split_feature",best_split_feature,"best_split_value",best_split_value)
        return best_gain, best_split_feature, best_split_value 

def main():
    wdbc = pd.read_csv('wdbc.data.txt',header=None)
    train , test = train_test_split(wdbc, test_size=0.2) 
    y_test = test.iloc[:,1].tolist()
    
    tree = BinaryTree()
    tree.fit(train)
    predictions = tree.make_predictions(test)
    tree.get_accuracy(predictions,y_test)

if __name__ == '__main__':
    main()



