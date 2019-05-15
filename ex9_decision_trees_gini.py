"""
Created on Thu Apr 14 22:05:57 2019

@author: Leora Betesh

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Node():
    def __init__(self, split_feature, split_value, left_child, right_child, node_level):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.depth = node_level 
        
    def display(self, spacing=" "):    
        print(spacing + "feature", self.split_feature,"value", self.split_value, "node level", self.depth)    
        print(spacing + '--> Left:')
        self.left_child.display(spacing + "  ")
        print(spacing + '--> Right:')
        self.right_child.display(spacing + "  ")
        
    def predict(self, row):
        if row[self.split_feature] > self.split_value:
            return self.left_child.predict(row)
        return self.right_child.predict(row)

class Leaf():
    def __init__(self, classValue):
        self.value = classValue
    
    def display(self, spacing=" "):
        print(spacing + "class",self.value)
        
    def predict(self, row):
        return self.value

class DecisionTreeClassifier():
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth
        
    def fit(self, train):
        self.train = train
        # The build_tree method returns a node.  Assign this as the root node.
        self.set_root(self.build_tree(train))
        #self.print_tree()
    
    def set_root(self, node):
        if self.root == None:
            self.root = node

    def build_tree(self, rows, node_level=0):
        node_level +=1
        
        if self.calc_gini(rows) == 0 or node_level > self.max_depth :
            #If gini is 0 we found a class, i.e. all items are in the same class.  
            #Take first class value (0th row, 1st col) and use that value to create leaf node.
            return Leaf(rows.iloc[0,1])
        
        best_split_feature, best_split_value = self.find_best_split(rows)
        left_rows, right_rows = self.data_split(rows, best_split_feature, best_split_value) 
                   
        left_child = self.build_tree(left_rows, node_level)
        right_child = self.build_tree(right_rows, node_level)
        
        return Node(best_split_feature, best_split_value, left_child, right_child, node_level)
    
    def print_tree(self):
        self.root.display(spacing="")
        
    def make_predictions(self, test):
        test_labels = []
        y_test = test.iloc[:,1].tolist()
  
        for rowNum, row_data in enumerate(test.itertuples(index=False)):
            test_labels.append(self.root.predict(row_data))
        
        self.get_accuracy(test_labels, y_test)
        
    def get_accuracy(self, predictions, Y_test):
        correct = np.asarray(Y_test) == np.asarray(predictions)
        print("prediction accuracy for tree with max depth of",self.max_depth, (np.sum(correct)/float(len(Y_test))) * 100.0)
    
    def class_counts(self, classes):
        counts = {}  # a dictionary of label -> count.
        for label in classes:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
      
    def data_split(self, dataset, feature_idx, split_val):
        #left_tree, right_tree = [], []
        left_tree = dataset[dataset[feature_idx] > split_val]
        right_tree = dataset[dataset[feature_idx] <= split_val]     
        return pd.DataFrame(left_tree), pd.DataFrame(right_tree)
     
    # The calc_gini impurity can be computed by summing the probability of an item being chosen
    # times the probability of a mistake in categorizing that item.
    def calc_gini(self, rows):    
        counts = self.class_counts(rows.iloc[:,1])
        probabilities =  [x / float(len(rows)) for x in counts.values()]      
        return 1 - sum([p**2 for p in probabilities])
    
    def find_best_split(self, rows):
        best_split_feature = 0
        best_split_value = 0
        best_gini = 1
        n_features = rows.shape[1] 
        
        for col in range(2, n_features):
            values = rows[col].unique()        
            for val in values:
                left_rows, right_rows = self.data_split(rows, col, val)
                p = (float(len(left_rows)) / (len(rows)))
                current_gini = self.calc_gini(left_rows)*p + self.calc_gini(right_rows)* (1-p)
                if current_gini < best_gini:
                    best_gini = current_gini
                    best_split_feature, best_split_value = col, val
        return best_split_feature, best_split_value 

def main():
    wdbc = pd.read_csv('wdbc.data.txt',header=None)
    train_data, test_data = train_test_split(wdbc, test_size=0.2, random_state=42) 
    
    #Create classifier objects with varied tree depths so we can determine optimal depth
    for max_depth in range(1,7):
        tree = DecisionTreeClassifier(max_depth)
        tree.fit(train_data)
        #tree.print_tree()
        tree.make_predictions(test_data)

if __name__ == '__main__':
    main()


