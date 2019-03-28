# Leora Betesh March 2019
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

numPoints = 200
    
def create_random_points(source_point, deviationFromPoint, numPoints):    
    pointArray = []    
    for i in range(numPoints):
        newCoords = [source_point[i] + random.random() * deviationFromPoint for i in range(2)]
        pointArray.append(newCoords)    
    return pointArray

def create_random_circular_points(source_point, radius, numPoints):    
    pointArray = []   
    for i in range(numPoints):
        angle = random.randint(0,360)
        noise = random.uniform(-.25,.25)
        newCoords = [(math.cos(angle) * radius) + noise, (math.sin(angle) * radius)+ noise]
        pointArray.append(newCoords)    
    return pointArray

def category_prediction(boundary,X,theta):
    #Return true or false values.  True for values that exceed the boundary.
    return sigmoid(np.dot(X,theta).reshape(X.shape[0])) > boundary

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def negative_log_likelihood(Y,Y_pred):
    #Add a tiny number to prevent zero values, as log on zero causes errors.
    epsilon = 1e-7
    class1_loss = np.abs(-(Y*np.log(Y_pred+epsilon)))
    class2_loss = np.abs((1-Y) * np.log(1-(Y_pred-epsilon)))
    return np.mean(class1_loss + class2_loss)

def calc_gradient(X,Y,weights,learning_rate, max_iterations,precision):
    #Save the loss values in an array so I can plot them easily
    loss = []
    theta = weights.copy()
    for i in range(max_iterations):
        hypothesis = np.dot(X,theta).reshape(X.shape[0])
        Y_pred = sigmoid(hypothesis)
        loss.append(negative_log_likelihood(Y,Y_pred))
        gradient = np.dot(Y_pred.reshape(Y.shape)-Y,X) / Y.shape
        theta = theta - (learning_rate * gradient).reshape(theta.shape)
        if loss[i] < precision:
            break
    plt.plot(loss)
    plt.show()
    print("Found acceptable loss at iteration",i+1,loss[i])
    return theta

def logistic_regression(arr1,arr2,addSquareFeatures):
    numPoints = 200
    boundary = .5
    learning_rate = .1
    max_iterations = 100
    bias = np.ones(numPoints * 2)
    precision = .1 
    
    X = np.column_stack((np.concatenate((arr1,arr2), axis=0),bias))
    Y = np.asarray(np.concatenate((np.ones(numPoints),np.zeros(numPoints)), axis=0))

    if addSquareFeatures:
        #Add x-coordinate squared and y-coodinate squared as features to boost results
        x_squared = X[:,0] ** 2
        y_squared = X[:,1] ** 2
        X = np.column_stack((X,x_squared,y_squared))
        orig_weights = np.ones(5).reshape(5,1)
    else:
        orig_weights = np.ones(3).reshape(3,1)
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  

    train_weights = calc_gradient(X_train,Y_train,orig_weights,learning_rate,max_iterations,precision)

    plt.scatter(arr1[:, 0], arr1[:, 1], s=15, label='Category 1')
    plt.scatter(arr2[:, 0], arr2[:, 1], s=15, label='Category 2')
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * train_weights[0] + (train_weights[2]*1.2))/train_weights[1]
    
    plt.plot(x_vals, y_vals, '--', c="green")
    
    plt.legend()
    plt.show()  
    
    train_precision = np.mean(category_prediction(boundary,X_train,train_weights)==Y_train)
    test_precision = np.mean(category_prediction(boundary,X_test,train_weights)==Y_test)
    print('Train precision: {} \nTest precision: {}'.format(train_precision, test_precision)) 
    

arr1 = np.array(create_random_points([0,0],4.5,numPoints))
arr2 = np.array(create_random_points([3,3],4.5,numPoints))
logistic_regression(arr1,arr2,False)

circle1 = np.array(create_random_circular_points([0,0],1,numPoints))
circle2 = np.array(create_random_circular_points([0,0],2,numPoints))    
logistic_regression(circle1,circle2,True)