# Leora Betesh March 2019
import numpy as np
from matplotlib import pyplot as plt

def calc_weights(X,Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
def calc_L1_loss(Y,Y_pred):
    return np.sum(np.abs(Y - Y_pred))/Y.shape[0]

def calc_L2_loss(Y,Y_pred):
    return np.mean((Y.reshape(Y.shape[0],1) - Y_pred)**2)

def gradient_descent_regular(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient):
    loss_vect = []
    found_good_match = max_iterations
    
    for i in range(max_iterations):
        gradient = np.dot(X,hypothesis-Y)
        theta = theta - learning_rate*gradient
        hypothesis = np.dot(X,theta)
        loss_vect.append(calc_L2_loss(Y, hypothesis))
        print("Iteration",i+1,"Loss:",loss_vect[i])
        if loss_vect[i] < .00001:
            found_good_match = i+1
            break
    plt.plot(loss_vect)
    plt.show()
    
    print("Regular - Final Weights achived at",found_good_match,theta.reshape(1,3))
    
def gradient_descent_momentum(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient,momentum):
    velocity = np.zeros(theta.shape)
    loss_vect = []
    found_good_match = max_iterations
    
    for i in range(max_iterations):
        gradient = np.dot(X,hypothesis-Y)
        velocity = (momentum * velocity) - (learning_rate * gradient)
        theta = theta + velocity
        hypothesis = np.dot(X,theta)
        loss_vect.append(calc_L2_loss(Y, hypothesis))
        print("Iteration",i+1,"Loss:",loss_vect[i])
        if loss_vect[i] < .00001:
            found_good_match = i+1
            break
    plt.plot(loss_vect)
    plt.show()
    
    print("Momentum - Final Weights achived at",found_good_match,theta.reshape(1,3))
    
def gradient_descent_nesterov(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient,momentum):
    velocity = np.zeros(theta.shape)
    loss_vect = []
    found_good_match = max_iterations
    
    for i in range(max_iterations):
        hypothesis = np.dot(X,theta)
        hypothesis_next = np.dot(X,theta+momentum*((momentum * velocity) - (learning_rate * gradient)))
        loss_vect.append(calc_L2_loss(Y, hypothesis))
        gradient = np.dot(X,hypothesis_next-Y)
        velocity = (momentum * velocity) - (learning_rate * gradient)
        theta = theta + velocity
        
        print("Iteration",i+1,"Loss:",loss_vect[i])
        if loss_vect[i] < .00001:
            found_good_match = i+1
            break
        
        
    plt.plot(loss_vect)
    plt.show()
    
    print("Nesterov - Final Weights achived at",found_good_match,theta.reshape(1,3))
    
def gradient_descent_exercise():
#    1. Calculate the hypothesis = X * theta
#    2. Calculate the loss 
#    3. Calculate the gradient 
#    4. Update the weights theta = theta - learning rate * gradient

    X0 = [1,1,1]
    X1 = [0,1,2]
    X2 = [0,1,4]
    Y = np.array([1,3,7]).reshape(3,1)
    X = np.column_stack((X0,X1,X2))
    
    theta = np.array([2,2,0]).reshape( (3,1))
    hypothesis = np.dot(X,theta).reshape((3,1))
    print("Loss using initial weights:",calc_L2_loss(Y, hypothesis))
    
    learning_rate = .1
    max_iterations = 100
    gradient = 0
    momentum = .9
    
    gradient_descent_regular(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient)
    gradient_descent_momentum(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient,momentum)
    gradient_descent_nesterov(X,Y,theta,hypothesis,learning_rate,max_iterations,gradient,momentum)

def linear_reg_exercise():
    X = np.array([[31,22],[22,21],[40,37],[26,25]])
    Y = np.array([2,3,8,12])
    
    #A: hypothesis is that θ1 * X1 + θ2 * X2 = Y
    W1 = calc_weights(X,Y)
    print_results(X,Y,W1,"Orig Weights")
    
    #B Add another feature: the age difference between the siblings.  Does this improve the fitting?
    age_diff = X[:,0] - X[:,1]
    np.reshape(age_diff, (4,1))
    X_age = np.column_stack((X,age_diff))
    W_age = calc_weights(X_age,Y)
    print_results(X_age,Y,W_age,"With age difference added as feature")
    
    #C Instead of the age difference, add the square of the difference as the new feature.  Does this improve the fitting?
    age_diff_squared = (X[:,0] - X[:,1]) ** 2
    np.reshape(age_diff_squared, (4,1))
    X_age_sq = np.column_stack((X,age_diff_squared))
    W_age_sq = calc_weights(X_age_sq,Y)
    print_results(X_age_sq,Y,W_age_sq, "With age diff squared as feature")
    
    #D Instead of the age difference, add a bias.  Does this improve the fitting?
    col_ones = np.ones((X.shape[0],1))
    X_ones = np.append(X,col_ones,axis=1)
    W_ones = calc_weights(X_ones,Y)
    print_results(X_ones,Y,W_ones,"With bias added")
    
    #E Combine features
    X_combined = np.column_stack((X,age_diff_squared,col_ones))
    W_combined = calc_weights(X_combined,Y)
    print_results(X_combined,Y,W_combined,"Best weights prediction when age diff squared and bias added")

def print_results(X,Y,weights, title="Weights"):
    print("\n---- {0} ----\n".format(title),weights)
    print("----Real Values----\n",Y)
    print("----Predicted Values----\n",np.dot(X,weights) )
    print("----L2 loss----\n",calc_L2_loss(Y,np.dot(X,weights)))
    
gradient_descent_exercise()
#linear_reg_exercise()