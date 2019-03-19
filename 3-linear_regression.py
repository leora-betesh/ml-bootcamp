# Leora Betesh March 2019
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#1.1

X = np.array([[31,22],[22,21],[40,37],[26,25]])
Y = np.array([2,3,8,12])

#This is the formula for theta = inverse of X transpoe X times X transpose times y
W1 = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

loss1 = np.sum((X@W1-Y)**2)/X.shape[0]
print("initial loss:", loss1)

#1.2 Add a bias of 1
col_ones = np.ones((X.shape[0],1))
XX = np.append(X,col_ones,axis=1)

W2 = np.linalg.inv(XX.T.dot(XX)).dot(XX.T).dot(Y)

loss2 = np.sum((XX@W2-Y)**2)/XX.shape[0]
print("loss with bias of 1 added:", loss2)

#2.1 Load the data from the file “data_for_linear_regression.csv”

#os.chdir("C:/LeoraProjects/bootcamp/3_logistic_regression")
lin_data = pd.read_csv("data_for_linear_regression.csv").dropna()

#2.2 Convert the data to numpy array. (use - pandas.values)
data_X = lin_data.values[:,0]
data_Y = lin_data.values[:,1]

#2.3 Show the data on the graph, use matplotlib.pyplot to show the data with scatter plot
# plt.scatter(data_X,data_Y)
# plt.show()

# 2.4 Now, take some data and try to use the solution from the previous question to find the trend line of this points.
# Remember because you want to try to find equation like this form: y = ax + b, So you need to add bias, add b as one's vector to x with
# numpy.hstack.
#data_X = data_X[:100]
#data_Y = data_Y[:100]

col_ones = np.ones((data_X.shape[0],1))
data_XX = np.column_stack((data_X,col_ones))

Weights = np.linalg.inv(data_XX.T.dot(data_XX)).dot(data_XX.T).dot(data_Y)

# 2.5
# After you find θ1 ,θ2 , plot the line from the equation - y = θ1∗X+θ2∗b
# use matplotlib.pyplot.hold to draw this line over last figure without deleting it.
#plt.plot(x_some, (W_some[1] + W_some[0]*x_some), '-r', label = 'y = {:.2f} + {:.2f}*x'.format(W_some[0], W_some[0]))
#plt.scatter(x_some,y_some)
#plt.show()

# 2.6
# Create a new plot with the same trend line but with the rest of the points. If you
# can’t see the plot well, try to use xlim(0,100) and ylim(0,100)
plt.plot(data_X, (Weights[1] + Weights[0]*data_X), '-r')
plt.scatter(data_X,data_Y)
plt.xlim(20,60)
plt.ylim(20,60)
plt.show()

#Calculate the loss

#The predicted y values are the x times the theta
y_pred = data_XX@Weights

#L1 mean absolute loss
l1 = np.sum(np.abs(data_Y - y_pred))/data_Y.shape[0]
print("L1 aka mean absolute error:",l1)

#L2 mean square loss
l2 = np.mean((data_Y - y_pred)**2)
print("L2 aka mean square loss:",l2)

