# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:48:33 2019

Created my own cov function to better understand what the numpy.cov function does, without the bells and whistles.
    
@author: Leora Betesh
"""
import numpy as np

def cov_lb(m):
    """
    Calculate how two variables vary together. 
  
    Parameters
    ----------
    m : array_like
        A 2-D array, whose each row is a variable, and each column a single observation of all those variables. 
        
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    """
    
    mean = m.mean(axis=1)
    m = m.astype(float)
    
    #Size of output matrix will be number of variables squared
    covMatrix = np.zeros((m.shape[0],m.shape[0]))
    
    #Traversing rows is traversing variables
    for rowNum in range(m.shape[0]):
        m[rowNum] -= mean[rowNum]
        variance = 0
        for colNum in range(m.shape[1]):
            variance += m[rowNum,colNum] ** 2
        
        #Along the diagonal we place the amount each variable varies by itself
        covMatrix[rowNum,rowNum] = variance 
        
        #Calculate the covariance of each variable with those that follow it.
        for covarNum in range(m.shape[0]-1,rowNum,-1):
            covMatrix[rowNum,covarNum] = np.sum(m[rowNum] * m[covarNum])
            covMatrix[covarNum,rowNum] = np.sum(m[rowNum] * m[covarNum])
    
    covMatrix /= (m.shape[1] -1)    
    return covMatrix

#TEST OUT COV_LB FUNCTION
#M4 = np.array([[1,2,3,4],[5,8,11,14],[2,4,6,8],[5,10,15,20]]) 
#print(cov_lb(M4))
#print(np.cov(M4))

