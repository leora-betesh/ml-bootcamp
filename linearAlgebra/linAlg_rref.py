# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:48:33 2019

@author: Leora Betesh
"""
import numpy as np
import sympy as sp
from numpy.linalg import inv

#***************************#
#2 - Row Echelon Form, with optional parameter to achieve *reduced* row echelon form
def rref_lb(m,reduced=False):
    colNum = 0
    pivot = 0
    pivot_points = []
    
    #If matrix is 0 matrix end process
    if np.count_nonzero(m) ==0:
        print("No reduction needed for Zero matrix")
        return True
    
    while pivot < m.shape[0] and colNum < m.shape[1]:
    
        #If last row is all zero then stop traversing rows and start to clean up the values above the pivots
        if np.count_nonzero(m[pivot,:]) == 0:
            break
        
        #If all entries in the column are zero, move on to next column
        if np.count_nonzero(m[pivot:,colNum]) == 0: 
            colNum = colNum + 1
            
        #If pivot point is zero swap rows
        elif m[pivot,colNum] == 0 and np.count_nonzero(m[pivot+1:,colNum]) > 0:
            nextNonZero = next((index for index,value in enumerate(m[pivot+1:]) if value != 0), None)
            m[pivot,:] = m[nextNonZero, :]
        
        else:
            #Hooray, we found a pivot point
            pivot_points.append((pivot,colNum))
            
            #If pivot point isn't already 1, edit the row to make it 1
            if m[pivot,colNum] != 1:
                #Divide the entire row by pivot point value 
                m[pivot,:] = m[pivot,:] / m[pivot,colNum]             
            
            #Now make everything under pivot 0
            for i in range(pivot+1,m.shape[0]):
                if m[i,colNum] !=0:
                    multiplier = - m[i,colNum]
                    m[i,:] = m[i,:] + (multiplier * m[pivot,:])
              
            pivot = pivot + 1
            colNum = colNum + 1  
     
    
    #If reduced row echelon form was selected, make everything above pivot points 0
    if reduced:
        #For each pivot point
        for i in reversed(pivot_points):
            #For each point above the pivot point
            for j in range(i[0],0,-1):
                if m[j-1,i[1]] !=0:
                    multiplier = - m[j-1,i[1]]
                    m[j-1,:] = m[j-1,:] + (multiplier * m[i[0],:])
            
#    print("RREF matrix:\n",np.matrix(m))    
#    print("\nPivot points:",pivot_points)
            
    return m

#Test our function on matrix
#m = sp.Matrix([[5,0,2,1],[3,3,2,-1],[1,6,2,-3]])
#print(rref_lb(m,True))

#***************************#
#3 Test Function by solving system of 10 equations with 10 unkowns, represented by A, solution is x.  
#  let b = A*x
#  If the functions I defined are correct, x_test should be I + x
np.random.seed(3)
A = sp.Matrix(np.random.uniform(low=-1,high=1, size=(10, 10)))
x = sp.Matrix(np.random.uniform(low=-1,high=1, size=(10, 1)))
b = np.dot(A,x)
A_b = np.column_stack((A,b))
x_test = np.matrix(rref_lb(A_b,True))
print(np.matrix(x),"\n\n",x_test)

#***************************#
#4 - Reduced Column Echelon Form
def rcef_lb(m,reduced=False):
    return rref_lb(m.T,reduced).T

#***************************#
#5 - test rcef and rref 
A = sp.Matrix([[1,2,-3],
               [-1,1,0],
               [0,-2,4]])

#Add identity matrix adjacent to A
AI_horizontal = np.column_stack((A,np.identity(3)))
#Add identity matrix below A
AI_vertical = np.row_stack((A,np.identity(3)))
#If our functions work correctly, both rref on AI horizontal and rcef on AI vertical should give us the inverse of A
print(np.matrix(rref_lb(AI_horizontal,True)))
print(np.matrix(rcef_lb(AI_vertical)))

#Change type to float so inverse function will accept it
A_float = np.array(A).astype(np.float64)
print(inv(A_float))

#ker_A = sp.Matrix([[-2/5,-1/5],
#                   [-4/15,8/15],
#                   [1,0],
#                   [0,1]])
#    
#print(np.matrix(rcef_lb(ker_A,True)))

