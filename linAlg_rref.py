# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:48:33 2019

@author: Leora Betesh
"""
import numpy as np
import sympy as sp

#Row Echelon Form, with optional parameter to achieve *reduced* row echelon form
def ref_lb(m,reduced):
      
    colNum = 0
    pivot = 0
    
    #List to store all pivot columns
    pivot_points = []
    
    print("Original Matrix:\n",np.matrix(m))
    
    #If matrix is 0 matrix end process
    if np.count_nonzero(m) ==0:
        print("No reduction needed for Zero matrix")
        return True
    
    while pivot < m.shape[0] and colNum < m.shape[1]:
    
        #if last row is all zero then stop traversing rows and start to clean up the values above the pivots
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
            
    print("RREF matrix:\n",np.matrix(m))    
    print("\nPivot points:",pivot_points)
            

matrix1 = sp.Matrix([[1,0,3,2,1],
                     [0,2,2,4,4],
                     [0,0,0,2,6]])
    
matrix2 = sp.Matrix([[1,2,0,0,2,-3],
                     [2,2,2,2,2,2],
                     [1,2,1,4,3,6],
                     [3,4,3,6,5,8]]
        )

matrix3 = sp.Matrix([[-2,4,-5,-8,4],
                     [ 2,1, 2, 5,4],
                     [2,-4,5,8,-4]]
        )

ref_lb(matrix1,True)
ref_lb(matrix2,True)
ref_lb(matrix3,True)
