# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:42:15 2019

@author: Leora Betesh
"""

import numpy as np
import sympy as sp
import scipy 

def get_null_space_basis_scipy(m):
    print("\n\nSeeking basis for matrix:\n",m)
    rref, pivot_cols = m.rref()
    
    print(scipy.linalg.orth(np.array(rref.T, dtype=np.float)).T)

def test_null_space_basis(m,b):
    zeroVector = np.zeros(m.shape[0])
    return (np.dot(m,b) == zeroVector).all()

def get_null_space_basis(m):
    print("\n\nSeeking basis for matrix:\n",m)
    rref, pivot_cols = m.rref()
    non_pivot_cols = list(set(range(rref.shape[1])) - set(pivot_cols))
    Basis = []

    for basis_num in range(len(non_pivot_cols)):
        #Each time set one of the independent variables to be 1 and the rest to 0
        x = np.zeros(rref.shape[1])
        x[non_pivot_cols[basis_num]] = 1
        
        for idx, p in enumerate(pivot_cols):
            pjtotal = 0
            for n in non_pivot_cols:
                pjtotal = pjtotal - rref[idx,n] * x[n]
            x[p] = pjtotal

        if test_null_space_basis(m,x):
            Basis.append(x)
    
    print("\nBasis for this matrix:\n",Basis)


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
    
    
get_null_space_basis(matrix1)
get_null_space_basis_scipy(matrix1)
#get_null_space_basis(matrix2)
#get_null_space_basis(matrix3)