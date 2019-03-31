# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:42:15 2019

@author: Leora Betesh
"""
import numpy as np

def print_traffic(roads,traffic):
    for i in range(len(traffic)):
        print("Traffic between cities {0} and {1} is {2} cars per minute".format(roads[i][0],roads[i][1],traffic[i]))

def create_matrix(roads,traffic):
    
    
L = ((1,2),(2,1),
     (1,3),(3,1),
     (1,5),(5,1),
     (3,5),(5,3),
     (2,4),(4,2),
     (2,6),(6,2),
     (3,4),(4,3),
     (4,6),(6,4))

v = np.ones(len(L))
calc_traffic(L,v)

nums = np.arange(1,len(L) / 2 +1)
v1 = np.sort(np.concatenate((nums, nums)))
calc_traffic(L,v1)

