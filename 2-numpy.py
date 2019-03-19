# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:38:13 2019

@author: Leora Betesh
"""
# 1
import numpy as np

#2
print(np.version.version)

#3
myArr = np.zeros(10, dtype="int")

#4
print(myArr.itemsize)

#6
myArr[4]=1
print(myArr)

#7
v = np.arange(10,50)
print(v)

#8
v_rev = v[::-1]
print(v_rev)

#9
v1 = np.arange(9)
m = v1.reshape(3,3)
print(m)

#10
v2 = np.array([1,2,0,0,4,0])
print(v2.nonzero())

#11
i = np.eye(3)
print(i)

#12
r = np.random.randint(1,10,size=(3,3,3))
print(r)

#13
r10 = np.random.randint(1,100,size=(10,10))
print(r10.min(),r10.max())

#14
r30 = np.random.randint(1,100,size=(1,30))
print(r30.mean())

#15
b = np.zeros((5,5))
b[:,0] = 1
b[:,-1] = 1
b[0,:]=1
b[-1,:]=1
print(b)

#16
b = np.pad(b,1,mode="constant",constant_values=0)
print(b)

#17
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
# Explanation of why these are not equal can be found here: https://docs.python.org/3.4/tutorial/floatingpoint.html
0.3 == 3 * 0.1

#18
d = np.zeros((5,5))
d = np.diag((1,2,3,4,7))
print(d)

#19
check = np.zeros((8,8))
check[::2,1::2] = 1
check[1::2,::2] = 1
print(check)

#20
s = np.arange(336)
s = s.reshape(6,7,8)
hundreth = np.unravel_index(100, (6,7,8))
print(s[hundreth])

#21
check2 = np.tile([[1,0],[0,1]], (4,4))
print(check2)

#22
r = np.random.rand(5,5)
print(r)
r -= np.min(r, axis=0)
r /= np.ptp(r, axis=0)
print(r)

#23 Create an array of 2x4 with dtype numpy.int16, print the dtype of the array
r = np.array((2,4), dtype="int16")
print(r.dtype)

#24
m1 = np.random.randint(-10,10,size=(5,3))
m2 = np.random.randint(-10,10,size=(3,2))
dotproduct = np.dot(m1,m2)
print(m1)
print(m2)
print(dotproduct)

#25
v1 = np.arange(1,11)
v1[(3 < v1) & (v1 <= 8)] *= -1
print(v1)

#26
#Author: Jake VanderPlas
#The numpy sum function has an extra argument that regular sum doesn't have for the axis (e.g. x-axis)
#So instead of -1 being added to the range, it is assumed to be the axis, resulting in a different sum value.
print(sum(range(5),-1))
#from numpy import *
print(sum(range(5),-1))

#27
Z = np.random.randint(0,10,size=(5,3))
# It is legal to raise a matrix to the power of another matrix
print(Z**Z)
# not valid
print(Z <- Z)
# 1j is a complex number.  It is legal to multiply it by a matrix.
print(1j*Z)

#28
a = np.array(0)
b = np.array(0)
print(a)
print(b)
#print(0/0)
#print(0//0)
#Division by zero gives error
#np.array(0) / np.array(0)
#np.array(0) // np.array(0)

#29
z = np.random.random(20)
x = np.copysign(z,-1)
print(z,np.round(z), np.ceil(z))
print(x,np.round(x),np.ceil(x))

#30
print(np.intersect1d(x, z))

#31
np.seterr(over='raise')
#Now it will complain
#np.int16(32000) * np.int16(3)
#Now it will be quiet
np.seterr(over='ignore')
np.int16(32000) * np.int16(3)

#32
#Left is undefined, while right is a complex number so these are not equal.
np.sqrt(-1) == np.emath.sqrt(-1)

#33
days3 = np.arange('2019-03-06', '2019-03-09', dtype='datetime64[D]')
print(days3[1])
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print(yesterday)
print(yesterday+1)

#34
Z = np.arange('2016-08', '2016-09', dtype='datetime64[D]')
print(Z)

#35 How to compute ((A+B)\*(-A/2)) in place (without copy)?
A = np.ones(5)
B = np.zeros(5)
np.multiply(np.add(A,B,out=B),np.negative(np.divide(A,2,out=A)),out = A)
print(A)

#36 Extract the integer part of a random array using 5 different methods
z = np.random.rand(10) * 10
print(z)

# -------1--------
print(np.rint(z))
# -------2--------
print(np.floor(z))
# -------3--------
print(np.modf(z)[1])
# -------4--------
print(z.astype(int))
# -------5--------
print (z - z%1)
# -------6--------
print (np.ceil(z)-1)
# -------7--------
print (np.trunc(z))

#37. Create a 5x5 matrix with row values ranging from 0 to 4
x = np.arange(0,5)
y = np.column_stack((x,x,x,x,x))
print(y)

#39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
x = np.arange(.1,1,.09)
print(x)

#40. Create a random vector of size 10 and sort it
r = np.random.randint(1,10,size=(1,10))
r.sort()
print(r)

#41. Print the minimum and maximum representable value for each numpy scalar type 

for dtype1 in np.ScalarType:
    if dtype1.__name__ == 'timedelta64':
        print("*******************************************")
        print(dtype1)
        print(" timedelta64 max and min are undefined")
    elif np.issubdtype(dtype1,np.int):
        print("*******************************************")
        print(dtype1)
        print(np.iinfo(dtype1))
    elif np.issubdtype(dtype1,float):
        print("*******************************************")
        print(dtype1)
        print(np.finfo(dtype1))
    else:
        print("*******************************************")
        print(dtype1)
        print("max and min are undefined")
        
#42. How to print all the values of an array? 
x = np.zeros((100, 100))
np.set_printoptions(threshold=np.nan)
print(x)

#43. How to find the closest value (to a given scalar) in an array? 
x = np.arange(100)
np.random.shuffle(x)
print(x)
value_to_find = np.random.uniform(0,100)
print("value_to_find: ",value_to_find)
index_smallest = np.rint(np.abs(x - value_to_find)).argmin()
print(x[index_smallest])

#44. Create a structured array representing a position (x,y) and a color (r,g,b) 
# my solution
x = np.array([((1,2), (200,100,0)), ((-1,-2), (70,60,0)), ((0,2), (100,5,70))])
print(x)

#45. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
# compare each set with every other set of points and end up with an array of 5000 comparisons 
r = np.random.randint(10,size=(100,2)) 
#print(r)
distances =np.ndarray((100,100))
#How can we do this only 5000 times rather than 10000?
for idx1 in range(r.shape[0]):
    for idx2 in range(r.shape[0]):
        if idx1 == idx2:
            distances[idx1,idx2] = 0
        else:
            #print(np.sqrt((r[idx1,0]-r[idx2,0])**2 + (r[idx1,1]-r[idx2,1])**2))
            distances[idx1,idx2] = np.sqrt((r[idx1,0]-r[idx2,0])**2 + (r[idx1,1]-r[idx2,1])**2)

#46. How to convert a float (32 bits) array into an integer (32 bits) in place?
x = np.arange(5, dtype=np.int32).astype(np.float32, copy=False)
print(x)

#48. What is the equivalent of enumerate for numpy arrays? 
x = np.arange(10)
for idx, value in np.ndenumerate(x):
    print(idx, value)

#49. Generate a generic 2D Gaussian-like array 
#I looked at docs.scipy.org for a sample of what to do
import matplotlib.pyplot as plt
    
mean = 400
stdev = 200/3   # 99.73% chance the sample will fall in your desired range
values = np.random.normal(mean, stdev, 800)
values.sort()
count, bins, ignored = plt.hist(values, 30, normed=True)
plt.plot(bins, 1/(stdev * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * stdev**2) ), linewidth=2, color='r')
plt.show()

#50. How to randomly place p elements in a 2D array? 
np.set_printoptions(threshold=100)
c = np.array(["pass"])
results = np.tile(c,(8,8))
print(results)
np.put(results,np.random.choice(range(64), 20),"fail")
print(results)

#51. Subtract the mean of each row of a matrix 
x = np.random.randint(100,size=(10,5)) 
print(x)
print("orig mean:", x.mean(axis=1, keepdims=True))
y = x - x.mean(axis=1, keepdims=True)
print(y)
#When you subtract the mean from each value the new mean will be close to zero
print("new mean:", y.mean(axis=1, keepdims=True))

#52. How to I sort an array by the nth column?
n=3 
x = np.random.randint(0,1000,(10,4))
print("unsorted:\n", x)
print("\nsorted by column {0} ascending\n".format(n),x[x[:,3].argsort()])
print("\nsorted by column {0} descending\n".format(n),x[(-x)[:,3].argsort()])

#53. How to tell if a given 2D array has null columns? 
x = np.random.random((4,4))
print(x)
print((~x.any(axis=0)).any())
#Now add some null values
x[0:4,1] = 0
print(x)
print((~x.any(axis=0)).any())

#54. Find the nearest value from a given value in an array 
x = np.arange(100)
np.random.shuffle(x)
print(x)
value_to_find = np.random.uniform(0,100)
print("value_to_find: ",value_to_find)
index_smallest = np.rint(np.abs(x - value_to_find)).argmin()
print(x[index_smallest])

#55. Create an array class that has a name attribute 
class Arr:
    
    def __init__(self, elements, name):
        self.elements = elements
        self.name = name

    def __repr__(self):
        return '%r' % self.elements
    
arr1 = Arr(np.ones(10),"random_arr1")
print(arr1)
print(arr1.name)

#56. Consider a given vector, how to add 1 to each element indexed by a second vector (be
#careful with repeated indices)? 
v1 = np.ones(10)
v2 = np.arange(3,8,2)
v1[v2] += 1
print(v1)

#57. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?
# X is a vector
X = [0,1,2,3,4,5,6,7,8,9]
# I are bins
I = [1,3,9,3,4,1,3,9,1,2]
"""
F accumulates all X[I] found in I.
F[0] = 0 .no o in I
F[1] = 0+5+8 X[I where I==1]
F[2] = 9
F{3] = 1+3+6...
 """
 
F = np.bincount(I,X)
print(F)

#58. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
#ubyte is unique byte.  each of the 3 rgb values is the size of a byte
np.set_printoptions(threshold=1000)
w,h = 5,5
#generate an image where the RGB values are randomly selected as values betweeen 0 and 255
#use as type int32 to allow for larger size
#what we want to do is 
I = np.random.randint(0,256,(h,w,3)).astype(np.ubyte)
'''
Multiply the red component by 256 squared, multiply the green component by 256 and leave the blue alone
When we add the three together we will get the three values all in one number.
Then we can easily compare the RGB's to find unique colors
'''
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))

#just for fun seeing if I can generate an image.  Didn't work and I need to move on.
#H = hex(F)
#ba = bytearray(H)
#f = open('c:/LeoraProjects/bootcamp/myimage.jpeg', 'wb')
#f.write(ba)
#f.close()

#59. Considering a four dimensions array, how to get sum over the last two axis at once? 
fourD = np.ones((2,2,2,2))
fourD[0,0] *= 3 
fourD[1,1] *= 4
fourD[0,1] *= 2 
fourD = np.array(fourD)
sumAll = fourD.sum() #40
sumAxis0 = fourD.sum(axis = 0) # 4,4,4,4
sumAxis1 = fourD.sum(axis = 1) # 5,5,5,5
sumAxis2 = fourD.sum(axis = 2) # 6,6,4,4
sumAxis3 = fourD.sum(axis = 3) # 6,6,4,4
sumTwoAtOnce = fourD.sum(axis=(1,2)) # 10,10,10,10
print(sumTwoAtOnce)
#60. Considering a one-dimensional vector D, how to compute means of subsets of D using a
#vector S of same size describing subset indices? (**hint**: np.bincount)
D = np.arange(5,11)
S = np.arange(6)
D_sums = np.bincount(S, weights=D)


#61. How to get the diagonal of a dot product? 
A = np.random.randint(1,5,(5,5))
B = np.random.randint(1,5,(5,5))
C = np.dot(A, B)
D = np.diag(C)
print(A)
print(B)
print(C)
print(D)

#62. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros
#interleaved between each value? 

#Method 1 np.append
v1 = np.arange(1,6)
v2 = np.zeros((5,1))
v1 = v1.reshape(5,1)
v1 = np.append(v1,v2,axis=1)
v1 = np.append(v1,v2,axis=1)
v1 = np.append(v1,v2,axis=1)
v1 = v1.reshape(1,20)
print(v1)

#Method 2 np.column_stack
v1 = np.arange(1,6)
v1 = np.column_stack((v1, np.zeros((v1.shape[0],3))))
v1 = v1.reshape(1,20)
print(v1)

#Method 3 np.concatenate
v1 = np.arange(1,6)
v1 = v1.reshape(5,1)
z = np.zeros((5,3))
v1 = np.concatenate((v1, z), axis=1)
v1 = v1.reshape(1,20)
print(v1)

#Method 4 - create zero array and sub with nums
v1 = np.arange(1,6)
z = np.zeros(len(v1) + (len(v1)-1)*(3))
z[::4] = v1
print(z)
    
#63. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions
#(5,5)? 
three_matrix553 = 3*np.ones((5,5,3))
two_matrix55 = 2*np.ones((5,5))
two_matrix551 = np.expand_dims(B,axis=2)
product_matrix = three_matrix553 * two_matrix551
print(product_matrix)

#64. How to swap two rows of an array? 
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)

#65. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of
#unique line segments composing all the triangles 
# Start with 10 rows of 3 sides
#For a triangle to be possible from 3 values, the sum of any two side lengths must be greater than the third side
tri = np.random.randint(0,100,(10,3))
# add in adjacent points
F = np.roll(tri.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(tri)


#66. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
C = np.bincount([1,1,2,3,4,4,6])
print(C)
A = np.repeat(np.arange(len(C)), C)
print(A)
D = np.bincount(A)
print(D)

#67. How to compute averages using a sliding window over an array? 


#68. Consider a one-dimensional array Z, build a two-dimensional array whose first row is
#(Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-
#1]) 

n = 10
x = np.arange(n)
master = np.stack(x)
print(master)
for i in range(1,n):
    new_row = np.roll(x,i)
    master = np.vstack((master,new_row))
print(master)

#69. How to negate a boolean, or to change the sign of a float inplace? 
b1 = True
b2 = not b1
b2 = not b2

orig_float = -3.45
pos_float = np.negative(orig_float)

#70. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance
#from p to each line i (P0[i],P1[i])?
P0 = np.random.randint(-50,50,(10,2))
P1 = np.random.randint(-50,50,(10,2))
p  = np.random.randint(-50,50,( 2))
distances = np.zeros(P0.shape[0])

i=0
while i < P0.shape[0]:
    point1 = P0[i]
    point2 = P1[i]

    #y = mx +b
    
    #  slope = Change in Y       divided by     Change in X    
    m = np.abs(point1[1] - point2[1]) / np.abs(point1[0] - point2[0])
    
    # b = y-mx
    b = point1[1] - (m*point1[0])
    
    m_new = -1/m
    b_new = p[1] - (m*p[0])
    
    #Where the two lines intercept
    x_junction = (b_new - b)/(m - m_new)
    y_junction = m * x_junction + b
    
    point_junction = np.array(x_junction,y_junction)
    
    distance = np.linalg.norm(point_junction-p)
    distances[i] = distance
    print("Distance between {0}, {1}, and {2} is {3} ".format(point1,point2,p,distance))
    
    i = i + 1
    
print("\n\n The given point {0} is closest to the line between {1} and {2}".format(p, P0[np.argmin(distances)],P1[np.argmin(distances)]))

#distances = np.zeros(10)
#for i in range(10):
#    x_diff = P0[i,0] - P1[i,0]
#    y_diff = P0[i,1] - P1[i,1]
#    
#    num = abs(y_diff*p[i,0] - x_diff*p[i,1] + P0[i,0]*P1[i,1] - P0[i,1]*P0[i,1])
#    distances[i] = math.sqrt(y_diff**2 + x_diff**2)
#    

#What in the world?    
T = P1 - P0
L = (T**2).sum(axis=1)
U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
U = U.reshape(len(U),1)
D = P0 + U*T - p
print(np.sqrt((D**2).sum(axis=1)))


#71. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute
#distance from each point j (P[j]) to each line i (P0[i],P1[i])? 


#72. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and
#centered on a given element (pad with a fill value when necessary) 

#73. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R =
#[[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? 
Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))

#74. Compute a matrix rank
 
#75. How to find the most frequent value in an array?

#76. Extract all the contiguous 3x3 blocks from a random 10x10 matrix 

#77. Create a 2D array subclass such that Z[i,j] == Z[j,i] 

#78. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to

#compute the sum of of the p matrix products at once? (result has shape (n,1)) 

#79. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? 
#80. How to implement the Game of Life using numpy arrays? 
#81. How to get the n largest values of an array 
#82. Given an arbitrary number of vectors, build the cartesian product (every combinations of
#every item) 
#83. How to create a record array from a regular array? 
#84. Consider a large vector Z, compute Z to the power of 3 using 3 different methods 
#85. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain
#elements of each row of B regardless of the order of the elements in B? 
#86. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) 
#87. Convert a vector of ints into a matrix binary representation 
#88. Given a two dimensional array, how to extract unique rows? 
#89. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul
#function 
#90. Considering a path described by two vectors (X,Y), how to sample it using equidistant
#samples ?
#91. Given an integer n and a 2D array X, select from X the rows which can be interpreted as
#draws from a multinomial distribution with n degrees, i.e., the rows which only contain
#integers and which sum to n. 

