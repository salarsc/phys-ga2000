#to know the least number that can be added in 1 in float 32 and float 64 we need to use finfo() method of the numpy
#this number is called machine epsilon:
import numpy as np   
eps32=np.finfo(np.float32).eps    
eps64=np.finfo(np.float64).eps    

print("the least number that can be added to 1 to change for float32 is{} and for float 64 is{}".format(eps32,eps64))

#now we test it by comparing the results when they added to on
n32=np.float32(1)
n64=np.float64(1)

print("here the equality between 1 added to epsilon for 32 bit is {} and for  epsilon64 bit is{}, thus proved".format(n32+eps32==1,n64+eps64==1))

#now we test to see if less than it can work

print("here the equality between 1 added to slightly lower than epsilon  this number for 2 bit is {} and for 64 bit is {}, thus proved".format(np.float32((n32+(eps32/2)))==1,np.float64((n64+(eps64/2)))==1))


import numpy as np

# Getting the minimum and maximum positive representable numbers for 32-bit and 64-bit floating-point types
min_32 = np.finfo(np.float32).tiny
max_32 = np.finfo(np.float32).max
min_64 = np.finfo(np.float64).tiny
max_64 = np.finfo(np.float64).max

print("the maximum for 32 bit is{} for 64 bit is{} , the minimum before underflow in 32 bit is {} in 64 bit is {}, thus we see how 64 bit numpy float is accurate than 32 bit numpy float ".format(max_32,max_64,min_32,min_64))

#we test below, if it overflows or underflows it gives an error
# Testing overflow for 32-bit float
overflow_32 = np.float32(max_32) * 2

# Testing underflow for 32-bit float
underflow_32 = np.float32(min_32) / 2

# Testing overflow for 64-bit float
overflow_64 = np.float64(max_64) * 2

# Testing underflow for 64-bit float
underflow_64 = np.float64(min_64) / 2

# Display results
overflow_32, underflow_32, overflow_64, underflow_64