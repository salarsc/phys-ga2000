import numpy as np


#by method of for loop


#we define the function to calculate it , L is number of atms , a is the distance within
def maconst(L, a):
    
    V_total = 0 # total electric potential
   
#loop for i , j ,k
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i == 0 and j == 0 and k == 0:
                    continue
                #check the even or odd for cl or na
                elif abs(i+j+k) % 2 == 0:
                    V_total += 1 / (np.sqrt(i**2+j**2+k**2))
                else:
                    V_total -= 1 / np.sqrt(i**2+j**2+k**2)
    
    M = V_total # Madelung constant
    return M

#test
print(maconst(20,0.000000000564))
#completely true




#by method of no loop (numpy array)



def maconstnum(L, a):
    # Generate i, j, k indices as 3D arrays
    i, j, k = np.meshgrid(np.arange(-L, L+1), np.arange(-L, L+1), np.arange(-L, L+1), indexing='ij')
    
    # Calculate the distances r = sqrt(i^2 + j^2 + k^2) for all combinations
    r = np.sqrt(i**2 + j**2 + k**2)
    
    # Mask to exclude the center (i, j, k) = (0, 0, 0)
    mask = (i != 0) | (j != 0) | (k != 0)
    
    # Apply the mask to r
    r = r[mask]
    
    # Calculate the signs based on the evenness or oddness of i + j + k
    signs = (-1) ** ((i + j + k)[mask] % 2)
    
    # Compute the Madelung constant
    V_total = np.sum(signs / r)
    
    return V_total

# Test
print(maconstnum(20, 0.000000000564))

#now we compare the speed 
%timeit (maconst(20,0.000000000564))

# Comparing the second function
%timeit maconstnum(20, 0.000000000564)
print("as it is obvious the second one that use numpy loops is faster, almost 100 times")

