import numpy as np
import matplotlib.pyplot as plt
import timeit




#manual user defined matrix multiplication function

def matmult(m1,m2):
    #dimentions:   
        m1i,m1j,m2i,m2j=m1.shape[0],m1.shape[1],m2.shape[0],m2.shape[1]
        assert m1j==m2i, "the matices do not have compatible dimentions"    
        #error generation using assert when the dimentions are not compatible    
        #result matrix m3
        #processing cyle counter
        counter=0
        
        m3=np.zeros((m1i,m2j),dtype=np.float32)
        
        for i in range(m1i):
            for j in range(m2j):
                for k in range (m1j):
                    m3[i,j]+=m1[i,k]+m2[k,j]
                    counter+=1
                    
        return m3,counter          



#now we create an array genertor,this will generate ones arrays of dimention (2,2) to (N,N) in each step of iteration(one array only , each time)
def generator(n):
    # Check if n is at least 2
    assert n >= 2, "n should be at least 2"
    
    for i in range(2, n+1,6):
        yield np.ones((i, i), dtype=np.float32)


#print([i for i in generator(100)])
#print([(a.shape[0],matmult(a,a)) for a in generator(100)])

#print(matmult(np.zeros((10,10)),np.zeros((10,10))))


#now we are going to plot the compexity map of the matmult function:
cmap=np.array([(a.shape[0],matmult(a,a)[1]) for a in generator(100)],dtype=np.int32)
#print(cmap)


#reference N^3 plot to compare:    
    
N = np.linspace(0, 100, 100)  # 100 points between 0 and 10
y = N**3  # N^3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plotting the N^3 function with a dashed line
ax1.plot(N, y, linestyle='-', color='r', label='reference N^3 curve')

x,y=zip(*cmap)
ax1.plot(x,y, linestyle='--',label='complexity curve', color='b') 
#plt.plot(x, y2, linestyle='--',label='cos(x)', color='r')  

ax1.set_xlabel('the matrix sizes(input N)')
ax1.set_ylabel('number of cycles C')

# Adding a title
ax1.set_title('Plot of complexity increase C  vs the input size N  \n for manual loop function',fontsize=8)





# Adding the formula (assuming we want to show a mathematical relation, e.g., "y = x^2")
formula = 'N^3'  # LaTeX style formula
ax1.text(3, 10, formula, fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.5))
ax1.legend()


# using numpy dot and ploting its complexity
#because we can not count its cycles as we have no direct access to its inside we use timing, but we use a trick and measure the execution time 



print('np.dot({} , {})'.format(20,20))

cmapdot=[]
for a in generator(30):
    # Measure time, passing 'a' as part of the setup code
    #AA = np.array({a.tolist()})
    time = timeit.timeit(lambda: np.dot(a, a))
    
    # Append the matrix size and corresponding time to the cmapdot list
    cmapdot.append((a.shape[0], time))
    

x_values, y_values = zip(*cmapdot)

# Plot the data
ax2.plot(x_values, y_values, marker='o', linestyle='-', label='dot function')

# Add labels and title
ax2.set_xlabel('the matrix sizes(input N)')
ax2.set_ylabel('time ')
ax2.text(0, 0, 'N^3', fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.5))
# Adding a title
ax2.set_title('Plot of complexity increase C  vs the input size N or time \n,for numpy dot(much faster)',fontsize=8)
plt.savefig("fig3.png")
# Add legend
#plt.legend()

#print(end_time-start_time) 
   