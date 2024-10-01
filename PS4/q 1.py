from gaussxw import gaussxw
from numpy import exp, linspace, arange
import matplotlib.pyplot as plt
from numpy import sum

# Part a: Writing the function and constants
tet_D = 428
ro = 6.022e28
V = 1000
kb = 1.38e-23

# Now using lambda we define the I the inside integral function
I = lambda x: x**4 * exp(x) / (exp(x) - 1)**2

def cv(T):
    # Initiating gaussian points and weights
    x, w = gaussxw(50)
    
    # Defining range
    a = 0
    b = tet_D / T
    
    # Scaling
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    
    s = 9*V*ro*kb * (1/b)**3 * sum(I(xp) * wp)
    return s

# Part b: Plotting heat capacity vs temperature
T_values = linspace(5, 500,100)
CV = [cv(T) for T in T_values]

# Part c: Testing the convergence for a fixed temperature (T = 300)
def cvn(T, N):
    # Initiating gaussian points and weights
    x, w = gaussxw(N)
    
    # Defining range
    a = 0
    b = tet_D / T
    
    # Scaling
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    
    s = 9*V*ro*kb * (1/b)**3 * sum(I(xp) * wp)
    return s

# part C Testing convergence for T = 300 and different N values
x = []
y = []
sampleT = 5
for i in arange(10, 51, 10):
    x.append(i)  # Append the current N value
    y.append(cvn(sampleT, i))  # Append the corresponding cvn value

# Creating two side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot: Heat capacity vs Temperature
ax1.plot(T_values, CV)
ax1.set_xlabel('Temperature T[K]')
ax1.set_ylabel('Cv[J/K]')    
ax1.set_title("Heat Capacity vs Temperature based on Debye's formula")

# Second plot: Convergence of cvn(T, N) for T = 300
ax2.plot(x, y, marker='o')
ax2.set_xlabel('Number of Sampling Points (N)')
ax2.set_ylabel('cvn(T, N)')
ax2.set_title(f'Convergence of cvn(T, N) for T = {sampleT}')
ax2.grid(True)

# Display the plots
plt.tight_layout()
plt.savefig("q1.png")
plt.show()



#yes it coverges