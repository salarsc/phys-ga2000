from math import factorial, sqrt, pi
import numpy as np
from pylab import plot, xlabel, ylabel, title, show,savefig
from gaussxw import gaussxw

# Part A: Hermite polynomial function
def H(n, x):
    if n == 0:
        return np.ones_like(x)  # H_0(x) = 1
    elif n == 1:
        return 2 * x  # H_1(x) = 2x
    else:
        H_nm1 = 2 * x  # H_1(x)
        H_nm2 = np.ones_like(x)  # H_0(x)
        for i in range(2, n+1):
            H_n = 2 * x * H_nm1 - 2 * (i - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n  # Update for next iteration
        return H_n

# Part A: Wavefunction definition
def phi(n, x):
    return (1 / sqrt(2**n * factorial(n) * sqrt(pi))) * np.exp(-x**2 / 2) * H(n, x)

# Plotting wavefunctions for n = 0, 1, 2, 3
x = np.linspace(-4, 4, 100)
n_values = [0, 1, 2, 3]
for n in n_values:
    plot(x, phi(n, x), label=f"n = {n}")
xlabel("x")
ylabel("wave function")
title("Wavefunction graph for n=0 to n=3 of quantum harmonic oscillator")
savefig("q3,1.png")
show()

# Part B: Plot wavefunction for n=30
x = np.linspace(-10, 10, 200)
plot(x, phi(30, x), label="n = 30")
xlabel("x")
ylabel("wave function")
title("Wavefunction graph for n=30 of quantum harmonic oscillator")
savefig("q3,2.png")
show()

# Part C: RMS calculation using Gaussian quadrature

# Define the function for integration
def f(n, x):
    return x**2 * phi(n, x)**2

def phi2(n):
    # Define the integral with the variable transformation z -> tan(z)
    fz = lambda z: f(n, np.tan(z)) / (np.cos(z)**2)  # Adjusted for the tan(z) substitution
    z, w = gaussxw(100)  # Get Gaussian quadrature points and weights

    # Scaling the points from -pi/2 to pi/2
    a = -pi/2
    b = pi/2
    z = 0.5 * (b - a) * z + 0.5 * (b + a)
    w = 0.5 * (b - a) * w

    # Perform the integration
    integral = np.sum(w * fz(z))
    return integral

# Calculate and print the result for n=5
phi2_value = phi2(5)
print(f"Phi2 for n=5: {phi2_value}")
#results are Phi2 for n=5: 5.499999999380583      
# Root-mean-square (RMS) value
rms = lambda n: sqrt(phi2(n))
rms_value = rms(5)
print(f"RMS for n=5: {rms_value}")
#result : RMS for n=5: 2.3452078797796547




from scipy.special import eval_hermite, roots_hermite

# Define the Hermite polynomial squared times x^2
def g(n, x):
    return x**2 * eval_hermite(n, x)**2

# Perform the Gauss-Hermite quadrature to evaluate the integral
def phi2hermit(n):
    # Get the roots and weights for Gauss-Hermite quadrature
    x, wh = roots_hermite(100)  # 100-point Gauss-Hermite quadrature for accuracy
    
    # Perform the weighted sum to evaluate the integral
    integral = np.sum(wh * g(n, x))
    
    # Multiply by the appropriate normalization factor
    integral *= 1 / (2**n * factorial(n) * sqrt(pi))
    
    return integral

# Perform the calculation for n=5 using Gauss-Hermite quadrature
n = 5
result = phi2hermit(n)
print(f"Exact integral result for n={n} using Gauss-Hermite quadrature: {result}")

# RMS value calculation: sqrt(<x^2>)
rms = sqrt(result)
print(f"RMS value for n={n}: {rms}")
