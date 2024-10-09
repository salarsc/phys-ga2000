import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# Define the function for the integrand
def integrand(x, n):
    return x**(n - 1) * np.exp(-x)

# Generate the x values
x_values = np.linspace(0, 5, 1000)

# Plot the curves for n = 2, 3, and 4
plt.figure(figsize=(10, 6))
for n in [2, 3, 4]:
    plt.plot(x_values, integrand(x_values, n), label=f'n = {n}')

# Add labels, legend, and title
plt.xlabel('x')
plt.ylabel('Integrand: $x^{n-1} e^{-x}$')
plt.title('Plot of the Integrand $x^{n-1} e^{-x}$ for n = 2, 3, 4')
plt.legend()
plt.grid(True)
plt.savefig("3.png")
plt.show()




# Define the integrand with the change of variables
def integrand_transformed(z, n, c=1):
    x = z / (c + z)
    dx_dz = c / (c + z)**2
    return np.exp((n - 1) * np.log(x) - x) * dx_dz

# Gamma function implementation
def gamma(a):
    result, _ = quad(integrand_transformed, 0, 1, args=(a, 1))
    return result

# Test the gamma function with known value
gamma_value = gamma(0.5)
print(f'Gamma(0.5) = {gamma_value}, Expected value ≈ 0.886')
print(f'Gamma(3) = {gamma(3)}, Expected value = 2')
print(f'Gamma(6) = {gamma(6)}, Expected value = 120')
print(f'Gamma(10) = {gamma(10)}, Expected value = 362880')