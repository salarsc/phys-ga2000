import jax
import jax.numpy as jnp
from math import *
from numpy import tanh, cosh, linspace
import matplotlib.pyplot as plt

# Define the function
f = lambda x: 1 + (1/2) * (tanh(2 * x))

# Initializing xi, xf, and h
xi = -2
xf = 2
h = 1e-5

# Generating the input values using np.linspace
x = linspace(xi, xf, int((xf - xi) / h), endpoint=True)

# Function output
y = f(x)

# Now using the method of central difference:
derivative = []
for i in range(1, len(x)-1):
    derivative.append((y[i+1] - y[i-1]) / (2*h))

# Analytical derivative
analytic = 1 / cosh(2 * x)**2

# Now we perform the JAX autodiff at the end
# Define the function in JAX
f_jax = lambda x: 1 + (1/2) * (jnp.tanh(2 * x))

# Use jax.vmap to compute the derivative for each point in the array
f_prime = jax.grad(f_jax)
f_prime_vmap = jax.vmap(f_prime)

# Compute the derivative for each value in x
jax_derivative = f_prime_vmap(jnp.array(x))

# Create two subplots to compare numerical with analytical, and JAX with analytical

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Numerical vs Analytical
ax1.plot(x[1:-1], derivative, "ro", label='Numerical', markersize=4, alpha=0.7)
ax1.plot(x, analytic, "b--", label='Analytical', linewidth=2, alpha=0.8)
ax1.set_title("Numerical vs Analytical Derivative")
ax1.set_xlabel("x")
ax1.set_ylabel("f'(x)")
ax1.legend()
ax1.grid(True)

# Plot 2: JAX Autodiff (Yellow dots) vs Analytical (Blue dashed)
ax2.plot(x, jax_derivative, "yo", label='Autodiff (JAX)', markersize=4, alpha=0.7)  # Yellow dots for JAX
ax2.plot(x, analytic, "b--", label='Analytical', linewidth=2, alpha=0.8)           # Blue dashed line for Analytical
ax2.set_title("JAX Autodiff vs Analytical Derivative")
ax2.set_xlabel("x")
ax2.set_ylabel("f'(x)")
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Save and show the figure
plt.savefig("q1_two_sided_plot_with_yellow_jax.png")
plt.show()

