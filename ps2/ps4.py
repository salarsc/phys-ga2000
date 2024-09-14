import numpy as np
import matplotlib.pyplot as plt

grid_size = 1000  # Grid resolution
max_iters = 100  # Maximum number of iterations
div_threshold = 2  # Threshold for divergence

# Create a grid of complex numbers z = u + iv over the range [-2, 2] for both u and v
u_vals = np.linspace(-2, 2, grid_size)
v_vals = np.linspace(-2, 2, grid_size)
U, V = np.meshgrid(u_vals, v_vals)
complex_grid = U + 1j * V  # Complex grid

# Initialize Z as an array of zeros (starting points)

print(complex_grid)
# Array to store the number of iterations before divergence
iterations_grid = np.zeros(complex_grid.shape, dtype=np.float32)
Z_vals = np.zeros_like(complex_grid)

# Perform the iteration over all points without explicit loops
for iteration in range(max_iters):
    Z_vals = Z_vals**2 + complex_grid  # Apply Mandelbrot iteration
    diverge_mask = np.abs(Z_vals) <= div_threshold  # Check for divergence
    iterations_grid[diverge_mask] += 1  # Count iterations for points that haven't diverged yet

# Plot the Mandelbrot set
plt.imshow(iterations_grid.T, cmap='binary', extent=[-2, 2, -2, 2])

plt.title('Mandelbrot Set')

# Define the axes
plt.xlabel('Real Part (u)')
plt.ylabel('Imaginary Part (v)')

# Show the plot
plt.show()