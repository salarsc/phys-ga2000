
#Hi, please using mouse magnify the simulator screen to see better.Thank you


from banded import banded
from pylab import *
from vpython import canvas, curve, rate, vector, color, label

# these are Simulation parameters
time_step = 1e-17
hbar = 1.0546e-36  # Reduced Planck constant
length = 1e-8      # Length of the system
mass = 9.109e-31   # Mass of the particle (electron)
num_points = 1000  # Number of grid points

grid_spacing = length / num_points

# now Coefficients for the finite difference scheme
coeff_a1 = 1 + time_step * hbar / (2 * mass * grid_spacing**2) * 1j
coeff_a2 = -time_step * hbar * 1j / (4 * mass * grid_spacing**2)
coeff_b1 = 1 - time_step * hbar / (2 * mass * grid_spacing**2) * 1j
coeff_b2 = time_step * hbar * 1j / (4 * mass * grid_spacing**2)

# now this is to Initialize the wave function
wave_function = zeros(num_points + 1, complex)

# Initial Gaussian wave packet
def initial_wave_packet(position):
    center = length / 2
    spread = 1e-10
    momentum = 5e10
    return exp(-(position - center)**2 / (2 * spread**2)) * exp(1j * momentum * position)

# we Create a position grid and initialize wave function
positions = linspace(0, length, num_points + 1)
wave_function[:] = initial_wave_packet(positions)
wave_function[[0, num_points]] = 0  # Apply boundary conditions

# now Set up the tridiagonal matrix
matrix_A = empty((3, num_points), complex)
matrix_A[0, :] = coeff_a2
matrix_A[1, :] = coeff_a1
matrix_A[2, :] = coeff_a2

# then we Create a canvas with a white background
scene = canvas(background=color.white, title="Wave Function Visualization")


wave_curve = curve(color=color.blue)


x_axis = curve(color=color.black)
y_axis = curve(color=color.black)


x_axis.append(vector(-length / 2, 0, 0))
x_axis.append(vector(length / 2, 0, 0))


y_axis.append(vector(0, -1e-8, 0))
y_axis.append(vector(0, 1e-8, 0))


label(pos=vector(length / 2 + 1e-9, -2e-10, 0), text="Position (X) [m]", color=color.black, box=False)
label(pos=vector(-length / 2 + 1e-9, 1e-9, 0), text="Amplitude (Real Part)", color=color.black, box=False, align="center")


for i in range(-2, 3):
    x_label_pos = vector(i * length / 4, -1e-10, 0)
    label(pos=x_label_pos, text=f"{i * length / 4:.1e}", color=color.black, box=False)

for i in range(-1, 2):
    y_label_pos = vector(0, i * 5e-9, 0)
    label(pos=y_label_pos, text=f"{i * 0.05:.2f}", color=color.black, box=False)

# Initialize the curve with zero positions
for pos in positions - length / 2:
    wave_curve.append(vector(pos, 0, 0))

# Main simulation loop
while True:
    rate(30)  # Limit the update rate to 30 frames per second

    # Update the wave curve dynamically to show the real part of the wave function
    for i in range(len(positions)):
        wave_curve.modify(i, vector(positions[i] - length / 2, real(wave_function[i]) * 1e-9, 0))

    # Update the wave function using the banded solver
    for _ in range(20):
        rhs = coeff_b1 * wave_function[1:num_points] + coeff_b2 * (wave_function[2:num_points + 1] + wave_function[0:num_points - 1])
        wave_function[1:num_points] = banded(matrix_A, rhs, 1, 1)
