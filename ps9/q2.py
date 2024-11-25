from math import sin, cos, sqrt, pi
from numpy import array, arange
from pylab import plot, xlabel, ylabel, title, legend, axhline, savefig, show

# Parameters
R = 0.08  # radius (m)
rho = 1.22  # air density (kg/m^3)
C = 0.47  # drag coefficient
g = 9.81  # gravity (m/s^2)
v0 = 100.0  # initial velocity (m/s)
theta = 30.0  # launch angle (degrees)
theta_rad = theta * pi / 180  # angle in radians

# Rescaled parameters
beta = (C * rho * R**2 * g) / (2 * v0**2)

# Initial conditions in rescaled variables
x0, y0 = 0.0, 0.0
vx0_prime = cos(theta_rad)  # Rescaled initial horizontal velocity
vy0_prime = sin(theta_rad)  # Rescaled initial vertical velocity

# Time parameters
t0, t_end = 0.0, 20.0  # start and end time (s)
dt = 0.01  # time step

# Function for derivatives in rescaled variables
def rescaled_derivatives(r_prime, t_prime):
    x_prime, y_prime, vx_prime, vy_prime = r_prime
    v_prime = sqrt(vx_prime**2 + vy_prime**2)
    dxdt_prime = vx_prime
    dydt_prime = vy_prime
    dvxdt_prime = -beta * vx_prime * v_prime
    dvydt_prime = -1 - beta * vy_prime * v_prime
    return array([dxdt_prime, dydt_prime, dvxdt_prime, dvydt_prime], float)

# Function for derivatives with drag
def derivatives(r, t, k):
    x, y, vx, vy = r
    v = sqrt(vx**2 + vy**2)
    dxdt = vx
    dydt = vy
    dvxdt = -k * vx * v
    dvydt = -g - k * vy * v
    return array([dxdt, dydt, dvxdt, dvydt], float)

# **Part b: Rescaled Trajectory**
r_prime = array([x0, y0, vx0_prime, vy0_prime], float)
tpoints_prime = arange(t0, t_end, dt)
xpoints_prime, ypoints_prime = [], []

for t_prime in tpoints_prime:
    xpoints_prime.append(r_prime[0])
    ypoints_prime.append(r_prime[1])
    if len(ypoints_prime) > 1 and r_prime[1] <= 0:
        x_last, x_prev = xpoints_prime[-1], xpoints_prime[-2]
        y_last, y_prev = ypoints_prime[-1], ypoints_prime[-2]
        x_ground = x_prev + (0 - y_prev) * (x_last - x_prev) / (y_last - y_prev)
        xpoints_prime[-1] = x_ground
        ypoints_prime[-1] = 0
        break
    k1 = dt * rescaled_derivatives(r_prime, t_prime)
    k2 = dt * rescaled_derivatives(r_prime + 0.5 * k1, t_prime + 0.5 * dt)
    k3 = dt * rescaled_derivatives(r_prime + 0.5 * k2, t_prime + 0.5 * dt)
    k4 = dt * rescaled_derivatives(r_prime + k3, t_prime + dt)
    r_prime += (k1 + 2 * k2 + 2 * k3 + k4) / 6

plot(xpoints_prime, ypoints_prime, label="Rescaled Trajectory")
axhline(0, color='black', linestyle='--', label="Ground")
xlabel("Rescaled Horizontal Distance (x')")
ylabel("Rescaled Vertical Distance (y')")
title("Rescaled Trajectory of a Cannonball")
legend()
savefig("rescaled_trajectory.png")
show()

# **Part c: Trajectories for Different Masses**
masses = [0.5, 1.0, 2.0]
for m in masses:
    k = 0.5 * C * rho * pi * R**2 / m  # Recalculate drag constant
    r = array([x0, y0, v0 * cos(theta_rad), v0 * sin(theta_rad)], float)
    tpoints = arange(t0, t_end, dt)
    xpoints, ypoints = [], []
    for t in tpoints:
        if r[1] < 0:
            break
        xpoints.append(r[0])
        ypoints.append(r[1])
        k1 = dt * derivatives(r, t, k)
        k2 = dt * derivatives(r + 0.5 * k1, t + 0.5 * dt, k)
        k3 = dt * derivatives(r + 0.5 * k2, t + 0.5 * dt, k)
        k4 = dt * derivatives(r + k3, t + dt, k)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    range_distance = xpoints[-1]
    print(f"Mass = {m:.2f} kg, Range = {range_distance:.2f} m")
    plot(xpoints, ypoints, label=f"m = {m:.2f} kg")

axhline(0, color='black', linestyle='--', label="Ground")
xlabel("Horizontal Distance (m)")
ylabel("Vertical Distance (m)")
title("Trajectories for Different Masses")
legend()
savefig("trajectories_different_masses.png")
show()

# **Part d: Range as a Function of Mass**
mass_values = arange(0.1, 5.1, 0.1)  # masses from 0.1 kg to 5.0 kg
ranges = []
for m in mass_values:
    k = 0.5 * C * rho * pi * R**2 / m  # Recalculate drag constant
    r = array([x0, y0, v0 * cos(theta_rad), v0 * sin(theta_rad)], float)
    tpoints = arange(t0, t_end, dt)
    xpoints, ypoints = [], []
    for t in tpoints:
        if r[1] < 0:
            break
        xpoints.append(r[0])
        ypoints.append(r[1])
        k1 = dt * derivatives(r, t, k)
        k2 = dt * derivatives(r + 0.5 * k1, t + 0.5 * dt, k)
        k3 = dt * derivatives(r + 0.5 * k2, t + 0.5 * dt, k)
        k4 = dt * derivatives(r + k3, t + dt, k)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    range_distance = xpoints[-1]
    ranges.append(range_distance)

plot(mass_values, ranges)
xlabel("Mass (kg)")
ylabel("Range (m)")
title("Range as a Function of Mass")
savefig("range_vs_mass.png")
show()
