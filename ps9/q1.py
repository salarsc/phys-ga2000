from math import sin
from numpy import array, arange
from pylab import plot, xlabel, ylabel, legend, title, savefig, show

# Function definitions for different oscillators
def harmonic_f(r, t, w):
    x, v = r[0], r[1]
    dxdt = v
    dvdt = -w**2 * x
    return array([dxdt, dvdt], float)

def anharmonic_f(r, t, w):
    x, v = r[0], r[1]
    dxdt = v
    dvdt = -w**2 * x**3
    return array([dxdt, dvdt], float)

def van_der_pol_f(r, t, w, mu):
    x, v = r[0], r[1]
    dxdt = v
    dvdt = mu * (1 - x**2) * v - w**2 * x
    return array([dxdt, dvdt], float)

# Runge-Kutta method
def runge_kutta(f, r, tpoints, h, params):
    results = []
    for t in tpoints:
        results.append(r.copy())
        k1 = h * f(r, t, *params)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h, *params)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h, *params)
        k4 = h * f(r + k3, t + h, *params)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return array(results)

# Plot solutions
def plot_oscillator_results(results, tpoints, labels, xlabel_text, ylabel_text, title_text=None, save_as=None):
    for result, label in zip(results, labels):
        plot(tpoints, result[:, 0], label=label)  # Plot x vs t
    xlabel(xlabel_text)
    ylabel(ylabel_text)
    legend()
    if title_text:
        title(title_text)
    if save_as:
        savefig(save_as)  # Save the plot to a file
    show()

# Phase space plot
def plot_phase_space(results, labels, title_text=None, save_as=None):
    for result, label in zip(results, labels):
        plot(result[:, 0], result[:, 1], label=label)  # Plot v (dx/dt) vs x
    xlabel("Displacement (x)")
    ylabel("Velocity (dx/dt)")
    legend()
    if title_text:
        title(title_text)
    if save_as:
        savefig(save_as)  # Save the plot to a file
    show()

# Parameters
w = 1.0  # Oscillator frequency
a, b = 0.0, 50.0  # Time range for harmonic and anharmonic oscillators
N = 1000
h = (b - a) / N
tpoints = arange(a, b, h)

# (a, b) Harmonic Oscillator
harmonic_initial_conditions = [
    {"x0": 1.0, "v0": 0.0, "label": "Harmonic, Amplitude = 1"},
    {"x0": 2.0, "v0": 0.0, "label": "Harmonic, Amplitude = 2"},
]
harmonic_results = []
for ic in harmonic_initial_conditions:
    r = array([ic["x0"], ic["v0"]], float)
    result = runge_kutta(harmonic_f, r, tpoints, h, [w])
    harmonic_results.append(result)

plot_oscillator_results(
    harmonic_results,
    tpoints,
    [ic["label"] for ic in harmonic_initial_conditions],
    "Time (t)",
    "Displacement (x)",
    title_text="Harmonic Oscillator",
    save_as="harmonic_oscillator.png"
)

# (c) Anharmonic Oscillator
anharmonic_initial_conditions = [
    {"x0": 1.0, "v0": 0.0, "label": "Anharmonic, Amplitude = 1"},
    {"x0": 2.0, "v0": 0.0, "label": "Anharmonic, Amplitude = 2"},
    {"x0": 0.5, "v0": 0.0, "label": "Anharmonic, Amplitude = 0.5"},
]
anharmonic_results = []
for ic in anharmonic_initial_conditions:
    r = array([ic["x0"], ic["v0"]], float)
    result = runge_kutta(anharmonic_f, r, tpoints, h, [w])
    anharmonic_results.append(result)

plot_oscillator_results(
    anharmonic_results,
    tpoints,
    [ic["label"] for ic in anharmonic_initial_conditions],
    "Time (t)",
    "Displacement (x)",
    title_text="Anharmonic Oscillator",
    save_as="anharmonic_oscillator.png"
)

# (d) Phase space plots for harmonic and anharmonic oscillators
plot_phase_space(
    harmonic_results,
    [ic["label"] for ic in harmonic_initial_conditions],
    title_text="Phase Space: Harmonic Oscillator",
    save_as="harmonic_phase_space.png"
)

plot_phase_space(
    anharmonic_results,
    [ic["label"] for ic in anharmonic_initial_conditions],
    title_text="Phase Space: Anharmonic Oscillator",
    save_as="anharmonic_phase_space.png"
)

# (e) Van der Pol Oscillator
mu_values = [1, 2, 4]  # Different mu values
b_vdp = 20.0  # Shorter time range for Van der Pol oscillator
tpoints_vdp = arange(a, b_vdp, h)
vdp_results = []
for mu in mu_values:
    r = array([1.0, 0.0], float)  # Initial conditions: x(0) = 1, v(0) = 0
    result = runge_kutta(van_der_pol_f, r, tpoints_vdp, h, [w, mu])
    vdp_results.append(result)

plot_phase_space(
    vdp_results,
    [f"Van der Pol, μ = {mu}" for mu in mu_values],
    title_text="Phase Space: Van der Pol Oscillator",
    save_as="van_der_pol_phase_space.png"
)
