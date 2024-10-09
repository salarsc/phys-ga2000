
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('signal.dat', delimiter='|', skiprows=1, usecols=[1, 2])
time = data[:, 0]
signal = data[:, 1]

# Part (a): Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='blue', label='Measured Signal')
plt.xlabel('Time')
plt.ylabel('Signal Strength')
plt.title('Scatter Plot of Measured Signal Data')
plt.legend()
plt.grid(True)
plt.savefig('data_plot.png')

# Part (b): Third-order polynomial fit using SVD
time_normalized = (time - np.mean(time)) / np.std(time)
T_third_order = np.column_stack([time_normalized**3, time_normalized**2, time_normalized, np.ones_like(time_normalized)])
U, Sigma, VT = np.linalg.svd(T_third_order, full_matrices=False)
Sigma_inv = np.diag(1 / Sigma)
coefficients_third_order = VT.T @ Sigma_inv @ U.T @ signal
t_fit = np.linspace(np.min(time_normalized), np.max(time_normalized), 100)
signal_fit = coefficients_third_order[0]*t_fit**3 + coefficients_third_order[1]*t_fit**2 + coefficients_third_order[2]*t_fit + coefficients_third_order[3]
plt.figure(figsize=(10, 6))
plt.scatter(time_normalized, signal, color='blue', label='Measured Signal')
plt.plot(t_fit, signal_fit, color='red', label='Third-Order Polynomial Fit')
plt.xlabel('Normalized Time')
plt.ylabel('Signal Strength')
plt.title('Third-Order Polynomial Fit to the Signal Data Using SVD')
plt.legend()
plt.grid(True)
plt.savefig('third_order_fit.png')

# Part (c): Residual analysis
signal_model = coefficients_third_order[0]*time_normalized**3 + coefficients_third_order[1]*time_normalized**2 + coefficients_third_order[2]*time_normalized + coefficients_third_order[3]
residuals = signal - signal_model
plt.figure(figsize=(10, 6))
plt.scatter(time_normalized, residuals, color='green', label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Normalized Time')
plt.ylabel('Residuals (Measured Signal - Model Prediction)')
plt.title('Residuals of the Third-Order Polynomial Fit')
plt.legend()
plt.grid(True)
plt.savefig('residuals.png')

# Part (d): 20th-order polynomial fit using SVD
order = 20
T_20th_order = np.column_stack([time_normalized**i for i in range(order, -1, -1)])
U_20th, Sigma_20th, VT_20th = np.linalg.svd(T_20th_order, full_matrices=False)
Sigma_20th_inv = np.diag(1 / Sigma_20th)
coefficients_20th_order = VT_20th.T @ Sigma_20th_inv @ U_20th.T @ signal
t_fit_20th = np.linspace(np.min(time_normalized), np.max(time_normalized), 100)
signal_fit_20th = sum(coefficients_20th_order[i] * t_fit_20th**(order - i) for i in range(order + 1))
plt.figure(figsize=(10, 6))
plt.scatter(time_normalized, signal, color='blue', label='Measured Signal')
plt.plot(t_fit_20th, signal_fit_20th, color='red', label='20th-Order Polynomial Fit')
plt.xlabel('Normalized Time')
plt.ylabel('Signal Strength')
plt.title('20th-Order Polynomial Fit to the Signal Data Using SVD')
plt.legend()
plt.grid(True)
plt.savefig('20th_order_fit.png')

# Part (e): Fourier series fit with increasing frequency
fundamental_frequency = 1 / (0.5 * (np.max(time) - np.min(time)))
num_harmonics = 10
T_fourier_low_to_high = np.column_stack(
    [np.ones_like(time)] + 
    [np.sin(2 * np.pi * k * fundamental_frequency * time) for k in range(1, num_harmonics + 1)] +
    [np.cos(2 * np.pi * k * fundamental_frequency * time) for k in range(1, num_harmonics + 1)]
)
U_fourier_low_to_high, Sigma_fourier_low_to_high, VT_fourier_low_to_high = np.linalg.svd(T_fourier_low_to_high, full_matrices=False)
Sigma_fourier_low_to_high_inv = np.diag(1 / Sigma_fourier_low_to_high)
coefficients_fourier_low_to_high = VT_fourier_low_to_high.T @ Sigma_fourier_low_to_high_inv @ U_fourier_low_to_high.T @ signal
signal_fit_fourier_low_to_high = coefficients_fourier_low_to_high[0]
for k in range(1, num_harmonics + 1):
    signal_fit_fourier_low_to_high += coefficients_fourier_low_to_high[k] * np.sin(2 * np.pi * k * fundamental_frequency * time)
    signal_fit_fourier_low_to_high += coefficients_fourier_low_to_high[k + num_harmonics] * np.cos(2 * np.pi * k * fundamental_frequency * time)
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='blue', label='Measured Signal')
plt.plot(time, signal_fit_fourier_low_to_high, color='red', label='Fourier Series Fit (Low to High Frequencies)')
plt.xlabel('Time')
plt.ylabel('Signal Strength')
plt.title('Fourier Series Fit to the Signal Data (Low to High Frequencies)')
plt.legend()
plt.grid(True)
plt.savefig('d.png')
plt.show()