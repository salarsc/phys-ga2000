import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft

# Load and prepare data from dow.txt
with open('dow.txt', 'r') as file:
    dow_prices = np.array([float(line.strip()) for line in file])

# Plot and save the original Dow Jones data with 10% filtered data
plt.figure(figsize=(12, 6))
plt.plot(dow_prices, label="Dow Jones Original", color='blue')
plt.title("Dow Jones Industrial Average (Daily Closing Values)")
plt.xlabel("Days")
plt.ylabel("Closing Value")

# Compute Fourier Transform to analyze frequency components
fourier_transformed = rfft(dow_prices)

# Filter to retain only the first 10% of Fourier coefficients
filtered_10pct = fourier_transformed.copy()
limit_10pct = int(0.1 * len(fourier_transformed))
filtered_10pct[limit_10pct:] = 0  # Set the rest to zero for smoothing

# Inverse Transform for 10% filtered data to bring back to time domain
smoothed_10pct = irfft(filtered_10pct)
plt.plot(smoothed_10pct, label="Smoothed Data (10% Coefficients)", color='orange')
plt.legend()
plt.savefig("q3-1.png")  # Save the figure as q3-1
plt.show()

# Further filter to retain only the first 2% of Fourier coefficients
filtered_2pct = fourier_transformed.copy()
limit_2pct = int(0.02 * len(fourier_transformed))
filtered_2pct[limit_2pct:] = 0  # Set the remaining coefficients to zero for extra smoothing

# Inverse Transform for 2% filtered data
smoothed_2pct = irfft(filtered_2pct)

# Plot original data with both 10% and 2% smoothed versions and save
plt.figure(figsize=(12, 6))
plt.plot(dow_prices, label="Dow Jones Original", color='blue')
plt.plot(smoothed_10pct, label="Smoothed Data (10% Coefficients)", color='orange')
plt.plot(smoothed_2pct, label="Highly Smoothed Data (2% Coefficients)", color='green')
plt.legend()
plt.title("Dow Jones Industrial Average: Smoothing via Fourier Filtering")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.savefig("q3-2.png")  # Save the figure as q3-2
plt.show()
