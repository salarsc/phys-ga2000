import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Set random seed for reproducibility
np.random.seed(42)

def generate_random_means(sample_size, total_samples):
    # Generate total_samples sets of sample_size exponentially distributed random variables
    random_samples = np.random.exponential(scale=1.0, size=(total_samples, sample_size))
    # Calculate the mean of each sample
    sample_averages = np.mean(random_samples, axis=1)
    return sample_averages

# Parameters
total_samples = 10000  # Number of samples to generate
sample_size_values = [1, 5, 10, 20, 50, 100, 200]  # Different sample sizes

# Store statistics
mean_values = []
variance_values = []
skewness_values = []
kurtosis_values = []

for sample_size in sample_size_values:
    sample_averages = generate_random_means(sample_size, total_samples)
    
    # Calculate statistics
    mean_values.append(np.mean(sample_averages))
    variance_values.append(np.var(sample_averages))
    skewness_values.append(skew(sample_averages))
    kurtosis_values.append(kurtosis(sample_averages))
    
    # Plot histogram for each sample_size
    plt.hist(sample_averages, bins=50, density=True, alpha=0.6, label=f'sample_size={sample_size}')
    
    # Show normal distribution curve for comparison
    x_range = np.linspace(0, 3, 100)
    normal_curve = (1/np.sqrt(2 * np.pi * (1/sample_size))) * np.exp(-0.5 * ((x_range - 1) ** 2) / (1/sample_size))
    plt.plot(x_range, normal_curve, label='Normal Approximation', linestyle='dashed')

    plt.title(f'Histogram of Sample Means for sample_size={sample_size}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Probability Density')
    plt.legend()
    
    # Save the histogram figure
    plt.savefig(f'histogram_sample_size_{sample_size}.png')
    plt.show()

# Plot how mean, variance, skewness, and kurtosis change with sample_size
plt.figure(figsize=(10, 6))

# Plot Mean
plt.subplot(2, 2, 1)
plt.plot(sample_size_values, mean_values, marker='o')
plt.axhline(1, color='r', linestyle='--', label='Expected Mean')
plt.title('Mean vs sample_size')
plt.xlabel('sample_size')
plt.ylabel('Mean')
plt.legend()

# Plot Variance
plt.subplot(2, 2, 2)
plt.plot(sample_size_values, variance_values, marker='o')
plt.axhline(0, color='r', linestyle='--', label='Expected Variance for large sample_size')
plt.title('Variance vs sample_size')
plt.xlabel('sample_size')
plt.ylabel('Variance')
plt.legend()

# Plot Skewness
plt.subplot(2, 2, 3)
plt.plot(sample_size_values, skewness_values, marker='o')
plt.axhline(0, color='r', linestyle='--', label='Expected Skewness for large sample_size')
plt.title('Skewness vs sample_size')
plt.xlabel('sample_size')
plt.ylabel('Skewness')
plt.legend()

# Plot Kurtosis
plt.subplot(2, 2, 4)
plt.plot(sample_size_values, kurtosis_values, marker='o')
plt.axhline(0, color='r', linestyle='--', label='Expected Kurtosis for large sample_size')
plt.title('Kurtosis vs sample_size')
plt.xlabel('sample_size')
plt.ylabel('Kurtosis')
plt.legend()

# Save the overall statistics figure
plt.tight_layout()
plt.savefig('statistics_vs_sample_size.png')
plt.show()

# Determine at which sample_size skewness and kurtosis reach ~1% of their initial value
initial_skewness = skewness_values[0]
initial_kurtosis = kurtosis_values[0]

for i, sample_size in enumerate(sample_size_values):
    if abs(skewness_values[i]) < 0.01 * abs(initial_skewness):
        print(f'Skewness reaches 1% of its initial value at sample_size = {sample_size}')
        break

for i, sample_size in enumerate(sample_size_values):
    if abs(kurtosis_values[i]) < 0.01 * abs(initial_kurtosis):
        print(f'Kurtosis reaches 1% of its initial value at sample_size = {sample_size}')
        break
