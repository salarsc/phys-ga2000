import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import time

# Part (a) - Plot the spectra of 5 nearby galaxies with Hydrogen Balmer lines
hdu_list = fits.open('specgrid.fits')

# Extract logwave (log10 λ for λ in Angstroms) and flux (spectrum data)
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# Close the file after reading
hdu_list.close()

wavelength = 10 ** logwave

# Select a handful of galaxies (for example, the first 5)
sample_flux = flux[:5, :]  # Taking the first 5 galaxies

# Define the wavelengths of the Hydrogen Balmer lines
balmer_lines = {
    'Hα (6563 Å)': 6563,
    'Hβ (4861 Å)': 4861,
    'Hγ (4341 Å)': 4341,
    'Hδ (4102 Å)': 4102
}

# Plot the spectra of the selected galaxies
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(wavelength, sample_flux[i], label=f'Galaxy {i+1}')

# Mark the Balmer lines with vertical tick markers on the x-axis
for line_name, line_wavelength in balmer_lines.items():
    plt.plot([line_wavelength], [0], marker='|', color='black', markersize=12, label=line_name)

# Adding plot details
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux (10^-17 erg s^-1 cm^-2 Å^-1)')
plt.title('Spectra of 5 Nearby Galaxies with Hydrogen Balmer Lines')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("parta.png")
plt.show()

# Result for Part (a): "parta.png" shows the spectra of 5 galaxies with labeled Hydrogen Balmer lines.

# Part (b) - Normalize the flux for each galaxy
normalization_factors = np.zeros(flux.shape[0])
delta_wavelength = np.gradient(wavelength)  # Approximate step sizes in wavelength
normalized_flux = np.zeros_like(flux)  # Create an array to store normalized fluxes

for i in range(flux.shape[0]):
    total_flux = np.trapz(flux[i, :], x=wavelength)
    normalization_factors[i] = total_flux
    normalized_flux[i, :] = flux[i, :] / total_flux

# Result for Part (b): Each galaxy's flux is normalized, with total_flux values stored in normalization_factors.

# Part (c) - Compute the mean spectrum and subtract from each galaxy's flux
mean_spectrum = np.mean(normalized_flux, axis=0)
residual_flux = normalized_flux - mean_spectrum

# Result for Part (c): Residual flux matrix is computed by subtracting the mean spectrum from each galaxy's flux.

# Part (d) - Compute the covariance matrix and perform eigen decomposition
N_galaxies = residual_flux.shape[0]  # Number of galaxies
covariance_matrix = (1 / N_galaxies) * np.dot(residual_flux.T, residual_flux)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Plot the first five eigenvectors (principal components)
plt.figure(figsize=(12, 8))
for i in range(5):
    plt.plot(wavelength, eigenvectors[:, i], label=f'Eigenvector {i+1}')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Amplitude')
plt.title('First 5 Eigenvectors from PCA')
plt.legend()
plt.grid(True)
plt.savefig("partd1.png")
plt.show()

# Result for Part (d): "partd1.png" shows the first 5 eigenvectors computed using PCA.

# Part (e) - Perform SVD and compare with covariance eigenvectors
start_svd_time = time.time()
U, W, Vt = svd(residual_flux, full_matrices=False)
end_svd_time = time.time()
V_svd = Vt.T  # Transpose to match dimensions

# Eigenvectors comparison
eigenvectors_cov = eigenvectors[:, :5]  # First five eigenvectors from covariance method
eigenvectors_svd = V_svd[:, :5]  # First five eigenvectors from SVD

print("Comparison of eigenvectors (first 5 columns):")
for i in range(5):
    print(f"Eigenvector {i+1}:")
    print(f"Covariance method: {eigenvectors_cov[:, i]}")
    print(f"SVD method: {eigenvectors_svd[:, i]}")
    print(f"Difference: {np.linalg.norm(eigenvectors_cov[:, i] - eigenvectors_svd[:, i])}")

# Computational time comparison
print("\nComputational time for SVD method: {:.5f} seconds".format(end_svd_time - start_svd_time))
start_eigen_time = time.time()
eigenvalues_cov, eigenvectors_cov_alt = np.linalg.eig(np.dot(residual_flux.T, residual_flux))
end_eigen_time = time.time()

# Sorting eigenvalues and eigenvectors for consistency
sorted_indices_alt = np.argsort(eigenvalues_cov)[::-1]
eigenvectors_cov_alt = eigenvectors_cov_alt[:, sorted_indices_alt]
print("\nComputational time for covariance eigen decomposition method: {:.5f} seconds".format(end_eigen_time - start_eigen_time))
print(f"\nSpeedup using SVD: {(end_eigen_time - start_eigen_time) / (end_svd_time - start_svd_time):.2f}x")

# Results for Part (e):
# Comparison of eigenvectors (first 5 columns):
# Eigenvector 1:
# Covariance method: [-0.02026115 -0.02068221 -0.01985255 ...  0.01344336  0.01334705 0.01304163]
# SVD method: [ 0.02026115  0.02068221  0.01985255 ... -0.01344335 -0.01334705 -0.01304163]
# Difference: 2.0

# Eigenvector 2:
# Covariance method: [-0.00732357 -0.00813623 -0.00777943 ...  0.01922882  0.019036  0.01868768]
# SVD method: [-0.00732357 -0.00813623 -0.00777943 ...  0.01922882  0.019036  0.01868768]
# Difference: 1.229170294436699e-07

# Eigenvector 3:
# Covariance method: [-0.00139955 -0.00154928 -0.00171205 ...  0.0031653  0.00309557 0.00284273]
# SVD method: [-0.00139955 -0.00154928 -0.00171205 ...  0.00316531  0.00309558 0.00284273]
# Difference: 5.713738815416036e-08

# Eigenvector 4:
# Covariance method: [-0.00354917 -0.00678993  0.00875472 ... -0.05514797 -0.04150251 -0.00300133]
# SVD method: [ 0.00354917  0.00678993 -0.00875472 ...  0.05514797  0.04150251  0.00300134]
# Difference: 2.0

# Eigenvector 5:
# Covariance method: [-0.01002841 -0.00844457 -0.01222887 ...  0.00291238 -0.00153644 -0.01354197]
# SVD method: [-0.01002839 -0.00844456 -0.01222887 ...  0.0029124  -0.00153643 -0.01354196]
# Difference: 4.5884468136137e-07

# Computational time for SVD method: 47.26358 seconds
# Computational time for covariance eigen decomposition method: 39.12994 seconds
# Speedup using SVD: 0.83x

# Part (e) - Plot eigenvectors from covariance and SVD methods for comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title('Covariance Method: First 5 Eigenvectors')
for i in range(5):
    ax1.plot(wavelength, eigenvectors_cov[:, i], label=f'Eigenvector {i+1}')
ax1.set_xlabel('Wavelength (Angstrom)')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)

ax2.set_title('SVD Method: First 5 Eigenvectors')
for i in range(5):
    ax2.plot(wavelength, eigenvectors_svd[:, i], label=f'Eigenvector {i+1}', linestyle='--')
ax2.set_xlabel('Wavelength (Angstrom)')
ax2.set_ylabel('Amplitude')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("partd2.png")
plt.show()

# Result for Part (e): "partd2.png" compares the first 5 eigenvectors from the covariance method and SVD.

# Part (f) - Compute condition number of R and covariance matrix C
cond_R = np.linalg.cond(residual_flux)
cov_matrix = np.dot(residual_flux.T, residual_flux)
cond_C = np.linalg.cond(cov_matrix)
print(f"Condition number of R: {cond_R:.2e}")
print(f"Condition number of C: {cond_C:.2e}")

# Result for Part (f): 
# - Condition number of R: 4.13e+06
# - Condition number of C: 3.62e+10

# Part (g) - Approximate spectra using the first Nc coefficients
Nc = 5
coefficients = np.dot(residual_flux, eigenvectors[:, :Nc])
approx_flux = np.dot(coefficients, eigenvectors[:, :Nc].T) + mean_spectrum
plt.figure(figsize=(10, 6))
plt.plot(wavelength, residual_flux[0, :] + mean_spectrum, label='Original Spectrum')
plt.plot(wavelength, approx_flux[0, :], label='Approximate Spectrum (Nc=5)', linestyle='--')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux')
plt.title('Original vs Approximate Spectrum (First Galaxy)')
plt.legend()
plt.grid(True)
plt.savefig("partg.png")
plt.show()

# Result for Part (g): "partg.png" shows the original vs approximate spectrum (using Nc=5 components) for the first galaxy.

# Part (h) - Plot c0 vs c1 and c0 vs c2
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(coefficients[:, 0], coefficients[:, 1], alpha=0.7)
plt.xlabel('c0')
plt.ylabel('c1')
plt.title('c0 vs c1')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(coefficients[:, 0], coefficients[:, 2], alpha=0.7)
plt.xlabel('c0')
plt.ylabel('c2')
plt.title('c0 vs c2')
plt.grid(True)
plt.tight_layout()
plt.savefig("parth.png")
plt.show()

# Result for Part (h): "parth.png" shows the scatter plot of c0 vs c1 and c0 vs c2.

# Part (i) - Compute and plot RMS residuals as a function of Nc
N_galaxies, N_wavelengths = residual_flux.shape
Nc_values = range(1, 21)
rms_residuals = []
for Nc in Nc_values:
    coefficients = np.dot(residual_flux, eigenvectors[:, :Nc])
    approx_flux = np.dot(coefficients, eigenvectors[:, :Nc].T) + mean_spectrum
    squared_residuals = (residual_flux + mean_spectrum - approx_flux) ** 2
    rms = np.sqrt(np.mean(squared_residuals))
    rms_residuals.append(rms)

plt.figure(figsize=(10, 6))
plt.plot(Nc_values, rms_residuals, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components (Nc)')
plt.ylabel('Root Mean Squared Residual')
plt.title('RMS Residuals vs Number of Principal Components (Nc)')
plt.grid(True)
plt.savefig("parti.png")
plt.show()

# Print the RMS residual for Nc = 20
print(f"Root Mean Squared Residual for Nc = 20: {rms_residuals[-1]:.6f}")

# Result for Part (i): 
# - RMS residual for Nc = 20: 0.000012
