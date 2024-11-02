import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# we get the survey data

survey_df = pd.read_csv("survey.csv")
ages_array = survey_df['age'].values
answers_array = survey_df['recognized_it'].values

# the logistic function to give probablity
def logistic_curve(age, intercept, coefficient):
    return 1 / (1 + np.exp(-(intercept + coefficient * age)))

# this is to  negative log-likelihood function
def neg_log_likelihood(params, ages_array, answers_array):
    intercept, coefficient = params
    predicted_probs = logistic_curve(ages_array, intercept, coefficient)
    epsilon = 1e-9  # Small constant to prevent log(0)
    log_likelihood_values = (
        answers_array * np.log(predicted_probs + epsilon) +
        (1 - answers_array) * np.log(1 - predicted_probs + epsilon)
    )
    return -np.sum(log_likelihood_values)

# Initialialization and optimization
initial_params = np.array([0.0, 0.0])
opt_result = minimize(neg_log_likelihood, initial_params, args=(ages_array, answers_array), method='BFGS')
opt_intercept, opt_coefficient = opt_result.x

# Calculate covariance matrix and standard errors
inv_hessian = opt_result.hess_inv
covariance_matrix = inv_hessian
standard_errors = np.sqrt(np.diag(covariance_matrix))

# Generate points for plotting the logistic curve
ages_for_plot = np.linspace(ages_array.min(), ages_array.max(), 100)
predicted_probabilities = logistic_curve(ages_for_plot, opt_intercept, opt_coefficient)

# Plot 
plt.figure(figsize=(10, 6))
plt.scatter(ages_array, answers_array, label="Survey Responses", color='blue', marker='o')
plt.plot(ages_for_plot, predicted_probabilities, label="Fitted Logistic Curve", color='red', linestyle='-')
plt.xlabel("Age (years)")
plt.ylabel("Probability of Recognizing")
plt.legend()
plt.title("Logistic Regression for Data")
plt.savefig("q1")
plt.show()

# print the optimized parameters and their statistics
print("Optimized Parameters:")
print(f"Intercept: {opt_intercept}, Coefficient: {opt_coefficient}")
print("Std Errors:", standard_errors)
print("Covariance Matrix:\n", covariance_matrix)
