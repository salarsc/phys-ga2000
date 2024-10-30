import jax.numpy as jnp
from jax import grad

# Constants in SI units
G = 6.674e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Masses and distances for each system
mass_earth = 5.974e24  # Mass of Earth (kg)
mass_moon = 7.348e22  # Mass of Moon (kg)
distance_earth_to_moon = 3.844e8  # Average distance from Earth to Moon (m)

mass_sun = 1.989e30  # Mass of Sun (kg)
distance_earth_to_sun = 1.496e11  # Average distance from Earth to Sun (m)

mass_jupiter = 1.898e27  # Mass of Jupiter (kg)
distance_sun_to_jupiter = 7.785e11  # Average distance from Sun to Jupiter (m)

# Function defining the rescaled L1 equation for finding L1 point in terms of r' and mass_ratio
def rescaled_L1_equation(r_prime, mass_ratio):
    """
    Defines the rescaled equation for finding the L1 point.
    
    Parameters:
    - r_prime: The distance ratio (r / R) where r is the distance to the L1 point, and R is the distance between the two large bodies.
    - mass_ratio: The ratio of the smaller mass to the larger mass (m / M).
    
    Returns:
    - The result of the equation when evaluated at the given r_prime and mass_ratio.
    """
    # Balances gravitational and centripetal forces at the L1 point
    return 1 / r_prime**2 - mass_ratio / (1 - r_prime)**2 - (1 + mass_ratio) * r_prime

# Compute the derivative of the L1 equation with respect to r_prime for Newton's method
L1_derivative = grad(rescaled_L1_equation, argnums=0)

# Newton's method function to solve for r_prime (distance ratio) that satisfies the L1 equation
def solve_L1_distance_ratio(mass_ratio, initial_guess=0.5, tolerance=1e-4, max_iterations=100):
    """
    Uses Newton's method to find the distance ratio r_prime for the L1 point.
    
    Parameters:
    - mass_ratio: The mass ratio (m / M) between the two large bodies.
    - initial_guess: An initial guess for r_prime to start the iterative solution.
    - tolerance: The acceptable tolerance for convergence of r_prime.
    - max_iterations: The maximum number of iterations to attempt.
    
    Returns:
    - The converged value of r_prime if successful.
    """
    r_prime = initial_guess
    for i in range(max_iterations):
        # Evaluate the function and its derivative at the current r_prime
        function_value = rescaled_L1_equation(r_prime, mass_ratio)
        derivative_value = L1_derivative(r_prime, mass_ratio)
        
        # Update r_prime using Newton's method formula
        r_prime_new = r_prime - function_value / derivative_value
        
        # Check if the change is within the tolerance, indicating convergence
        if jnp.abs(r_prime_new - r_prime) < tolerance:
            return r_prime_new
        
        r_prime = r_prime_new
    
    # Raise an error if convergence wasn't reached within max_iterations
    raise ValueError("Newton's method did not converge")

# Define cases with the mass ratio and distances for each system
systems = {
    "Earth-Moon": (mass_moon / mass_earth, distance_earth_to_moon),
    "Earth-Sun": (mass_earth / mass_sun, distance_earth_to_sun),
    "Sun-Jupiter": (mass_jupiter / mass_sun, distance_sun_to_jupiter)
}

# Dictionary to store the results
L1_distances = {}

# Calculate the L1 distance for each system
for system_name, (mass_ratio, distance_between_bodies) in systems.items():
    # Find r_prime, the ratio of the L1 distance to the total distance between bodies
    r_prime_solution = solve_L1_distance_ratio(mass_ratio)
    
    # Convert r_prime to the actual L1 distance by multiplying by the distance between the two bodies
    L1_distance = r_prime_solution * distance_between_bodies
    L1_distances[system_name] = L1_distance

# Display the results for each system
for system_name, L1_distance in L1_distances.items():
    print(f"The L1 distance for the {system_name} system is approximately {L1_distance:.4f} meters.")
