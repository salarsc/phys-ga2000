import math
from scipy.optimize import brent as scipy_brent

def brents_method_minimize(f, lower_bound, upper_bound, tolerance=1e-5, max_iterations=100):
    """
    Finds the minimum of a unimodal function f within the interval [lower_bound, upper_bound]
    using Brent's Method.

    Parameters:
    - f (callable): The function to minimize. It should be unimodal within the interval.
    - lower_bound (float): The start of the interval to search within.
    - upper_bound (float): The end of the interval to search within.
    - tolerance (float, optional): How precise the result should be. Smaller means more precise.
    - max_iterations (int, optional): The maximum number of steps the algorithm will take.

    Returns:
    - float: The x-value where the function f reaches its minimum within the interval.
    """
    # Golden ratio factor to decide the next step
    golden_ratio_factor = (3 - math.sqrt(5)) / 2

    # Initialize three points: current best (current_x), and two others (previous_w and previous_v)
    current_x = previous_w = previous_v = lower_bound + golden_ratio_factor * (upper_bound - lower_bound)
    current_fx = previous_w_fx = previous_v_fx = f(current_x)

    # Initialize step sizes
    step = previous_step = upper_bound - lower_bound

    for iteration in range(max_iterations):
        # Find the midpoint of the current interval
        midpoint = 0.5 * (lower_bound + upper_bound)

        # Calculate the acceptable tolerance based on current_x to avoid division by zero
        absolute_tolerance = tolerance * abs(current_x) + 1e-10
        # Double the tolerance for relative comparisons
        relative_tolerance = 2 * absolute_tolerance

        # Check if we've narrowed down enough to stop
        if abs(current_x - midpoint) <= relative_tolerance - 0.5 * (upper_bound - lower_bound):
            print(f"Converged after {iteration} iterations.")
            break  # We're close enough to the minimum

        # Variables to hold the parameters for a possible parabolic step
        parabolic_p = parabolic_q = parabolic_r = 0.0

        # Decide whether to attempt a parabolic fit based on the previous step size
        if abs(previous_step) > absolute_tolerance:
            # Attempt to fit a parabola through the last three points to guess where the minimum is
            parabolic_r = (current_x - previous_w) * (current_fx - previous_v_fx)
            parabolic_q = (current_x - previous_v) * (current_fx - previous_w_fx)
            parabolic_p = (current_x - previous_v) * parabolic_q - (current_x - previous_w) * parabolic_r
            parabolic_q = 2.0 * (parabolic_q - parabolic_r)

            if parabolic_q > 0:
                parabolic_p = -parabolic_p  # Make sure the step is towards the minimum

            parabolic_q = abs(parabolic_q)

            # Check if the parabolic step is a good move
            if (abs(parabolic_p) < abs(0.5 * parabolic_q * previous_step)) and \
               (parabolic_p > parabolic_q * (lower_bound - current_x)) and \
               (parabolic_p < parabolic_q * (upper_bound - current_x)):
                # Calculate the parabolic step
                parabolic_step = parabolic_p / parabolic_q
                trial_u = current_x + parabolic_step

                # Make sure the new point isn't too close to the boundaries
                if (trial_u - lower_bound) < relative_tolerance or (upper_bound - trial_u) < relative_tolerance:
                    parabolic_step = absolute_tolerance if midpoint > current_x else -absolute_tolerance
            else:
                # If parabolic step isn't suitable, use the golden section step
                previous_step = step
                parabolic_step = golden_ratio_factor * (upper_bound - current_x) if current_x < midpoint else golden_ratio_factor * (lower_bound - current_x)
        else:
            # If the previous step was too small, stick with the golden section step
            previous_step = step
            parabolic_step = golden_ratio_factor * (upper_bound - current_x) if current_x < midpoint else golden_ratio_factor * (lower_bound - current_x)

        # Decide where to try next
        if abs(parabolic_step) >= absolute_tolerance:
            trial_u = current_x + parabolic_step
        else:
            # If the parabolic step is too tiny, take a step of minimum size in the right direction
            trial_u = current_x + (absolute_tolerance if parabolic_step > 0 else -absolute_tolerance)

        # Evaluate the function at the new trial point
        trial_fu = f(trial_u)

        # Decide whether to update our current best point
        if trial_fu <= current_fx:
            if trial_u < current_x:
                upper_bound = current_x  # Narrow the upper bound
            else:
                lower_bound = current_x  # Narrow the lower bound

            # Shift the previous points
            previous_v, previous_v_fx = previous_w, previous_w_fx
            previous_w, previous_w_fx = current_x, current_fx
            current_x, current_fx = trial_u, trial_fu  # Update the current best
        else:
            # If the new point isn't better, just narrow the bounds
            if trial_u < current_x:
                lower_bound = trial_u
            else:
                upper_bound = trial_u

            # Update the auxiliary points based on where the trial point falls
            if trial_fu <= previous_w_fx or previous_w == current_x:
                previous_v, previous_v_fx = previous_w, previous_w_fx
                previous_w, previous_w_fx = trial_u, trial_fu
            elif trial_fu <= previous_v_fx or previous_v == current_x or previous_v == previous_w:
                previous_v, previous_v_fx = trial_u, trial_fu

    else:
        print("Reached maximum iterations without full convergence.")

    return current_x

# Define the function we want to minimize
def test_function(x):
    """
    The function to minimize: y = (x - 0.3)^2 * e^x

    Parameters:
    - x (float): The input value.

    Returns:
    - float: The computed y value.
    """
    return (x - 0.3)**2 * math.exp(x)

# Set the interval where we search for the minimum
lower_bound = 0
upper_bound = 1

# Find the minimum using our custom Brent's method
min_x_custom = brents_method_minimize(test_function, lower_bound, upper_bound)
print(f"Brent's Method Implementation:\nMinimum x = {min_x_custom}")

# Find the minimum using SciPy's built-in Brent's method for comparison
min_x_scipy = scipy_brent(test_function, brack=(lower_bound, upper_bound), tol=1e-5)
print(f"\nSciPy's brent method:\nMinimum x = {min_x_scipy}")

# Calculate and display how different the two results are
difference_x = abs(min_x_custom - min_x_scipy)
print(f"\nDifference in x:\nΔx = {difference_x}")
