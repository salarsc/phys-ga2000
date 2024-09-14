import math
import numpy as np

# Part a
def standard_quadratic_equation_solver(coef_a, coef_b, coef_c):
    solution_plus = ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) ) / (2*coef_a)
    solution_minus = ( -coef_b - np.sqrt(coef_b**2 - 4*coef_a*coef_c) ) / (2*coef_a)
    return solution_minus, solution_plus

# Solution for 0.001x^2 + 1000x + 0.001 = 0
print("part a solouions:", standard_quadratic_equation_solver(0.001, 1000, 0.001))

# Part b
def modified_quadratic_equation_solver(coef_a, coef_b, coef_c):
    # This version multiplies ( -b +- np.sqrt(b**2 - 4*a*c) ) to top and bottom
    solution_plus = (2*coef_c) / ( -coef_b - np.sqrt(coef_b**2 - 4*coef_a*coef_c) )
    solution_minus = (2*coef_c) / ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) )
    return solution_minus, solution_plus

print("part c solouions:",modified_quadratic_equation_solver(0.001, 1000, 0.001))

# Part c: Combining both methods for better accuracy in both ranges
def quadratic(coef_a, coef_b, coef_c):
    
    
#part a has a limit accuracy for large minus sigh but good accuaracy for the small negative one
#the part b is reverse
#thus we in part c combine these two, to get accurate results for both ranges.
    solution_plus = ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) ) / (2*coef_a)
    solution_minus = (2*coef_c) / ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) )
    return solution_minus, solution_plus

print("part c solouions:",quadratic(0.001, 1000, 0.001))
