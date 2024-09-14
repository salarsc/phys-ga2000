
import numpy as np
# Part c: Combining both methods for better accuracy in both ranges
def quadratic(coef_a, coef_b, coef_c):
    
    
#part a has a limit accuracy for large minus sigh but good accuaracy for the small negative one
#the part b is reverse
#thus we in part c combine these two, to get accurate results for both ranges.
    solution_plus = ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) ) / (2*coef_a)
    solution_minus = (2*coef_c) / ( -coef_b + np.sqrt(coef_b**2 - 4*coef_a*coef_c) )
    return solution_minus, solution_plus

print("part c solouions:",quadratic(0.001, 1000, 0.001))