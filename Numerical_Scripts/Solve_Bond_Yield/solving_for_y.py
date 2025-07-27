import numpy as np
import solving_for_y_funcs as y_funcs

# Params
P = 100  # Principle
C = 6  # Coupon payment
M = 96  # Market price
T = 2  # Number of time periods in years
N = 1000  # Number of iterations
a = 0.01  # Min guess point
b = 0.2  # Max guess point
tol = 1e-10  # Error tolerance

y_solution = y_funcs.yield_to_maturity_bisection_MarloVersion(P, M, C, T, a, b, N, tol)
y_solution_gpt = y_funcs.solve_yield_bisection(P, C, T, M)
print(f"Text book solution: y* = {y_solution*100:.2f}")
print(f"GPT solution: y* = {y_solution_gpt*100:.2f}")
print(f"Error = {np.abs(y_solution - y_solution_gpt)}")
