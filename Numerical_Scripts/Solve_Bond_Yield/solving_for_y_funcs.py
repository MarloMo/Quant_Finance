import numpy as np


def discreet_compounding(P, C, T, y):
    f_y = P * (1 + y) ** (-T) + np.sum([C * (1 + y) ** (-i) for i in range(1, T + 1)])

    return f_y


def yield_to_maturity_bisection_MarloVersion(P, M, C, T, a, b, N, tol):
    for i in range(N):
        m = (a + b) * 0.5
        f_a = discreet_compounding(P, C, T, a)
        f_b = discreet_compounding(P, C, T, b)
        f_m = discreet_compounding(P, C, T, m)

        if f_a > M and f_m < M:
            b = m
            f_b = discreet_compounding(P, C, T, b)
            if np.abs(M - f_b) < tol:
                return b
        elif f_m > M and f_b < M:
            a = m
            f_a = discreet_compounding(P, C, T, a)
            if np.abs(M - f_a) < tol:
                return a


def solve_yield_bisection(P, C, T, M, a=0.01, b=0.2, tol=1e-10, max_iter=1000):
    """
    Find yield y such that y_funcs.discreet_compounding(P, C, T, y) ≈ M using bisection.

    Parameters
    ----------
    P : float
        Principal amount.
    C : float
        Coupon payment.
    T : int
        Number of periods.
    M : float
        Target price.
    a, b : float
        Initial lower and upper guesses for yield.
    tol : float
        Absolute tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    y : float
        Approximated yield.
    """

    # Initial evaluations
    f_a = discreet_compounding(P, C, T, a)
    f_b = discreet_compounding(P, C, T, b)

    # Ensure that M is between f_a and f_b
    if (f_a - M) * (f_b - M) > 0:
        raise ValueError("Initial interval does not bracket the solution.")

    for i in range(max_iter):
        m = 0.5 * (a + b)
        f_m = discreet_compounding(P, C, T, m)

        # Check convergence
        if abs(f_m - M) < tol:
            return m

        # Narrow the bracket
        if (f_a - M) * (f_m - M) < 0:
            b = m
            f_b = f_m
        else:
            a = m
            f_a = f_m

    raise RuntimeError("Bisection method did not converge within max_iter iterations.")
