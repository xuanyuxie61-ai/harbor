"""
Utility Module
==============
Common utilities, numerical stability checks, and helper functions.

Incorporates:
- Runge function tests (from 1051_runge) for numerical accuracy validation
- General numerical utilities
"""

import numpy as np


def runge_function(x):
    """
    Runge function: f(x) = 1 / (1 + 25*x^2).
    From 1051_runge.
    Classic example of polynomial interpolation failure on equally-spaced nodes.
    Used here to validate numerical accuracy of spectral methods.

    Parameters
    ----------
    x : float or ndarray
        Input.

    Returns
    -------
    float or ndarray
        Function value.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_derivative(x):
    """
    First derivative of Runge function.
    From 1051_runge.
    """
    x = np.asarray(x, dtype=float)
    return -50.0 * x / ((1.0 + 25.0 * x * x) ** 2)


def runge_second_derivative(x):
    """
    Second derivative of Runge function.
    From 1051_runge.
    """
    x = np.asarray(x, dtype=float)
    return (50.0 * (75.0 * x * x - 1.0)) / ((1.0 + 25.0 * x * x) ** 3)


def runge_antiderivative(x):
    """
    Antiderivative of Runge function.
    From 1051_runge.
    integral dx/(1+25x^2) = (1/5) * arctan(5x)
    """
    x = np.asarray(x, dtype=float)
    return 0.2 * np.arctan(5.0 * x)


def numerical_differentiation(f, x, h=None):
    """
    Numerical differentiation using central differences.

    Formula:
    f'(x) = (f(x+h) - f(x-h)) / (2h) - (h^2/6) * f'''(xi)

    Parameters
    ----------
    f : callable
        Function.
    x : float
        Point.
    h : float, optional
        Step size. Auto-selected if None.

    Returns
    -------
    float
        Derivative estimate.
    """
    if h is None:
        h = np.sqrt(np.finfo(float).eps) * max(abs(x), 1.0)
    h = max(h, 1e-15)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def numerical_integration(f, a, b, n=1000):
    """
    Numerical integration using Simpson's rule.

    Formula:
    integral_a^b f(x) dx ~ (h/3) * [f0 + 4f1 + 2f2 + 4f3 + ... + fn]

    Parameters
    ----------
    f : callable
        Function.
    a, b : float
        Integration limits.
    n : int
        Number of subintervals (must be even).

    Returns
    -------
    float
        Integral estimate.
    """
    n = max(n - n % 2, 2)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])

    result = y[0] + y[-1]
    result += 4.0 * np.sum(y[1:-1:2])
    result += 2.0 * np.sum(y[2:-1:2])
    return result * h / 3.0


def golden_section_search(f, a, b, tol=1e-6, max_iter=100):
    """
    Golden section search for unimodal function minimization.

    Parameters
    ----------
    f : callable
        Function to minimize.
    a, b : float
        Interval.
    tol : float
        Tolerance.
    max_iter : int
        Max iterations.

    Returns
    -------
    float
        Minimum point.
    float
        Minimum value.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            return (a + b) / 2.0, f((a + b) / 2.0)

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = f(d)

    return (a + b) / 2.0, f((a + b) / 2.0)


def check_numerical_stability(operation_name, values, max_value=1e15, min_value=-1e15):
    """
    Check for numerical stability issues.

    Parameters
    ----------
    operation_name : str
        Name of operation.
    values : ndarray
        Values to check.
    max_value, min_value : float
        Bounds.

    Returns
    -------
    bool
        True if stable.
    """
    values = np.asarray(values)
    if np.any(np.isnan(values)):
        print(f"WARNING: NaN detected in {operation_name}")
        return False
    if np.any(np.isinf(values)):
        print(f"WARNING: Inf detected in {operation_name}")
        return False
    if np.any(values > max_value) or np.any(values < min_value):
        print(f"WARNING: Value out of bounds in {operation_name}")
        return False
    return True


def linear_interpolation(x, xp, fp):
    """
    1D linear interpolation with boundary extrapolation.

    Parameters
    ----------
    x : float or ndarray
        Points to evaluate.
    xp : ndarray
        Data points.
    fp : ndarray
        Data values.

    Returns
    -------
    float or ndarray
        Interpolated values.
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

    if np.isscalar(x):
        if x <= xp[0]:
            return float(fp[0])
        if x >= xp[-1]:
            return float(fp[-1])
        idx = np.searchsorted(xp, x)
        t = (x - xp[idx - 1]) / (xp[idx] - xp[idx - 1])
        return fp[idx - 1] + t * (fp[idx] - fp[idx - 1])

    result = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= xp[0]:
            result[i] = fp[0]
        elif xi >= xp[-1]:
            result[i] = fp[-1]
        else:
            idx = np.searchsorted(xp, xi)
            t = (xi - xp[idx - 1]) / (xp[idx] - xp[idx - 1])
            result[i] = fp[idx - 1] + t * (fp[idx] - fp[idx - 1])
    return result


def compute_rmse(predicted, actual):
    """
    Root Mean Square Error.

    Formula:
    RMSE = sqrt(mean((predicted - actual)^2))
    """
    return np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2))


def compute_correlation(x, y):
    """
    Pearson correlation coefficient.

    Formula:
    r = cov(x, y) / (sigma_x * sigma_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if denominator < 1e-15:
        return 0.0
    return numerator / denominator
