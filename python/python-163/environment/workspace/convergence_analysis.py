"""
convergence_analysis.py
=======================
Numerical error analysis, convergence checking, and iterative solver
monitoring for the THM geothermal simulation.

Incorporates algorithms from:
  - 338_errors: numerical error estimation and root-finding convergence

Mathematical formulation:
For a numerical solution u_h approximating exact solution u:

  L^2 \text{ error: } \|u - u_h\|_{L^2} = \left(\int_\Omega |u - u_h|^2 d\Omega\right)^{1/2}

  L^\infty \text{ error: } \|u - u_h\|_\infty = \max_\Omega |u - u_h|

Convergence rate p for grid refinement:
  \|u - u_h\| \approx C h^p

  p = \frac{\log(\|e_{h_1}\| / \|e_{h_2}\|)}{\log(h_1 / h_2)}

For iterative solver convergence:
  \|r^{(k)}\|_2 = \|b - A x^{(k)}\|_2

  \text{Relative residual: } \frac{\|r^{(k)}\|_2}{\|b\|_2} < \text{tol}

For the Newton-Raphson method:
  x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}

  \text{Convergence criterion: } |x_{n+1} - x_n| < \text{tol}
"""

import numpy as np


class ConvergenceMonitor:
    """Monitor convergence of iterative solvers."""

    def __init__(self, tol=1.0e-8, max_iter=1000):
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.residuals = []
        self.iterates = []

    def reset(self):
        self.residuals = []
        self.iterates = []

    def check(self, residual, iterate=None):
        """
        Check if convergence criteria are met.

        Parameters
        ----------
        residual : float
            Current residual norm.
        iterate : np.ndarray, optional
            Current iterate.

        Returns
        -------
        converged : bool
        reason : str
        """
        self.residuals.append(float(residual))
        if iterate is not None:
            self.iterates.append(iterate.copy())

        if residual < self.tol:
            return True, "Converged: residual below tolerance."
        if len(self.residuals) >= self.max_iter:
            return True, "Stopped: maximum iterations reached."
        if len(self.residuals) > 1:
            if self.residuals[-1] > self.residuals[-2] * 10:
                return True, "Divergence detected."
        return False, "Continuing."

    def convergence_rate(self):
        """
        Estimate convergence rate from residual history.

        Returns
        -------
        rate : float
            Estimated convergence rate (residual ratio).
        """
        if len(self.residuals) < 2:
            return np.nan
        return self.residuals[-1] / self.residuals[-2]


def l2_error(u_exact, u_numeric):
    """
    Compute discrete L^2 error.
    """
    u_exact = np.asarray(u_exact, dtype=np.float64)
    u_numeric = np.asarray(u_numeric, dtype=np.float64)
    diff = u_exact - u_numeric
    return np.sqrt(np.mean(diff ** 2))


def linf_error(u_exact, u_numeric):
    """
    Compute discrete L^\infty error.
    """
    u_exact = np.asarray(u_exact, dtype=np.float64)
    u_numeric = np.asarray(u_numeric, dtype=np.float64)
    return np.max(np.abs(u_exact - u_numeric))


def convergence_rate(errors, h_values):
    """
    Compute observed convergence rate from error-grid size data.

    Parameters
    ----------
    errors : np.ndarray
        Error values.
    h_values : np.ndarray
        Grid sizes.

    Returns
    -------
    rates : np.ndarray
        Convergence rates between successive points.
    """
    errors = np.asarray(errors, dtype=np.float64)
    h_values = np.asarray(h_values, dtype=np.float64)
    if len(errors) != len(h_values):
        raise ValueError("errors and h_values must have the same length.")
    if len(errors) < 2:
        return np.array([])

    rates = np.zeros(len(errors) - 1)
    for i in range(len(rates)):
        if errors[i] <= 0 or errors[i + 1] <= 0 or h_values[i] <= 0 or h_values[i + 1] <= 0:
            rates[i] = np.nan
        else:
            rates[i] = np.log(errors[i + 1] / errors[i]) / np.log(h_values[i + 1] / h_values[i])
    return rates


def newton_raphson(f, df, x0, tol=1.0e-12, max_iter=50):
    """
    Solve f(x) = 0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : callable
        Function.
    df : callable
        Derivative.
    x0 : float
        Initial guess.
    tol : float
        Tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    root : float
    converged : bool
    iterations : int
    """
    x = float(x0)
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1.0e-30:
            return x, False, i
        dx = fx / dfx
        x_new = x - dx
        if abs(dx) < tol * max(abs(x_new), 1.0):
            return x_new, True, i + 1
        x = x_new
    return x, False, max_iter


def bisection(f, a, b, tol=1.0e-12, max_iter=100):
    """
    Solve f(x) = 0 on [a, b] using bisection.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) < tol or (b - a) < tol:
            return c, True, i + 1
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0, False, max_iter


def richardson_extrapolation(values, h_values, p_expected):
    """
    Richardson extrapolation for error estimation.

    Given approximations A(h) with A(h) = A_0 + C h^p + O(h^{p+1}):

    A_0 \approx \frac{h_2^p A(h_1) - h_1^p A(h_2)}{h_2^p - h_1^p}

    Parameters
    ----------
    values : np.ndarray
        Approximate values A(h).
    h_values : np.ndarray
        Grid sizes.
    p_expected : float
        Expected order.

    Returns
    -------
    extrapolated : float
        Extrapolated value A_0.
    error_estimate : float
        Estimated error.
    """
    if len(values) < 2 or len(h_values) < 2:
        raise ValueError("Need at least two values for extrapolation.")
    h1, h2 = h_values[0], h_values[1]
    A1, A2 = values[0], values[1]
    hp1 = h1 ** p_expected
    hp2 = h2 ** p_expected
    extrapolated = (hp2 * A1 - hp1 * A2) / (hp2 - hp1)
    error_estimate = abs(A1 - A2) / (abs(hp2 / hp1 - 1.0))
    return extrapolated, error_estimate


def matrix_exponential_error(A, dt, method="taylor"):
    """
    Estimate error in matrix exponential computation.

    For Taylor series: e^{A dt} = I + A dt + (A dt)^2/2! + ...
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")

    I = np.eye(n)
    if method == "taylor":
        # 4th order Taylor
        Adt = A * dt
        exp_approx = I + Adt + (Adt @ Adt) / 2.0 + (Adt @ Adt @ Adt) / 6.0
        # Error bound for next term
        next_term_norm = np.linalg.norm((Adt @ Adt @ Adt @ Adt) / 24.0, ord=2)
        return exp_approx, next_term_norm
    elif method == "pade":
        # Scaled Taylor (simplified)
        exp_approx = I + A * dt
        error = np.linalg.norm((A * dt) @ (A * dt), ord=2) / 2.0
        return exp_approx, error
    else:
        raise ValueError("Unknown method.")
