# -*- coding: utf-8 -*-
"""
================================================================================
Synaptic Nonlinearity Approximation Module
================================================================================

This module implements multivariate polynomial approximations of synaptic
transmission nonlinearities, particularly NMDA receptor current-voltage
relationships and calcium-dependent release probability functions.

Biological Context:
-------------------
NMDA receptor currents exhibit strong voltage dependence due to Mg²⁺ block.
The current-voltage relationship is highly nonlinear and can be approximated
by multivariate polynomials in voltage V, Mg²⁺ concentration, and gating
variables.

Mathematical Model:
-------------------
The NMDA current is:

    I_NMDA(V, [Mg²⁺]) = g_NMDA · P_open · (V - E_NMDA) · M_g(V)

where the Mg²⁺ block is:

    M_g(V) = 1 / (1 + [Mg²⁺]/K_d · exp(-γ·V))

with typical parameters:
    K_d ≈ 3.57 mM (dissociation constant at 0 mV)
    γ ≈ 0.062 mV⁻¹ (voltage dependence)
    E_NMDA ≈ 0 mV (reversal potential)

Polynomial Approximation:
-------------------------
We expand M_g(V) as a multivariate polynomial:

    M_g(V, [Mg²⁺]) ≈ Σ_{α,β} c_{αβ} · V^α · [Mg²⁺]^β

For numerical stability, we use a graded lexicographic ordering of monomials
and evaluate via Horner-like schemes.

Calcium-Dependent Release:
--------------------------
The release probability as a function of Ca²⁺ concentration is often modeled
by a Hill function:

    P_rel([Ca²⁺]) = [Ca²⁺]^n / ([Ca²⁺]^n + K_D^n)

which can also be approximated by polynomials in a restricted range.

================================================================================
"""

import numpy as np
from typing import List, Tuple, Optional


def mono_next_grlex(m: int, x: np.ndarray) -> np.ndarray:
    """
    Return the next monomial in graded lexicographic order.

    For m-dimensional exponents, the grlex order first sorts by total degree,
    then lexicographically.

    Parameters
    ----------
    m : int
        Spatial dimension.
    x : np.ndarray
        Current exponent vector.

    Returns
    -------
    x_next : np.ndarray
        Next exponent vector, or None if x is the last.
    """
    x = np.asarray(x, dtype=int).copy()

    # Find the first index from the right that can be incremented
    for i in range(m - 1, -1, -1):
        if x[i] > 0:
            # Decrement x[i], increment x[i-1] (or wrap)
            if i > 0:
                x[i] -= 1
                x[i - 1] += 1
                # Move all trailing mass to the end
                t = np.sum(x[i:])
                x[i] = t
                x[i + 1:] = 0
                return x
            else:
                # Last monomial
                return None

    # All zeros: first monomial, return (1,0,...,0) or handle separately
    if np.all(x == 0):
        x_next = np.zeros(m, dtype=int)
        x_next[m - 1] = 1
        return x_next

    return None


def mono_total_next_grlex(m: int, n: int, x: np.ndarray) -> np.ndarray:
    """
    Return the next monomial with total degree exactly n in grlex order.

    Parameters
    ----------
    m : int
        Dimension.
    n : int
        Total degree.
    x : np.ndarray
        Current exponent vector.

    Returns
    -------
    x_next : np.ndarray
        Next exponent vector, or None.
    """
    x = np.asarray(x, dtype=int).copy()

    if np.sum(x) != n:
        raise ValueError("Current monomial does not have the specified total degree.")

    # Similar to mono_next_grlex but constrained to total degree n
    for i in range(m - 1, -1, -1):
        if x[i] > 0:
            if i > 0:
                x[i] -= 1
                x[i - 1] += 1
                # Redistribute remaining degree
                remaining = n - np.sum(x[:i])
                x[i] = remaining
                x[i + 1:] = 0
                if np.sum(x) == n:
                    return x
            else:
                return None

    return None


def mono_upto_enum(m: int, n: int) -> int:
    """
    Count the number of monomials in m variables with total degree <= n.

    Formula:
        C(m+n, n) = (m+n)! / (m! · n!)

    Parameters
    ----------
    m : int
        Dimension.
    n : int
        Maximum total degree.

    Returns
    -------
    count : int
        Number of monomials.
    """
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative.")

    count = 1
    for i in range(1, m + 1):
        count = count * (n + i) // i
    return count


def mono_value(m: int, n_points: int, f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate monomials x^f at multiple points.

    Parameters
    ----------
    m : int
        Spatial dimension.
    n_points : int
        Number of evaluation points.
    f : np.ndarray
        Exponent vector, shape (m,).
    x : np.ndarray
        Evaluation points, shape (m, n_points).

    Returns
    -------
    values : np.ndarray
        Monomial values, shape (n_points,).
    """
    f = np.asarray(f)
    x = np.asarray(x)

    if x.shape[0] != m:
        raise ValueError(f"x must have {m} rows, got {x.shape[0]}.")

    values = np.ones(n_points)
    for dim in range(m):
        if f[dim] > 0:
            values *= x[dim, :] ** f[dim]

    return values


def polynomial_value(
    m: int,
    coefficients: np.ndarray,
    exponents: List[np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a multivariate polynomial at points x.

    P(x) = Σ_j c_j · x^{e_j}

    Parameters
    ----------
    m : int
        Spatial dimension.
    coefficients : np.ndarray
        Coefficients c_j.
    exponents : list of np.ndarray
        Exponent vectors e_j.
    x : np.ndarray
        Evaluation points, shape (m, n_points) or (m,).

    Returns
    -------
    p : np.ndarray
        Polynomial values.
    """
    coefficients = np.asarray(coefficients)
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    n_points = x.shape[1]
    p = np.zeros(n_points)

    for j, c in enumerate(coefficients):
        if abs(c) < 1e-18:
            continue
        e = exponents[j]
        v = mono_value(m, n_points, e, x)
        p += c * v

    if squeeze:
        return p[0]
    return p


def build_nmda_block_polynomial(
    degree_v: int = 4,
    degree_mg: int = 2,
) -> Tuple[np.ndarray, List[np.ndarray], callable]:
    """
    Build a polynomial approximation of the NMDA Mg²⁺ block function.

    M_g(V, [Mg²⁺]) = 1 / (1 + [Mg²⁺]/K_d · exp(-γ·V))

    Parameters
    ----------
    degree_v : int
        Degree in voltage.
    degree_mg : int
        Degree in Mg²⁺ concentration.

    Returns
    -------
    coefficients : np.ndarray
        Polynomial coefficients.
    exponents : list
        Exponent vectors.
    exact_func : callable
        Exact function for comparison.
    """
    K_d = 3.57  # mM
    gamma = 0.062  # 1/mV
    V_range = (-80.0, 40.0)
    Mg_range = (0.1, 10.0)

    def exact_func(x):
        """x[0] = V [mV], x[1] = [Mg²⁺] [mM]"""
        V = x[0]
        Mg = x[1]
        return 1.0 / (1.0 + Mg / K_d * np.exp(-gamma * V))

    # Fit polynomial by least squares on a grid
    n_v = degree_v + 1
    n_mg = degree_mg + 1

    V_grid = np.linspace(V_range[0], V_range[1], 50)
    Mg_grid = np.linspace(Mg_range[0], Mg_range[1], 30)
    VV, MMg = np.meshgrid(V_grid, Mg_grid)

    n_samples = VV.size
    X = np.vstack([VV.ravel(), MMg.ravel()])
    Y = exact_func(X)

    # Build design matrix for polynomial terms
    exponents = []
    for i in range(degree_v + 1):
        for j in range(degree_mg + 1):
            exponents.append(np.array([i, j]))

    n_terms = len(exponents)
    A = np.zeros((n_samples, n_terms))
    for idx, e in enumerate(exponents):
        A[:, idx] = mono_value(2, n_samples, e, X)

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return coeffs, exponents, exact_func


def evaluate_nmda_current(
    V: np.ndarray,
    Mg: float,
    g_nmda: float = 1.0,
    E_nmda: float = 0.0,
    coeffs: Optional[np.ndarray] = None,
    exponents: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate NMDA current using polynomial approximation and exact formula.

    I_NMDA = g_NMDA · (V - E_NMDA) · M_g(V, [Mg²⁺])

    Parameters
    ----------
    V : np.ndarray
        Membrane potential [mV].
    Mg : float
        Mg²⁺ concentration [mM].
    g_nmda : float
        NMDA conductance.
    E_nmda : float
        NMDA reversal potential [mV].
    coeffs : np.ndarray, optional
        Polynomial coefficients.
    exponents : list, optional
        Polynomial exponents.

    Returns
    -------
    I_exact : np.ndarray
        Current from exact formula.
    I_poly : np.ndarray
        Current from polynomial approximation.
    """
    if coeffs is None or exponents is None:
        coeffs, exponents, exact_func = build_nmda_block_polynomial()
    else:
        K_d = 3.57
        gamma = 0.062

        def exact_func(x):
            return 1.0 / (1.0 + x[1] / K_d * np.exp(-gamma * x[0]))

    V = np.asarray(V)
    n_points = V.size

    X = np.vstack([V, np.full(n_points, Mg)])

    M_exact = exact_func(X)
    M_poly = polynomial_value(2, coeffs, exponents, X)

    # Clip polynomial approximation to physical range [0,1]
    M_poly = np.clip(M_poly, 0.0, 1.0)

    I_exact = g_nmda * (V - E_nmda) * M_exact
    I_poly = g_nmda * (V - E_nmda) * M_poly

    return I_exact, I_poly


def compute_polynomial_approximation_error(
    n_test: int = 100,
) -> dict:
    """
    Compute error metrics for the NMDA polynomial approximation.

    Parameters
    ----------
    n_test : int
        Number of test points.

    Returns
    -------
    metrics : dict
        Error metrics.
    """
    coeffs, exponents, exact_func = build_nmda_block_polynomial()

    rng = np.random.default_rng(42)
    V_test = rng.uniform(-80.0, 40.0, n_test)
    Mg_test = rng.uniform(0.1, 10.0, n_test)

    X_test = np.vstack([V_test, Mg_test])
    Y_exact = exact_func(X_test)
    Y_poly = polynomial_value(2, coeffs, exponents, X_test)
    Y_poly = np.clip(Y_poly, 0.0, 1.0)

    abs_err = np.abs(Y_poly - Y_exact)
    rel_err = abs_err / (np.abs(Y_exact) + 1e-15)

    return {
        "max_abs_error": np.max(abs_err),
        "mean_abs_error": np.mean(abs_err),
        "max_rel_error": np.max(rel_err),
        "mean_rel_error": np.mean(rel_err),
        "rmse": np.sqrt(np.mean((Y_poly - Y_exact) ** 2)),
    }


def hill_function_polynomial_approximation(
    n: float = 4.0,
    K_D: float = 1.0,
    degree: int = 8,
    ca_range: Tuple[float, float] = (0.0, 5.0),
) -> Tuple[np.ndarray, List[np.ndarray], callable]:
    """
    Approximate the Hill function for Ca²⁺-dependent release by a polynomial.

    P_rel([Ca²⁺]) = [Ca²⁺]^n / ([Ca²⁺]^n + K_D^n)

    Parameters
    ----------
    n : float
        Hill coefficient.
    K_D : float
        Dissociation constant.
    degree : int
        Polynomial degree.
    ca_range : tuple
        [Ca²⁺] range.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients.
    exponents : list
        Exponent vectors.
    exact_func : callable
        Exact Hill function.
    """
    def exact_func(x):
        ca = x[0]
        ca = np.maximum(ca, 0.0)
        return ca ** n / (ca ** n + K_D ** n)

    ca_grid = np.linspace(ca_range[0], ca_range[1], 100)
    X = ca_grid.reshape(1, -1)
    Y = exact_func(X)

    exponents = [np.array([i]) for i in range(degree + 1)]
    A = np.zeros((ca_grid.size, degree + 1))
    for idx, e in enumerate(exponents):
        A[:, idx] = mono_value(1, ca_grid.size, e, X)

    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return coeffs, exponents, exact_func


if __name__ == "__main__":
    metrics = compute_polynomial_approximation_error()
    print("NMDA polynomial approximation errors:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")

    V = np.linspace(-80.0, 40.0, 50)
    I_exact, I_poly = evaluate_nmda_current(V, Mg=1.0)
    print(f"\nMax current difference: {np.max(np.abs(I_exact - I_poly)):.6e}")
