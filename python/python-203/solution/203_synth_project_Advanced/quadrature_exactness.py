"""
quadrature_exactness.py -- Multidimensional Quadrature Exactness Testing
==========================================================================
Tests the polynomial exactness of quadrature rules by comparing numerical
integration of monomials against analytical values. Supports mixed rules
(Gauss-Legendre, Gauss-Gegenbauer, Gauss-Hermite, Gauss-Laguerre).

Algorithm:
1. Enumerate all multi-indices of total degree d using composition generation
2. For each monomial x^alpha, compute both:
   - Exact integral (analytically via Gamma/Beta functions)
   - Quadrature approximation (weighted sum at quadrature points)
3. Report the relative error; a rule of order n should be exact for degree <= 2n-1

Seed references:
  - 804_nint_exactness_mixed: composition enumeration, exact monomial integrals,
    mixed quadrature rules, hypergeometric function
  - 462_gegenbauer_polynomial: hypergeometric evaluation, Gauss-Gegenbauer nodes
  - 235_cube_monte_carlo: cube01_monomial_integral for reference
"""
import numpy as np
from scipy.special import gamma as gamma_func, factorial
from typing import Tuple
import polynomial_chaos as pc


# ---------------------------------------------------------------------------
# Integer composition generator
# ---------------------------------------------------------------------------
def comp_next(n: int, k: int, comp: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Generate the next composition of integer n into k parts.

    A composition of n into k parts is an ordered tuple (c_1, ..., c_k)
    of non-negative integers summing to n.

    Uses the "stars and bars" algorithm in lexicographic order.

    Parameters
    ----------
    n : int
        Integer to compose.
    k : int
        Number of parts.
    comp : ndarray, shape (k,)
        Current composition (modified in place).

    Returns
    -------
    finished : int
        1 if all compositions have been generated, 0 otherwise.
    comp : ndarray
        Next composition.
    """
    comp = comp.copy()

    if k < 1:
        return 1, comp

    # Find the rightmost nonzero entry (excluding the last)
    i = k - 2
    while i >= 0 and comp[i] == 0:
        i -= 1

    if i < 0:
        # All weight is in the last entry; done
        return 1, comp

    # Decrease comp[i] by 1 and move all weight from comp[i+1:] to comp[k-1]
    comp[i] -= 1
    moved = comp[i + 1]
    comp[i + 1] = 0
    if i + 1 < k - 1:
        # Zero out everything between i+1 and k-2
        for j in range(i + 2, k - 1):
            moved += comp[j]
            comp[j] = 0
    comp[k - 1] += moved + 1

    # Check if we've returned to the initial composition
    if comp[k - 1] == n:
        return 1, comp

    return 0, comp


def generate_compositions(n: int, k: int) -> list:
    """Generate all compositions of n into k non-negative parts."""
    if k <= 0:
        return []
    if k == 1:
        return [np.array([n])]
    compositions = []
    comp = np.zeros(k, dtype=int)
    comp[0] = n
    compositions.append(comp.copy())

    while True:
        finished, comp = comp_next(n, k, comp)
        if finished:
            break
        compositions.append(comp.copy())

    return compositions


# ---------------------------------------------------------------------------
# Exact monomial integrals for different weight functions
# ---------------------------------------------------------------------------
def monomial_integral_legendre(exponent: int) -> float:
    """
    Exact integral of x^n over [-1, 1] with unit weight:
        integral_{-1}^{1} x^n dx = (1 + (-1)^n) / (n + 1)
    """
    if exponent % 2 == 1:
        return 0.0
    return 2.0 / (exponent + 1)


def monomial_integral_gegenbauer(exponent: int, alpha: float) -> float:
    """
    Exact integral of x^n with Gegenbauer weight (1-x^2)^{alpha-1/2}:
        integral_{-1}^{1} x^n (1-x^2)^{alpha-1/2} dx
        = 0                           if n is odd
        = B((n+1)/2, alpha+1/2)      if n is even
    where B is the Beta function.
    """
    if exponent % 2 == 1:
        return 0.0
    m = exponent // 2
    return gamma_func(m + 0.5) * gamma_func(alpha + 0.5) / gamma_func(m + alpha + 1.0)


def monomial_integral_hermite(exponent: int) -> float:
    """
    Exact integral of x^n with Hermite weight exp(-x^2):
        integral_{-inf}^{inf} x^n exp(-x^2) dx
        = 0                     if n is odd
        = (n-1)!! * sqrt(pi) / 2^{n/2}   if n is even (double factorial)
    Actually: = Gamma((n+1)/2) for even n.
    """
    if exponent % 2 == 1:
        return 0.0
    return gamma_func((exponent + 1) / 2.0)


def monomial_integral_laguerre(exponent: int) -> float:
    """
    Exact integral of x^n with Laguerre weight exp(-x) on [0, inf):
        integral_0^inf x^n exp(-x) dx = n! = Gamma(n+1)
    """
    return gamma_func(exponent + 1.0)


# ---------------------------------------------------------------------------
# Quadrature exactness test
# ---------------------------------------------------------------------------
def test_gegenbauer_exactness(max_degree: int = 15, alpha: float = 0.5,
                              n_quad_override: int = None) -> dict:
    """
    Test the exactness of Gauss-Gegenbauer quadrature for polynomials
    of increasing degree.

    A Gauss-Gegenbauer rule with n points should integrate polynomials
    of degree <= 2n-1 exactly (up to floating-point precision).

    Parameters
    ----------
    max_degree : int
        Maximum polynomial degree to test.
    alpha : float
        Gegenbauer parameter.
    n_quad_override : int or None
        Override the number of quadrature points. If None, uses
        n = ceil((max_degree + 1) / 2) + 1 points.

    Returns
    -------
    results : dict
        'degrees': tested degrees
        'exact_values': analytical integral values
        'quad_values': quadrature approximation values
        'errors': relative errors
        'n_quad': number of quadrature points used
        'theoretical_exactness': 2*n_quad - 1
    """
    if n_quad_override is not None:
        n_quad = n_quad_override
    else:
        n_quad = (max_degree + 2) // 2 + 1

    nodes, weights = pc.gauss_gegenbauer(n_quad, alpha)
    w = (1.0 - nodes ** 2) ** (alpha - 0.5)

    degrees = []
    exact_values = []
    quad_values = []
    errors = []

    for deg in range(max_degree + 1):
        # Exact integral
        exact = monomial_integral_gegenbauer(deg, alpha)

        # Quadrature approximation
        quad = np.sum(weights * w * nodes ** deg)

        # Relative error
        if abs(exact) > 1e-14:
            rel_err = abs(quad - exact) / abs(exact)
        else:
            rel_err = abs(quad - exact)

        degrees.append(deg)
        exact_values.append(exact)
        quad_values.append(quad)
        errors.append(rel_err)

    return {
        'degrees': np.array(degrees),
        'exact_values': np.array(exact_values),
        'quad_values': np.array(quad_values),
        'errors': np.array(errors),
        'n_quad': n_quad,
        'theoretical_exactness': 2 * n_quad - 1
    }


def test_multidim_exactness(dim: int = 2, max_total_degree: int = 6,
                            alpha: float = 0.5) -> dict:
    """
    Test exactness of tensor-product Gauss-Gegenbauer quadrature in
    multiple dimensions.

    For a tensor product of 1D rules each exact to degree 2n-1,
    the product rule is exact for all polynomials of total degree <= 2n-1.

    Parameters
    ----------
    dim : int
        Spatial dimension.
    max_total_degree : int
        Maximum total polynomial degree to test.
    alpha : float
        Gegenbauer parameter.

    Returns
    -------
    results : dict with 'degrees', 'errors', 'max_error', etc.
    """
    n_1d = (max_total_degree + 2) // 2 + 1
    nodes_1d, weights_1d = pc.gauss_gegenbauer(n_1d, alpha)
    w_1d = (1.0 - nodes_1d ** 2) ** (alpha - 0.5)

    # Tensor product quadrature
    if dim == 1:
        nodes = nodes_1d.reshape(-1, 1)
        weights = weights_1d * w_1d
    elif dim == 2:
        n_total = n_1d ** 2
        nodes = np.zeros((n_total, 2))
        weights = np.zeros(n_total)
        idx = 0
        for i in range(n_1d):
            for j in range(n_1d):
                nodes[idx] = [nodes_1d[i], nodes_1d[j]]
                weights[idx] = weights_1d[i] * w_1d[i] * weights_1d[j] * w_1d[j]
                idx += 1
    else:
        # For higher dimensions, use a smaller rule
        n_1d = max(3, (max_total_degree + 2) // 2)
        n_total = n_1d ** dim
        # Generate multi-index for tensor product
        from itertools import product as iterproduct
        indices = list(iterproduct(range(n_1d), repeat=dim))
        nodes = np.array([[nodes_1d[idx[d]] for d in range(dim)] for idx in indices])
        weights = np.array([
            np.prod([weights_1d[idx[d]] * w_1d[idx[d]] for d in range(dim)])
            for idx in indices
        ])

    all_compositions = []
    all_exact = []
    all_quad = []
    all_errors = []

    for total_deg in range(max_total_degree + 1):
        comps = generate_compositions(total_deg, dim)
        for comp in comps:
            # Exact integral (product of 1D integrals)
            exact = 1.0
            for d in range(dim):
                exact *= monomial_integral_gegenbauer(comp[d], alpha)

            # Quadrature approximation
            quad = 0.0
            for i in range(len(weights)):
                mon_val = 1.0
                for d in range(dim):
                    mon_val *= nodes[i, d] ** comp[d]
                quad += weights[i] * mon_val

            rel_err = abs(quad - exact) / max(abs(exact), 1e-15)

            all_compositions.append(comp)
            all_exact.append(exact)
            all_quad.append(quad)
            all_errors.append(rel_err)

    return {
        'dim': dim,
        'n_1d_points': n_1d,
        'max_total_degree': max_total_degree,
        'theoretical_exactness': 2 * n_1d - 1,
        'compositions': all_compositions,
        'exact_values': np.array(all_exact),
        'quad_values': np.array(all_quad),
        'errors': np.array(all_errors),
        'max_error': float(np.max(all_errors)) if all_errors else 0.0,
        'n_tests': len(all_errors)
    }


def cube_monomial_integral_reference(exponents: np.ndarray) -> float:
    """
    Exact integral of monomial x^e over [0,1]^d:
        integral = prod_i 1/(e_i + 1)

    Reference: cube01_monomial_integral from seed project 235.
    """
    return float(np.prod(1.0 / (exponents + 1)))
