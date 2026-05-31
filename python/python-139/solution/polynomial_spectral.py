"""
Spectral representation of concentration profiles using orthogonal polynomials.

Adapted from polynomial_conversion (Bernstein, Legendre, Chebyshev, Gegenbauer,
Hermite, Laguerre basis conversions).  In membrane science, spectral methods
allow high-accuracy approximation of steep concentration gradients near interfaces.
"""

import numpy as np
from scipy.special import eval_legendre, eval_chebyt, factorial


def build_legendre_matrix(n, x_nodes):
    """
    Build the Vandermonde-like matrix for Legendre polynomial basis of degree n
    evaluated at x_nodes in [-1,1].
    """
    x = np.asarray(x_nodes, dtype=float)
    m = len(x)
    V = np.zeros((m, n + 1), dtype=float)
    for k in range(n + 1):
        V[:, k] = eval_legendre(k, x)
    return V


def monomial_to_legendre_matrix(n):
    """
    Return the (n+1) x (n+1) conversion matrix from monomial basis
    {1, x, x^2, ..., x^n} to shifted Legendre basis on [0,1].
    """
    # Shifted Legendre P_k^*(x) = P_k(2x-1)
    # Use recursion and symbolic coefficients truncated to n+1
    A = np.eye(n + 1, dtype=float)
    # For each shifted Legendre polynomial, express in monomials via binomial sums
    # P_k(2x-1) = sum_{j=0}^k c_{k,j} x^j
    C = np.zeros((n + 1, n + 1), dtype=float)
    for k in range(n + 1):
        # Use three-term recursion for shifted Legendre on [0,1]
        # P_0^* = 1
        # P_1^* = 2x - 1
        if k == 0:
            C[0, 0] = 1.0
        elif k == 1:
            C[1, 0] = -1.0
            C[1, 1] = 2.0
        else:
            # (k) P_k^* = (2k-1)(2x-1) P_{k-1}^* - (k-1) P_{k-2}^*
            # => P_k^* = ((2k-1)(2x-1) P_{k-1}^* - (k-1) P_{k-2}^*) / k
            prev = np.zeros(n + 1, dtype=float)
            prev2 = np.zeros(n + 1, dtype=float)
            prev[:k] = C[k - 1, :k]
            prev2[:k - 1] = C[k - 2, :k - 1]
            # Multiply prev by (2x-1): convolve with [-1, 2]
            temp = np.zeros(n + 1, dtype=float)
            temp[0] = -prev[0]
            for j in range(1, n + 1):
                temp[j] = 2.0 * prev[j - 1] - (prev[j] if j <= k - 1 else 0.0)
            temp = temp * (2.0 * k - 1.0)
            temp[:k - 1] -= (k - 1.0) * prev2[:k - 1]
            C[k, :] = temp / float(k)
    # Now C[k,j] is coefficient of x^j in P_k^*(x)
    # Monomial -> shifted Legendre: A = C^{-T} ... actually if
    # f(x) = sum_j a_j x^j = sum_k b_k P_k^*(x)
    # then a = C^T b, so b = (C^T)^{-1} a.
    A = np.linalg.inv(C.T)
    return A


def legendre_to_monomial_matrix(n):
    """
    Inverse of monomial_to_legendre_matrix.
    """
    A = monomial_to_legendre_matrix(n)
    return np.linalg.inv(A)


def bernstein_to_monomial_matrix(n):
    """
    Bernstein basis B_{i,n}(x) = C(n,i) x^i (1-x)^{n-i}.
    Express each Bernstein polynomial in monomials.
    """
    B = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        # (1-x)^{n-i} = sum_{k=0}^{n-i} C(n-i,k) (-1)^k x^k
        for k in range(n - i + 1):
            coeff = factorial(n) / (factorial(i) * factorial(n - i))
            sign = 1.0 if k % 2 == 0 else -1.0
            coeff *= sign * (factorial(n - i) / (factorial(k) * factorial(n - i - k)))
            B[i, i + k] = coeff
    # Monomial -> Bernstein: inverse of B^T? Let's be careful.
    # f = sum_i c_i B_i = sum_j a_j x^j.
    # Here B[i,j] = coeff of x^j in B_i.
    # Then a = B^T c, so c = (B^T)^{-1} a.
    return np.linalg.inv(B.T)


def monomial_to_bernstein_matrix(n):
    return np.linalg.inv(bernstein_to_monomial_matrix(n))


def chebyshev_to_monomial_matrix(n):
    """
    Chebyshev T_k(x) on [-1,1]; convert to monomials.
    T_0=1, T_1=x, T_{k}=2x T_{k-1} - T_{k-2}.
    """
    C = np.zeros((n + 1, n + 1), dtype=float)
    C[0, 0] = 1.0
    if n >= 1:
        C[1, 1] = 1.0
    for k in range(2, n + 1):
        # T_k = 2x T_{k-1} - T_{k-2}
        prev = np.zeros(n + 1, dtype=float)
        prev2 = np.zeros(n + 1, dtype=float)
        prev[:k] = C[k - 1, :k]
        prev2[:k - 1] = C[k - 2, :k - 1]
        # multiply prev by x: shift up by 1
        temp = np.zeros(n + 1, dtype=float)
        temp[1:k + 1] = 2.0 * prev[:k]
        temp[:k - 1] -= prev2[:k - 1]
        C[k, :] = temp
    # Monomial -> Chebyshev
    return np.linalg.inv(C.T)


def monomial_to_chebyshev_matrix(n):
    return np.linalg.inv(chebyshev_to_monomial_matrix(n))


def gegenbauer_to_monomial_matrix(n, alpha):
    """
    Gegenbauer (ultraspherical) polynomials C_k^{(\alpha)}(x).
    Recurrence: k C_k = 2(k+\alpha-1) x C_{k-1} - (k+2\alpha-2) C_{k-2}
    """
    C = np.zeros((n + 1, n + 1), dtype=float)
    C[0, 0] = 1.0
    if n >= 1:
        C[1, 1] = 2.0 * alpha
    for k in range(2, n + 1):
        prev = np.zeros(n + 1, dtype=float)
        prev2 = np.zeros(n + 1, dtype=float)
        prev[:k] = C[k - 1, :k]
        prev2[:k - 1] = C[k - 2, :k - 1]
        temp = np.zeros(n + 1, dtype=float)
        temp[1:k + 1] = 2.0 * (k + alpha - 1.0) * prev[:k]
        temp[:k - 1] -= (k + 2.0 * alpha - 2.0) * prev2[:k - 1]
        C[k, :] = temp / float(k)
    return np.linalg.inv(C.T)


def monomial_to_gegenbauer_matrix(n, alpha):
    return np.linalg.inv(gegenbauer_to_monomial_matrix(n, alpha))


def spectral_interpolate_legendre(coeffs, x_query):
    """
    Evaluate a function given by Legendre coefficients at query points.
    """
    x = np.asarray(x_query, dtype=float)
    n = len(coeffs) - 1
    val = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        val += coeffs[k] * eval_legendre(k, x)
    return val


def map_domain_01_to_m1p1(x):
    """Map x in [0,1] to [-1,1]."""
    return 2.0 * np.asarray(x, dtype=float) - 1.0


def map_domain_m1p1_to_01(x):
    """Map x in [-1,1] to [0,1]."""
    return 0.5 * (np.asarray(x, dtype=float) + 1.0)
