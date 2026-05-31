"""
orthogonal_polynomials.py
=========================
Orthogonal polynomial basis generation and root-finding for gPC.

Fused from seed projects:
- 990_r8poly : Chebyshev / Legendre polynomial coefficient generation & Horner evaluation
- 203_companion_matrix : Companion matrix construction in orthogonal bases for root-finding

Mathematical foundation
-----------------------
For a weight function w(x) on [-1,1], orthogonal polynomials {P_n} satisfy
    \int_{-1}^{1} P_m(x) P_n(x) w(x) dx = h_n \delta_{mn}

Three classical families are implemented:
1. Legendre  : w(x)=1,   recurrence (n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}
2. Chebyshev (1st kind): w(x)=1/\sqrt{1-x^2}, recurrence T_{n+1}=2x T_n - T_{n-1}
3. Hermite   : w(x)=e^{-x^2} on (-\infty,\infty), recurrence H_{n+1}=2x H_n - 2n H_{n-1}

Gauss quadrature nodes are the eigenvalues of the Jacobi / symmetric tridiagonal
matrix (or the companion matrix in the monomial basis).  For Legendre/Chebyshev
the Jacobi matrix J has
    J_{i,i}   = 0
    J_{i,i+1} = J_{i+1,i} = \beta_i
where \beta_i = i / \sqrt{4i^2-1} for Legendre and \beta_i = 1/2 for Chebyshev.
"""

import numpy as np
from numpy.polynomial import polynomial as P


def legendre_coeffs(n):
    """
    Return the monomial coefficients of the Legendre polynomial P_n(x).
    Coefficients are ordered from lowest to highest degree.

    Uses Bonnet's recurrence in the coefficient domain:
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    p_nm2 = np.array([1.0])          # P_0
    p_nm1 = np.array([0.0, 1.0])     # P_1
    for k in range(1, n):
        # x * P_k  -> shift coefficients right by 1
        xpk = np.concatenate(([0.0], p_nm1))
        p_n = ((2.0 * k + 1.0) * xpk - k * np.concatenate((p_nm2, [0.0, 0.0]))[:len(xpk)]) / (k + 1.0)
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def chebyshev_coeffs(n):
    """
    Return the monomial coefficients of the Chebyshev polynomial T_n(x).

    Recurrence: T_0=1, T_1=x, T_n = 2x T_{n-1} - T_{n-2}
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    t_nm2 = np.array([1.0])
    t_nm1 = np.array([0.0, 1.0])
    for k in range(1, n):
        xtm1 = np.concatenate(([0.0], t_nm1))
        t_n = 2.0 * xtm1 - np.concatenate((t_nm2, [0.0, 0.0]))[:len(xtm1)]
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


def hermite_coeffs(n):
    """
    Return the monomial coefficients of the physicists' Hermite polynomial H_n(x).

    Recurrence: H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    h_nm2 = np.array([1.0])
    h_nm1 = np.array([0.0, 2.0])
    for k in range(1, n):
        xhm1 = np.concatenate(([0.0], h_nm1))
        h_n = 2.0 * xhm1 - 2.0 * k * np.concatenate((h_nm2, [0.0, 0.0]))[:len(xhm1)]
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def poly_horner(coeffs, x):
    """
    Evaluate a polynomial with monomial coefficients (low->high) using Horner's rule.
    p(x) = c_0 + x*(c_1 + x*(c_2 + ...))
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    if coeffs.size == 0:
        return np.zeros_like(x)
    result = np.full_like(x, coeffs[-1])
    for c in coeffs[-2::-1]:
        result = result * x + c
    return result


def legendre_eval(n, x):
    """Evaluate Legendre polynomial P_n at points x via stable three-term recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    p_nm2 = np.ones_like(x)
    p_nm1 = x.copy()
    for k in range(1, n):
        p_n = ((2.0 * k + 1.0) * x * p_nm1 - k * p_nm2) / (k + 1.0)
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def chebyshev_eval(n, x):
    """Evaluate Chebyshev polynomial T_n at points x via stable recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    t_nm2 = np.ones_like(x)
    t_nm1 = x.copy()
    for _ in range(1, n):
        t_n = 2.0 * x * t_nm1 - t_nm2
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


def hermite_eval(n, x):
    """Evaluate physicists' Hermite polynomial H_n at points x."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(1, n):
        h_n = 2.0 * x * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def jacobi_matrix_legendre(n):
    """
    Construct the n×n symmetric tridiagonal Jacobi matrix for Legendre polynomials.
    Eigenvalues are the Gauss-Legendre nodes; eigenvector weights give quadrature weights.

    J_{i,i} = 0
    J_{i,i+1} = i / sqrt(4i^2 - 1)   for i = 1,...,n-1
    """
    if n < 1:
        raise ValueError("n must be positive")
    J = np.zeros((n, n))
    for i in range(1, n):
        beta = i / np.sqrt(4.0 * i * i - 1.0)
        J[i - 1, i] = beta
        J[i, i - 1] = beta
    return J


def jacobi_matrix_chebyshev(n):
    """
    Construct the n×n symmetric tridiagonal Jacobi matrix for Chebyshev (1st kind).
    J_{i,i}=0, J_{i,i+1}=1/2.
    """
    if n < 1:
        raise ValueError("n must be positive")
    J = np.zeros((n, n))
    for i in range(1, n):
        J[i - 1, i] = 0.5
        J[i, i - 1] = 0.5
    return J


def gauss_legendre_nodes_weights(n):
    """
    Compute Gauss-Legendre quadrature nodes x_i and weights w_i for exactness degree 2n-1.

    The nodes x_i are the eigenvalues of the Jacobi matrix J, and the weights are
        w_i = 2 * (v_i^{(1)})^2
    where v_i^{(1)} is the first component of the normalized eigenvector.
    """
    J = jacobi_matrix_legendre(n)
    eigvals, eigvecs = np.linalg.eigh(J)
    # Nodes in ascending order
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    v1 = eigvecs[0, idx]
    w = 2.0 * v1 ** 2
    return x, w


def gauss_chebyshev_nodes_weights(n):
    """
    Compute Gauss-Chebyshev quadrature nodes and weights.
    Exact for polynomials up to degree 2n-1 with weight 1/sqrt(1-x^2).
    """
    J = jacobi_matrix_chebyshev(n)
    eigvals, eigvecs = np.linalg.eigh(J)
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    v1 = eigvecs[0, idx]
    w = np.pi * v1 ** 2
    return x, w


def normalized_legendre_eval(n, x):
    """
    Evaluate the normalized Legendre polynomial \tilde{P}_n(x) = P_n(x) / sqrt(2/(2n+1)).
    This makes {\tilde{P}_n} an orthonormal basis on [-1,1] with weight 1.
    """
    return legendre_eval(n, x) / np.sqrt(2.0 / (2.0 * n + 1.0))


def polynomial_roots_via_companion(coeffs):
    """
    Compute all roots of a monic polynomial from its coefficients using the
    Frobenius companion matrix eigenvalue problem.

    For p(x) = c_0 + c_1 x + ... + c_{n-1} x^{n-1} + x^n,
    the companion matrix C has C_{i+1,i}=1 and last row = -[c_0,...,c_{n-1}].
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    n = len(coeffs) - 1
    if n < 1:
        return np.array([], dtype=complex)
    # Make monic if not already
    if abs(coeffs[-1]) < 1e-15:
        raise ValueError("Leading coefficient is numerically zero")
    coeffs = coeffs / coeffs[-1]
    C = np.zeros((n, n), dtype=complex)
    if n > 1:
        C[1:, :-1] = np.eye(n - 1)
    C[-1, :] = -coeffs[:-1]
    roots = np.linalg.eigvals(C)
    return roots


def test_orthogonal_polynomials():
    """Self-test with orthogonality checks."""
    x, w = gauss_legendre_nodes_weights(20)
    for m in range(6):
        for n in range(6):
            val = np.sum(legendre_eval(m, x) * legendre_eval(n, x) * w)
            expect = 2.0 / (2.0 * m + 1.0) if m == n else 0.0
            assert np.isclose(val, expect, atol=1e-12), f"Legendre ortho fail ({m},{n})"
    print("orthogonal_polynomials: all self-tests passed")


if __name__ == "__main__":
    test_orthogonal_polynomials()
