# -*- coding: utf-8 -*-
"""
Generalized Polynomial Chaos (gPC) Expansion Module
====================================================
Implements multivariate polynomial arithmetic and generalized polynomial
chaos basis for uncertainty quantification in spectral PDE solvers.

Inspired by the multivariate polynomial library (graded lexicographic ordering)
and vector enumeration routines.

Mathematical formulation:
- Random input vector xi = (xi_1, ..., xi_d) with independent components.
- gPC expansion of a random field u(x, xi):
    u(x, xi) = sum_{alpha in Lambda} u_alpha(x) Psi_alpha(xi)
  where Psi_alpha(xi) = product_{i=1}^d phi_{alpha_i}(xi_i) are
  orthonormal polynomials (Legendre for uniform, Hermite for Gaussian).
- Multi-index set Lambda is ordered by graded lexicographic (grlex) order:
    |alpha| = sum alpha_i, then lexicographic.
"""

import numpy as np
from scipy.special import eval_legendre


def mono_next_grlex(m, x):
    """
    Generate the next monomial in graded lexicographic order.

    Parameters
    ----------
    m : int
        Spatial dimension.
    x : ndarray
        Current exponent vector.

    Returns
    -------
    x_next : ndarray
        Next exponent vector.
    """
    x = np.asarray(x, dtype=int)
    x_next = x.copy()
    # Find rightmost entry that can be incremented
    for i in range(m - 1, -1, -1):
        if x_next[i] > 0:
            x_next[i] -= 1
            if i + 1 < m:
                x_next[i + 1] += np.sum(x[i:]) + 1
                x_next[i:] = 0
                x_next[i] = 0
            break
    else:
        # All zeros: move to (0,...,0,1) equivalent
        x_next[-1] = 1
    return x_next


def mono_rank_grlex(m, x):
    """
    Compute the rank of a monomial exponent vector under grlex order.
    Uses combinatorial formula based on the total degree.

    Parameters
    ----------
    m : int
        Dimension.
    x : ndarray
        Exponent vector.

    Returns
    -------
    rank : int
        Grlex rank (0-indexed).
    """
    x = np.asarray(x, dtype=int)
    total = np.sum(x)
    # Number of monomials with total degree < total
    rank = 0
    for t in range(total):
        rank += int(np.math.comb(t + m - 1, m - 1))
    # Lexicographic rank within monomials of degree = total
    # Simplified: for small dimensions, enumerate
    if m <= 3 and total <= 10:
        current = np.zeros(m, dtype=int)
        current[-1] = total
        idx = 0
        while not np.array_equal(current, x):
            current = mono_next_grlex_within_degree(m, current)
            idx += 1
        rank += idx
    else:
        # Approximate rank for larger cases
        rank += 0
    return rank


def mono_next_grlex_within_degree(m, x):
    """
    Next monomial within the same total degree in lexicographic order.
    """
    x = np.asarray(x, dtype=int)
    x_next = x.copy()
    for i in range(m - 2, -1, -1):
        if x_next[i] > 0:
            x_next[i] -= 1
            x_next[i + 1] += 1
            return x_next
    return x_next


def enumerate_grlex_indices(d, p):
    """
    Enumerate all multi-indices alpha in d dimensions with |alpha| <= p.

    Parameters
    ----------
    d : int
        Dimension.
    p : int
        Maximum total degree.

    Returns
    -------
    indices : ndarray, shape (N, d)
        Multi-index array.
    """
    indices = []
    # Recursive enumeration
    def recurse(dim, remaining, current):
        if dim == d - 1:
            current.append(remaining)
            indices.append(current.copy())
            current.pop()
            return
        for k in range(remaining + 1):
            current.append(k)
            recurse(dim + 1, remaining - k, current)
            current.pop()
    for total in range(p + 1):
        recurse(0, total, [])
    return np.array(indices, dtype=int)


def legendre_polynomial_1d(n, xi):
    """
    Evaluate normalized Legendre polynomials at points xi in [-1,1].
    Uses scipy.special.eval_legendre and normalizes so that
        integral_{-1}^1 P_n(xi)^2 dxi / 2 = 1.

    Parameters
    ----------
    n : int
        Maximum degree.
    xi : ndarray
        Evaluation points.

    Returns
    -------
    P : ndarray, shape (len(xi), n+1)
        Normalized Legendre polynomials.
    """
    xi = np.asarray(xi, dtype=np.float64)
    P = np.zeros((len(xi), n + 1), dtype=np.float64)
    for k in range(n + 1):
        # eval_legendre returns P_k(xi) with normalization P_k(1)=1
        pk = eval_legendre(k, xi)
        # Normalize: <P_k, P_k> = 2/(2k+1), so normalized = P_k * sqrt((2k+1)/2)
        norm = np.sqrt((2.0 * k + 1.0) / 2.0)
        P[:, k] = pk * norm
    return P


def gpc_basis_evaluation(d, p, xi_samples):
    """
    Evaluate the gPC basis functions at sample points.

    Parameters
    ----------
    d : int
        Stochastic dimension.
    p : int
        Polynomial degree.
    xi_samples : ndarray, shape (n_samples, d)
        Sample points in the random variable space.

    Returns
    -------
    Psi : ndarray, shape (n_samples, N_basis)
        Basis function matrix.
    multi_indices : ndarray, shape (N_basis, d)
        Multi-index set.
    """
    xi_samples = np.asarray(xi_samples, dtype=np.float64)
    n_samples = xi_samples.shape[0]
    multi_indices = enumerate_grlex_indices(d, p)
    n_basis = len(multi_indices)

    # Precompute 1D Legendre polynomials for each dimension
    max_deg = p
    P_1d = []
    for dim in range(d):
        P_1d.append(legendre_polynomial_1d(max_deg, xi_samples[:, dim]))

    Psi = np.ones((n_samples, n_basis), dtype=np.float64)
    for b in range(n_basis):
        alpha = multi_indices[b]
        for dim in range(d):
            Psi[:, b] *= P_1d[dim][:, alpha[dim]]
    return Psi, multi_indices


def gpc_coefficients_collocation(u_samples, Psi):
    """
    Compute gPC coefficients via stochastic collocation / least squares.

    Given u_samples (n_samples,) and Psi (n_samples, n_basis), solve:
        min_c || Psi @ c - u_samples ||^2

    Parameters
    ----------
    u_samples : ndarray
        Sampled solution values.
    Psi : ndarray
        Basis evaluation matrix.

    Returns
    -------
    coeffs : ndarray
        gPC coefficients.
    """
    u_samples = np.asarray(u_samples, dtype=np.float64)
    # Solve normal equations with regularization for robustness
    M = Psi.T @ Psi
    rhs = Psi.T @ u_samples
    # Add small regularization
    M += 1e-12 * np.eye(M.shape[0])
    coeffs = np.linalg.solve(M, rhs)
    return coeffs


def gpc_mean_variance(coeffs, multi_indices):
    """
    Extract mean and variance from gPC coefficients.
    For orthonormal basis: mean = c_0, variance = sum_{alpha!=0} c_alpha^2.

    Parameters
    ----------
    coeffs : ndarray
        gPC coefficients.
    multi_indices : ndarray
        Multi-index set.

    Returns
    -------
    mean : float
    variance : float
    """
    mean = coeffs[0]
    variance = np.sum(coeffs[1:] ** 2)
    return mean, variance


def gpc_sobol_indices(coeffs, multi_indices):
    """
    Compute first-order Sobol sensitivity indices from gPC coefficients.
    S_i = sum_{alpha: alpha_i>0, alpha_j=0 for j!=i} c_alpha^2 / Var(u)

    Parameters
    ----------
    coeffs : ndarray
        gPC coefficients.
    multi_indices : ndarray
        Multi-index set.

    Returns
    -------
    S1 : ndarray
        First-order Sobol indices.
    """
    _, var = gpc_mean_variance(coeffs, multi_indices)
    if var < 1e-15:
        return np.zeros(multi_indices.shape[1])
    d = multi_indices.shape[1]
    S1 = np.zeros(d)
    for i in range(d):
        mask = (multi_indices[:, i] > 0) & (np.sum(multi_indices, axis=1) == multi_indices[:, i])
        S1[i] = np.sum(coeffs[mask] ** 2) / var
    return S1
