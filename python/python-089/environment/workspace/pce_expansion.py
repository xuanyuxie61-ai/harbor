"""
Polynomial Chaos Expansion (PCE) Module
========================================
Based on project 853_pce_legendre.

Implements generalized Polynomial Chaos Expansion using normalized
Legendre polynomials for uniform random variables in stochastic
structural mechanics.

Key formulas:
- Multidimensional Legendre basis:
  Psi_alpha(xi) = prod_{i=1}^N L_{alpha_i}(xi_i)
- Stochastic Galerkin projection:
  <Psi_alpha, Psi_beta> = delta_{alpha,beta} ||Psi_alpha||^2
- Random field representation:
  u(x,xi) = sum_{alpha} u_alpha(x) Psi_alpha(xi)
"""

import numpy as np
from math import comb, factorial
from itertools import combinations_with_replacement


def legendre_polynomial(n, x):
    """
    Evaluate normalized Legendre polynomials L_0(x) through L_n(x).
    Normalization: integral_{-1}^{1} L_i(x) L_j(x) dx = delta_{ij}.
    
    Uses the three-term recurrence:
    (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
    with normalization factor sqrt((2k+1)/2).
    
    Parameters
    ----------
    n : int
        Maximum degree.
    x : float or array_like
        Evaluation point(s) in [-1, 1].
    
    Returns
    -------
    L : ndarray
        Array of shape (n+1, ...) with L[k] = L_k(x).
    """
    x = np.asarray(x, dtype=float)
    L = np.zeros((n + 1,) + x.shape)
    L[0] = 1.0 / np.sqrt(2.0)
    if n >= 1:
        L[1] = np.sqrt(3.0 / 2.0) * x
    for k in range(1, n):
        c1 = (2.0 * k + 1.0) / (k + 1.0)
        c2 = float(k) / (k + 1.0)
        # normalization adjustment for orthonormal basis
        norm_k = np.sqrt((2.0 * k + 1.0) / 2.0)
        norm_kp1 = np.sqrt((2.0 * (k + 1) + 1.0) / 2.0)
        norm_km1 = np.sqrt((2.0 * (k - 1) + 1.0) / 2.0)
        # For orthonormal Legendre:
        # (k+1)/sqrt(2k+3) L_{k+1} = (2k+1)/sqrt(2k+1) x L_k - k/sqrt(2k-1) L_{k-1}
        # Simplified recurrence for orthonormal version:
        ak = np.sqrt((2.0 * k + 1.0) * (2.0 * k + 3.0)) / (k + 1.0)
        bk = - np.sqrt((2.0 * k + 3.0) / (2.0 * k - 1.0)) * k / (k + 1.0)
        L[k + 1] = ak * x * L[k] + bk * L[k - 1]
    return L


def legendre_linear_product(p, e=0):
    """
    Compute table T_ij = integral_{-1}^{1} x^e * L_i(x) * L_j(x) dx
    for 0 <= i, j <= p using Gauss-Legendre quadrature.
    
    This is essential for stochastic Galerkin projections with
    spatially varying random coefficients.
    
    Parameters
    ----------
    p : int
        Maximum polynomial degree.
    e : int, optional
        Exponent of x in weight. Default 0.
    
    Returns
    -------
    table : ndarray, shape (p+1, p+1)
        The product integral table.
    """
    order = p + 1 + (e + 1) // 2
    xq, wq = np.polynomial.legendre.leggauss(order)
    table = np.zeros((p + 1, p + 1))
    L_all = legendre_polynomial(p, xq)  # shape (p+1, nquad)
    for k in range(order):
        if e == 0:
            contrib = wq[k] * np.outer(L_all[:, k], L_all[:, k])
        else:
            contrib = wq[k] * (xq[k] ** e) * np.outer(L_all[:, k], L_all[:, k])
        table += contrib
    # Numerical cleanup
    table[np.abs(table) < 1e-14] = 0.0
    return table


def multi_index_count(n_dim, p):
    """
    Number of N-variate polynomials of total degree <= p.
    
    NY = C(N+P, N) = (N+P)! / (N! P!)
    
    Parameters
    ----------
    n_dim : int
        Stochastic dimension.
    p : int
        Maximum total degree.
    
    Returns
    -------
    count : int
        Number of multi-indices.
    """
    return comb(n_dim + p, n_dim)


def generate_multi_indices(n_dim, p):
    """
    Generate all multi-indices alpha in N^N with |alpha| <= p.
    Uses lexicographic ordering based on compositions (comp_next logic).
    
    Parameters
    ----------
    n_dim : int
        Stochastic dimension.
    p : int
        Maximum total degree.
    
    Returns
    -------
    indices : ndarray, shape (NY, n_dim)
        Each row is a multi-index alpha.
    """
    # Use recursion to generate compositions
    if n_dim == 1:
        return np.arange(p + 1).reshape(-1, 1)
    indices = []
    for total in range(p + 1):
        # Generate all compositions of 'total' into 'n_dim' parts
        def compositions(k, n, prefix):
            if n == 1:
                yield prefix + [k]
            else:
                for i in range(k + 1):
                    yield from compositions(k - i, n - 1, prefix + [i])
        for comp in compositions(total, n_dim, []):
            indices.append(comp)
    return np.array(indices, dtype=int)


def evaluate_pce_basis(n_dim, p, xi, indices=None):
    """
    Evaluate all PCE basis functions at sample points xi.
    
    Psi_alpha(xi) = prod_{i=1}^N L_{alpha_i}(xi_i)
    
    Parameters
    ----------
    n_dim : int
        Stochastic dimension.
    p : int
        Maximum total degree.
    xi : ndarray, shape (n_samples, n_dim)
        Random variable samples in [-1, 1]^N.
    indices : ndarray, optional
        Precomputed multi-indices.
    
    Returns
    -------
    psi : ndarray, shape (n_samples, NY)
        Basis function values.
    indices : ndarray
        Multi-indices used.
    norms : ndarray
        Squared L2 norms of each basis function (should be 1 for orthonormal).
    """
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    if indices is None:
        indices = generate_multi_indices(n_dim, p)
    n_basis = indices.shape[0]
    n_samples = xi.shape[0]
    
    # Evaluate 1D Legendre polynomials for each dimension
    L_1d = []
    for d in range(n_dim):
        Ld = legendre_polynomial(p, xi[:, d])  # shape (p+1, n_samples)
        L_1d.append(Ld)
    
    psi = np.ones((n_samples, n_basis))
    for j, alpha in enumerate(indices):
        for d in range(n_dim):
            if alpha[d] > 0:
                psi[:, j] *= L_1d[d][alpha[d], :]
    
    # For orthonormal Legendre on uniform [-1,1], each basis norm = 1
    norms = np.ones(n_basis)
    return psi, indices, norms


def assemble_stochastic_galerkin_system(K_det, M_det, C_det, 
                                         n_dim, p, E_mean, E_std,
                                         kl_modes, kl_eigenvalues):
    """
    Assemble the stochastic Galerkin system for random vibration.
    
    For a lognormal Young's modulus:
    E(x,xi) = E_mean * exp( sum_i sqrt(lambda_i) * phi_i(x) * xi_i )
    
    Using a first-order PCE approximation:
    E(x,xi) ≈ E_0(x) + sum_i E_i(x) * xi_i
    
    The stochastic stiffness operator becomes block-coupled:
    [K_{alpha,beta}] = <Psi_alpha, E(x,cdot) Psi_beta> K_det
    
    Parameters
    ----------
    K_det, M_det, C_det : ndarray
        Deterministic stiffness, mass, damping matrices (sparse-like dense for small problems).
    n_dim : int
        Stochastic dimension (number of KL modes).
    p : int
        PCE polynomial order.
    E_mean, E_std : float
        Mean and standard deviation of Young's modulus.
    kl_modes : ndarray
        KL mode shapes evaluated at nodes.
    kl_eigenvalues : ndarray
        KL eigenvalues.
    
    Returns
    -------
    K_sg, M_sg, C_sg : ndarray
        Block-structured stochastic Galerkin matrices.
    indices : ndarray
        Multi-index table.
    """
    n_dof = K_det.shape[0]
    indices = generate_multi_indices(n_dim, p)
    n_basis = indices.shape[0]
    
    # For small problems, construct dense block matrices
    K_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    M_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    C_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    
    # Mass and damping are deterministic in this model
    for i in range(n_basis):
        ii = slice(i * n_dof, (i + 1) * n_dof)
        M_sg[ii, ii] = M_det
        C_sg[ii, ii] = C_det
    
    # Stiffness: K_sg[(k,i), (l,j)] = K_det[i,j] * <Psi_k, E Psi_l>
    # For first-order expansion E ≈ E_0 + sum_m E_m xi_m:
    # <Psi_k, E_0 Psi_l> = E_0 * delta_{kl}
    # <Psi_k, xi_m Psi_l> requires 1D Legendre product integral
    
    E0 = E_mean
    # Precompute 1D linear product tables
    table = legendre_linear_product(p, e=1)
    
    for k in range(n_basis):
        for l in range(n_basis):
            alpha = indices[k]
            beta = indices[l]
            
            # Deterministic part: E_0 * delta_{alpha,beta}
            coeff = E0 if np.array_equal(alpha, beta) else 0.0
            
            # Stochastic coupling through KL modes
            # <Psi_alpha, xi_m Psi_beta> = prod_d <L_{alpha_d}, xi_d^{delta_{dm}} L_{beta_d}>
            for m in range(n_dim):
                # Check if all other dimensions match
                match = True
                factor = 1.0
                for d in range(n_dim):
                    if d == m:
                        # <L_{alpha_m}, xi_m L_{beta_m}>
                        if alpha[d] <= p and beta[d] <= p:
                            factor *= table[alpha[d], beta[d]]
                        else:
                            match = False
                            break
                    else:
                        if alpha[d] != beta[d]:
                            match = False
                            break
                        # <L_{alpha_d}, L_{beta_d}> = delta_{alpha_d,beta_d} = 1
                        factor *= 1.0
                
                if match:
                    Em_coeff = E_std * np.sqrt(kl_eigenvalues[m])
                    coeff += Em_coeff * factor
            
            if abs(coeff) > 1e-14:
                kk = slice(k * n_dof, (k + 1) * n_dof)
                ll = slice(l * n_dof, (l + 1) * n_dof)
                K_sg[kk, ll] += coeff * K_det
    
    return K_sg, M_sg, C_sg, indices


def pce_moments(coefficients, indices, norms=None):
    """
    Compute mean and variance from PCE coefficients.
    
    Mean = coefficient of Psi_0
    Variance = sum_{alpha != 0} coeff_alpha^2 * ||Psi_alpha||^2
    
    Parameters
    ----------
    coefficients : ndarray
        PCE coefficients, shape (n_dof, n_basis) or (n_basis,).
    indices : ndarray
        Multi-indices.
    norms : ndarray, optional
        Basis norms squared.
    
    Returns
    -------
    mean, variance : ndarray or float
    """
    coefficients = np.asarray(coefficients)
    if norms is None:
        norms = np.ones(indices.shape[0])
    
    if coefficients.ndim == 1:
        mean = coefficients[0]
        variance = np.sum(coefficients[1:] ** 2 * norms[1:])
    else:
        mean = coefficients[:, 0]
        variance = np.sum(coefficients[:, 1:] ** 2 * norms[1:], axis=1)
    return mean, variance


def kl_expansion_1d(n_modes, length, correlation_length, x_coords):
    """
    Generate 1D Karhunen-Loeve modes for exponential covariance:
    C(x1,x2) = exp(-|x1-x2|/Lc)
    
    Analytical eigenvalues and eigenfunctions for 1D exponential kernel:
    lambda_n = 2*Lc / (1 + w_n^2 * Lc^2)
    phi_n(x) = cos(w_n * x) / sqrt(...) or sin(...)
    
    Parameters
    ----------
    n_modes : int
        Number of KL modes to retain.
    length : float
        Domain length.
    correlation_length : float
        Correlation length Lc.
    x_coords : ndarray
        Spatial coordinates.
    
    Returns
    -------
    eigenvalues : ndarray
    modes : ndarray, shape (n_modes, n_points)
    """
    x = np.asarray(x_coords)
    Lc = correlation_length
    L = length
    
    eigenvalues = np.zeros(n_modes)
    modes = np.zeros((n_modes, len(x)))
    
    # Approximate with analytical solution on [0, L]
    for n in range(n_modes):
        if n % 2 == 0:
            # Even modes (cosine-like)
            w = (n + 1) * np.pi / L
            eigenvalues[n] = 2.0 * Lc / (1.0 + (w * Lc) ** 2)
            phi = np.cos(w * x)
        else:
            # Odd modes (sine-like)
            w = (n + 1) * np.pi / L
            eigenvalues[n] = 2.0 * Lc / (1.0 + (w * Lc) ** 2)
            phi = np.sin(w * x)
        # Normalize
        norm = np.sqrt(np.trapezoid(phi ** 2, x))
        if norm > 1e-12:
            phi = phi / norm
        modes[n] = phi
    
    return eigenvalues, modes
