# -*- coding: utf-8 -*-
"""
Random Field Generation for Stochastic PDEs
============================================
Generates random coefficients and forcing terms for spectral PDE solvers
using Wishart covariance sampling and Monte Carlo validation.

Inspired by:
- AS 53 Wishart variate generator (wshrt)
- Fly simulation Monte Carlo sampling (inverse transform)

Mathematical formulation:
- Karhunen-Loeve expansion of a random field:
    nu(x, xi) = nu_bar(x) + sum_{k=1}^M sqrt(lambda_k) phi_k(x) xi_k
  where (lambda_k, phi_k) are eigenpairs of the covariance kernel.
- For this module, we use a simplified parametric representation with
  Wishart-sampled covariance matrices to model spatial correlation.
"""

import numpy as np


def box_muller_normal(n):
    """
    Generate n standard normal variates using Box-Muller transform.
    Adapted from rnorm.m in AS 53.

    Parameters
    ----------
    n : int
        Number of variates (rounded up to even if odd).

    Returns
    -------
    z : ndarray
        Standard normal samples.
    """
    n2 = (n + 1) // 2
    u1 = np.random.rand(n2)
    u2 = np.random.rand(n2)
    # Box-Muller
    r = np.sqrt(-2.0 * np.log(u1 + 1e-15))
    theta = 2.0 * np.pi * u2
    z = np.zeros(2 * n2)
    z[0::2] = r * np.cos(theta)
    z[1::2] = r * np.sin(theta)
    return z[:n]


def wishart_variate_chol(D, n, np_dim):
    """
    Generate a random Wishart variate given the Cholesky factor D.
    Adapted from wshrt.m (AS 53).

    Parameters
    ----------
    D : ndarray, shape (np_dim, np_dim)
        Upper-triangular Cholesky factor of covariance Sigma.
    n : int
        Degrees of freedom.
    np_dim : int
        Dimension.

    Returns
    -------
    SA : ndarray, shape (np_dim, np_dim)
        Wishart sample (upper triangular part).
    """
    D = np.asarray(D, dtype=np.float64)
    np_dim = int(np_dim)
    nnp = np_dim * (np_dim + 1) // 2
    sb = box_muller_normal(nnp)
    sa = np.zeros(nnp, dtype=np.float64)

    # Wilson-Hilferty chi-square approximation for diagonal
    ns = 0
    for i in range(1, np_dim + 1):
        df = np_dim - i + 1
        ns += i
        u1 = 2.0 / (9.0 * df)
        u2 = 1.0 - u1
        u1 = np.sqrt(u1)
        sb[ns - 1] = np.sqrt(df * abs((u2 + sb[ns - 1] * u1) ** 3))

    rn = float(n)
    # First triangular multiplication
    sa_out = np.zeros(nnp, dtype=np.float64)
    for i in range(1, np_dim + 1):
        nr = i * (i - 1) // 2 + 1
        for j in range(i, np_dim + 1):
            ip = nr
            nq = j * (j - 1) // 2 + i - 1
            c = 0.0
            for k in range(i, j + 1):
                ip += k - 1
                nq += 1
                c += sb[ip - 1] * D[i - 1, k - 1]
            sa_out[ip - 1] = c

    # Second multiplication and scaling
    SA = np.zeros((np_dim, np_dim), dtype=np.float64)
    idx = 0
    for j in range(np_dim):
        for i in range(j + 1):
            SA[i, j] = sa_out[idx]
            idx += 1

    # Reconstruct symmetric matrix: SA = SA @ SA.T / rn
    # Actually wshrt computes a packed upper-triangular form.
    # For simplicity, we compute the full symmetric matrix.
    W = SA @ SA.T / rn
    return W


def generate_random_diffusion_field(x, d_stochastic=3, mean_val=0.1,
                                    fluctuation=0.05, correlation_length=0.3):
    """
    Generate a random diffusion coefficient field nu(x, xi) using a
    truncated Karhunen-Loeve expansion with Wishart-sampled covariance.

    nu(x, xi) = mean_val + fluctuation * sum_k sqrt(lambda_k) phi_k(x) xi_k

    Parameters
    ----------
    x : ndarray
        Spatial grid points.
    d_stochastic : int
        Number of KL modes (stochastic dimension).
    mean_val : float
        Mean diffusion coefficient.
    fluctuation : float
        Fluctuation amplitude.
    correlation_length : float
        Correlation length of the random field.

    Returns
    -------
    nu_base : ndarray
        Mean field.
    kl_modes : ndarray, shape (len(x), d_stochastic)
        KL modes.
    kl_eigenvalues : ndarray
        KL eigenvalues.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    L = x[-1] - x[0]

    # Exponential covariance kernel: C(x,y) = exp(-|x-y|/lc)
    # Analytical KL eigenvalues and eigenfunctions on [-L/2, L/2]
    # For simplicity, use discrete eigen-decomposition
    X, Y = np.meshgrid(x, x)
    C = np.exp(-np.abs(X - Y) / (correlation_length * L + 1e-15))

    # Sample covariance structure using Wishart for robustness
    # Use eigenvalue perturbation to ensure positive definiteness
    eig_C = np.linalg.eigvalsh(C)
    min_eig = np.min(eig_C)
    reg = max(1e-6, -min_eig + 1e-6)
    C_reg = C + reg * np.eye(N)
    D_chol = np.linalg.cholesky(C_reg)
    W = wishart_variate_chol(D_chol, n=min(N, 20), np_dim=N)
    # Use Wishart sample to perturb the covariance
    C_perturbed = C_reg + 0.001 * W
    C_perturbed = 0.5 * (C_perturbed + C_perturbed.T)

    eigenvalues, eigenvectors = np.linalg.eigh(C_perturbed)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[idx], 0, None)
    eigenvectors = eigenvectors[:, idx]

    d = min(d_stochastic, N)
    kl_eigenvalues = eigenvalues[:d] / eigenvalues[0]
    kl_modes = eigenvectors[:, :d]

    nu_base = np.full_like(x, mean_val)
    return nu_base, kl_modes, kl_eigenvalues


def sample_random_field_at_xi(x, xi, nu_base, kl_modes, kl_eigenvalues, fluctuation):
    """
    Evaluate the random field at specific random variable samples xi.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    xi : ndarray, shape (d,)
        Random variable sample.
    nu_base : ndarray
        Mean field.
    kl_modes : ndarray
        KL modes.
    kl_eigenvalues : ndarray
        KL eigenvalues.
    fluctuation : float
        Fluctuation amplitude.

    Returns
    -------
    nu : ndarray
        Realized diffusion field.
    """
    nu = nu_base.copy()
    d = len(xi)
    for k in range(d):
        nu += fluctuation * np.sqrt(kl_eigenvalues[k]) * kl_modes[:, k] * xi[k]
    # Ensure positivity for physical diffusion coefficient
    nu = np.clip(nu, 1e-6, 10.0)
    return nu


def monte_carlo_statistic(samples):
    """
    Compute Monte Carlo mean, variance, and confidence interval.
    Inspired by fly_simulation statistical estimation.

    Parameters
    ----------
    samples : ndarray
        Array of samples.

    Returns
    -------
    stats : dict
        mean, var, std, ci_95
    """
    samples = np.asarray(samples, dtype=np.float64)
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    std = np.sqrt(var)
    n = len(samples)
    ci_95 = 1.96 * std / np.sqrt(max(n, 1))
    return {"mean": mean, "variance": var, "std": std, "ci_95": ci_95, "n": n}
