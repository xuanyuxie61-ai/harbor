"""
eigenvalue_analysis.py
======================
Eigenvalue and buckling analysis for damaged composite laminates.

Incorporates core algorithms from:
- 1206_test_eigen : Generation of nonsymmetric matrices with prescribed
    eigenstructure via Schur decomposition (A = Q^T T Q).

Scientific role:
    Performs linearized buckling analysis and modal analysis of composite
    structures with damage-induced stiffness degradation. The buckling
    load factor lambda_b is found by solving the generalized eigenvalue
    problem:
        (K + lambda * K_sigma) * phi = 0
    where K is the tangent stiffness and K_sigma is the geometric
    stiffness (stress stiffness) matrix.

Key formulas:
-----------
1. Generalized eigenvalue problem:
    K * phi = lambda * M * phi
    For buckling: K * phi = -lambda * K_sigma * phi

2. Rayleigh quotient:
    lambda = (phi^T K phi) / (phi^T M phi)

3. Nonsymmetric test matrix generation:
    A = Q^T * T * Q
    where Q is a random orthogonal matrix and T is upper triangular
    with prescribed eigenvalues.

4. Buckling load factor:
    lambda_b = min_i |lambda_i|
    where lambda_i are eigenvalues of (-K^{-1} K_sigma).

5. Mode shape normalization:
    phi_norm = phi / sqrt(phi^T M phi)
"""

import numpy as np


def r8mat_orth_uniform(n):
    """
    Generate a random orthogonal matrix Q using QR decomposition.
    """
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def r8triangular_upper_gen(n, lambda_mean=0.0, lambda_dev=1.0):
    """
    Generate an upper triangular matrix T with random diagonal entries
    drawn from N(lambda_mean, lambda_dev^2) and random upper entries.
    """
    T = np.triu(np.random.randn(n, n))
    diag = lambda_mean + lambda_dev * np.random.randn(n)
    np.fill_diagonal(T, diag)
    return T


def r8nsymm_gen(n, lambda_mean=0.0, lambda_dev=1.0):
    """
    Generate a real nonsymmetric matrix A = Q^T * T * Q with prescribed
    eigenstructure (eigenvalues are diagonal of T).

    Parameters
    ----------
    n : int
    lambda_mean, lambda_dev : float
        Parameters for eigenvalue distribution.

    Returns
    -------
    A, Q, T : ndarray
    """
    T = r8triangular_upper_gen(n, lambda_mean, lambda_dev)
    Q = r8mat_orth_uniform(n)
    A = Q.T @ T @ Q
    return A, Q, T


def generalized_eigenvalue_buckling(K, K_sigma, n_modes=5):
    """
    Solve the linearized buckling eigenvalue problem.

    (K + lambda * K_sigma) * phi = 0
    => K^{-1} K_sigma phi = -1/lambda phi

    Parameters
    ----------
    K : ndarray, shape (n, n)
        Tangent stiffness (must be positive definite).
    K_sigma : ndarray, shape (n, n)
        Geometric stiffness.
    n_modes : int
        Number of buckling modes to extract.

    Returns
    -------
    lambdas : ndarray
        Buckling load factors (positive values).
    modes : ndarray
        Mode shapes, columns are eigenvectors.
    """
    n = K.shape[0]
    n_modes = min(n_modes, n)

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        # Regularize if singular
        K_reg = K + 1e-8 * np.eye(n) * np.max(np.abs(K))
        K_inv = np.linalg.inv(K_reg)

    M = K_inv @ K_sigma
    # M may be nonsymmetric; use real Schur if needed, but for
    # structural applications K and K_sigma are symmetric, so M is
    # symmetric in the K-inner product. Use eigh on the symmetric part.
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)

    # Buckling factors: lambda = -1/mu where mu are eigenvalues of M
    # We want the smallest positive lambda
    with np.errstate(divide='ignore', invalid='ignore'):
        lambdas = -1.0 / eigvals

    # Sort by magnitude and keep positive ones
    valid = lambdas > 0
    if not np.any(valid):
        return np.array([]), np.zeros((n, 0))

    lambdas_valid = lambdas[valid]
    eigvecs_valid = eigvecs[:, valid]
    idx = np.argsort(lambdas_valid)
    lambdas_sorted = lambdas_valid[idx][:n_modes]
    modes_sorted = eigvecs_valid[:, idx][:, :n_modes]

    # Normalize modes
    for i in range(modes_sorted.shape[1]):
        norm = np.linalg.norm(modes_sorted[:, i])
        if norm > 1e-14:
            modes_sorted[:, i] /= norm

    return lambdas_sorted, modes_sorted


def modal_analysis(K, M_mass, n_modes=10):
    """
    Free vibration modal analysis.

    Solve K * phi = omega^2 * M_mass * phi

    Parameters
    ----------
    K : ndarray
        Stiffness matrix.
    M_mass : ndarray
        Mass matrix.
    n_modes : int

    Returns
    -------
    frequencies : ndarray
        Natural frequencies (Hz).
    modes : ndarray
        Mode shapes.
    """
    n = K.shape[0]
    n_modes = min(n_modes, n)

    K_sym = 0.5 * (K + K.T)
    M_sym = 0.5 * (M_mass + M_mass.T)

    # Add small regularization for numerical stability
    reg = 1e-12 * np.eye(n)
    K_sym += reg * np.max(np.abs(K_sym))
    M_sym += reg * np.max(np.abs(M_sym))

    eigvals, eigvecs = np.linalg.eigh(K_sym, M_sym)

    # Natural frequencies in Hz: f = sqrt(lambda) / (2*pi)
    # Keep positive eigenvalues
    positive = eigvals > 1e-10
    eigvals_pos = eigvals[positive]
    eigvecs_pos = eigvecs[:, positive]

    idx = np.argsort(eigvals_pos)
    omega = np.sqrt(eigvals_pos[idx][:n_modes])
    frequencies = omega / (2.0 * np.pi)
    modes = eigvecs_pos[:, idx][:, :n_modes]

    # Mass-normalize modes: phi^T M phi = 1
    for i in range(modes.shape[1]):
        m_norm = np.sqrt(modes[:, i] @ M_sym @ modes[:, i])
        if m_norm > 1e-14:
            modes[:, i] /= m_norm

    return frequencies, modes


def damage_sensitive_buckling(K_undamaged, K_damaged, K_sigma,
                              damage_ratio_threshold=0.1):
    """
    Analyze how damage affects buckling capacity.

    Computes buckling load factors for undamaged and damaged states,
    and identifies critical plies based on stiffness reduction.

    Parameters
    ----------
    K_undamaged : ndarray
    K_damaged : ndarray
    K_sigma : ndarray
    damage_ratio_threshold : float

    Returns
    -------
    report : dict
    """
    lambda_u, modes_u = generalized_eigenvalue_buckling(K_undamaged, K_sigma, n_modes=3)
    lambda_d, modes_d = generalized_eigenvalue_buckling(K_damaged, K_sigma, n_modes=3)

    report = {
        'buckling_undamaged': lambda_u[0] if len(lambda_u) > 0 else np.inf,
        'buckling_damaged': lambda_d[0] if len(lambda_d) > 0 else np.inf,
        'reduction_ratio': 0.0,
        'critical': False
    }

    if len(lambda_u) > 0 and len(lambda_d) > 0 and lambda_u[0] > 0:
        reduction = (lambda_u[0] - lambda_d[0]) / lambda_u[0]
        report['reduction_ratio'] = reduction
        report['critical'] = (reduction > damage_ratio_threshold)

    return report
