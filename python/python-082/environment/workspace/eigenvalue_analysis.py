
import numpy as np


def r8mat_orth_uniform(n):
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def r8triangular_upper_gen(n, lambda_mean=0.0, lambda_dev=1.0):
    T = np.triu(np.random.randn(n, n))
    diag = lambda_mean + lambda_dev * np.random.randn(n)
    np.fill_diagonal(T, diag)
    return T


def r8nsymm_gen(n, lambda_mean=0.0, lambda_dev=1.0):
    T = r8triangular_upper_gen(n, lambda_mean, lambda_dev)
    Q = r8mat_orth_uniform(n)
    A = Q.T @ T @ Q
    return A, Q, T


def generalized_eigenvalue_buckling(K, K_sigma, n_modes=5):
    n = K.shape[0]
    n_modes = min(n_modes, n)

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:

        K_reg = K + 1e-8 * np.eye(n) * np.max(np.abs(K))
        K_inv = np.linalg.inv(K_reg)

    M = K_inv @ K_sigma



    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)



    with np.errstate(divide='ignore', invalid='ignore'):
        lambdas = -1.0 / eigvals


    valid = lambdas > 0
    if not np.any(valid):
        return np.array([]), np.zeros((n, 0))

    lambdas_valid = lambdas[valid]
    eigvecs_valid = eigvecs[:, valid]
    idx = np.argsort(lambdas_valid)
    lambdas_sorted = lambdas_valid[idx][:n_modes]
    modes_sorted = eigvecs_valid[:, idx][:, :n_modes]


    for i in range(modes_sorted.shape[1]):
        norm = np.linalg.norm(modes_sorted[:, i])
        if norm > 1e-14:
            modes_sorted[:, i] /= norm

    return lambdas_sorted, modes_sorted


def modal_analysis(K, M_mass, n_modes=10):
    n = K.shape[0]
    n_modes = min(n_modes, n)

    K_sym = 0.5 * (K + K.T)
    M_sym = 0.5 * (M_mass + M_mass.T)


    reg = 1e-12 * np.eye(n)
    K_sym += reg * np.max(np.abs(K_sym))
    M_sym += reg * np.max(np.abs(M_sym))

    eigvals, eigvecs = np.linalg.eigh(K_sym, M_sym)



    positive = eigvals > 1e-10
    eigvals_pos = eigvals[positive]
    eigvecs_pos = eigvecs[:, positive]

    idx = np.argsort(eigvals_pos)
    omega = np.sqrt(eigvals_pos[idx][:n_modes])
    frequencies = omega / (2.0 * np.pi)
    modes = eigvecs_pos[:, idx][:, :n_modes]


    for i in range(modes.shape[1]):
        m_norm = np.sqrt(modes[:, i] @ M_sym @ modes[:, i])
        if m_norm > 1e-14:
            modes[:, i] /= m_norm

    return frequencies, modes


def damage_sensitive_buckling(K_undamaged, K_damaged, K_sigma,
                              damage_ratio_threshold=0.1):
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
