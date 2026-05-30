# -*- coding: utf-8 -*-

import numpy as np


def box_muller_normal(n):
    n2 = (n + 1) // 2
    u1 = np.random.rand(n2)
    u2 = np.random.rand(n2)

    r = np.sqrt(-2.0 * np.log(u1 + 1e-15))
    theta = 2.0 * np.pi * u2
    z = np.zeros(2 * n2)
    z[0::2] = r * np.cos(theta)
    z[1::2] = r * np.sin(theta)
    return z[:n]


def wishart_variate_chol(D, n, np_dim):
    D = np.asarray(D, dtype=np.float64)
    np_dim = int(np_dim)
    nnp = np_dim * (np_dim + 1) // 2
    sb = box_muller_normal(nnp)
    sa = np.zeros(nnp, dtype=np.float64)


    ns = 0
    for i in range(1, np_dim + 1):
        df = np_dim - i + 1
        ns += i
        u1 = 2.0 / (9.0 * df)
        u2 = 1.0 - u1
        u1 = np.sqrt(u1)
        sb[ns - 1] = np.sqrt(df * abs((u2 + sb[ns - 1] * u1) ** 3))

    rn = float(n)

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


    SA = np.zeros((np_dim, np_dim), dtype=np.float64)
    idx = 0
    for j in range(np_dim):
        for i in range(j + 1):
            SA[i, j] = sa_out[idx]
            idx += 1




    W = SA @ SA.T / rn
    return W


def generate_random_diffusion_field(x, d_stochastic=3, mean_val=0.1,
                                    fluctuation=0.05, correlation_length=0.3):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    L = x[-1] - x[0]




    X, Y = np.meshgrid(x, x)
    C = np.exp(-np.abs(X - Y) / (correlation_length * L + 1e-15))



    eig_C = np.linalg.eigvalsh(C)
    min_eig = np.min(eig_C)
    reg = max(1e-6, -min_eig + 1e-6)
    C_reg = C + reg * np.eye(N)
    D_chol = np.linalg.cholesky(C_reg)
    W = wishart_variate_chol(D_chol, n=min(N, 20), np_dim=N)

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
    nu = nu_base.copy()
    d = len(xi)
    for k in range(d):
        nu += fluctuation * np.sqrt(kl_eigenvalues[k]) * kl_modes[:, k] * xi[k]

    nu = np.clip(nu, 1e-6, 10.0)
    return nu


def monte_carlo_statistic(samples):
    samples = np.asarray(samples, dtype=np.float64)
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    std = np.sqrt(var)
    n = len(samples)
    ci_95 = 1.96 * std / np.sqrt(max(n, 1))
    return {"mean": mean, "variance": var, "std": std, "ci_95": ci_95, "n": n}
