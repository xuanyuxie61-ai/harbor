# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


HBARC = 197.3269804
K_B = 8.617333262e-11


def beta_decay_rates(temperature, rho_n, rho_p, electron_chemical_potential):
    if temperature <= 0.0:
        return 0.0, 0.0


    G_F = 1.1663787e-11


    m_n = 939.565
    m_p = 938.272
    k_fn = (3.0 * np.pi**2 * rho_n)**(1.0/3.0)
    k_fp = (3.0 * np.pi**2 * rho_p)**(1.0/3.0)


    mu_n = m_n + k_fn**2 / (2.0 * m_n)
    mu_p = m_p + k_fp**2 / (2.0 * m_p)

    Q = mu_n - mu_p - electron_chemical_potential



    eta = Q / temperature


    if eta > 10.0:
        f_plus = eta**5 / 120.0
        f_minus = 0.0
    elif eta < -10.0:
        f_plus = 0.0
        f_minus = (-eta)**5 / 120.0
    else:
        f_plus = np.exp(eta) / (1.0 + np.exp(eta)) * (temperature / 1.0)**5
        f_minus = np.exp(-eta) / (1.0 + np.exp(-eta)) * (temperature / 1.0)**5

    prefactor = G_F**2 / (np.pi**3)
    lambda_plus = prefactor * f_plus * 1e42
    lambda_minus = prefactor * f_minus * 1e42


    lambda_plus = max(0.0, min(lambda_plus, 1e20))
    lambda_minus = max(0.0, min(lambda_minus, 1e20))

    return lambda_plus, lambda_minus


def diffusion_coefficient(temperature, viscosity, radius):
    if viscosity <= 0.0 or radius <= 0.0:
        return 0.0
    D = K_B * temperature / (6.0 * np.pi * viscosity * radius)
    return max(0.0, D)


def fd_reaction_diffusion_1d(rho_n0, rho_p0, dx, dt, n_steps, D_n, D_p,
                              lambda_plus, lambda_minus, bc_type='neumann'):
    N = len(rho_n0)
    if len(rho_p0) != N:
        raise ValueError("rho_n0和rho_p0长度必须相同")
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx和dt必须为正")

    rho_n = np.array(rho_n0, dtype=float)
    rho_p = np.array(rho_p0, dtype=float)


    cfl_n = D_n * dt / dx**2
    cfl_p = D_p * dt / dx**2
    if cfl_n > 0.5 or cfl_p > 0.5:

        dt_new = 0.4 * dx**2 / max(D_n, D_p)
        n_steps = int(n_steps * dt / dt_new) + 1
        dt = dt_new

    history_n = [rho_n.copy()]
    history_p = [rho_p.copy()]

    lambda_plus = np.asarray(lambda_plus)
    lambda_minus = np.asarray(lambda_minus)
    if lambda_plus.ndim == 0:
        lambda_plus = np.full(N, lambda_plus)
    if lambda_minus.ndim == 0:
        lambda_minus = np.full(N, lambda_minus)

    for _ in range(n_steps):
        rho_n_new = rho_n.copy()
        rho_p_new = rho_p.copy()

        for i in range(1, N - 1):

            lap_n = (rho_n[i + 1] - 2.0 * rho_n[i] + rho_n[i - 1]) / dx**2
            lap_p = (rho_p[i + 1] - 2.0 * rho_p[i] + rho_p[i - 1]) / dx**2


            R_n = lambda_plus[i] * rho_p[i] - lambda_minus[i] * rho_n[i]
            R_p = -lambda_plus[i] * rho_p[i] + lambda_minus[i] * rho_n[i]


            rho_n_new[i] = rho_n[i] + dt * (D_n * lap_n + R_n)
            rho_p_new[i] = rho_p[i] + dt * (D_p * lap_p + R_p)


        if bc_type == 'neumann':
            rho_n_new[0] = rho_n_new[1]
            rho_n_new[-1] = rho_n_new[-2]
            rho_p_new[0] = rho_p_new[1]
            rho_p_new[-1] = rho_p_new[-2]
        elif bc_type == 'dirichlet':
            rho_n_new[0] = rho_n[0]
            rho_n_new[-1] = rho_n[-1]
            rho_p_new[0] = rho_p[0]
            rho_p_new[-1] = rho_p[-1]
        elif bc_type == 'periodic':
            rho_n_new[0] = rho_n_new[-2]
            rho_n_new[-1] = rho_n_new[1]
            rho_p_new[0] = rho_p_new[-2]
            rho_p_new[-1] = rho_p_new[1]


        rho_n_new = np.maximum(rho_n_new, 0.0)
        rho_p_new = np.maximum(rho_p_new, 0.0)

        rho_n = rho_n_new
        rho_p = rho_p_new

        if len(history_n) < 1000:
            history_n.append(rho_n.copy())
            history_p.append(rho_p.copy())

    return rho_n, rho_p, np.array(history_n), np.array(history_p)


def fe_reaction_diffusion_2d(nodes, elements, rho_n0, rho_p0, dt, n_steps,
                             D_n, D_p, lambda_plus, lambda_minus,
                             bc_nodes=None):
    n_nodes = nodes.shape[0]
    rho_n = np.array(rho_n0, dtype=float)
    rho_p = np.array(rho_p0, dtype=float)


    m_hat = np.zeros(n_nodes)
    K = csr_matrix((n_nodes, n_nodes))

    row_ind = []
    col_ind = []
    data_k = []

    for elem in range(elements.shape[0]):
        idx = elements[elem]
        xi, yi = nodes[idx[0]]
        xj, yj = nodes[idx[1]]
        xk, yk = nodes[idx[2]]

        area = abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)) / 2.0
        if area < 1e-15:
            continue


        m_i = area / 3.0
        for i in range(3):
            m_hat[idx[i]] += m_i



        h1 = (xi - xj) * (yk - yj) - (xk - xj) * (yi - yj)
        h2 = (xj - xk) * (yi - yk) - (xi - xk) * (yj - yk)
        h3 = (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi)


        h1 = max(abs(h1), 1e-15) * np.sign(h1) if h1 != 0 else 1e-15
        h2 = max(abs(h2), 1e-15) * np.sign(h2) if h2 != 0 else 1e-15
        h3 = max(abs(h3), 1e-15) * np.sign(h3) if h3 != 0 else 1e-15

        s1 = (yj - yi) * (yk - yj) + (xi - xj) * (xj - xk)
        s2 = (yj - yi) * (yi - yk) + (xi - xj) * (xk - xi)
        s3 = (yk - yj) * (yi - yk) + (xj - xk) * (xk - xi)
        t1 = (yj - yi)**2 + (xi - xj)**2
        t2 = (yk - yj)**2 + (xj - xk)**2
        t3 = (yi - yk)**2 + (xk - xi)**2


        local_k = {
            (idx[0], idx[0]): area * t2 / (h1 * h1),
            (idx[1], idx[1]): area * t3 / (h2 * h2),
            (idx[2], idx[2]): area * t1 / (h3 * h3),
            (idx[0], idx[1]): area * s3 / (h1 * h2),
            (idx[1], idx[0]): area * s3 / (h1 * h2),
            (idx[0], idx[2]): area * s2 / (h1 * h3),
            (idx[2], idx[0]): area * s2 / (h1 * h3),
            (idx[1], idx[2]): area * s1 / (h2 * h3),
            (idx[2], idx[1]): area * s1 / (h2 * h3),
        }

        for (i, j), val in local_k.items():
            row_ind.append(i)
            col_ind.append(j)
            data_k.append(val)

    K = csr_matrix((data_k, (row_ind, col_ind)), shape=(n_nodes, n_nodes))


    m_inv = 1.0 / (m_hat + 1e-15)



    I = csr_matrix(np.eye(n_nodes))

    for step in range(n_steps):

        R_n = lambda_plus * rho_p - lambda_minus * rho_n
        R_p = -lambda_plus * rho_p + lambda_minus * rho_n


        diff_n = -K.dot(rho_n)
        diff_p = -K.dot(rho_p)


        rho_n = rho_n + dt * m_inv * (diff_n + R_n)
        rho_p = rho_p + dt * m_inv * (diff_p + R_p)


        if bc_nodes is not None:
            rho_n[bc_nodes] = rho_n0[bc_nodes]
            rho_p[bc_nodes] = rho_p0[bc_nodes]


        rho_n = np.maximum(rho_n, 0.0)
        rho_p = np.maximum(rho_p, 0.0)


        if not np.all(np.isfinite(rho_n)) or not np.all(np.isfinite(rho_p)):
            raise RuntimeError(f"FE求解在step={step}发散")

    return rho_n, rho_p


def entropy_production_rate(rho_n, rho_p, grad_mu_n, grad_mu_p, T, D_n, D_p):
    if T <= 0.0:
        return 0.0
    J_n = -D_n * rho_n * grad_mu_n / T
    J_p = -D_p * rho_p * grad_mu_p / T
    X_n = -grad_mu_n
    X_p = -grad_mu_p
    sigma = J_n * X_n + J_p * X_p
    return sigma


if __name__ == '__main__':

    N = 100
    x = np.linspace(0, 10, N)
    rho_n0 = np.exp(-(x - 5)**2)
    rho_p0 = np.ones(N) * 0.1
    rho_n, rho_p, _, _ = fd_reaction_diffusion_1d(
        rho_n0, rho_p0, dx=x[1]-x[0], dt=0.001, n_steps=100,
        D_n=1.0, D_p=0.5, lambda_plus=0.1, lambda_minus=0.05
    )
    print(f"1D FD test: rho_n range = [{rho_n.min():.4f}, {rho_n.max():.4f}]")
