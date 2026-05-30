# -*- coding: utf-8 -*-

import numpy as np
from constants import G_PAIRING


def bcs_occupation(epsilon, lambda_, Delta):
    E = np.sqrt((epsilon - lambda_) ** 2 + Delta ** 2)

    E = np.where(E < 1e-14, 1e-14, E)
    u2 = 0.5 * (1.0 + (epsilon - lambda_) / E)
    v2 = 1.0 - u2

    u2 = np.clip(u2, 0.0, 1.0)
    v2 = np.clip(v2, 0.0, 1.0)
    return u2, v2, E


def conjugate_gradient_solve(A, b, x0=None, tol=1e-12, maxiter=None):
    n = b.size
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    b = b.reshape(n)
    x = x.reshape(n)

    r = b - A @ x
    p = r.copy()
    rsold = float(r @ r)

    for k in range(maxiter):
        Ap = A @ p
        alpha = rsold / max(float(p @ Ap), 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(r @ r)
        if np.sqrt(rsnew) < tol:
            return x, {'iterations': k + 1, 'residual': np.sqrt(rsnew)}
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, {'iterations': maxiter, 'residual': np.sqrt(rsnew), 'converged': False}


def solve_hfb_bcs(epsilon, target_N, G=None,
                  Delta0=1.0, lambda0=None,
                  tol=1e-10, max_iter=200):
    epsilon = np.asarray(epsilon, dtype=float)
    n_sp = epsilon.size
    if n_sp == 0:
        raise ValueError("Empty single-particle spectrum.")

    if G is None:

        A_est = max(int(target_N * 2), 1)
        G = 25.0 / A_est

    if lambda0 is None:
        lambda_ = float(np.median(epsilon))
    else:
        lambda_ = float(lambda0)

    Delta = float(Delta0)

    for it in range(max_iter):
        u2, v2, E = bcs_occupation(epsilon, lambda_, Delta)


        Delta_new = G * np.sum(u2 * v2)

        def particle_number(lambda_):
            _, v2_tmp, _ = bcs_occupation(epsilon, lambda_, Delta_new)
            return 2.0 * np.sum(v2_tmp)


        N_current = particle_number(lambda_)
        if abs(N_current - target_N) > 0.01:

            lambda_plus = lambda_ + 1.0
            lambda_minus = lambda_ - 1.0
            N_plus = particle_number(lambda_plus)
            N_minus = particle_number(lambda_minus)
            for _ in range(50):
                if abs(N_current - target_N) < 0.001:
                    break

                dN = N_plus - N_minus
                if abs(dN) < 1e-12:
                    break
                lambda_new = lambda_minus + (target_N - N_minus) * (lambda_plus - lambda_minus) / dN
                lambda_new = max(epsilon.min() - 5.0, min(epsilon.max() + 5.0, lambda_new))
                N_new = particle_number(lambda_new)
                if N_new > target_N:
                    lambda_plus = lambda_new
                    N_plus = N_new
                else:
                    lambda_minus = lambda_new
                    N_minus = N_new
                lambda_ = lambda_new
                N_current = N_new

        Delta_old = Delta
        Delta = Delta_new

        if abs(Delta - Delta_old) < tol and abs(N_current - target_N) < 0.01:
            converged = True
            break
    else:
        converged = False

    u2, v2, E = bcs_occupation(epsilon, lambda_, Delta)


    E_pair = -(Delta ** 2) / G if G > 0 else 0.0


    E_total = 2.0 * np.sum(v2 * epsilon) + E_pair


    s = -2.0 * np.sum(u2 * np.log(np.clip(u2, 1e-30, 1.0))
                     + v2 * np.log(np.clip(v2, 1e-30, 1.0)))

    return {
        'lambda': lambda_,
        'Delta': Delta,
        'u2': u2,
        'v2': v2,
        'E': E,
        'E_pair': E_pair,
        'E_total': E_total,
        'entropy': s,
        'converged': converged,
        'iterations': it + 1,
        'particle_number': 2.0 * np.sum(v2)
    }


def hfb_density_matrix(u2, v2, phi):

    weights = np.sqrt(np.clip(v2, 0.0, 1.0))
    weighted = phi * weights[np.newaxis, :]
    rho = weighted @ weighted.T
    return rho


def hfb_pairing_tensor(u2, v2, phi):
    uv = np.sqrt(np.clip(u2 * v2, 0.0, 0.25))
    weighted = phi * uv[np.newaxis, :]
    kappa = weighted @ weighted.T
    return kappa
