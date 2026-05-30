# -*- coding: utf-8 -*-

import numpy as np


def st_to_ge(nst, ist, jst, Ast):
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    Ast = np.asarray(Ast, dtype=float)

    m = int(np.max(ist)) if len(ist) > 0 else 0
    n = int(np.max(jst)) if len(jst) > 0 else 0
    Age = np.zeros((m, n), dtype=float)

    for kst in range(nst):
        i = ist[kst] - 1
        j = jst[kst] - 1
        if 0 <= i < m and 0 <= j < n:
            Age[i, j] += Ast[kst]

    return Age


def assemble_sparse_st(connections, values, n_nodes):
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for local_K, global_dof in zip(values, connections):
        n_loc = len(global_dof)
        for i in range(n_loc):
            for j in range(n_loc):
                gi = global_dof[i]
                gj = global_dof[j]
                if 0 <= gi < n_nodes and 0 <= gj < n_nodes:
                    A[gi, gj] += local_K[i, j]
    return A


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    b = np.asarray(b, dtype=float)
    A = np.asarray(A, dtype=float)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))

    if rs_old < tol * tol:
        return x, {"iter": 0, "residual": np.sqrt(rs_old)}

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (np.dot(p, Ap) + 1e-15)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))

        if np.sqrt(rs_new) < tol:
            return x, {"iter": k + 1, "residual": np.sqrt(rs_new)}

        beta = rs_new / (rs_old + 1e-15)
        p = r + beta * p
        rs_old = rs_new

    return x, {"iter": max_iter, "residual": np.sqrt(rs_old), "converged": False}


def bicgstab(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    b = np.asarray(b, dtype=float)
    A = np.asarray(A, dtype=float)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    r0 = r.copy()
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    p = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)

    for k in range(max_iter):
        rho = np.dot(r0, r)
        if abs(rho) < 1e-15:
            return x, {"iter": k, "residual": np.linalg.norm(r), "converged": False}

        beta = (rho / (rho_old + 1e-15)) * (alpha / (omega + 1e-15))
        p = r + beta * (p - omega * v)
        v = A @ p
        alpha = rho / (np.dot(r0, v) + 1e-15)
        s = r - alpha * v

        if np.linalg.norm(s) < tol:
            x = x + alpha * p
            return x, {"iter": k + 1, "residual": np.linalg.norm(s), "converged": True}

        t = A @ s
        omega = np.dot(t, s) / (np.dot(t, t) + 1e-15)
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_old = rho

        if np.linalg.norm(r) < tol:
            return x, {"iter": k + 1, "residual": np.linalg.norm(r), "converged": True}

    return x, {"iter": max_iter, "residual": np.linalg.norm(r), "converged": False}


def solve_pressure_poisson(p, div_u, dx, dy, dz, tol=1e-8, max_iter=500):
    nx, ny, nz = p.shape


    def apply_laplacian(phi):
        result = np.zeros_like(phi)
        result[1:-1, 1:-1, 1:-1] = (
            (phi[2:, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / dx ** 2
            + (phi[1:-1, 2:, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / dy ** 2
            + (phi[1:-1, 1:-1, 2:] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / dz ** 2
        )
        return result


    b = div_u.flatten()
    x0 = p.flatten()


    n = len(b)
    x = x0.copy()
    r = b - apply_laplacian(x.reshape(nx, ny, nz)).flatten()


    diag_val = -2.0 * (1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2)
    if abs(diag_val) < 1e-15:
        diag_val = -1.0

    z = r / diag_val
    p_vec = z.copy()
    rz_old = np.dot(r, z)

    for k in range(max_iter):
        Ap = apply_laplacian(p_vec.reshape(nx, ny, nz)).flatten()
        alpha = rz_old / (np.dot(p_vec, Ap) + 1e-15)
        x = x + alpha * p_vec
        r = r - alpha * Ap

        if np.linalg.norm(r) < tol:
            break

        z = r / diag_val
        rz_new = np.dot(r, z)
        beta = rz_new / (rz_old + 1e-15)
        p_vec = z + beta * p_vec
        rz_old = rz_new

    return x.reshape(nx, ny, nz)
