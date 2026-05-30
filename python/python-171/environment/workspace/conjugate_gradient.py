# -*- coding: utf-8 -*-

import numpy as np
import math






def conjugate_gradient(matvec, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    p = r.copy()
    rs_old = float(r @ r)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(rs_old) / norm_b]

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            if verbose:
                print(f"CG break at iteration {k}: pAp too small.")
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        residual_history.append(math.sqrt(rs_new) / norm_b)

        if math.sqrt(rs_new) / norm_b < tol:
            if verbose:
                print(f"CG converged at iteration {k+1}.")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info






def preconditioned_cg(matvec, preconditioner, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = float(r @ z)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(float(r @ r)) / norm_b]

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            if verbose:
                print(f"PCG break at iteration {k}: pAp too small.")
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner(r)
        rz_new = float(r @ z)
        residual_history.append(math.sqrt(float(r @ r)) / norm_b)

        if math.sqrt(float(r @ r)) / norm_b < tol:
            if verbose:
                print(f"PCG converged at iteration {k+1}.")
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info






def flexible_cg(matvec, preconditioner, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = float(r @ z)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(float(r @ r)) / norm_b]
    z_list = [z.copy()]
    Az_list = []

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner(r)
        z_list.append(z.copy())
        Az_list.append(Ap.copy())

        residual_history.append(math.sqrt(float(r @ r)) / norm_b)
        if math.sqrt(float(r @ r)) / norm_b < tol:
            break


        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info






def restarted_cg(matvec, b, restart=50, max_iter=1000, tol=1e-10, verbose=False):
    b = np.asarray(b, dtype=float).flatten()
    x = np.zeros(b.size, dtype=float)
    total_iter = 0
    full_history = []

    while total_iter < max_iter:
        x, info = conjugate_gradient(matvec, b, x0=x, max_iter=restart, tol=tol, verbose=False)
        total_iter += info['iterations']
        full_history.extend(info['residual_history'])
        if info['converged']:
            break
        if info['iterations'] == 0:
            break

    info = {
        'iterations': total_iter,
        'residual_history': full_history,
        'final_residual': full_history[-1] if full_history else 1.0,
        'converged': full_history[-1] < tol if full_history else False
    }
    return x, info






def solve_with_format(fmt, data, b, extra=None, preconditioner=None, tol=1e-10, max_iter=None):
    from sparse_matrix import SparseMatrixOperator

    if extra is None:
        extra = {}
    n = len(b)
    op = SparseMatrixOperator(fmt, data, n, extra)
    matvec = op.matvec

    if preconditioner is None:
        x, info = conjugate_gradient(matvec, b, max_iter=max_iter, tol=tol)
    else:
        x, info = preconditioned_cg(matvec, preconditioner, b, max_iter=max_iter, tol=tol)
    return x, info
