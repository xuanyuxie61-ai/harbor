# -*- coding: utf-8 -*-

import numpy as np
from numpy.polynomial import polynomial as P


def weno5_reconstruct(v):
    v = np.asarray(v, dtype=np.float64)
    nx = v.shape[0]
    v_half = np.zeros_like(v)

    eps = 1e-12

    d0, d1, d2 = 0.1, 0.6, 0.3

    for i in range(2, nx - 2):

        p0 = (1.0 / 3.0) * v[i - 2] - (7.0 / 6.0) * v[i - 1] + (11.0 / 6.0) * v[i]
        p1 = (-1.0 / 6.0) * v[i - 1] + (5.0 / 6.0) * v[i] + (1.0 / 3.0) * v[i + 1]
        p2 = (1.0 / 3.0) * v[i] + (5.0 / 6.0) * v[i + 1] - (1.0 / 6.0) * v[i + 2]


        is0 = (13.0 / 12.0) * (v[i - 2] - 2.0 * v[i - 1] + v[i]) ** 2 \
              + 0.25 * (v[i - 2] - 4.0 * v[i - 1] + 3.0 * v[i]) ** 2
        is1 = (13.0 / 12.0) * (v[i - 1] - 2.0 * v[i] + v[i + 1]) ** 2 \
              + 0.25 * (v[i - 1] - v[i + 1]) ** 2
        is2 = (13.0 / 12.0) * (v[i] - 2.0 * v[i + 1] + v[i + 2]) ** 2 \
              + 0.25 * (3.0 * v[i] - 4.0 * v[i + 1] + v[i + 2]) ** 2


        a0 = d0 / (eps + is0) ** 2
        a1 = d1 / (eps + is1) ** 2
        a2 = d2 / (eps + is2) ** 2
        wsum = a0 + a1 + a2
        w0 = a0 / wsum
        w1 = a1 / wsum
        w2 = a2 / wsum

        v_half[i] = w0 * p0 + w1 * p1 + w2 * p2


    if nx > 2:
        v_half[0] = v[0]
        v_half[1] = v[1]
        v_half[-2] = v[-2]
        v_half[-1] = v[-1]
    return v_half


def weno5_neg_reconstruct(v):
    v = np.asarray(v, dtype=np.float64)
    nx = v.shape[0]
    v_half = np.zeros_like(v)

    eps = 1e-12
    d0, d1, d2 = 0.1, 0.6, 0.3

    for i in range(2, nx - 2):
        p0 = (1.0 / 3.0) * v[i + 2] - (7.0 / 6.0) * v[i + 1] + (11.0 / 6.0) * v[i]
        p1 = (-1.0 / 6.0) * v[i + 1] + (5.0 / 6.0) * v[i] + (1.0 / 3.0) * v[i - 1]
        p2 = (1.0 / 3.0) * v[i] + (5.0 / 6.0) * v[i - 1] - (1.0 / 6.0) * v[i - 2]

        is0 = (13.0 / 12.0) * (v[i + 2] - 2.0 * v[i + 1] + v[i]) ** 2 \
              + 0.25 * (v[i + 2] - 4.0 * v[i + 1] + 3.0 * v[i]) ** 2
        is1 = (13.0 / 12.0) * (v[i + 1] - 2.0 * v[i] + v[i - 1]) ** 2 \
              + 0.25 * (v[i + 1] - v[i - 1]) ** 2
        is2 = (13.0 / 12.0) * (v[i] - 2.0 * v[i - 1] + v[i - 2]) ** 2 \
              + 0.25 * (3.0 * v[i] - 4.0 * v[i - 1] + v[i - 2]) ** 2

        a0 = d0 / (eps + is0) ** 2
        a1 = d1 / (eps + is1) ** 2
        a2 = d2 / (eps + is2) ** 2
        wsum = a0 + a1 + a2
        w0 = a0 / wsum
        w1 = a1 / wsum
        w2 = a2 / wsum

        v_half[i] = w0 * p0 + w1 * p1 + w2 * p2

    if nx > 2:
        v_half[0] = v[0]
        v_half[1] = v[1]
        v_half[-2] = v[-2]
        v_half[-1] = v[-1]
    return v_half


def weno5_derivative(phi, dx, axis=0):
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        vp = weno5_reconstruct(v)
        vm = weno5_neg_reconstruct(v)

        dphi = np.zeros_like(v)

        for i in range(2, nx - 2):
            dphi[i] = 0.5 * ((vp[i] - vp[i - 1]) + (vm[i] - vm[i - 1])) / dx

        dphi[0] = (v[1] - v[0]) / dx
        dphi[1] = (v[2] - v[1]) / dx
        dphi[-2] = (v[-1] - v[-2]) / dx
        dphi[-1] = (v[-1] - v[-2]) / dx
        return dphi
    else:
        dphi = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            vp = weno5_reconstruct(col)
            vm = weno5_neg_reconstruct(col)
            for i in range(2, nx - 2):
                dphi[i, j] = 0.5 * ((vp[i] - vp[i - 1]) + (vm[i] - vm[i - 1])) / dx
            dphi[0, j] = (col[1] - col[0]) / dx
            dphi[1, j] = (col[2] - col[1]) / dx
            dphi[-2, j] = (col[-1] - col[-2]) / dx
            dphi[-1, j] = (col[-1] - col[-2]) / dx
        if axis == 1:
            dphi = dphi.T
        return dphi


def tvd_rk3_step(phi0, dt, rhs_func):
    phi0 = np.asarray(phi0, dtype=np.float64)





    raise NotImplementedError("HOLE_1: TVD-RK3 step implementation missing")


def central_diff_4th(phi, dx, axis=0):
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        d = np.zeros_like(v)
        if nx >= 5:
            for i in range(2, nx - 2):
                d[i] = (-v[i + 2] + 8.0 * v[i + 1] - 8.0 * v[i - 1] + v[i - 2]) / (12.0 * dx)

        if nx >= 3:
            d[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dx)
            d[1] = (v[2] - v[0]) / (2.0 * dx)
            d[-2] = (v[-1] - v[-3]) / (2.0 * dx)
            d[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dx)
        elif nx == 2:
            d[0] = (v[1] - v[0]) / dx
            d[1] = (v[1] - v[0]) / dx
        return d
    else:
        d = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            if nx >= 5:
                for i in range(2, nx - 2):
                    d[i, j] = (-col[i + 2] + 8.0 * col[i + 1] - 8.0 * col[i - 1] + col[i - 2]) / (12.0 * dx)
            if nx >= 3:
                d[0, j] = (-3.0 * col[0] + 4.0 * col[1] - col[2]) / (2.0 * dx)
                d[1, j] = (col[2] - col[0]) / (2.0 * dx)
                d[-2, j] = (col[-1] - col[-3]) / (2.0 * dx)
                d[-1, j] = (3.0 * col[-1] - 4.0 * col[-2] + col[-3]) / (2.0 * dx)
        if axis == 1:
            d = d.T
        return d


def central_diff_2nd(phi, dx, axis=0):
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        d = np.zeros_like(v)
        if nx >= 3:
            for i in range(1, nx - 1):
                d[i] = (v[i + 1] - v[i - 1]) / (2.0 * dx)
            d[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dx)
            d[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dx)
        elif nx == 2:
            d[0] = (v[1] - v[0]) / dx
            d[1] = d[0]
        return d
    else:
        d = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            if nx >= 3:
                for i in range(1, nx - 1):
                    d[i, j] = (col[i + 1] - col[i - 1]) / (2.0 * dx)
                d[0, j] = (-3.0 * col[0] + 4.0 * col[1] - col[2]) / (2.0 * dx)
                d[-1, j] = (3.0 * col[-1] - 4.0 * col[-2] + col[-3]) / (2.0 * dx)
        if axis == 1:
            d = d.T
        return d


def laplacian_2d(phi, dx, dy):
    phi = np.asarray(phi, dtype=np.float64)
    nx, ny = phi.shape
    lap = np.zeros_like(phi)


    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap[i, j] = (phi[i + 1, j] - 2.0 * phi[i, j] + phi[i - 1, j]) / (dx * dx) \
                        + (phi[i, j + 1] - 2.0 * phi[i, j] + phi[i, j - 1]) / (dy * dy)


    for j in range(ny):
        lap[0, j] = lap[1, j]
        lap[-1, j] = lap[-2, j]
    for i in range(nx):
        lap[i, 0] = lap[i, 1]
        lap[i, -1] = lap[i, -2]
    return lap


def cplx_cholesky_decompose(A):
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("cplx_cholesky_decompose: A must be square")

    if not np.allclose(A, A.conj().T, atol=1e-12):
        raise ValueError("cplx_cholesky_decompose: A must be Hermitian")

    L = np.zeros_like(A, dtype=np.complex128)
    for j in range(n):
        diag_val = A[j, j] - np.sum(np.abs(L[j, :j]) ** 2)
        if diag_val.real <= 0:
            raise ValueError("cplx_cholesky_decompose: Matrix is not positive definite")
        L[j, j] = np.sqrt(diag_val)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j].conj())) / L[j, j]
    return L


def cplx_solve_lower_triangular(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.complex128)
    for i in range(n):
        val = b[i] - np.sum(L[i, :i] * y[:i])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("cplx_solve_lower_triangular: zero diagonal element")
        y[i] = val / L[i, i]
    return y


def cplx_solve_upper_triangular(U, b):
    n = U.shape[0]
    x = np.zeros_like(b, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        val = b[i] - np.sum(U[i, i + 1:] * x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise ValueError("cplx_solve_upper_triangular: zero diagonal element")
        x[i] = val / U[i, i]
    return x


def cplx_lu_factor(A):
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("cplx_lu_factor: A must be square")

    L = np.eye(n, dtype=np.complex128)
    U = A.copy()
    P = np.eye(n, dtype=np.complex128)

    for k in range(n - 1):

        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("cplx_lu_factor: singular matrix")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U, P


def cplx_qr_factor(A):
    A = np.asarray(A, dtype=np.complex128)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.complex128)
    R = np.zeros((n, n), dtype=np.complex128)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            raise ValueError("cplx_qr_factor: rank deficient")
        R[j, j] = norm_v
        Q[:, j] = v / norm_v
    return Q, R
