# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()
    if max_iter is None:
        max_iter = n

    if callable(A):
        Av = A
    else:
        A_mat = np.asarray(A, dtype=float)
        def Av(v):
            return A_mat.dot(v)

    r = b - Av(x)
    p = r.copy()
    rsold = float(r.dot(r))
    bnorm = norm(b)
    if bnorm < 1e-15:
        bnorm = 1.0

    for k in range(max_iter):
        Ap = Av(p)
        pAp = float(p.dot(Ap))
        if abs(pAp) < 1e-20:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(r.dot(r))
        if np.sqrt(rsnew) / bnorm < tol:
            return x, np.sqrt(rsnew), k + 1
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, norm(r), max_iter


def mult_givens(c, s, k, g):
    g = np.asarray(g, dtype=float).copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def gmres_restart(Ax, b, x0=None, max_iter=100, restart=30, tol_abs=1e-10, tol_rel=1e-6):
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()

    if restart > n:
        restart = n

    delta = 0.001
    r = b - Ax(x)
    rho = norm(r)
    bnorm = norm(b)
    if bnorm < 1e-15:
        bnorm = 1.0
    rho_tol = rho * tol_rel

    itr_used = 0

    for itr in range(max_iter):
        r = b - Ax(x)
        rho = norm(r)
        if rho <= rho_tol and rho <= tol_abs:
            break

        v = np.zeros((n, restart + 1))
        v[:, 0] = r / rho if rho > 1e-15 else r
        g = np.zeros(restart + 1)
        g[0] = rho
        H = np.zeros((restart + 1, restart))
        c = np.zeros(restart)
        s = np.zeros(restart)

        for k in range(restart):
            w = Ax(v[:, k])
            av = norm(w)
            for j in range(k + 1):
                H[j, k] = np.dot(v[:, j], w)
                w -= H[j, k] * v[:, j]
            H[k + 1, k] = norm(w)

            if av + delta * H[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], w)
                    H[j, k] += htmp
                    w -= htmp * v[:, j]
                H[k + 1, k] = norm(w)

            if H[k + 1, k] > 1e-15:
                v[:, k + 1] = w / H[k + 1, k]

            y = H[:k + 2, k].copy()
            for j in range(k):
                y = mult_givens(c[j], s[j], j, y)
            H[:k + 2, k] = y

            mu = np.sqrt(H[k, k] ** 2 + H[k + 1, k] ** 2)
            if mu < 1e-15:
                mu = 1.0
            c[k] = H[k, k] / mu
            s[k] = -H[k + 1, k] / mu
            H[k, k] = c[k] * H[k, k] - s[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g = mult_givens(c[k], s[k], k, g)
            rho = abs(g[k + 1])
            itr_used += 1

            if rho <= rho_tol and rho <= tol_abs:
                break

        k_use = k if (rho <= rho_tol and rho <= tol_abs) else restart - 1
        y = np.zeros(k_use + 1)
        y[k_use] = g[k_use] / H[k_use, k_use]
        for i in range(k_use - 1, -1, -1):
            y[i] = (g[i] - np.dot(H[i, i + 1:k_use + 1], y[i + 1:k_use + 1])) / H[i, i]
        x += v[:, :k_use + 1].dot(y)

        if rho <= rho_tol and rho <= tol_abs:
            break

    return x, rho, itr_used


def cholesky_factor(a, n, eta=1e-09):
    nn = len(a)
    req = n * (n + 1) // 2
    if n <= 0:
        return np.array([]), 0, 1
    if nn < req:
        return np.array([]), 0, 3

    u = np.zeros(req, dtype=float)
    nullty = 0
    j = 0
    k = 0
    ii = 0

    for icol in range(n):
        ii += icol + 1
        x = eta * eta * a[ii - 1]
        l = 0
        kk = 0
        for irow in range(icol + 1):
            kk += irow + 1
            k += 1
            w = a[k - 1]
            m = j
            for i in range(irow):
                l += 1
                w -= u[l - 1] * u[m]
                m += 1
            l += 1
            if irow == icol:
                break
            if abs(u[l - 1]) > 1e-15:
                u[k - 1] = w / u[l - 1]
            else:
                u[k - 1] = 0.0
                if abs(x * a[k - 1]) < w * w:
                    return np.array([]), 0, 2
        if abs(w) <= abs(eta * a[k - 1]):
            u[k - 1] = 0.0
            nullty += 1
        else:
            if w < 0.0:
                return np.array([]), 0, 2
            u[k - 1] = np.sqrt(w)
        j += icol + 1

    return u, nullty, 0


def cholesky_solve_dense(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]

    a = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            a[idx] = A[i, j]
            idx += 1
    u, nullty, ifault = cholesky_factor(a, n)
    if ifault != 0:

        return np.linalg.solve(A, b)


    U = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            U[i, j] = u[idx]
            idx += 1


    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= U[j, i] * y[j]
        if abs(U[i, i]) < 1e-15:
            y[i] = 0.0
        else:
            y[i] /= U[i, i]


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        if abs(U[i, i]) < 1e-15:
            x[i] = 0.0
        else:
            x[i] /= U[i, i]
    return x
