
import numpy as np


def mult_givens(c, s, k, g):
    g = np.asarray(g, dtype=np.float64).copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def sparse_mv(a, ia, ja, x, n, nz_num):
    w = np.zeros(n, dtype=np.float64)
    for k in range(nz_num):
        i = ia[k]
        j = ja[k]
        w[i] += a[k] * x[j]
    return w


def restarted_gmres(a, ia, ja, x0, rhs, n, nz_num, itr_max, mr, tol_abs, tol_rel, verbose=0):
    if n <= 0:
        raise ValueError("n必须为正整数")
    if mr <= 0 or mr > n:
        raise ValueError("mr必须满足 0 < mr <= n")
    if nz_num < 0:
        raise ValueError("nz_num不能为负")

    delta = 0.001
    x = np.asarray(x0, dtype=np.float64).copy()
    rhs = np.asarray(rhs, dtype=np.float64)

    if len(x) != n or len(rhs) != n:
        raise ValueError("向量维度必须与n一致")

    itr_used = 0
    rho_tol = None

    for itr in range(1, itr_max + 1):

        r = rhs - sparse_mv(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if verbose:
            print(f"  ITR = {itr:8d}  Residual = {rho:.6e}")

        if itr == 1:
            rho_tol = rho * tol_rel


        v = np.zeros((n, mr + 1), dtype=np.float64)
        if rho < 1e-30:
            converged = True
            final_res = rho
            return x, converged, itr_used, final_res
        v[:, 0] = r / rho

        g = np.zeros(mr + 1, dtype=np.float64)
        g[0] = rho
        h = np.zeros((mr + 1, mr), dtype=np.float64)
        c = np.zeros(mr, dtype=np.float64)
        s = np.zeros(mr, dtype=np.float64)

        k_copy = 0
        for k in range(mr):
            k_copy = k

            v[:, k + 1] = sparse_mv(a, ia, ja, v[:, k], n, nz_num)
            av_norm = np.linalg.norm(v[:, k + 1])


            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])


            if av_norm + delta * h[k + 1, k] <= av_norm + 1e-15:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] > 1e-30:
                v[:, k + 1] /= h[k + 1, k]


            if k > 0:
                y = h[:k + 2, k].copy()
                for j in range(k):
                    y = mult_givens(c[j], s[j], j, y)
                h[:k + 2, k] = y

            mu = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            if mu < 1e-30:
                mu = 1.0
            c[k] = h[k, k] / mu
            s[k] = -h[k + 1, k] / mu
            h[k, k] = c[k] * h[k, k] - s[k] * h[k + 1, k]
            h[k + 1, k] = 0.0
            g[:k + 2] = mult_givens(c[k], s[k], k, g[:k + 2])

            rho = abs(g[k + 1])
            itr_used += 1

            if verbose:
                print(f"  K =   {k + 1:8d}  Residual = {rho:.6e}")

            if rho <= rho_tol and rho <= tol_abs:
                break


        k_solve = k_copy
        if k_solve >= mr:
            k_solve = mr - 1
        y = np.zeros(k_solve + 1, dtype=np.float64)
        y[k_solve] = g[k_solve] / h[k_solve, k_solve]
        for i in range(k_solve - 1, -1, -1):
            y[i] = (g[i] - np.dot(h[i, i + 1:k_solve + 1], y[i + 1:k_solve + 1])) / h[i, i]


        x += v[:, :k_solve + 1] @ y

        if rho <= rho_tol and rho <= tol_abs:
            converged = True
            final_res = rho
            return x, converged, itr_used, final_res

    converged = False
    final_res = rho
    return x, converged, itr_used, final_res


def gmres_dense(A, b, x0=None, tol=1e-10, maxiter=None, restart=None):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)
    if maxiter is None:
        maxiter = n
    if restart is None:
        restart = min(n, 50)


    rowind, colind = np.nonzero(np.abs(A) > 1e-15)
    nz_num = len(rowind)
    a_vals = A[rowind, colind]

    x, converged, itr_used, final_res = restarted_gmres(
        a_vals, rowind.astype(np.int64), colind.astype(np.int64),
        x0, b, n, nz_num, maxiter, restart, tol, tol, verbose=0
    )
    return x, converged, itr_used, final_res
