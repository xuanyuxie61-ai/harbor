
import numpy as np


def mult_givens(c, s, k, g):
    g = g.copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def ax_crs(a, ia, ja, x, n, nz_num):
    y = np.zeros(n)
    for k in range(nz_num):
        y[ia[k]] += a[k] * x[ja[k]]
    return y


def mgmres(a, ia, ja, x0, rhs, n, nz_num, itr_max, mr, tol_abs, tol_rel, verbose=False):
    if n < 1:
        return x0.copy()
    if mr > n:
        mr = n
    if mr < 1:
        mr = min(10, n)

    delta = 0.001
    x = x0.copy()
    rho_tol = None
    itr_used = 0

    for itr in range(1, itr_max + 1):
        r = rhs - ax_crs(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if verbose:
            print(f"  ITR = {itr:8d}  Residual = {rho:e}")

        if itr == 1:
            rho_tol = rho * tol_rel

        v = np.zeros((n, mr + 1))
        v[:, 0] = r / rho if rho > 1e-30 else r

        g = np.zeros(mr + 1)
        g[0] = rho

        h = np.zeros((mr + 1, mr))
        c = np.zeros(mr)
        s = np.zeros(mr)

        converged = False
        for k in range(mr):
            k_copy = k
            v[:, k + 1] = ax_crs(a, ia, ja, v[:, k], n, nz_num)
            av = np.linalg.norm(v[:, k + 1])

            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])


            if av + delta * h[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] > 1e-15:
                v[:, k + 1] = v[:, k + 1] / h[k + 1, k]


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
            g = mult_givens(c[k], s[k], k, g)

            rho = abs(g[k + 1])
            itr_used += 1

            if verbose:
                print(f"  K =   {k + 1:8d}  Residual = {rho:e}")

            if rho <= rho_tol and rho <= tol_abs:
                converged = True
                break



        k_solve = k_copy - 1
        if converged:
            k_solve = k - 1
        y = np.zeros(k_solve + 1)
        for i in range(k_solve, -1, -1):
            denom = h[i, i]
            if abs(denom) < 1e-30:
                y[i] = 0.0
            else:
                y[i] = (g[i] - np.dot(h[i, i + 1:k_solve + 1], y[i + 1:k_solve + 1])) / denom

        for i in range(n):
            x[i] += np.dot(v[i, :k_solve + 1], y[:k_solve + 1])

        if converged:
            break

    if verbose:
        print(f"\nMGMRES")
        print(f"  Iterations = {itr_used}")
        print(f"  Final residual = {rho:e}")

    return x


def build_dispersion_matrix_crs(n, dt, beta2, beta3, beta4=0.0):
    if n < 5 or dt <= 0:
        raise ValueError("build_dispersion_matrix_crs: invalid parameters")

    rows = []
    cols = []
    vals = []

    c2 = 1j * beta2 / (2.0 * dt ** 2)
    c3 = -1j * beta3 / (12.0 * dt ** 3)
    c4 = beta4 / (24.0 * dt ** 4) if beta4 != 0.0 else 0.0

    for i in range(n):

        for di, coef in [(-1, c2), (0, -2.0 * c2), (1, c2)]:
            j = i + di
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
                vals.append(coef)


        for di, coef in [(-2, -c3), (-1, 2.0 * c3), (1, -2.0 * c3), (2, c3)]:
            j = i + di
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
                vals.append(coef)


        if beta4 != 0.0:
            for di, coef in [(-2, c4), (-1, -4.0 * c4), (0, 6.0 * c4), (1, -4.0 * c4), (2, c4)]:
                j = i + di
                if 0 <= j < n:
                    rows.append(i)
                    cols.append(j)
                    vals.append(coef)


    from collections import defaultdict
    merged = defaultdict(float)
    for r, c, v in zip(rows, cols, vals):
        merged[(r, c)] += v

    nz_num = len(merged)
    a = np.zeros(nz_num, dtype=complex)
    ia = np.zeros(nz_num, dtype=int)
    ja = np.zeros(nz_num, dtype=int)

    for idx, ((r, c), v) in enumerate(sorted(merged.items())):
        a[idx] = v
        ia[idx] = r
        ja[idx] = c

    return a, ia, ja, nz_num
