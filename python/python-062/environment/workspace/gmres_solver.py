
import numpy as np


def givens_rotation(v1, v2):
    if abs(v1) < 1e-15:
        cs = 0.0
        sn = 1.0
    else:
        t = np.sqrt(v1**2 + v2**2)
        cs = abs(v1) / t
        sn = cs * v2 / v1
    return cs, sn


def apply_givens_rotation(h, cs, sn, k):
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp

    cs_k, sn_k = givens_rotation(h[k], h[k + 1])
    h[k] = cs_k * h[k] + sn_k * h[k + 1]
    h[k + 1] = 0.0
    return h, cs_k, sn_k


def arnoldi_iteration(A_func, Q, k):
    n = Q.shape[0]
    q = A_func(Q[:, k])
    h = np.zeros(k + 2, dtype=np.float64)

    for i in range(k + 1):
        h[i] = np.dot(q, Q[:, i])
        q = q - h[i] * Q[:, i]

    h[k + 1] = np.linalg.norm(q)
    if h[k + 1] < 1e-15:
        q = np.zeros(n, dtype=np.float64)
    else:
        q = q / h[k + 1]

    return h, q


def gmres_solve(A_func, b, x0=None, max_iter=50, tol=1e-8, restart=None):
    n = len(b)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return np.zeros(n), [0.0], True

    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    if restart is None:
        restart = max_iter

    x = np.copy(x0)
    residuals = []

    for outer in range(max_iter // restart + 1):
        r = b - A_func(x)
        r_norm = np.linalg.norm(r)
        error = r_norm / b_norm
        residuals.append(error)

        if error <= tol:
            return x, residuals, True

        m = min(restart, max_iter - outer * restart)

        Q = np.zeros((n, m + 1), dtype=np.float64)
        Q[:, 0] = r / r_norm

        H = np.zeros((m + 1, m), dtype=np.float64)
        cs = np.zeros(m, dtype=np.float64)
        sn = np.zeros(m, dtype=np.float64)
        e1 = np.zeros(m + 1, dtype=np.float64)
        e1[0] = r_norm
        beta = np.copy(e1)

        for k in range(m):
            h_col, q_new = arnoldi_iteration(A_func, Q, k)
            H[:k + 2, k] = h_col
            Q[:, k + 1] = q_new

            h_col[:k + 2], cs[k], sn[k] = apply_givens_rotation(
                h_col[:k + 2].copy(), cs, sn, k)
            H[:k + 2, k] = h_col[:k + 2]

            beta[k + 1] = -sn[k] * beta[k]
            beta[k] = cs[k] * beta[k]
            error = abs(beta[k + 1]) / b_norm
            residuals.append(error)

            if error <= tol:

                y = np.linalg.solve(H[:k + 1, :k + 1], beta[:k + 1])
                x = x + Q[:, :k + 1] @ y
                return x, residuals, True


        y = np.linalg.solve(H[:m, :m], beta[:m])
        x = x + Q[:, :m] @ y

    return x, residuals, False


def build_poisson_matrix(nx, ny, nz, dx, dy, dz):
    N = nx * ny * nz
    A = np.zeros((N, N), dtype=np.float64)

    def idx(i, j, k):
        return i + nx * (j + ny * k)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = idx(i, j, k)
                coeff = 0.0

                if i > 0:
                    A[n, idx(i - 1, j, k)] = 1.0 / dx**2
                    coeff += 1.0 / dx**2
                if i < nx - 1:
                    A[n, idx(i + 1, j, k)] = 1.0 / dx**2
                    coeff += 1.0 / dx**2
                if j > 0:
                    A[n, idx(i, j - 1, k)] = 1.0 / dy**2
                    coeff += 1.0 / dy**2
                if j < ny - 1:
                    A[n, idx(i, j + 1, k)] = 1.0 / dy**2
                    coeff += 1.0 / dy**2
                if k > 0:
                    A[n, idx(i, j, k - 1)] = 1.0 / dz**2
                    coeff += 1.0 / dz**2
                if k < nz - 1:
                    A[n, idx(i, j, k + 1)] = 1.0 / dz**2
                    coeff += 1.0 / dz**2

                A[n, n] = -coeff



    center = N // 2
    A[center, :] = 0.0
    A[center, center] = 1.0

    return A
