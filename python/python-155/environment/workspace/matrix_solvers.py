import numpy as np
from typing import Optional, Tuple





def r8to_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if n < 1:
        return np.array([])
    b = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i:
                b[i] += a[j - i] * x[j]
            else:
                b[i] += a[n + (i - j) - 1] * x[j]
    return b


def r8to_sl(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if n < 1:
        return np.array([])
    first_row = a[:n].copy()
    first_col = np.zeros(n, dtype=float)
    first_col[0] = first_row[0]
    if len(a) > n:
        first_col[1:min(n, len(a) - n + 1)] = a[n:min(len(a), 2 * n - 1)]
    try:
        from scipy.linalg import solve_toeplitz
        return solve_toeplitz((first_col, first_row), b)
    except Exception:

        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if j >= i:
                    A[i, j] = first_row[j - i]
                else:
                    A[i, j] = first_col[i - j]
        return np.linalg.solve(A, b)





def r8vm_mv(m: int, n: int, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    b = np.zeros(m, dtype=float)
    for i in range(m):
        for j in range(n):
            b[i] += (x[j] ** i) * v[j]
    return b


def r8vm_sl(n: int, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    if n < 1:
        return np.array([])

    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(x[i], x[j]):
                raise ValueError(f"Vandermonde singular: duplicate node x[{i}] = x[{j}]")


    V = np.vander(x, n, increasing=True)
    A = V.T
    return np.linalg.solve(A, b)





def r83_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    b = np.zeros(n, dtype=float)
    b[0] = a[1, 0] * x[0]
    if n > 1:
        b[0] += a[0, 0] * x[1]
        b[n - 1] = a[2, n - 1] * x[n - 2] + a[1, n - 1] * x[n - 1]
    for i in range(1, n - 1):
        b[i] = a[2, i] * x[i - 1] + a[1, i] * x[i] + a[0, i] * x[i + 1]
    return b


def r83_cg(n: int, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
           tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    r = b - r83_mv(n, a, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        ap = r83_mv(n, a, p)
        alpha = safe_divide(rs_old, np.dot(p, ap), 0.0)
        if alpha == 0.0:
            break
        x += alpha * p
        r -= alpha * ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        beta = safe_divide(rs_new, rs_old, 0.0)
        p = r + beta * p
        rs_old = rs_new
    return x


def r83_cr_fa(n: int, a: np.ndarray) -> np.ndarray:
    if n < 1:
        return np.zeros((3, 1))
    a_cr = np.zeros((3, 2 * n + 1), dtype=float)
    a_cr[0, :n] = a[0, :n].copy()
    a_cr[1, :n] = a[1, :n].copy()
    a_cr[2, :n] = a[2, :n].copy()



    for i in range(n):
        if np.isclose(a_cr[1, i], 0.0):
            raise ValueError(f"Zero pivot at {i} in cyclic reduction")
        if i + 1 < n:
            factor = a_cr[2, i + 1] / a_cr[1, i]
            a_cr[2, i + 1] = factor
            a_cr[1, i + 1] -= factor * a_cr[0, i]
    return a_cr


def r83_cr_sl(n: int, a_cr: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = b.copy()

    for i in range(1, n):
        x[i] -= a_cr[2, i] * x[i - 1]

    for i in range(n - 1, -1, -1):
        if np.isclose(a_cr[1, i], 0.0):
            raise ValueError(f"Zero pivot at {i} in back substitution")
        x[i] /= a_cr[1, i]
        if i > 0:
            x[i - 1] -= a_cr[0, i - 1] * x[i]
    return x


def r83_jac_sl(n: int, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
               max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=float)
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= a[2, i] * x[i - 1]
            if i < n - 1:
                s -= a[0, i] * x[i + 1]
            if np.isclose(a[1, i], 0.0):
                raise ValueError(f"Zero diagonal at {i}")
            x_new[i] = s / a[1, i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def r83_gs_sl(n: int, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
              max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= a[2, i] * x[i - 1]
            if i < n - 1:
                s -= a[0, i] * x_old[i + 1]
            if np.isclose(a[1, i], 0.0):
                raise ValueError(f"Zero diagonal at {i}")
            x[i] = s / a[1, i]
        if np.linalg.norm(x - x_old) < tol:
            break
    return x





def cg_rc(n: int, b: np.ndarray, x: np.ndarray, r: np.ndarray, z: np.ndarray,
          p: np.ndarray, q: np.ndarray, job: int, rho: float = 0.0,
          rho_old: float = 0.0, iter_count: int = 0) -> Tuple:
    tol = 1e-10
    max_iter = n * 10

    if job == 0:

        return x, r, z, p, q, 3, rho, rho_old, iter_count

    if job == 3:

        iter_count = 0
        rho = 0.0
        return x, r, z, p, q, 2, rho, rho_old, iter_count

    if job == 2:

        z = r.copy()
        rho_old = rho
        rho = np.dot(r, z)
        if iter_count == 0:
            p = z.copy()
        else:
            if np.isclose(rho_old, 0.0):
                beta = 0.0
            else:
                beta = rho / rho_old
            p = z + beta * p
        return x, r, z, p, q, 1, rho, rho_old, iter_count

    if job == 1:

        alpha = safe_divide(rho, np.dot(p, q), 0.0)
        x = x + alpha * p
        r = r - alpha * q
        iter_count += 1
        if iter_count > max_iter:
            return x, r, z, p, q, 4, rho, rho_old, iter_count
        if np.linalg.norm(r) < tol:
            return x, r, z, p, q, 4, rho, rho_old, iter_count
        return x, r, z, p, q, 2, rho, rho_old, iter_count

    return x, r, z, p, q, 4, rho, rho_old, iter_count


def cg_rc_solve(n: int, A_mult: callable, b: np.ndarray,
                x0: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    r = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    q = np.zeros(n, dtype=float)
    job = 0
    rho = 0.0
    rho_old = 0.0
    it = 0
    while True:
        x, r, z, p, q, job, rho, rho_old, it = cg_rc(
            n, b, x, r, z, p, q, job, rho, rho_old, it
        )
        if job == 1:
            q = A_mult(p)
        elif job == 2:
            z = r.copy()
        elif job == 3:
            r = b - A_mult(x)
        elif job == 4:
            break

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):

            A = np.zeros((n, n), dtype=float)
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1.0
                A[:, j] = A_mult(ej)
            return np.linalg.solve(A, b)
    return x





def wathen_order(nx: int, ny: int) -> int:
    return 3 * nx * ny + 2 * nx + 2 * ny + 1


def wathen(nx: int, ny: int) -> np.ndarray:
    n = wathen_order(nx, ny)
    A = np.zeros((n, n), dtype=float)
    em = np.array([
        [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
        [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
        [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
        [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
        [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
        [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
        [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
        [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
    ]) / 9.0
    rho = 1.0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):

            nn = np.array([
                3 * j * nx + 2 * i + 2 * j + 1,
                3 * j * nx + 2 * i + 2 * j,
                3 * j * nx + 2 * i + 2 * j - 1,
                3 * (j - 1) * nx + 2 * i + 2 * j - 1,
                3 * (j - 1) * nx + 2 * i + 2 * j - 2,
                3 * (j - 1) * nx + 2 * i + 2 * j - 3,
                3 * (j - 1) * nx + 2 * i + 2 * (j - 1),
                3 * (j - 1) * nx + 2 * i + 2 * (j - 1) + 1
            ], dtype=int)

            nn = nn - 1

            nn = np.clip(nn, 0, n - 1)
            for k in range(8):
                for l in range(8):
                    A[nn[k], nn[l]] += rho * em[k, l]
    return A


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if np.isclose(b, 0.0):
        return default
    return a / b
