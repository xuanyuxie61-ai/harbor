
import numpy as np
from typing import Callable, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator


def detq_orthogonal(a: np.ndarray) -> Tuple[float, int]:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    tol = 1.0e-4
    ifault = 0

    if n <= 0:
        return 0.0, 1


    a2 = a.flatten(order='F').copy()
    d = 1.0
    r_idx = 0

    for k in range(1, n + 1):
        q = r_idx
        x = a2[q]
        y = np.sign(x)
        d *= y
        y = -1.0 / (x + y)
        x = abs(x) - 1.0

        if tol < abs(x):
            if 0.0 < x:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault

            for i in range(k, n):
                q += n
                x_val = a2[q] * y
                p = r_idx
                s = q
                for j in range(k, n):
                    p += 1
                    s += 1
                    a2[s] += x_val * a2[p]

        r_idx += n + 1

    return d, ifault


def assemble_sparse_jacobian_2d(n: int,
                                 u: np.ndarray,
                                 diff_func: Callable[[np.ndarray], np.ndarray],
                                 reaction_deriv: Callable[[np.ndarray], np.ndarray],
                                 xleft: float = -1.0,
                                 xright: float = 1.0) -> csr_matrix:
    numnodes = n * n
    x = np.linspace(xleft, xright, n)
    h = x[1] - x[0]
    h2 = h * h

    D = diff_func(u)
    Rprime = reaction_deriv(u)

    row_idx = []
    col_idx = []
    data = []

    def add_entry(r, c, val):
        row_idx.append(r)
        col_idx.append(c)
        data.append(float(val))

    k = 0
    for i in range(n):
        for j in range(n):

            D_e = D[k] if j == n - 1 else 0.5 * (D[k] + D[k + 1])
            D_w = D[k] if j == 0 else 0.5 * (D[k] + D[k - 1])
            D_n = D[k] if i == n - 1 else 0.5 * (D[k] + D[k + n])
            D_s = D[k] if i == 0 else 0.5 * (D[k] + D[k - n])


            center = 0.0
            if j > 0:
                center += D_w / h2
                add_entry(k, k - 1, -D_w / h2)
            else:

                center += D_w / h2

            if j < n - 1:
                center += D_e / h2
                add_entry(k, k + 1, -D_e / h2)
            else:
                center += D_e / h2

            if i > 0:
                center += D_s / h2
                add_entry(k, k - n, -D_s / h2)
            else:
                center += D_s / h2

            if i < n - 1:
                center += D_n / h2
                add_entry(k, k + n, -D_n / h2)
            else:
                center += D_n / h2

            center += Rprime[k]
            add_entry(k, k, center)
            k += 1

    J = csr_matrix((data, (row_idx, col_idx)), shape=(numnodes, numnodes))
    return J


def nonlinear_residual_2d(n: int,
                          u: np.ndarray,
                          diff_func: Callable[[np.ndarray], np.ndarray],
                          reaction_func: Callable[[np.ndarray], np.ndarray],
                          source: np.ndarray,
                          xleft: float = -1.0,
                          xright: float = 1.0) -> np.ndarray:
    numnodes = n * n
    x = np.linspace(xleft, xright, n)
    h = x[1] - x[0]
    h2 = h * h

    D = diff_func(u)
    R = reaction_func(u)

    F = np.zeros(numnodes)
    k = 0
    for i in range(n):
        for j in range(n):
            D_e = D[k] if j == n - 1 else 0.5 * (D[k] + D[k + 1])
            D_w = D[k] if j == 0 else 0.5 * (D[k] + D[k - 1])
            D_n = D[k] if i == n - 1 else 0.5 * (D[k] + D[k + n])
            D_s = D[k] if i == 0 else 0.5 * (D[k] + D[k - n])

            val = 0.0
            u_k = u[k]
            if j > 0:
                val -= D_w * (u[k - 1] - u_k) / h2
            else:
                val -= D_w * (-u_k) / h2

            if j < n - 1:
                val -= D_e * (u[k + 1] - u_k) / h2
            else:
                val -= D_e * (-u_k) / h2

            if i > 0:
                val -= D_s * (u[k - n] - u_k) / h2
            else:
                val -= D_s * (-u_k) / h2

            if i < n - 1:
                val -= D_n * (u[k + n] - u_k) / h2
            else:
                val -= D_n * (-u_k) / h2

            F[k] = val + R[k] - source[k]
            k += 1

    return F


def newton_krylov_solve(n: int,
                        u0: np.ndarray,
                        diff_func: Callable[[np.ndarray], np.ndarray],
                        reaction_func: Callable[[np.ndarray], np.ndarray],
                        reaction_deriv: Callable[[np.ndarray], np.ndarray],
                        source: np.ndarray,
                        xleft: float = -1.0,
                        xright: float = 1.0,
                        tol: float = 1.0e-8,
                        max_iter: int = 50,
                        alpha: float = 1.0) -> Tuple[np.ndarray, int, float]:
    u = u0.copy()
    numnodes = n * n

    for it in range(max_iter):
        F = nonlinear_residual_2d(n, u, diff_func, reaction_func, source, xleft, xright)
        res_norm = np.linalg.norm(F)

        if res_norm < tol:
            return u, it, res_norm

        J = assemble_sparse_jacobian_2d(n, u, diff_func, reaction_deriv, xleft, xright)


        if numnodes <= 400:
            try:
                delta = spsolve(J, -F)
            except Exception:
                delta, info = gmres(J, -F, rtol=tol, maxiter=200)
                if info != 0:

                    delta = -F * 0.01
        else:
            delta, info = gmres(J, -F, rtol=tol, maxiter=500)
            if info != 0:
                delta = -F * 0.01


        alpha_local = alpha
        for _ in range(5):
            u_trial = u + alpha_local * delta
            F_trial = nonlinear_residual_2d(n, u_trial, diff_func, reaction_func, source, xleft, xright)
            if np.linalg.norm(F_trial) < res_norm:
                break
            alpha_local *= 0.5

        u = u + alpha_local * delta


        u = np.maximum(u, 0.0)
        u = np.minimum(u, 1.0e3)

    return u, max_iter, np.linalg.norm(nonlinear_residual_2d(n, u, diff_func, reaction_func, source, xleft, xright))


def gel_effect_diffusion(conversion: np.ndarray,
                         D0: float = 1.0e-4,
                         beta: float = 2.5,
                         c_crit: float = 0.8) -> np.ndarray:
    c = np.asarray(conversion)
    D = np.zeros_like(c)
    mask1 = c < c_crit
    mask2 = ~mask1
    D[mask1] = D0 * ((1.0 - c[mask1]) ** beta)
    D[mask2] = D0 * ((1.0 - c_crit) ** beta) * np.exp(-10.0 * (c[mask2] - c_crit))
    D = np.maximum(D, 1.0e-8 * D0)
    return D


def nonlinear_source_reaction(conversion: np.ndarray,
                              k0: float = 1.0,
                              activation: float = 10.0) -> np.ndarray:
    c = np.asarray(conversion)
    c = np.clip(c, 0.0, 1.0)
    return k0 * c * (1.0 - c) * (1.0 + activation * c ** 2)


def nonlinear_source_derivative(conversion: np.ndarray,
                                k0: float = 1.0,
                                activation: float = 10.0) -> np.ndarray:
    c = np.asarray(conversion)
    c = np.clip(c, 0.0, 1.0)
    return k0 * (1.0 - 2.0 * c + 3.0 * activation * c ** 2 - 4.0 * activation * c ** 3)
