
import numpy as np
from typing import Callable, Optional, Tuple


def conjugate_gradient(A_matvec: Callable[[np.ndarray], np.ndarray],
                       b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       maxiter: int = 2000) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A_matvec(x)
    p = r.copy()
    rsold = float(np.dot(r, r))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-30:
        norm_b = 1.0

    for k in range(maxiter):
        Ap = A_matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(np.dot(r, r))
        resid = np.sqrt(rsnew) / norm_b
        if resid < tol:
            return x, k + 1, resid
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, maxiter, np.sqrt(float(np.dot(r, r))) / norm_b


def incomplete_cholesky_prec(A: np.ndarray, drop_tol: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    for j in range(n):
        diag_val = A[j, j] - np.dot(L[j, :j], L[j, :j])
        if diag_val <= 0.0:
            diag_val = abs(diag_val) + 1e-10
        L[j, j] = np.sqrt(diag_val)
        for i in range(j + 1, n):
            if abs(A[i, j]) > drop_tol:
                offdiag = A[i, j] - np.dot(L[i, :j], L[j, :j])
                L[i, j] = offdiag / L[j, j]
    return L


def pcg_solver(A_matvec: Callable[[np.ndarray], np.ndarray],
               b: np.ndarray,
               preconditioner_solve: Callable[[np.ndarray], np.ndarray],
               x0: Optional[np.ndarray] = None,
               tol: float = 1e-10,
               maxiter: int = 2000) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A_matvec(x)
    z = preconditioner_solve(r)
    p = z.copy()
    rzold = float(np.dot(r, z))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-30:
        norm_b = 1.0

    for k in range(maxiter):
        Ap = A_matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rzold / pAp
        x += alpha * p
        r -= alpha * Ap
        resid = float(np.linalg.norm(r)) / norm_b
        if resid < tol:
            return x, k + 1, resid
        z = preconditioner_solve(r)
        rznew = float(np.dot(r, z))
        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew

    return x, maxiter, float(np.linalg.norm(r)) / norm_b






def build_radial_diffusion_operator(n: int, dr: float, dt: float, eta: float,
                                     theta_cn: float = 0.5) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    coeff = theta_cn * dt * eta / (dr * dr)
    for i in range(n):
        A[i, i] = 1.0 + 2.0 * coeff
        if i > 0:
            A[i, i - 1] = -coeff
        if i < n - 1:
            A[i, i + 1] = -coeff

    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[n - 1, n - 1] = 1.0
    A[n - 1, n - 2] = 0.0
    return A


def solve_radial_diffusion_cg(rhs: np.ndarray, n: int, dr: float, dt: float,
                               eta: float, theta_cn: float = 0.5,
                               tol: float = 1e-10, maxiter: int = 2000) -> np.ndarray:
    A = build_radial_diffusion_operator(n, dr, dt, eta, theta_cn)

    def matvec(v):
        return A @ v


    diag = np.diag(A).copy()
    diag[diag == 0.0] = 1.0

    def prec_solve(r):
        return r / diag

    x, iters, resid = pcg_solver(matvec, rhs, prec_solve, tol=tol, maxiter=maxiter)
    return x





def _self_test():

    n = 50
    A = np.diag(np.arange(1, n + 1, dtype=float))
    b = np.ones(n, dtype=float)
    x, iters, resid = conjugate_gradient(lambda v: A @ v, b, tol=1e-12)
    assert resid < 1e-10
    x_exact = 1.0 / np.arange(1, n + 1, dtype=float)
    assert np.linalg.norm(x - x_exact) < 1e-8


    n = 32
    dr = 1.0 / (n - 1)
    dt = 0.01
    eta = 1.0
    rhs = np.ones(n, dtype=float)
    rhs[0] = 0.0
    rhs[-1] = 0.0
    x = solve_radial_diffusion_cg(rhs, n, dr, dt, eta)
    assert not np.isnan(x).any()
    assert not np.isinf(x).any()

    print(f"cg_solver: CG converged in {iters} iterations, resid={resid:.4e}")
    print("cg_solver: self-test passed.")


if __name__ == "__main__":
    _self_test()
