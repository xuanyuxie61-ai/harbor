
import numpy as np
from typing import Tuple, Callable


def newton_solve_scalar(
    f: Callable[[float], float],
    fp: Callable[[float], float],
    a0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    small: float = 1e-8
) -> Tuple[float, float, int, str]:
    a = float(a0)
    fa = f(a)
    fpa = fp(a)
    big = 100.0 * abs(fa)
    it = 0

    while True:
        if it >= max_iter:
            return a, fa, it, "Failure: too many steps!"

        if abs(fpa) <= small:
            return a, fa, it, "Divergence: derivative value too small!"

        b = a - fa / fpa
        fb = f(b)
        it += 1
        a = b
        fa = fb
        fpa = fp(a)

        if abs(fa) >= big:
            return a, fa, it, "Divergence: function value grew too large!"

        if abs(fa) <= tol:
            return a, fa, it, "Convergence: very small function value!"


def newton_solve_system(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 30,
    damping: float = 1.0
) -> Tuple[np.ndarray, float, int, str]:
    u = u0.copy().astype(float)
    it = 0

    for it in range(1, max_iter + 1):
        R = residual(u)
        J = jacobian(u)

        res_norm = float(np.linalg.norm(R, np.inf))
        if res_norm <= tol:
            return u, res_norm, it, "Convergence: residual below tolerance"


        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:

            lam = 1e-6
            J_reg = J + lam * np.eye(J.shape[0])
            try:
                delta = np.linalg.solve(J_reg, -R)
            except np.linalg.LinAlgError:
                return u, res_norm, it, "Failure: singular Jacobian"


        alpha = damping
        for _ in range(10):
            u_new = u + alpha * delta

            u_new = np.where(u_new < 0, 0.0, u_new)
            R_new = residual(u_new)
            new_norm = float(np.linalg.norm(R_new, np.inf))
            if new_norm < res_norm or alpha < 1e-4:
                u = u_new
                break
            alpha *= 0.5
        else:
            u = u + alpha * delta
            u = np.where(u < 0, 0.0, u)

    res_norm = float(np.linalg.norm(residual(u), np.inf))
    return u, res_norm, it, "Failure: max iterations reached"


def coupled_tumor_nutrient_residual(
    u: np.ndarray, D: float, k_c: float, Km: float,
    lambda_prolif: float, lambda_death: float, rho_max: float,
    laplacian_matrix: np.ndarray
) -> np.ndarray:
    N = u.shape[0] // 2
    C = u[:N]
    rho = u[N:]


    C_safe = np.where(C < 0, 0.0, C)
    rho_safe = np.where(rho < 0, 0.0, rho)

    denom = Km + C_safe
    denom = np.where(denom < 1e-15, 1e-15, denom)

    R_C = D * (laplacian_matrix @ C_safe) - k_c * rho_safe * C_safe / denom


    R_rho = lambda_prolif * rho_safe * (1.0 - rho_safe / rho_max) * C_safe / denom - lambda_death * rho_safe

    return np.concatenate([R_C, R_rho])


def coupled_tumor_nutrient_jacobian(
    u: np.ndarray, D: float, k_c: float, Km: float,
    lambda_prolif: float, lambda_death: float, rho_max: float,
    laplacian_matrix: np.ndarray
) -> np.ndarray:
    N = u.shape[0] // 2
    C = u[:N]
    rho = u[N:]

    C_safe = np.where(C < 0, 0.0, C)
    rho_safe = np.where(rho < 0, 0.0, rho)

    denom = Km + C_safe
    denom = np.where(denom < 1e-15, 1e-15, denom)
    denom2 = denom ** 2


    dRc_dC = D * laplacian_matrix - np.diag(k_c * rho_safe * Km / denom2)


    dRc_drho = -np.diag(k_c * C_safe / denom)


    dRrho_dC = np.diag(lambda_prolif * rho_safe * Km / denom2)


    dRrho_drho_diag = (lambda_prolif * (1.0 - 2.0 * rho_safe / rho_max) * C_safe / denom -
                       lambda_death)
    dRrho_drho = np.diag(dRrho_drho_diag)

    J = np.block([[dRc_dC, dRc_drho],
                  [dRrho_dC, dRrho_drho]])
    return J


def solve_coupled_steady_state(
    N: int = 32,
    D: float = 1.0,
    k_c: float = 0.5,
    Km: float = 0.1,
    lambda_prolif: float = 0.3,
    lambda_death: float = 0.1,
    rho_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float, int, str]:

    h = 1.0 / (N + 1)
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0 / (h * h)
        if i > 0:
            L[i, i - 1] = 1.0 / (h * h)
        if i < N - 1:
            L[i, i + 1] = 1.0 / (h * h)


    x_grid = np.linspace(0.0, 1.0, N)
    C0 = np.sin(np.pi * x_grid) * 0.8 + 0.2
    rho0 = np.ones(N) * 0.3
    u0 = np.concatenate([C0, rho0])


    source_term = np.sin(np.pi * x_grid) * 2.0 + 1.0

    def res_func(u):
        R = coupled_tumor_nutrient_residual(
            u, D, k_c, Km, lambda_prolif, lambda_death, rho_max, L
        )

        R[:N] += source_term
        return R

    def jac_func(u):
        return coupled_tumor_nutrient_jacobian(
            u, D, k_c, Km, lambda_prolif, lambda_death, rho_max, L
        )


    u, res_norm, it, status = newton_solve_system(
        res_func, jac_func, u0, tol=1e-9, max_iter=60, damping=0.6
    )

    C = u[:N]
    rho = u[N:]
    C = np.clip(C, 0.0, None)
    rho = np.clip(rho, 0.0, rho_max * 1.5)


    if np.mean(rho) < 1e-4:
        C_avg = float(np.mean(C))
        denom = Km + C_avg
        if denom > 1e-15:
            growth_term = lambda_prolif * C_avg / denom
            if growth_term > lambda_death:
                rho_avg = rho_max * (1.0 - lambda_death / growth_term)

                rho = rho_avg * (C / (C_avg + 1e-15))
                rho = np.clip(rho, 0.0, rho_max)

    return C, rho, res_norm, it, status
