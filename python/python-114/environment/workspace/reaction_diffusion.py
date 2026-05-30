
import numpy as np
from typing import Tuple, Optional, Callable


def laplacian9_torus(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    nxp, nyp = A.shape

    if nxp < 3 or nyp < 3:
        raise ValueError("grid dimensions must be at least 3")

    denom = 6.0 * dx * dx


    L = (
        1.0 * np.roll(np.roll(A, -1, axis=0), -1, axis=1)
        + 4.0 * np.roll(A, -1, axis=0)
        + 1.0 * np.roll(np.roll(A, -1, axis=0), 1, axis=1)
        + 4.0 * np.roll(A, -1, axis=1)
        - 20.0 * A
        + 4.0 * np.roll(A, 1, axis=1)
        + 1.0 * np.roll(np.roll(A, 1, axis=0), -1, axis=1)
        + 4.0 * np.roll(A, 1, axis=0)
        + 1.0 * np.roll(np.roll(A, 1, axis=0), 1, axis=1)
    ) / denom

    return L


def laplacian5_torus(A: np.ndarray, dx: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return (
        np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
        + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
        - 4.0 * A
    ) / (dx * dx)


def gray_scott_step(
    u: np.ndarray,
    v: np.ndarray,
    du: float,
    dv: float,
    f: float,
    k: float,
    dt: float,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)


    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)


    Lu = laplacian5_torus(u, dx)
    Lv = laplacian5_torus(v, dx)


    reaction = u * v * v

    dudt = du * Lu - reaction + f * (1.0 - u)
    dvdt = dv * Lv + reaction - (f + k) * v


    u_new = u + dt * dudt
    v_new = v + dt * dvdt


    u_new = np.clip(u_new, 0.0, 1.0)
    v_new = np.clip(v_new, 0.0, 1.0)

    return u_new, v_new


def porous_medium_solution(
    x: np.ndarray,
    t: float,
    m: float = 3.0,
    delta: float = 1.0 / 75.0,
    c: float = np.sqrt(3.0) / 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if t + delta <= 0:
        raise ValueError("t + delta must be positive")
    if m <= 1.0:
        raise ValueError("m must be > 1 for porous medium equation")

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2

    positive = factor > 0.0

    u = np.zeros_like(x)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)

    if np.any(positive):
        fp = factor[positive]
        xp = x[positive]

        u[positive] = (t + delta) ** (-beta) * fp ** alpha

        ut[positive] = (
            2.0 * alpha * beta * gamma * (t + delta) ** (-1.0 - 3.0 * beta)
            * xp ** 2 * fp ** (alpha - 1.0)
            - beta * (t + delta) ** (-1.0 - beta) * fp ** alpha
        )

        ux[positive] = (
            -2.0 * alpha * gamma
            * (t + delta) ** (-3.0 * beta)
            * xp
            * fp ** (alpha - 1.0)
        )

        uxx[positive] = (
            4.0 * (alpha - 1.0) * alpha * gamma ** 2
            * (t + delta) ** (-5.0 * beta)
            * xp ** 2
            * fp ** (alpha - 2.0)
            - 2.0 * alpha * gamma
            * (t + delta) ** (-3.0 * beta)
            * fp ** (alpha - 1.0)
        )


    return u, ut, ux, uxx


def porous_medium_residual(
    x: np.ndarray,
    t: float,
    m: float = 3.0,
) -> np.ndarray:
    u, ut, ux, uxx = porous_medium_solution(x, t, m)


    eps = 1e-14
    u_safe = np.where(u > eps, u, eps)

    R = ut - m * (m - 1.0) * u_safe ** (m - 2.0) * ux ** 2 - m * u_safe ** (m - 1.0) * uxx


    R = np.where(u > eps, R, 0.0)
    return R


def simulate_gamma_h2ax_wave(
    nx: int = 128,
    ny: int = 128,
    nt: int = 2000,
    f: float = 0.03,
    k: float = 0.062,
    du: float = 0.16,
    dv: float = 0.08,
    dt: float = 1.0,
    dx: float = 10.0,
) -> dict:
    np.random.seed(7)

    u = np.ones((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)


    xm, ym = nx // 2, ny // 2
    patch = nx // 8
    v[xm - patch:xm + patch, ym - patch:ym + patch] = 1.0
    u[xm - patch:xm + patch, ym - patch:ym + patch] = 0.0

    v += np.random.randn(nx, ny) * 0.01
    v = np.clip(v, 0.0, 1.0)


    radii = []
    thresholds = [0.5]

    for it in range(nt):
        u, v = gray_scott_step(u, v, du, dv, f, k, dt, dx, dx)


        if it % 100 == 0:
            coords = np.argwhere(v > 0.3)
            if len(coords) > 0:
                dists = np.sqrt((coords[:, 0] - xm) ** 2 + (coords[:, 1] - ym) ** 2) * dx
                radii.append(float(np.max(dists)))
            else:
                radii.append(0.0)


    velocity = 0.0
    if len(radii) > 1:
        velocity = (radii[-1] - radii[0]) / (len(radii) * 100.0 * dt + 1e-12)

    total_gamma = float(np.sum(v)) * dx * dx

    return {
        "u_final": u,
        "v_final": v,
        "wave_velocity": velocity,
        "total_gamma_h2ax": total_gamma,
        "max_v": float(np.max(v)),
    }


def simulate_parp1_nonlinear_diffusion(
    nx: int = 256,
    nt: int = 400,
    dt: float = 0.01,
    dx: float = 0.1,
    m: float = 3.0,
) -> dict:
    x = np.linspace(-nx * dx / 2, nx * dx / 2, nx)

    delta_pm = 1.0 / 75.0
    c_pm = np.sqrt(3.0) / 15.0
    t0 = 1.0

    u, _, _, _ = porous_medium_solution(x, t0, m=m, delta=delta_pm, c=c_pm)
    u = np.maximum(u, 0.0)

    for _ in range(nt):

        u_padded = np.pad(u, 1, mode='constant')
        grad_u = (u_padded[2:] - u_padded[:-2]) / (2.0 * dx)
        D = m * np.maximum(u, 1e-12) ** (m - 1.0)
        flux = -D * grad_u


        flux_padded = np.pad(flux, 1, mode='constant')
        div_flux = (flux_padded[2:] - flux_padded[:-2]) / (2.0 * dx)

        u_new = u + dt * (-div_flux)
        u_new = np.maximum(u_new, 0.0)
        u = u_new


    t_final = nt * dt + t0
    u_exact, _, _, _ = porous_medium_solution(x, t_final, m=m, delta=delta_pm, c=c_pm)


    mask = u_exact > 1e-12
    if np.any(mask):
        l2_error = float(np.sqrt(np.mean((u[mask] - u_exact[mask]) ** 2)))
    else:
        l2_error = 0.0

    return {
        "u_numerical": u,
        "u_exact": u_exact,
        "x_grid": x,
        "l2_error": l2_error,
        "total_mass_numerical": float(np.sum(u) * dx),
        "total_mass_exact": float(np.sum(u_exact) * dx),
    }
