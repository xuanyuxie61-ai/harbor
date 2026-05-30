
import numpy as np


def gray_scott_step(u: np.ndarray, v: np.ndarray,
                    Du: float, Dv: float, f: float, k: float,
                    dx: float, dy: float, dt: float) -> tuple:
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")

    nx, ny = u.shape

    def laplacian9_torus(field: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(field)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            for j in range(ny):
                jm = (j - 1) % ny
                jp = (j + 1) % ny
                lap[i, j] = (
                    field[im, jm] + field[im, j] + field[im, jp]
                    + field[i, jm] - 8.0 * field[i, j] + field[i, jp]
                    + field[ip, jm] + field[ip, j] + field[ip, jp]
                ) / (3.0 * dx * dy)
        return lap

    uLap = laplacian9_torus(u)
    vLap = laplacian9_torus(v)

    dudt = Du * uLap - u * v ** 2 + f * (1.0 - u)
    dvdt = Dv * vLap + u * v ** 2 - (f + k) * v


    max_dt_diff = 0.25 * min(dx * dx / (Du + 1e-12), dy * dy / (Dv + 1e-12))
    if dt > max_dt_diff:
        dt = max_dt_diff * 0.9

    u_new = np.clip(u + dt * dudt, 0.0, 1.0)
    v_new = np.clip(v + dt * dvdt, 0.0, 1.0)

    return u_new, v_new


def gray_scott_simulation(nx: int = 64, ny: int = 64,
                          f: float = 0.035, k: float = 0.060,
                          Du: float = 0.16, Dv: float = 0.08,
                          dx: float = 1.0, dy: float = 1.0,
                          dt: float = 1.0, n_steps: int = 2000) -> tuple:
    u = np.ones((nx, ny), dtype=float)
    v = np.zeros((nx, ny), dtype=float)


    xm = nx // 2
    ym = ny // 2
    u[xm - 5:xm + 5, ym - 5:ym + 5] = 0.0
    v[xm - 5:xm + 5, ym - 5:ym + 5] = 1.0

    for step in range(n_steps):
        u, v = gray_scott_step(u, v, Du, Dv, f, k, dx, dy, dt)

    return u, v


def porous_medium_exact(x: np.ndarray, t: float,
                        c: float = np.sqrt(3.0) / 15.0,
                        delta: float = 1.0 / 75.0,
                        m: float = 3.0) -> tuple:
    x = np.asarray(x, dtype=float)
    if t + delta <= 0:
        raise ValueError("t+delta must be positive")
    if m <= 1.0:
        raise ValueError("m must be > 1")

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2

    u = np.zeros_like(x, dtype=float)
    ut = np.zeros_like(x, dtype=float)
    ux = np.zeros_like(x, dtype=float)
    uxx = np.zeros_like(x, dtype=float)

    mask = factor > 0.0
    if np.any(mask):
        f = factor[mask]
        u[mask] = (t + delta) ** (-beta) * f ** alpha
        ut[mask] = (2.0 * alpha * beta * gamma * (t + delta) ** (-1.0 - 3.0 * beta)
                    * x[mask] ** 2 * f ** (alpha - 1.0)
                    - beta * (t + delta) ** (-1.0 - beta) * f ** alpha)
        ux[mask] = (-2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta)
                    * x[mask] * f ** (alpha - 1.0))
        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma ** 2
                     * (t + delta) ** (-5.0 * beta) * x[mask] ** 2 * f ** (alpha - 2.0)
                     - 2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta) * f ** (alpha - 1.0))

    return u, ut, ux, uxx


def protein_diffusion_reaction_1d(n_sites: int = 100,
                                   n_steps: int = 5000,
                                   D: float = 0.1,
                                   k_on: float = 0.05,
                                   k_off: float = 0.01,
                                   dx: float = 1.0,
                                   dt: float = 0.1) -> tuple:
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if D <= 0 or dx <= 0 or dt <= 0:
        raise ValueError("Physical parameters must be positive")


    cfl = D * dt / (dx ** 2)
    if cfl > 0.5:
        dt = 0.45 * dx ** 2 / D
        cfl = D * dt / (dx ** 2)

    P = np.zeros(n_sites, dtype=float)
    B = np.zeros(n_sites, dtype=float)
    S_total = np.ones(n_sites, dtype=float)


    P[:10] = 1.0
    P[-10:] = 1.0

    for _ in range(n_steps):

        P_diff = np.zeros_like(P)
        for i in range(1, n_sites - 1):
            P_diff[i] = D * (P[i - 1] - 2.0 * P[i] + P[i + 1]) / (dx ** 2)

        P_diff[0] = D * (P[1] - P[0]) / (dx ** 2)
        P_diff[-1] = D * (P[-2] - P[-1]) / (dx ** 2)

        S = np.maximum(S_total - B, 0.0)
        binding = k_on * P * S
        unbinding = k_off * B

        P_new = P + dt * (P_diff - binding + unbinding)
        B_new = B + dt * (binding - unbinding)

        P = np.clip(P_new, 0.0, None)
        B = np.clip(B_new, 0.0, S_total)

    return P, B


def compute_reaction_front_velocity(D: float, k_on: float,
                                    P0: float = 1.0) -> float:
    if D < 0 or k_on < 0 or P0 < 0:
        raise ValueError("Parameters must be non-negative")
    return 2.0 * np.sqrt(D * k_on * P0)
