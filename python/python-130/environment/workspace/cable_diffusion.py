# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def build_laplacian_1d(n: int, h: float, bc: str = "DD") -> np.ndarray:
    if n < 3:
        raise ValueError(f"Number of points n={n} must be >= 3.")
    if h <= 0.0:
        raise ValueError(f"Grid spacing h={h} must be positive.")
    valid_bcs = ("DD", "DN", "ND", "NN", "PP")
    if bc not in valid_bcs:
        raise ValueError(f"Boundary condition '{bc}' not in {valid_bcs}.")

    L = np.zeros((n, n))
    inv_h2 = 1.0 / (h * h)

    if bc == "DD":

        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "DN":

        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2

        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "ND":


        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "NN":

        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "PP":

        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        L[0, n - 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, 0] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    return L


def apply_laplacian_1d(n: int, h: float, u: np.ndarray, bc: str = "DD") -> np.ndarray:
    if u.shape[0] != n:
        raise ValueError(f"Input length {u.shape[0]} does not match n={n}.")
    if h <= 0.0:
        raise ValueError("Grid spacing h must be positive.")

    Lu = np.zeros_like(u)
    inv_h2 = 1.0 / (h * h)

    if bc == "DD":
        Lu[0] = (2.0 * u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1]) * inv_h2
    elif bc == "DN":
        Lu[0] = (2.0 * u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + u[n - 1]) * inv_h2
    elif bc == "ND":
        Lu[0] = (u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1]) * inv_h2
    elif bc == "NN":
        Lu[0] = (u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + u[n - 1]) * inv_h2
    elif bc == "PP":
        Lu[0] = (2.0 * u[0] - u[1] - u[n - 1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1] - u[0]) * inv_h2
    else:
        raise ValueError(f"Invalid boundary condition: {bc}")

    return Lu


def cable_diffusion_step(
    c: np.ndarray,
    D: float,
    h: float,
    dt: float,
    gamma: float,
    source: Optional[np.ndarray] = None,
    bc: str = "DD",
) -> np.ndarray:
    if D < 0.0:
        raise ValueError("Diffusion coefficient D must be non-negative.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if gamma < 0.0:
        raise ValueError("Degradation rate gamma must be non-negative.")

    n = c.shape[0]
    Lu = apply_laplacian_1d(n, h, c, bc)

    c_new = c + dt * (D * Lu - gamma * c)
    if source is not None:
        if source.shape[0] != n:
            raise ValueError("Source term length must match concentration length.")
        c_new = c_new + dt * source


    c_new = np.maximum(c_new, 0.0)

    return c_new


def laplacian_eigenvalues(n: int, h: float, bc: str = "DD") -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3.")
    if h <= 0.0:
        raise ValueError("h must be positive.")

    L = build_laplacian_1d(n, h, bc)
    eigvals = np.linalg.eigvalsh(L)
    return eigvals


def stability_limit(n: int, h: float, D: float, bc: str = "DD") -> float:
    if D <= 0.0:
        raise ValueError("D must be positive for stability analysis.")
    eigvals = laplacian_eigenvalues(n, h, bc)
    lambda_max = np.max(np.abs(eigvals))
    dt_max = 2.0 / (D * lambda_max) if lambda_max > 0 else np.inf
    return dt_max


def simulate_protein_diffusion(
    n: int = 64,
    length: float = 100.0,
    D: float = 0.1,
    gamma: float = 0.01,
    dt: float = 0.1,
    t_final: float = 50.0,
    bc: str = "DD",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("n must be >= 3.")
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if dt <= 0.0 or t_final <= 0.0:
        raise ValueError("dt and t_final must be positive.")

    h = length / (n + 1.0)
    x = np.linspace(h, length - h, n)


    dt_max = stability_limit(n, h, D, bc)
    if dt > dt_max:
        dt = dt_max * 0.9
        print(f"[cable_diffusion] dt adjusted to {dt:.4f} for stability (limit={dt_max:.4f})")

    nt = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, t_final, nt + 1)


    x0 = length / 2.0
    sigma0 = length / 20.0
    c = np.exp(-((x - x0) ** 2) / (2.0 * sigma0 ** 2))


    source = np.zeros(n)
    source += 0.5 * np.exp(-((x - 0.3 * length) ** 2) / (2.0 * (h * 2.0) ** 2))
    source += 0.5 * np.exp(-((x - 0.7 * length) ** 2) / (2.0 * (h * 2.0) ** 2))

    c_history = np.zeros((nt + 1, n))
    c_history[0, :] = c

    for step in range(nt):
        c = cable_diffusion_step(c, D, h, dt, gamma, source, bc)
        c_history[step + 1, :] = c

    return x, t, c_history


if __name__ == "__main__":
    x, t, c_hist = simulate_protein_diffusion()
    print(f"Cable diffusion: max concentration at t={t[-1]:.1f} ms = {np.max(c_hist[-1]):.6f}")
