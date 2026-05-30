
import numpy as np
from typing import Optional

from ice_constitutive_model import (
    ICE_DENSITY, GRAVITY, GLEN_N, rate_factor_arrhenius
)


def compute_diffusivity_sia(H: np.ndarray,
                            surface: np.ndarray,
                            dx: float, dy: float,
                            temperature: float = 253.15) -> tuple:
    H = np.asarray(H, dtype=np.float64)
    surface = np.asarray(surface, dtype=np.float64)

    if H.shape != surface.shape:
        raise ValueError("H and surface must have the same shape.")


    grad_s_x = np.zeros_like(surface)
    grad_s_y = np.zeros_like(surface)

    grad_s_x[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2.0 * dx)
    grad_s_y[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2.0 * dy)


    grad_s_x[:, 0] = (surface[:, 1] - surface[:, 0]) / dx
    grad_s_x[:, -1] = (surface[:, -1] - surface[:, -2]) / dx
    grad_s_y[0, :] = (surface[1, :] - surface[0, :]) / dy
    grad_s_y[-1, :] = (surface[-1, :] - surface[-2, :]) / dy


    grad_mag = np.sqrt(grad_s_x ** 2 + grad_s_y ** 2)
    grad_mag = np.maximum(grad_mag, 1e-12)





    raise NotImplementedError("Hole 2: 请实现 compute_diffusivity_sia 核心公式")


def explicit_euler_step_sia(H: np.ndarray,
                            bedrock: np.ndarray,
                            accumulation: np.ndarray,
                            dx: float, dy: float,
                            dt: float,
                            temperature: float = 253.15) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    bedrock = np.asarray(bedrock, dtype=np.float64)
    accumulation = np.asarray(accumulation, dtype=np.float64)

    if not (H.shape == bedrock.shape == accumulation.shape):
        raise ValueError("H, bedrock, and accumulation must have the same shape.")

    surface = bedrock + H
    D, gx, gy = compute_diffusivity_sia(H, surface, dx, dy, temperature)

    ny, nx = H.shape
    rhs = np.zeros_like(H)


    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            D_e = 0.5 * (D[i, j] + D[i, j + 1])
            D_w = 0.5 * (D[i, j] + D[i, j - 1])
            flux_e = D_e * (surface[i, j + 1] - surface[i, j]) / dx
            flux_w = D_w * (surface[i, j] - surface[i, j - 1]) / dx


            D_n = 0.5 * (D[i, j] + D[i + 1, j])
            D_s = 0.5 * (D[i, j] + D[i - 1, j])
            flux_n = D_n * (surface[i + 1, j] - surface[i, j]) / dy
            flux_s = D_s * (surface[i, j] - surface[i - 1, j]) / dy

            div_flux = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy
            rhs[i, j] = accumulation[i, j] + div_flux


    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[:, -1] = 0.0

    H_new = H + dt * rhs
    H_new = np.maximum(H_new, 0.0)

    return H_new


def adaptive_cfl_timestep_sia(H: np.ndarray,
                              bedrock: np.ndarray,
                              dx: float, dy: float,
                              temperature: float = 253.15,
                              cfl_safety: float = 0.25) -> float:
    surface = bedrock + H
    D, _, _ = compute_diffusivity_sia(H, surface, dx, dy, temperature)
    D_max = np.max(D)

    if D_max < 1e-20:
        return 1e7

    dt_max = cfl_safety * 0.5 * min(dx ** 2, dy ** 2) / D_max
    dt_max = max(dt_max, 1.0)
    dt_max = min(dt_max, 1e7)
    return dt_max


def solve_sia_evolution(H0: np.ndarray,
                        bedrock: np.ndarray,
                        accumulation: np.ndarray,
                        dx: float, dy: float,
                        total_time: float,
                        temperature: float = 253.15,
                        output_interval: Optional[int] = None) -> tuple:
    H = H0.copy()
    t = 0.0

    history = []
    if output_interval is not None and output_interval > 0:
        history.append(H.copy())
    step = 0

    while t < total_time:
        dt = adaptive_cfl_timestep_sia(H, bedrock, dx, dy, temperature)
        if t + dt > total_time:
            dt = total_time - t

        H = explicit_euler_step_sia(H, bedrock, accumulation, dx, dy, dt, temperature)
        t += dt
        step += 1

        if output_interval is not None and step % output_interval == 0:
            history.append(H.copy())

    if history:
        return H, np.array(history)
    return H, None


def ice_volume(H: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(H) * dx * dy)


def ice_area(H: np.ndarray, dx: float, dy: float, threshold: float = 1.0) -> float:
    return float(np.sum(H > threshold) * dx * dy)
