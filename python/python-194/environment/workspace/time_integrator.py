
import numpy as np
from typing import Tuple, Callable, Optional
from sparse_matrix import BandedSPDMatrix, banded_cholesky_solve


def semi_implicit_euler_step(
    u: np.ndarray,
    A: BandedSPDMatrix,
    f: np.ndarray,
    dt: float
) -> np.ndarray:
    n = A.n
    M = BandedSPDMatrix(n, A.ml)
    for j in range(n):
        for i in range(j, min(n, j + A.ml + 1)):
            v = A.get(i, j)
            M.set(i, j, v * dt)
            if i == j:
                M.set(i, j, v * dt + 1.0)
    rhs = u + dt * f
    return banded_cholesky_solve(M, rhs)


def velocity_verlet_step(
    u: np.ndarray,
    v: np.ndarray,
    accel_func: Callable[[np.ndarray], np.ndarray],
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = accel_func(u)
    v_half = v + 0.5 * dt * a
    u_new = u + dt * v_half
    a_new = accel_func(u_new)
    v_new = v_half + 0.5 * dt * a_new
    return u_new, v_new, a_new


def adaptive_time_stepping(
    u0: np.ndarray,
    t_span: Tuple[float, float],
    dt_init: float,
    rhs_func: Callable[[float, np.ndarray], np.ndarray],
    A_band: BandedSPDMatrix,
    tol: float = 1e-4,
    gamma_max: float = 2.0,
    dt_min: float = 1e-6,
    dt_max: float = 0.1,
    max_steps: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = t_start
    dt = dt_init
    u = u0.copy()

    t_hist = [t]
    u_hist = [u.copy()]
    dt_hist = [dt]

    blowup_counter = 0
    max_blowup = 3

    step = 0
    while t < t_end and step < max_steps:
        step += 1
        dt = min(dt, t_end - t)
        if dt <= 0:
            break

        f = rhs_func(t, u)


        M = BandedSPDMatrix(A_band.n, A_band.ml)
        for j in range(A_band.n):
            for i in range(j, min(A_band.n, j + A_band.ml + 1)):
                v = A_band.get(i, j)
                M.set(i, j, v * dt)
                if i == j:
                    M.set(i, j, v * dt + 1.0)
        rhs = u + dt * f
        try:
            u_full = banded_cholesky_solve(M, rhs)
        except Exception:

            dt *= 0.5
            if dt < dt_min:
                break
            continue


        dt2 = 0.5 * dt
        M2 = BandedSPDMatrix(A_band.n, A_band.ml)
        for j in range(A_band.n):
            for i in range(j, min(A_band.n, j + A_band.ml + 1)):
                v = A_band.get(i, j)
                M2.set(i, j, v * dt2)
                if i == j:
                    M2.set(i, j, v * dt2 + 1.0)

        rhs1 = u + dt2 * f
        try:
            u_half = banded_cholesky_solve(M2, rhs1)
        except Exception:
            dt *= 0.5
            if dt < dt_min:
                break
            continue

        f2 = rhs_func(t + dt2, u_half)
        rhs2 = u_half + dt2 * f2
        try:
            u_rich = banded_cholesky_solve(M2, rhs2)
        except Exception:
            dt *= 0.5
            if dt < dt_min:
                break
            continue


        err_est = float(np.linalg.norm(u_full - u_rich))
        norm_u = float(np.linalg.norm(u_full))
        rel_err = err_est / max(norm_u, 1e-15)


        norm_prev = float(np.linalg.norm(u))
        gamma = norm_u / max(norm_prev, 1e-15)

        if gamma > gamma_max:
            blowup_counter += 1
            if blowup_counter >= max_blowup:

                dt *= 0.25
                blowup_counter = 0
                if dt < dt_min:

                    dt = dt_min
                    t += dt
                    u = u_full
                    t_hist.append(t)
                    u_hist.append(u.copy())
                    dt_hist.append(dt)
                    break
                continue
        else:
            blowup_counter = max(0, blowup_counter - 1)


        if rel_err <= tol:
            t += dt
            u = u_full
            t_hist.append(t)
            u_hist.append(u.copy())
            dt_hist.append(dt)

            if rel_err > 0:
                factor = min(2.0, max(0.5, np.sqrt(tol / (rel_err + 1e-15))))
            else:
                factor = 2.0
            dt = min(dt * factor, dt_max)
        else:

            dt *= 0.5
            if dt < dt_min:
                dt = dt_min
                t += dt
                u = u_full
                t_hist.append(t)
                u_hist.append(u.copy())
                dt_hist.append(dt)
                break

    return np.array(t_hist), np.array(u_hist), np.array(dt_hist)


def transient_stokes_step(
    u_n: np.ndarray,
    v_n: np.ndarray,
    p_n: np.ndarray,
    A_viscous: BandedSPDMatrix,
    B_div: np.ndarray,
    f_n: np.ndarray,
    dt: float,
    nu: float = 1.0,
    rho: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
















    raise NotImplementedError("Hole 2: transient_stokes_step 需要补全分步投影法")
