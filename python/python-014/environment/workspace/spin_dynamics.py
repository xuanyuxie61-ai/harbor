
import numpy as np
from typing import Tuple, Callable, Optional
from utils import EPS_MACHINE, clip_spin_norm
from spin_quaternion import q_rotate_vector, axis_angle_to_q


def effective_field(
    J: np.ndarray,
    spins: np.ndarray,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> np.ndarray:
    N = spins.shape[0]
    H = J @ spins
    if H_ext is not None:
        H = H + H_ext
    if K_anis > EPS_MACHINE and anisotropy_axis is not None:
        n = np.array(anisotropy_axis, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > EPS_MACHINE:
            n = n / n_norm
        proj = np.sum(spins * n, axis=1, keepdims=True)
        H = H + 2.0 * K_anis * proj * n
    return H


def llg_rhs(
    spins: np.ndarray,
    J: np.ndarray,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> np.ndarray:
    H = effective_field(J, spins, H_ext, anisotropy_axis, K_anis)

    cross = np.cross(spins, H)

    dot = np.sum(spins * H, axis=1, keepdims=True)
    damping = spins * dot - H
    dSdt = -gamma * cross + alpha * damping

    tangent_proj = dSdt - spins * np.sum(dSdt * spins, axis=1, keepdims=True)
    return tangent_proj


def llg_rhs_flat(t: float, y_flat: np.ndarray, J: np.ndarray, **kwargs) -> np.ndarray:
    N = J.shape[0]
    spins = y_flat.reshape((N, 3))

    spins = np.array([clip_spin_norm(s) for s in spins])
    dSdt = llg_rhs(spins, J, **kwargs)
    return dSdt.ravel()


def euler_integrate_llg(
    J: np.ndarray,
    spins0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    h = (tf - t0) / n_steps
    N = spins0.shape[0]
    t_arr = np.linspace(t0, tf, n_steps + 1)
    spins_traj = np.zeros((n_steps + 1, N, 3), dtype=float)
    spins = spins0.copy()
    spins_traj[0] = spins

    for i in range(n_steps):
        dSdt = llg_rhs(spins, J, gamma, alpha, H_ext, anisotropy_axis, K_anis)
        spins = spins + h * dSdt

        for j in range(N):
            spins[j] = clip_spin_norm(spins[j])
        spins_traj[i + 1] = spins

    return t_arr, spins_traj


def trapezoidal_integrate_llg(
    J: np.ndarray,
    spins0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    h = (tf - t0) / n_steps
    N = spins0.shape[0]
    t_arr = np.linspace(t0, tf, n_steps + 1)
    spins_traj = np.zeros((n_steps + 1, N, 3), dtype=float)
    spins = spins0.copy()
    spins_traj[0] = spins

    kwargs = {
        "gamma": gamma,
        "alpha": alpha,
        "H_ext": H_ext,
        "anisotropy_axis": anisotropy_axis,
        "K_anis": K_anis,
    }

    for i in range(n_steps):
        f_old = llg_rhs(spins, J, **kwargs)

        spins_new = spins + h * f_old
        for j in range(N):
            spins_new[j] = clip_spin_norm(spins_new[j])


        for _newton in range(newton_max_iter):
            f_new = llg_rhs(spins_new, J, **kwargs)
            residual = spins_new - spins - 0.5 * h * (f_old + f_new)

            delta = -residual
            spins_new = spins_new + 0.5 * delta
            for j in range(N):
                spins_new[j] = clip_spin_norm(spins_new[j])
            if np.linalg.norm(delta) < newton_tol:
                break

        spins = spins_new
        spins_traj[i + 1] = spins

    return t_arr, spins_traj


def brusselator_like_spin_pump(
    t: float,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
) -> np.ndarray:
    u, v = y[0], y[1]
    dudt = a + u * u * v - (b + 1.0) * u
    dvdt = b * u - u * u * v
    return np.array([dudt, dvdt])


def integrate_brusselator_pump(
    a: float = 1.0,
    b: float = 3.0,
    y0: np.ndarray = None,
    t_span: Tuple[float, float] = (0.0, 20.0),
    n_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    if y0 is None:
        y0 = np.array([0.5, 1.0])
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_traj = np.zeros((n_steps + 1, 2), dtype=float)
    y = y0.copy()
    y_traj[0] = y
    for i in range(n_steps):
        dydt = brusselator_like_spin_pump(t_arr[i], y, a, b)
        y = y + h * dydt
        y_traj[i + 1] = y
    return t_arr, y_traj


def fisher_kpp_domain_wall_exact(
    t: float,
    x: np.ndarray,
    a: float = 1.0,
    c: float = 5.0 / np.sqrt(6.0),
    k: float = np.sqrt(6.0) / 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = x - c * t
    exp_kz = np.exp(k * z)
    denom = 1.0 + a * exp_kz
    u = 1.0 / (denom ** 2)
    ut = 2.0 * c * a * k * exp_kz / (denom ** 3)
    ux = -2.0 * a * k * exp_kz / (denom ** 3)
    uxx = 6.0 * (a ** 2) * (k ** 2) * np.exp(2.0 * k * z) / (denom ** 4) - \
          2.0 * a * (k ** 2) * exp_kz / (denom ** 3)
    return u, ut, ux, uxx


def domain_wall_magnetization(
    t: float,
    x: np.ndarray,
    Ms: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, ut, ux, uxx = fisher_kpp_domain_wall_exact(t, x)
    Mz = Ms * (2.0 * u - 1.0)
    dMz_dt = 2.0 * Ms * ut
    dMz_dx = 2.0 * Ms * ux
    d2Mz_dx2 = 2.0 * Ms * uxx
    return Mz, dMz_dt, dMz_dx, d2Mz_dx2


def compute_magnetization_trajectory(spins_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = spins_traj.shape[1]
    M = np.mean(spins_traj, axis=1)
    return M[:, 0], M[:, 1], M[:, 2]
