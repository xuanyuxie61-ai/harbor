
import numpy as np
from typing import Tuple, Callable
from utils import EPS_MACHINE, EPS_SQRT
from spin_quaternion import axis_angle_to_q, q_rotate_vector


def local_min_brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsi: float = EPS_SQRT,
    t: float = 1e-10,
    max_calls: int = 500,
) -> Tuple[float, float, int]:
    c_ratio = 0.5 * (3.0 - np.sqrt(5.0))
    sa, sb = a, b
    x = sa + c_ratio * (b - a)
    w, v = x, x
    e = 0.0
    fx = f(x)
    calls = 1
    fw, fv = fx, fx

    while calls < max_calls:
        m = 0.5 * (sa + sb)
        tol = epsi * abs(x) + t
        t2 = 2.0 * tol
        if abs(x - m) <= t2 - 0.5 * (sb - sa):
            break

        r_val = 0.0
        q_val = 0.0
        p_val = 0.0
        if tol < abs(e):
            r_val = (x - w) * (fx - fv)
            q_val = (x - v) * (fx - fw)
            p_val = (x - v) * q_val - (x - w) * r_val
            q_val = 2.0 * (q_val - r_val)
            if 0.0 < q_val:
                p_val = -p_val
            q_val = abs(q_val)
            r_val = e
            e = d


        if calls == 1:
            if x < m:
                e = sb - x
            else:
                e = sa - x
            d = c_ratio * e
        else:
            if (
                abs(p_val) < abs(0.5 * q_val * r_val)
                and q_val * (sa - x) < p_val < q_val * (sb - x)
            ):

                d = p_val / q_val
                u = x + d
                if (u - sa) < t2 or (sb - u) < t2:
                    if x < m:
                        d = tol
                    else:
                        d = -tol
            else:

                if x < m:
                    e = sb - x
                else:
                    e = sa - x
                d = c_ratio * e

        if tol <= abs(d):
            u = x + d
        elif 0.0 < d:
            u = x + tol
        else:
            u = x - tol

        fu = f(u)
        calls += 1

        if fu <= fx:
            if u < x:
                sb = x
            else:
                sa = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u
            if fu <= fw or abs(w - x) < EPS_MACHINE:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or abs(v - x) < EPS_MACHINE or abs(v - w) < EPS_MACHINE:
                v = u
                fv = fu

    return x, fx, calls


def line_search_spin_rotation(
    J: np.ndarray,
    spins: np.ndarray,
    site_idx: int,
    axis: np.ndarray,
    a: float = -np.pi,
    b: float = np.pi,
) -> Tuple[float, np.ndarray, float]:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < EPS_MACHINE:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / norm

    N = spins.shape[0]

    def energy_at_angle(theta: float) -> float:
        q = axis_angle_to_q(axis, theta)
        s_rot = q_rotate_vector(q, spins[site_idx])
        new_spins = spins.copy()
        new_spins[site_idx] = s_rot



        raise NotImplementedError("Hole_2: 请实现 energy_at_angle 中的能量计算")

    theta_opt, e_min, _ = local_min_brent(energy_at_angle, a, b)
    q_opt = axis_angle_to_q(axis, theta_opt)
    spin_new = q_rotate_vector(q_opt, spins[site_idx])
    return theta_opt, spin_new, e_min


def greedy_relaxation(
    J: np.ndarray,
    spins: np.ndarray,
    n_sweeps: int = 10,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, list]:
    N = spins.shape[0]
    spins = spins.copy()
    history = []
    e_old = float("inf")
    for sweep in range(n_sweeps):
        for i in range(N):

            H_i = J[i, :] @ spins
            H_norm = np.linalg.norm(H_i)
            if H_norm < EPS_MACHINE:
                continue
            axis = H_i / H_norm
            _, spins[i], _ = line_search_spin_rotation(J, spins, i, axis)

        H = J @ spins
        e_new = 0.5 * np.sum(spins * H) + 0.05 * np.sum(spins[:, 2] ** 2)
        history.append(float(e_new))
        if abs(e_old - e_new) < tol:
            break
        e_old = e_new
    return spins, float(e_new), history


def simulated_annealing_spin_glass(
    J: np.ndarray,
    spins_init: np.ndarray,
    T_init: float = 2.0,
    T_final: float = 1e-4,
    cooling_rate: float = 0.995,
    steps_per_T: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, float, list]:
    np.random.seed(seed)
    spins = spins_init.copy()
    N = spins.shape[0]
    H = J @ spins
    e_current = 0.5 * np.sum(spins * H) + 0.05 * np.sum(spins[:, 2] ** 2)
    spins_best = spins.copy()
    e_best = e_current
    history = [e_current]
    T = T_init

    while T > T_final:
        for _ in range(steps_per_T):
            i = np.random.randint(N)

            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + EPS_MACHINE)
            theta = np.random.uniform(-0.5, 0.5)
            q = axis_angle_to_q(axis, theta)
            s_old = spins[i].copy()
            s_new = q_rotate_vector(q, s_old)
            spins[i] = s_new
            H_new = J @ spins
            e_new = 0.5 * np.sum(spins * H_new) + 0.05 * np.sum(spins[:, 2] ** 2)
            delta_e = e_new - e_current
            if delta_e < 0.0 or np.random.rand() < np.exp(-delta_e / T):
                e_current = e_new
                if e_current < e_best:
                    e_best = e_current
                    spins_best = spins.copy()
            else:
                spins[i] = s_old
        history.append(e_current)
        T *= cooling_rate

    return spins_best, e_best, history
