
import numpy as np
from typing import Callable, Tuple, Optional



RK45_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
    [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0],
    [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0],
    [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0],
])

RK45_C = np.array([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0])

RK45_B5 = np.array([16.0 / 135.0, 0.0, 6656.0 / 12825.0,
                     28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0])

RK45_B4 = np.array([25.0 / 216.0, 0.0, 1408.0 / 2565.0,
                     2197.0 / 4104.0, -1.0 / 5.0, 0.0])


def semiclassical_rhs(
    state: np.ndarray,
    band_energies_func: Callable,
    E_field: np.ndarray,
    B_field: float,
    band_index: int,
) -> np.ndarray:
    k = state[0:2]
    hbar = 0.6582119
    e_charge = 1.0


    B_eff = e_charge * B_field * 1.519e-4 / hbar






    raise NotImplementedError("Hole 3: implement semiclassical RHS (group velocity + Lorentz force)")


def rk45_step(
    f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    s = 6
    k = np.zeros((s, y.size))

    for i in range(s):
        yi = y.copy()
        for j in range(i):
            yi += h * RK45_A[i, j] * k[j]
        k[i] = f(yi)

    y5 = y + h * np.dot(RK45_B5, k)
    y4 = y + h * np.dot(RK45_B4, k)
    error = np.abs(y5 - y4)


    tol = 1e-6
    scale = tol + tol * np.abs(y5)
    err_norm = np.linalg.norm(error / scale)
    if err_norm == 0.0:
        h_new = 2.0 * h
    else:
        h_new = h * min(5.0, max(0.1, 0.9 * (1.0 / err_norm) ** 0.2))

    return y5, error, h_new


def integrate_trajectory(
    band_energies_func: Callable,
    k0: np.ndarray,
    r0: np.ndarray,
    E_field: np.ndarray,
    B_field: float,
    band_index: int,
    t_max: float = 1000.0,
    h_init: float = 1.0,
    h_min: float = 0.01,
    h_max: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.concatenate([np.asarray(k0, dtype=float),
                        np.asarray(r0, dtype=float)])
    t = 0.0
    h = h_init

    t_list = [t]
    y_list = [y.copy()]

    def rhs(state):
        return semiclassical_rhs(
            state, band_energies_func, E_field, B_field, band_index
        )

    while t < t_max:
        y_next, error, h_suggest = rk45_step(rhs, y, t, h)
        err_max = np.max(error)
        if err_max > 1e-3 and h > h_min:

            h = max(h_suggest, h_min)
            continue

        y = y_next
        t += h
        h = max(h_min, min(h_max, h_suggest))

        t_list.append(t)
        y_list.append(y.copy())

        if len(t_list) > 50000:
            break

    t_array = np.array(t_list)
    y_array = np.array(y_list)
    k_array = y_array[:, 0:2]
    r_array = y_array[:, 2:4]
    return t_array, k_array, r_array


def cyclotron_frequency(
    effective_mass: float,
    B_field: float,
) -> float:
    m_e = 5.685e-5

    return 0.1759 * B_field / effective_mass


def lax_wendroff_predictor_corrector(
    f: Callable,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    y_star = y + 0.5 * h * f(y)
    y_new = y + h * f(y_star)
    return y_new
