
import numpy as np
from typing import Dict, Tuple, List
from radial_solver import evolve_radial_modes, alpha_effect_source, omega_effect_source
from adaptive_rk import rk45_adaptive
from special_functions import safe_div





def differential_rotation_profile(r: np.ndarray, r_icb: float, r_cmb: float,
                                   Omega0: float, shear_strength: float) -> np.ndarray:
    d = r_cmb - r_icb
    if d <= 0.0:
        return np.zeros_like(r)
    x = (r - r_icb) / d
    return Omega0 * (1.0 - shear_strength * x)


def alpha_effect_profile(r: np.ndarray, r_icb: float, r_cmb: float,
                          alpha0: float) -> np.ndarray:
    d = r_cmb - r_icb
    if d <= 0.0:
        return np.zeros_like(r)
    alpha = alpha0 * np.sin(np.pi * (r - r_icb) / d)
    alpha[r <= r_icb] = 0.0
    alpha[r >= r_cmb] = 0.0
    return alpha





def induction_rhs(state: np.ndarray, r: np.ndarray,
                  r_icb: float, r_cmb: float,
                  eta: float, alpha0: float,
                  Omega0: float, shear_strength: float,
                  mode_list: List[Tuple[int, int]]) -> np.ndarray:
    n_r = len(r)
    n_modes = len(mode_list)
    rhs = np.zeros_like(state)


    T_modes = {}
    P_modes = {}
    offset = 0
    for key in mode_list:
        T_modes[key] = state[offset: offset + n_r]
        offset += n_r
        P_modes[key] = state[offset: offset + n_r]
        offset += n_r


    Omega_profile = differential_rotation_profile(r, r_icb, r_cmb, Omega0, shear_strength)
    alpha_profile = alpha_effect_profile(r, r_icb, r_cmb, alpha0)







    raise NotImplementedError("Hole_1: induction_rhs 核心循环待实现")

    return rhs





def encode_state(T_modes: Dict[Tuple[int, int], np.ndarray],
                 P_modes: Dict[Tuple[int, int], np.ndarray],
                 mode_list: List[Tuple[int, int]]) -> np.ndarray:
    n_r = len(T_modes[mode_list[0]])
    state = np.zeros(len(mode_list) * 2 * n_r, dtype=float)
    offset = 0
    for key in mode_list:
        state[offset: offset + n_r] = T_modes[key]
        offset += n_r
        state[offset: offset + n_r] = P_modes[key]
        offset += n_r
    return state


def decode_state(state: np.ndarray,
                 mode_list: List[Tuple[int, int]],
                 n_r: int) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    T_modes = {}
    P_modes = {}
    offset = 0
    for key in mode_list:
        T_modes[key] = state[offset: offset + n_r].copy()
        offset += n_r
        P_modes[key] = state[offset: offset + n_r].copy()
        offset += n_r
    return T_modes, P_modes





def run_kinematic_dynamo(
    r: np.ndarray,
    r_icb: float,
    r_cmb: float,
    eta: float,
    alpha0: float,
    Omega0: float,
    shear_strength: float,
    l_max: int,
    t_end: float,
    dt_init: float,
    save_interval: float,
    adaptive_tol: float = 1e-6
) -> Tuple[List[float], List[Dict[Tuple[int, int], complex]], List[Dict[Tuple[int, int], complex]]]:
    n_r = len(r)
    mode_list = []
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            mode_list.append((l, m))

    n_modes = len(mode_list)


    T_modes = {}
    P_modes = {}
    for key in mode_list:
        l, m = key

        P_init = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb)) * (l == 1 and m == 0)
        P_init[r <= r_icb] = 0.0
        P_init[r >= r_cmb] = 0.0
        P_modes[key] = P_init


        rng = np.random.default_rng(seed=42 + l * 100 + abs(m))
        T_modes[key] = 0.01 * rng.random(n_r) * (np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb)))

    state0 = encode_state(T_modes, P_modes, mode_list)


    def rhs_func(t, y):
        return induction_rhs(y, r, r_icb, r_cmb, eta, alpha0,
                             Omega0, shear_strength, mode_list)


    print(f"[Dynamo] Starting simulation: l_max={l_max}, modes={n_modes}, t_end={t_end/1e3/365.25/24/3600:.1f} kyrs")
    t_array, y_array, e_array = rk45_adaptive(rhs_func, (0.0, t_end), state0,
                                               dt_init=dt_init, tol=adaptive_tol)
    print(f"[Dynamo] Simulation complete: {len(t_array)} steps, final dt={t_array[-1]-t_array[-2] if len(t_array)>1 else 0:.3e} s")


    times = []
    T_history = []
    P_history = []
    next_save = 0.0

    for i in range(len(t_array)):
        if t_array[i] >= next_save or i == 0 or i == len(t_array) - 1:
            times.append(t_array[i])
            Ti, Pi = decode_state(y_array[i], mode_list, n_r)

            T_coeffs = {key: float(np.mean(Ti[key])) + 0.0j for key in mode_list}
            P_coeffs = {key: float(np.mean(Pi[key])) + 0.0j for key in mode_list}
            T_history.append(T_coeffs)
            P_history.append(P_coeffs)
            next_save += save_interval

    return times, T_history, P_history





def _self_test():
    r_icb = 1221e3
    r_cmb = 3480e3
    n_r = 16
    r = np.linspace(r_icb, r_cmb, n_r)
    eta = 2.0
    alpha0 = 0.5
    Omega0 = 7.292e-5
    shear = 1.0

    mode_list = [(1, 0), (2, 0)]
    n_modes = len(mode_list)
    state = np.zeros(n_modes * 2 * n_r, dtype=float)
    state[n_r: 2 * n_r] = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb))

    rhs = induction_rhs(state, r, r_icb, r_cmb, eta, alpha0, Omega0, shear, mode_list)
    assert rhs.shape == state.shape
    assert not np.isnan(rhs).any()


    times, T_hist, P_hist = run_kinematic_dynamo(
        r, r_icb, r_cmb, eta, alpha0, Omega0, shear,
        l_max=2, t_end=1e4 * 365.25 * 24 * 3600, dt_init=1e3 * 365.25 * 24 * 3600,
        save_interval=5e3 * 365.25 * 24 * 3600
    )
    assert len(times) > 0
    print("dynamo_induction: self-test passed.")


if __name__ == "__main__":
    _self_test()
