
import numpy as np
from parameters import (
    get_fusion_params, MD, MT, QE, DT_ENERGY_FUS
)


def dt_reactivity_bosch_hale(Ti_keV):
    Ti = np.asarray(Ti_keV, dtype=float)
    Ti = np.clip(Ti, 0.1, 100.0)


    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.35000e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5
    bg = 34.3827
    mr = 1124656.0

    theta = Ti / (1.0 - Ti * (c2 + Ti * (c4 + Ti * c6)) /
                   (1.0 + Ti * (c3 + Ti * (c5 + Ti * c7))))
    xi = (bg * bg / (4.0 * theta)) ** (1.0 / 3.0)

    sigmav = c1 * theta * np.sqrt(xi / (mr * Ti ** 3)) * np.exp(-3.0 * xi)
    return sigmav


def simplified_reactivity(Ti_keV):
    Ti = np.asarray(Ti_keV, dtype=float)
    Ti = np.clip(Ti, 0.2, 200.0)
    return 1.0e-18 * np.exp(-18.0 / (Ti ** 0.35))


def fusion_derivative(t, y, k_eff, tau_p, tau_He, S_D, S_T):
    y = np.asarray(y, dtype=float)
    if y.shape != (4,):
        raise ValueError("状态向量 y 必须为 4 维 [n_D, n_T, n_He, n_n]")
    nD, nT, nHe, nn = y


    nD = max(nD, 0.0)
    nT = max(nT, 0.0)
    nHe = max(nHe, 0.0)
    nn = max(nn, 0.0)

    reaction_rate = k_eff * nD * nT

    dnD_dt = -reaction_rate + S_D - nD / tau_p
    dnT_dt = -reaction_rate + S_T - nT / tau_p
    dnHe_dt = reaction_rate - nHe / tau_He
    dnn_dt = reaction_rate - nn / tau_He

    return np.array([dnD_dt, dnT_dt, dnHe_dt, dnn_dt], dtype=float)


def simulate_fusion_burn(fusion_params=None, Ti_keV=15.0, n_steps=2000):
    if fusion_params is None:
        fusion_params = get_fusion_params()

    t0 = fusion_params["t0"]
    y0 = fusion_params["y0"].copy()
    tstop = fusion_params["tstop"]


    if len(y0) == 3:
        y0 = np.array([y0[0], y0[1], y0[2], 0.0])

    k_eff = simplified_reactivity(Ti_keV)
    tau_p = 2.0
    tau_He = 5.0
    S_D = 5.0e18
    S_T = 5.0e18
    P_heat = 5.0e5

    def deriv(t, y):
        return fusion_derivative(t, y, k_eff, tau_p, tau_He, S_D, S_T)


    h = (tstop - t0) / n_steps
    t_arr = np.linspace(t0, tstop, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, 4))
    y_arr[0, :] = y0

    for n in range(n_steps):
        yn = y_arr[n, :]
        dydt = deriv(t_arr[n], yn)
        y_arr[n + 1, :] = yn + h * dydt

        y_arr[n + 1, :] = np.maximum(y_arr[n + 1, :], 0.0)


    nD = y_arr[:, 0]
    nT = y_arr[:, 1]
    P_fus_arr = nD * nT * k_eff * DT_ENERGY_FUS * QE / 4.0


    Q_factor = np.zeros_like(P_fus_arr)
    nonzero = P_heat > 1e-30
    Q_factor[nonzero] = P_fus_arr[nonzero] / P_heat

    return t_arr, y_arr, P_fus_arr, Q_factor


def compute_bremsstrahlung(n_e, T_e_eV, Z_eff=1.8):
    C_b = 1.69e-38
    T_e = np.asarray(T_e_eV)
    T_e_safe = np.where(T_e < 1.0, 1.0, T_e)
    return C_b * Z_eff * np.asarray(n_e) ** 2 * np.sqrt(T_e_safe)


def compute_alpha_heating(n_e, Ti_keV):
    E_alpha = 3.52e6
    sigmav = simplified_reactivity(Ti_keV)
    return 0.25 * np.asarray(n_e) ** 2 * sigmav * E_alpha * QE


def lawson_criterion(Ti_keV, eta=0.3):
    Ti = np.asarray(Ti_keV)
    sigmav = simplified_reactivity(Ti)


    E_fus_J = DT_ENERGY_FUS * QE
    Ti_J = Ti * 1e3 * QE
    ntau = 3.0 * Ti_J / (0.25 * sigmav * E_fus_J * eta)
    return ntau
