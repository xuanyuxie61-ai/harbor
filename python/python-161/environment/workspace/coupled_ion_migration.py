
import numpy as np
from typing import Callable, Tuple


def ode1_euler(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = tspan
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")
    h = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for k in range(n_steps):
        ydot = f(t[k], y[k, :])

        ydot = np.where(np.isfinite(ydot), ydot, 0.0)
        y[k + 1, :] = y[k, :] + h * ydot

        y[k + 1, :] = np.maximum(y[k + 1, :], 0.0)

    return t, y


def coupled_ion_carrier_system(
    t: float,
    y: np.ndarray,
    mu_ion: float,
    D_ion: float,
    mu_e: float,
    mu_h: float,
    E_ext: float,
    G_light: float,
    k_rec: float,
    k_ion_trap: float,
    n_ion_eq: float,
) -> np.ndarray:
    n_ion, n_e, n_h = y[0], y[1], y[2]


    E_ion = 1e-15 * (n_ion - n_ion_eq)
    E_eff = E_ext - E_ion


    J_ion = q * mu_ion * n_ion * E_eff - q * D_ion * (n_ion - n_ion_eq) / 1e-4

    d_n_ion_dt = -J_ion / (q * 1e-4)



    separation_eff = min(abs(E_eff) / (abs(E_ext) + 1e-10), 1.0)
    G_eff = G_light * separation_eff


    R_total = k_rec * n_e * n_h + k_ion_trap * n_ion * n_e

    d_n_e_dt = G_eff - R_total
    d_n_h_dt = G_eff - R_total

    return np.array([d_n_ion_dt, d_n_e_dt, d_n_h_dt])



q = 1.602176634e-19


def solve_hysteresis_cycle(
    voltage_sweep: np.ndarray,
    time_per_step: float = 1e-3,
    mu_ion: float = 1e-10,
    thickness: float = 5e-5,
    n_ion0: float = 1e16,
    n_e0: float = 1e10,
    n_h0: float = 1e10,
    G_light: float = 1e21,
    k_rec: float = 1e-10,
    k_ion_trap: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_v = len(voltage_sweep)
    if n_v < 2:
        raise ValueError("电压扫描点必须 ≥ 2")

    V = voltage_sweep
    J = np.zeros(n_v)
    n_ion_t = np.zeros(n_v)
    E_ion_t = np.zeros(n_v)

    y = np.array([n_ion0, n_e0, n_h0], dtype=float)
    n_ion_eq = n_ion0

    for i in range(n_v):
        V_step = V[i]
        E_ext = V_step / thickness


        def f(t, yy):
            return coupled_ion_carrier_system(
                t, yy, mu_ion, 1e-12, 20.0, 10.0,
                E_ext, G_light, k_rec, k_ion_trap, n_ion_eq,
            )


        _, y_hist = ode1_euler(f, (0.0, time_per_step), y, n_steps=50)
        y = y_hist[-1, :]

        n_ion, n_e, n_h = y

        n_e = max(n_e, 1e12)
        n_h = max(n_h, 1e12)
        n_ion_t[i] = n_ion
        E_ion = 1e-15 * (n_ion - n_ion_eq)
        E_ion_t[i] = E_ion
        E_eff = E_ext - E_ion


        J_ohm = q * (20.0 * n_e + 10.0 * n_h) * abs(E_eff)
        J[i] = J_ohm * 1e3

    return V, J, n_ion_t, E_ion_t


def predprey_style_ion_dynamics(
    tspan: Tuple[float, float] = (0.0, 100.0),
    n_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = 0.1, 0.02, 0.15, 0.01
    y0 = np.array([20.0, 10.0])

    def f(t, y):
        V_I, n_e = y

        V_I = min(V_I, 1e6)
        n_e = min(n_e, 1e6)
        dV = a * V_I - b * V_I * n_e
        dne = -c * n_e + d * V_I * n_e
        return np.array([dV, dne])

    t, y_hist = ode1_euler(f, tspan, y0, n_steps)
    return t, y_hist[:, 0], y_hist[:, 1]


if __name__ == "__main__":

    t, V_I, n_e = predprey_style_ion_dynamics()
    print(f"碘空位浓度范围: [{V_I.min():.2f}, {V_I.max():.2f}]")
    print(f"电子浓度范围: [{n_e.min():.2f}, {n_e.max():.2f}]")


    V_fwd = np.linspace(0.0, 1.0, 21)
    V_rev = np.linspace(1.0, 0.0, 21)
    V_full = np.concatenate([V_fwd, V_rev])
    V, J, n_ion_t, E_ion_t = solve_hysteresis_cycle(V_full, time_per_step=5e-4)
    print(f"最大电流密度: {J.max():.3f} mA/cm^2")
    print(f"离子浓度变化: {n_ion_t.min():.3e} -> {n_ion_t.max():.3e}")
