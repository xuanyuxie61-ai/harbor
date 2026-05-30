
import numpy as np
from typing import Tuple, Optional

from ice_constitutive_model import ICE_DENSITY, GRAVITY


def ice_stream_rhs(state: np.ndarray,
                   t: float,
                   params: dict) -> np.ndarray:
    u, v, N = state

    H = params.get('H', 1000.0)
    alpha = params.get('alpha', 0.001)
    delta = params.get('delta', 0.1)
    alpha_u = params.get('alpha_u', 0.01)
    beta_u = params.get('beta_u', 1e-8)
    gamma = params.get('gamma', 0.5)
    omega = params.get('omega', 2.0 * np.pi / (365.25 * 86400.0))
    C_weertman = params.get('C_weertman', 1e-4)
    u0 = params.get('u0', 1.0)
    a_drain = params.get('a_drain', 0.1)
    b_drain = params.get('b_drain', 1e-6)
    c_drain = params.get('c_drain', 1e-10)
    d_season = params.get('d_season', 0.05)
    omega_season = params.get('omega_season', 2.0 * np.pi / (365.25 * 86400.0))
    m_eff = params.get('m_eff', 1.0)


    tau_drive = ICE_DENSITY * GRAVITY * H * np.sin(alpha)


    denom = np.abs(u) + u0
    tau_drag = C_weertman * (N ** 3) * u / denom


    forcing = gamma * np.cos(omega * t)
    season_forcing = d_season * np.cos(omega_season * t)


    u_clipped = np.clip(u, -1e4, 1e4)
    N_clipped = np.clip(N, 1.0, 1e7)
    duffing = -alpha_u * u_clipped - beta_u * (u_clipped ** 3)


    du_dt = v
    tau_drag_safe = C_weertman * (N_clipped ** 3) * u_clipped / (np.abs(u_clipped) + u0)
    dv_dt = (tau_drive - tau_drag_safe - delta * v + duffing + forcing) / m_eff
    dN_dt = a_drain - b_drain * u_clipped - c_drain * (N_clipped ** 3) + season_forcing

    return np.array([du_dt, dv_dt, dN_dt], dtype=np.float64)


def rk4_step(y: np.ndarray, t: float, dt: float, rhs_func) -> np.ndarray:
    k1 = rhs_func(y, t)
    k2 = rhs_func(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ice_stream_oscillation(y0: np.ndarray,
                                 t_span: Tuple[float, float],
                                 dt: float,
                                 params: dict) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span

    if dt <= 0:
        raise ValueError("dt must be positive.")

    nt = int(np.ceil((t_end - t_start) / dt)) + 1
    t_array = np.linspace(t_start, t_end, nt)
    y_array = np.zeros((nt, 3), dtype=np.float64)
    y_array[0] = y0

    y = y0.copy()
    for i in range(1, nt):
        y = rk4_step(y, t_array[i - 1], dt, lambda state, t: ice_stream_rhs(state, t, params))

        y[0] = np.clip(y[0], -5000.0, 5000.0)
        y[1] = np.clip(y[1], -1000.0, 1000.0)
        y[2] = np.clip(y[2], 1e3, 1e7)
        y_array[i] = y

    return t_array, y_array


def detect_stick_slip_events(t_array: np.ndarray,
                             y_array: np.ndarray,
                             velocity_threshold: float = 0.1) -> dict:
    u = y_array[:, 0]


    above = u > velocity_threshold
    below = u <= velocity_threshold

    slip_events = []
    stick_events = []
    for i in range(1, len(u)):
        if below[i - 1] and above[i]:
            slip_events.append(i)
        if above[i - 1] and below[i]:
            stick_events.append(i)


    period = None
    if len(slip_events) >= 2:
        intervals = np.diff(t_array[slip_events])
        period = float(np.mean(intervals))

    stats = {
        'slip_events': slip_events,
        'stick_events': stick_events,
        'mean_velocity': float(np.mean(np.abs(u))),
        'max_velocity': float(np.max(np.abs(u))),
        'oscillation_period_estimate': period,
    }
    return stats


def basal_shear_stress_from_state(state: np.ndarray,
                                   params: dict) -> float:
    u, _, N = state
    C = params.get('C_weertman', 1e-4)
    u0 = params.get('u0', 1.0)
    n = params.get('n_weertman', 1.0)
    denom = np.abs(u) + u0
    tau_b = C * (N ** n) * u / denom
    return float(tau_b)


def driving_stress_from_params(params: dict) -> float:
    H = params.get('H', 1000.0)
    alpha = params.get('alpha', 0.001)
    return ICE_DENSITY * GRAVITY * H * np.sin(alpha)
