
import numpy as np
from typing import Tuple, List


def ymdf_to_jed_gregorian(y: int, m: int, d: int, f: float = 0.0) -> float:
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524
    jed = jd - 2400000.5 + f
    return float(jed)


def jed_to_ymdhms_common(jed: float) -> Tuple[int, int, int, int, int, float]:
    jd = jed + 2400000.5
    jd_int = int(np.floor(jd))
    f = jd - jd_int

    if jd_int < 2299161:
        a = jd_int
    else:
        alpha = int((jd_int - 1867216.25) / 36524.25)
        a = jd_int + 1 + alpha - int(alpha / 4)

    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e)
    if e < 14:
        month = e - 1
    else:
        month = e - 13
    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    hours = f * 24.0
    hour = int(hours)
    minutes = (hours - hour) * 60.0
    minute = int(minutes)
    second = (minutes - minute) * 60.0
    return year, month, day, hour, minute, second


def compute_tidal_phases(jed: float) -> dict:

    d = jed - 2451545.0


    T = d / 36525.0
    s = 218.316 + 13.176396 * d
    h = 280.460 + 0.985647 * d
    p = 83.353 + 0.111404 * d
    N_prime = 125.045 - 0.052954 * d


    deg2rad = np.pi / 180.0
    s_rad = (s % 360.0) * deg2rad
    h_rad = (h % 360.0) * deg2rad
    p_rad = (p % 360.0) * deg2rad
    N_rad = (N_prime % 360.0) * deg2rad
    tau = (15.0 * ((d % 1.0) * 24.0) - s) % 360.0
    tau_rad = tau * deg2rad

    phases = {
        'M2': 2.0 * tau_rad - 2.0 * s_rad + 2.0 * h_rad,
        'S2': 2.0 * tau_rad,
        'K1': tau_rad + h_rad,
        'O1': tau_rad - 2.0 * s_rad + h_rad,
        'N2': 2.0 * tau_rad - 3.0 * s_rad + p_rad + 2.0 * h_rad,
    }
    return phases


def generate_tidal_elevation(
    t_hours: np.ndarray,
    jed_base: float,
    amplitudes: dict = None,
) -> np.ndarray:
    if amplitudes is None:
        amplitudes = {
            'M2': 0.80,
            'S2': 0.35,
            'K1': 0.25,
            'O1': 0.20,
            'N2': 0.15,
        }

    phases_base = compute_tidal_phases(jed_base)
    omega = {
        'M2': 2.0 * np.pi / 12.4206012,
        'S2': 2.0 * np.pi / 12.0,
        'K1': 2.0 * np.pi / 23.934472,
        'O1': 2.0 * np.pi / 25.819338,
        'N2': 2.0 * np.pi / 12.658348,
    }

    t = np.asarray(t_hours, dtype=float)
    eta = np.zeros_like(t)
    for comp, A in amplitudes.items():
        eta += A * np.cos(omega[comp] * t + phases_base[comp])
    return eta


def generate_tidal_velocity(
    t_hours: np.ndarray,
    jed_base: float,
    max_velocity: float = 2.5,
    phase_lag_deg: float = 45.0,
) -> np.ndarray:
    eta = generate_tidal_elevation(t_hours, jed_base)

    dt = np.diff(t_hours, prepend=t_hours[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    deta = np.gradient(eta, t_hours)

    u_raw = deta
    u_max = np.max(np.abs(u_raw))
    if u_max < 1e-12:
        return np.zeros_like(t_hours)
    u = max_velocity * u_raw / u_max
    return u
