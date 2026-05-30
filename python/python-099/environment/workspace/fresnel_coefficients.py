
import numpy as np
import math
from utils import fresnel_cos, fresnel_sin, llsq_fit, llsq_fit_through_origin, safe_divide, clamp


def fresnel_reflection_coefficient(
    n1: complex, n2: complex, theta1: float, polarization: str = "TE"
) -> complex:
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    n1 = complex(n1)
    n2 = complex(n2)

    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1

    sin_theta2 = complex(
        clamp(sin_theta2.real, -1.0, 1.0),
        sin_theta2.imag,
    )
    theta2 = np.arcsin(sin_theta2)

    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)

    if polarization.upper() == "TE":
        num = n1 * cos_theta1 - n2 * cos_theta2
        den = n1 * cos_theta1 + n2 * cos_theta2
    elif polarization.upper() == "TM":
        num = n2 * cos_theta1 - n1 * cos_theta2
        den = n2 * cos_theta1 + n1 * cos_theta2
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")

    r = safe_divide(num, den, fallback=0.0)
    return complex(r)


def fresnel_transmission_coefficient(
    n1: complex, n2: complex, theta1: float, polarization: str = "TE"
) -> complex:
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    n1 = complex(n1)
    n2 = complex(n2)

    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1
    sin_theta2 = complex(
        clamp(sin_theta2.real, -1.0, 1.0),
        sin_theta2.imag,
    )
    theta2 = np.arcsin(sin_theta2)

    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)

    if polarization.upper() == "TE":
        den = n1 * cos_theta1 + n2 * cos_theta2
        num = 2.0 * n1 * cos_theta1
    elif polarization.upper() == "TM":
        den = n2 * cos_theta1 + n1 * cos_theta2
        num = 2.0 * n1 * cos_theta1
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")

    t = safe_divide(num, den, fallback=0.0)
    return complex(t)


def reflection_power_ratio(r: complex) -> float:
    return abs(r) ** 2


def transmission_power_ratio(
    r: complex, n1: complex, n2: complex, theta1: float
) -> float:
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    t = 1.0 + r


    n1 = complex(n1)
    n2 = complex(n2)
    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1
    sin_theta2 = complex(clamp(sin_theta2.real, -1.0, 1.0), sin_theta2.imag)
    theta2 = np.arcsin(sin_theta2)
    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)


    den = n1 * cos_theta1 + n2 * cos_theta2
    num = 2.0 * n1 * cos_theta1
    t_exact = safe_divide(num, den, fallback=0.0)
    t_exact = complex(t_exact)

    ratio = safe_divide(
        (n2.real / max(n1.real, 1e-15)) * abs(t_exact) ** 2 * (cos_theta2.real / cos_theta1),
        1.0,
        fallback=0.0,
    )

    return clamp(float(ratio), 0.0, 2.0)


def diffraction_correction_fresnel(
    x: float, y: float, z: float, k0: float
) -> complex:
    x = max(float(x), 1e-6)
    y = float(y)
    z = float(z)
    k0 = max(float(k0), 1e-6)

    v = math.sqrt(2.0 * k0 / math.pi) * (y + z) / math.sqrt(x)
    c_v = fresnel_cos(v)
    s_v = fresnel_sin(v)

    D_f = 0.5 * (1.0 + 1j) * (c_v + 1j * s_v)
    return D_f


def invert_reflection_for_epsilon(
    frequencies: np.ndarray,
    measured_R: np.ndarray,
    theta: float = 0.0,
    polarization: str = "TE",
) -> tuple:
    frequencies = np.asarray(frequencies, dtype=float)
    measured_R = np.asarray(measured_R, dtype=float)
    if frequencies.size != measured_R.size or frequencies.size < 3:
        raise ValueError("Need at least 3 frequency points.")


    measured_R = np.clip(measured_R, 1e-6, 1.0 - 1e-6)
    omega = 2.0 * math.pi * frequencies


    xi = 1.0 / (omega ** 2)








    sqrt_R = np.sqrt(measured_R)
    slope, residual = llsq_fit_through_origin(xi, sqrt_R)
    A_fit = 4.0 * slope


    predicted = slope * xi
    rms = np.sqrt(np.mean((sqrt_R - predicted) ** 2))

    return A_fit, residual, rms


def multilayer_reflection_stack(
    n_layers: np.ndarray,
    d_layers: np.ndarray,
    omega: float,
    theta0: float = 0.0,
    polarization: str = "TE",
) -> complex:















    raise NotImplementedError("Hole 2: multilayer_reflection_stack not implemented.")
