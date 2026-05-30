
import numpy as np
from typing import Tuple


def fresnel_integrals(x: float, eps: float = 1e-15) -> Tuple[float, float]:
    xa = abs(x)
    px = np.pi * xa
    t = 0.5 * px * xa
    t2 = t * t

    if xa == 0.0:
        return 0.0, 0.0
    elif xa < 2.5:

        r = xa
        c = r
        for k in range(1, 50):
            r = (-0.5 * r * (4.0 * k - 3.0) / k
                 / (2.0 * k - 1.0) / (4.0 * k + 1.0) * t2)
            c += r
            if abs(r) < abs(c) * eps:
                break

        s = xa * t / 3.0
        r = s
        for k in range(1, 50):
            r = (-0.5 * r * (4.0 * k - 1.0) / k
                 / (2.0 * k + 1.0) / (4.0 * k + 3.0) * t2)
            s += r
            if abs(r) < abs(s) * eps:
                if x < 0.0:
                    c = -c
                    s = -s
                return float(c), float(s)
        if x < 0.0:
            c = -c
            s = -s
        return float(c), float(s)
    elif xa < 4.5:
        m = int(np.floor(42.0 + 1.75 * t))
        su = 0.0
        c = 0.0
        s_val = 0.0
        f1 = 0.0
        f0 = 1.0e-100
        for k in range(m, -1, -1):
            f = (2.0 * k + 3.0) * f0 / t - f1
            if k == int(k / 2) * 2:
                c += f
            else:
                s_val += f
            su += (2.0 * k + 1.0) * f * f
            f1 = f0
            f0 = f
        q = np.sqrt(su)
        c = c * xa / q
        s_val = s_val * xa / q
        if x < 0.0:
            c = -c
            s_val = -s_val
        return float(c), float(s_val)
    else:

        r = 1.0
        f = 1.0
        for k in range(1, 20):
            r = (-0.25 * r * (4.0 * k - 1.0) * (4.0 * k - 3.0) / t2)
            f += r
        r = 1.0 / (px * xa)
        g = r
        for k in range(1, 12):
            r = (-0.25 * r * (4.0 * k + 1.0) * (4.0 * k - 1.0) / t2)
            g += r
        t0 = t - np.floor(t / (2.0 * np.pi)) * 2.0 * np.pi
        c = 0.5 + (f * np.sin(t0) - g * np.cos(t0)) / px
        s_val = 0.5 - (f * np.cos(t0) + g * np.sin(t0)) / px
        if x < 0.0:
            c = -c
            s_val = -s_val
        return float(c), float(s_val)


def fresnel_diffraction_1d(aperture_field: np.ndarray,
                           x_aperture: np.ndarray,
                           x_observation: np.ndarray,
                           wavelength: float,
                           z: float) -> np.ndarray:
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fresnel_diffraction_1d: wavelength and z must be > 0")
    k = 2.0 * np.pi / wavelength
    dx = x_aperture[1] - x_aperture[0]
    prefactor = np.exp(1j * k * z) / np.sqrt(1j * wavelength * z) * dx

    diff = x_observation[:, np.newaxis] - x_aperture[np.newaxis, :]
    phase = np.exp(1j * k * diff ** 2 / (2.0 * z))
    E_out = prefactor * np.sum(aperture_field[np.newaxis, :] * phase, axis=1)
    return E_out


def fiber_output_diffraction(E_fundamental: np.ndarray,
                              r_fiber: np.ndarray,
                              r_out: np.ndarray,
                              wavelength: float,
                              z: float,
                              core_radius: float) -> np.ndarray:
    from scipy.special import j0
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fiber_output_diffraction: wavelength and z must be > 0")
    k = 2.0 * np.pi / wavelength
    E_out = np.zeros_like(r_out, dtype=complex)
    dr = r_fiber[1] - r_fiber[0]
    for i, r in enumerate(r_out):
        integrand = E_fundamental * j0(k * r * r_fiber / z) * np.exp(1j * k * r_fiber ** 2 / (2.0 * z)) * r_fiber
        E_out[i] = (2.0 * np.pi * k / (1j * z)) * np.sum(integrand) * dr
    return E_out


def fresnel_number(a: float, wavelength: float, z: float) -> float:
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fresnel_number: wavelength and z must be > 0")
    return a * a / (wavelength * z)
