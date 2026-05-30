
import numpy as np
import math
from special_functions import cos_power_int


def disk_unit_sample(n, radius=1.0):
    rng = np.random.default_rng(42)
    r = radius * np.sqrt(rng.random(n))
    theta = 2.0 * math.pi * rng.random(n)
    points = np.zeros((n, 2), dtype=float)
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)
    return points


def rayleigh_integral_piston(observer, disk_samples, u_n, k, rho0=1.225, c0=343.0):
    omega = k * c0
    n_samp = disk_samples.shape[0]
    disk_area = math.pi * (np.max(np.linalg.norm(disk_samples, axis=1)) ** 2)
    dS = disk_area / n_samp

    if np.isscalar(u_n):
        u_n = np.full(n_samp, u_n)

    p = 0.0 + 0.0j
    for i in range(n_samp):
        r_prime = np.array([disk_samples[i, 0], disk_samples[i, 1], 0.0])
        R_vec = observer - r_prime
        R = np.linalg.norm(R_vec)
        if R < 1e-6:
            R = 1e-6
        p += u_n[i] * np.exp(-1j * k * R) / R * dS

    p = p * (1j * rho0 * omega / (2.0 * math.pi))
    return p


def piston_directivity_factor(ka, n_points=180):
    from scipy.special import j1

    theta = np.linspace(0, math.pi / 2, n_points)
    dtheta = theta[1] - theta[0]

    D = np.zeros_like(theta)
    for i, th in enumerate(theta):
        s = math.sin(th)
        if ka * s < 1e-6:
            D[i] = 1.0
        else:
            D[i] = 2.0 * j1(ka * s) / (ka * s)


    integrand = D ** 2 * np.sin(theta)
    integral = np.trapz(integrand, theta)

    if integral < 1e-12:
        return 0.0


    omni_int = 2.0 * math.pi * integral
    di = 10.0 * math.log10(4.0 * math.pi / omni_int)
    return di


def piston_radiation_resistance(ka):
    from scipy.special import j1
    if ka < 1e-6:
        return (ka ** 2) / 2.0
    return 1.0 - 2.0 * j1(2.0 * ka) / (2.0 * ka)


def piston_radiation_reactance(ka):
    from scipy.special import struve
    if ka < 1e-6:
        return 8.0 * ka / (3.0 * math.pi)
    return 2.0 * struve(1, 2.0 * ka) / (2.0 * ka)
