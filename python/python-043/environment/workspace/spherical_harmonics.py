
import numpy as np
from typing import Tuple, List


def associated_legendre(l: int, m: int, x: float) -> float:
    x = float(x)
    if abs(x) > 1.0 + 1e-12:
        raise ValueError("|x| must be <= 1 for Legendre functions")
    x = max(-1.0, min(1.0, x))
    m = abs(m)
    if m > l:
        return 0.0


    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m:
        return pmm


    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1


    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def spherical_harmonic_normalization(l: int, m: int) -> float:
    m = abs(m)

    log_num = 0.0
    for k in range(l - m + 1, l + m + 1):
        log_num += np.log(float(k))

    fact_ratio = np.exp(-log_num) if log_num < 700 else 0.0
    if m == 0:
        fact_ratio = 1.0 / np.math.factorial(l) / np.math.factorial(l)
        fact_ratio = 1.0

        num = 1.0
        den = 1.0
        for k in range(1, l + m + 1):
            den *= k
        for k in range(1, l - m + 1):
            num *= k
        fact_ratio = num / den
    else:
        num = 1.0
        den = 1.0
        for k in range(1, l + m + 1):
            den *= k
        for k in range(1, l - m + 1):
            num *= k
        fact_ratio = num / den

    return np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) * fact_ratio)


def scalar_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    x = np.cos(theta)
    plm = associated_legendre(l, m, x)
    N = spherical_harmonic_normalization(l, m)
    ylm = N * plm * np.exp(1j * m * phi)
    return ylm


def toroidal_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> Tuple[complex, complex, complex]:
    ylm = scalar_spherical_harmonic(l, m, theta, phi)


    dtheta = 1e-6
    thp = min(np.pi - 1e-8, theta + dtheta)
    thm = max(1e-8, theta - dtheta)
    ylm_p = scalar_spherical_harmonic(l, m, thp, phi)
    ylm_m = scalar_spherical_harmonic(l, m, thm, phi)
    dylm_dtheta = (ylm_p - ylm_m) / (thp - thm)

    T_r = 0.0 + 0.0j
    sin_t = max(1e-15, np.sin(theta))
    T_theta = 1j * m * ylm / sin_t
    T_phi = -dylm_dtheta

    return (T_r, T_theta, T_phi)


def poloidal_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> Tuple[complex, complex, complex]:
    ylm = scalar_spherical_harmonic(l, m, theta, phi)

    dtheta = 1e-6
    thp = min(np.pi - 1e-8, theta + dtheta)
    thm = max(1e-8, theta - dtheta)
    ylm_p = scalar_spherical_harmonic(l, m, thp, phi)
    ylm_m = scalar_spherical_harmonic(l, m, thm, phi)
    dylm_dtheta = (ylm_p - ylm_m) / (thp - thm)

    sin_t = max(1e-15, np.sin(theta))
    P_r = 0.0 + 0.0j
    P_theta = dylm_dtheta
    P_phi = 1j * m * ylm / sin_t

    return (P_r, P_theta, P_phi)





def gauss_legendre_grid(n_theta: int) -> Tuple[np.ndarray, np.ndarray]:

    x, w = np.polynomial.legendre.leggauss(n_theta)

    theta = np.arccos(x)
    return theta, w


def spherical_harmonic_analysis(field: np.ndarray, l_max: int) -> np.ndarray:
    n_theta, n_phi = field.shape
    theta, w = gauss_legendre_grid(n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    coeffs = np.zeros((l_max + 1, 2 * l_max + 1), dtype=complex)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            integral = 0.0 + 0.0j
            for it in range(n_theta):
                sin_t = np.sin(theta[it])
                for ip in range(n_phi):
                    ylm_conj = np.conj(scalar_spherical_harmonic(l, m, theta[it], phi[ip]))
                    integral += field[it, ip] * ylm_conj * w[it] * dphi * sin_t
            coeffs[l, m + l_max] = integral

    return coeffs


def spherical_harmonic_synthesis(coeffs: np.ndarray, n_theta: int, n_phi: int) -> np.ndarray:
    l_max = coeffs.shape[0] - 1
    theta, _ = gauss_legendre_grid(n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    field = np.zeros((n_theta, n_phi), dtype=complex)
    for it in range(n_theta):
        for ip in range(n_phi):
            val = 0.0 + 0.0j
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    val += coeffs[l, m + l_max] * scalar_spherical_harmonic(l, m, theta[it], phi[ip])
            field[it, ip] = val

    return np.real(field)







def mauersberger_lowes_spectrum(coeffs: np.ndarray) -> np.ndarray:
    l_max = coeffs.shape[0] - 1
    spectrum = np.zeros(l_max + 1, dtype=float)
    for l in range(l_max + 1):
        energy = 0.0
        for m in range(-l, l + 1):
            c = coeffs[l, m + l_max]
            energy += abs(c) ** 2
        spectrum[l] = (l + 1.0) * energy
    return spectrum





def dipole_inclination(g10: float, g11: float, h11: float) -> float:
    denom = np.sqrt(g11 ** 2 + h11 ** 2)
    if denom < 1e-30:
        return np.pi / 2.0 if g10 > 0 else -np.pi / 2.0
    return np.arctan2(2.0 * g10, denom)


def dipole_moment(g10: float, g11: float, h11: float, radius: float) -> float:
    mu0 = 4.0 * np.pi * 1.0e-7
    return (4.0 * np.pi / mu0) * (radius ** 3) * np.sqrt(g10 ** 2 + g11 ** 2 + h11 ** 2)





def _self_test():

    theta, w = gauss_legendre_grid(32)
    n_phi = 64
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    integral = 0.0
    for it in range(len(theta)):
        for ip in range(n_phi):
            y = scalar_spherical_harmonic(0, 0, theta[it], phi[ip])
            integral += abs(y) ** 2 * np.sin(theta[it]) * w[it] * dphi
    assert abs(integral - 1.0) < 1e-3, f"Normalization failed: {integral}"


    val = scalar_spherical_harmonic(1, 0, np.pi / 2.0, 0.0)
    assert abs(val - 0.0) < 1e-10


    inc = dipole_inclination(1.0, 0.0, 0.0)
    assert abs(inc - np.pi / 2.0) < 1e-10

    print("spherical_harmonics: self-test passed.")


if __name__ == "__main__":
    _self_test()
