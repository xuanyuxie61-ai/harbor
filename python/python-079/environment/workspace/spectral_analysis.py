
import numpy as np
from typing import List, Tuple, Optional
from utils import check_well_posed_diophantine, gcd_vector






def diophantine_nd_nonnegative_solutions(
    a: np.ndarray, b: int
) -> List[np.ndarray]:
    a = np.asarray(a, dtype=int)
    if not check_well_posed_diophantine(a, b):
        return []
    m = len(a)
    solutions = []
    _backtrack_diophantine(a, b, 0, np.zeros(m, dtype=int), solutions)
    return solutions


def _backtrack_diophantine(
    a: np.ndarray,
    remaining: int,
    idx: int,
    current: np.ndarray,
    solutions: List[np.ndarray],
):
    m = len(a)
    if idx == m - 1:
        if remaining % a[idx] == 0:
            current[idx] = remaining // a[idx]
            solutions.append(current.copy())
        return
    max_val = remaining // a[idx]
    for val in range(max_val, -1, -1):
        current[idx] = val
        _backtrack_diophantine(a, remaining - val * a[idx], idx + 1, current, solutions)


def diophantine_solution_count(a: np.ndarray, b: int) -> int:
    return len(diophantine_nd_nonnegative_solutions(a, b))






def wavenumber_discrete_constraint_bragg(
    wavelength: float,
    column_spacing: float,
    incidence_angle: float,
    max_order: int = 5,
) -> List[Tuple[int, float]]:
    k = 2.0 * np.pi / wavelength
    cos_theta = np.cos(incidence_angle)
    if abs(cos_theta) < 1e-12:
        return []
    target = 2.0 * k * cos_theta
    solutions = []
    for n in range(1, max_order + 1):
        required = n * (2.0 * np.pi / column_spacing)
        error = abs(target - required)
        relative_error = error / abs(target) if abs(target) > 1e-12 else error
        if relative_error < 0.1:
            solutions.append((n, relative_error))
    return solutions


def wavenumber_discrete_constraint_floquet(
    domain_lengths: np.ndarray,
    max_modes: int = 3,
    omega: float = 1.0,
    h: float = 100.0,
) -> List[dict]:
    domain_lengths = np.asarray(domain_lengths, dtype=float)
    if len(domain_lengths) < 2:
        raise ValueError("domain_lengths 至少包含两个元素")
    Lx, Ly = domain_lengths[0], domain_lengths[1]
    g = 9.80665
    omega2 = omega * omega
    modes = []
    for h1 in range(-max_modes, max_modes + 1):
        for h2 in range(-max_modes, max_modes + 1):
            if h1 == 0 and h2 == 0:
                continue
            kx = h1 * 2.0 * np.pi / Lx
            ky = h2 * 2.0 * np.pi / Ly
            k_mag = np.sqrt(kx ** 2 + ky ** 2)
            if k_mag < 1e-12:
                continue
            kh = k_mag * h
            if kh > 100:
                tanh_kh = 1.0
            else:
                tanh_kh = np.tanh(kh)
            omega_k = np.sqrt(g * k_mag * tanh_kh)
            rel_err = abs(omega_k - omega) / omega
            if rel_err < 0.15:
                modes.append(
                    {
                        "h1": h1,
                        "h2": h2,
                        "kx": kx,
                        "ky": ky,
                        "k_mag": k_mag,
                        "omega_computed": omega_k,
                        "relative_error": rel_err,
                    }
                )

    modes.sort(key=lambda x: x["relative_error"])
    return modes


def generate_allowed_wavenumbers_diophantine(
    a_coeffs: np.ndarray,
    b_total: int,
    domain_scale: float = 100.0,
) -> np.ndarray:
    solutions = diophantine_nd_nonnegative_solutions(a_coeffs, b_total)
    wavenumbers = []
    base_k = 2.0 * np.pi / domain_scale
    for sol in solutions:
        k_vec = np.zeros(len(a_coeffs))
        for i, xi in enumerate(sol):
            k_vec[i] = xi * base_k
        wavenumbers.append(k_vec)
    return np.array(wavenumbers)






def response_spectrum_rao(
    omega: np.ndarray,
    omega_n: float,
    zeta: float,
    wave_spectrum: np.ndarray,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    wave_spectrum = np.asarray(wave_spectrum, dtype=float)
    if len(omega) != len(wave_spectrum):
        raise ValueError("omega 与 wave_spectrum 长度不一致")
    r = omega / omega_n
    r = np.where(r <= 0, 1e-12, r)
    H2 = 1.0 / ((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)
    return H2 * wave_spectrum


def significant_response_from_spectrum(
    response_spectrum: np.ndarray, omega: np.ndarray
) -> float:
    if len(omega) < 2:
        return 0.0
    m0 = np.trapezoid(response_spectrum, omega)
    m0 = max(m0, 0.0)
    return 2.0 * np.sqrt(m0)


def spectral_moments(
    spectrum: np.ndarray, omega: np.ndarray, max_order: int = 4
) -> List[float]:
    moments = []
    for n in range(max_order + 1):
        integrand = (omega ** n) * spectrum
        mn = np.trapezoid(integrand, omega)
        moments.append(float(mn))
    return moments


def spectral_bandwidth_params(
    spectrum: np.ndarray, omega: np.ndarray
) -> dict:
    moments = spectral_moments(spectrum, omega, max_order=4)
    m0, m1, m2, _, m4 = moments
    epsilon = 0.0
    if m0 > 1e-15 and m4 > 1e-15:
        epsilon = np.sqrt(max(0.0, 1.0 - (m2 ** 2) / (m0 * m4)))
    T01 = 2.0 * np.pi * m0 / m1 if m1 > 1e-15 else 0.0
    T02 = 2.0 * np.pi * np.sqrt(m0 / m2) if m2 > 1e-15 else 0.0
    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "m4": m4,
        "epsilon": epsilon,
        "T01": T01,
        "T02": T02,
    }






def diffraction_transfer_function_diophantine(
    panel_ks: np.ndarray,
    incident_k: float,
    a_coeffs: np.ndarray,
    b_constraint: int,
) -> np.ndarray:
    solutions = diophantine_nd_nonnegative_solutions(a_coeffs, b_constraint)
    if len(solutions) == 0:
        return np.ones(len(panel_ks))
    allowed_deltas = set()
    for sol in solutions:
        allowed_deltas.add(int(np.sum(sol)))
    transfer = np.zeros(len(panel_ks))
    for i, pk in enumerate(panel_ks):
        delta = abs(int(round(pk - incident_k)))
        if delta in allowed_deltas:
            transfer[i] = 1.0
        else:
            transfer[i] = 0.1
    return transfer
