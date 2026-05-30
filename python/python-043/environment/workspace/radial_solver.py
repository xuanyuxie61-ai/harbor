
import numpy as np
from typing import Tuple
from cg_solver import solve_radial_diffusion_cg






def build_spherical_radial_laplacian(r: np.ndarray) -> np.ndarray:
    n = len(r)
    L = np.zeros((n, n), dtype=float)

    for i in range(1, n - 1):
        dr_plus = r[i + 1] - r[i]
        dr_minus = r[i] - r[i - 1]
        dr_avg = 0.5 * (dr_plus + dr_minus)
        r_plus_sq = (0.5 * (r[i] + r[i + 1])) ** 2
        r_minus_sq = (0.5 * (r[i] + r[i - 1])) ** 2

        coeff_plus = r_plus_sq / (dr_plus * dr_avg * r[i] ** 2)
        coeff_minus = r_minus_sq / (dr_minus * dr_avg * r[i] ** 2)
        coeff_center = -(coeff_plus + coeff_minus)

        L[i, i - 1] = coeff_minus
        L[i, i] = coeff_center
        L[i, i + 1] = coeff_plus


    return L





def alpha_effect_source(r: np.ndarray, r_icb: float, r_cmb: float,
                        alpha0: float, l: int) -> np.ndarray:
    d = r_cmb - r_icb
    f = np.sin(np.pi * (r - r_icb) / d)
    f[r <= r_icb] = 0.0
    f[r >= r_cmb] = 0.0
    return alpha0 * f * np.sqrt(l * (l + 1.0)) / r


def omega_effect_source(r: np.ndarray, r_icb: float, r_cmb: float,
                        omega_shear: float, Omega: float, B_tor: np.ndarray) -> np.ndarray:
    dBdr = np.gradient(B_tor, r)
    return omega_shear * Omega * dBdr





def step_radial_diffusion_cn(B: np.ndarray, r: np.ndarray,
                              dt: float, eta: float, l: int,
                              source: np.ndarray,
                              r_icb: float, r_cmb: float,
                              theta_cn: float = 0.5) -> np.ndarray:
    n = len(r)
    L = build_spherical_radial_laplacian(r)


    decay = -l * (l + 1.0) * eta / (r ** 2)
    for i in range(n):
        L[i, i] += decay[i]


    L[0, :] = 0.0
    L[0, 0] = 1.0
    L[-1, :] = 0.0

    L[-1, -1] = 1.0
    L[-1, -2] = -1.0

    I = np.eye(n, dtype=float)
    A = I - theta_cn * dt * eta * L
    B_rhs = (I + (1.0 - theta_cn) * dt * eta * L) @ B + dt * source


    B_rhs[0] = 0.0
    B_rhs[-1] = 0.0


    dr = np.mean(np.diff(r))
    B_new = solve_radial_diffusion_cg(B_rhs, n, dr, dt, eta, theta_cn=theta_cn)


    B_new[0] = 0.0

    return B_new





def evolve_radial_modes(B_modes: dict, r: np.ndarray, dt: float, eta: float,
                        r_icb: float, r_cmb: float,
                        alpha0: float, omega_shear: float, Omega: float) -> dict:
    B_new = {}
    for key, B in B_modes.items():
        l, m = key

        src_alpha = alpha_effect_source(r, r_icb, r_cmb, alpha0, l)

        src_omega = omega_effect_source(r, r_icb, r_cmb, omega_shear, Omega, B)
        source = src_alpha + src_omega
        B_new[key] = step_radial_diffusion_cn(B, r, dt, eta, l, source, r_icb, r_cmb)
    return B_new





def _self_test():
    n = 32
    r_icb = 1221e3
    r_cmb = 3480e3
    r = np.linspace(r_icb, r_cmb, n)
    B = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb))
    dt = 1e4 * 365.25 * 24 * 3600
    eta = 2.0
    l = 2
    source = np.zeros(n, dtype=float)
    B_new = step_radial_diffusion_cn(B, r, dt, eta, l, source, r_icb, r_cmb)
    assert not np.isnan(B_new).any()
    assert B_new[0] == 0.0
    assert not np.isinf(B_new).any()


    modes = {(1, 0): B.copy(), (2, 0): B.copy()}
    modes_new = evolve_radial_modes(modes, r, dt, eta, r_icb, r_cmb,
                                     alpha0=0.5, omega_shear=1.0, Omega=7.29e-5)
    assert len(modes_new) == 2
    print("radial_solver: self-test passed.")


if __name__ == "__main__":
    _self_test()
