
import numpy as np
from scipy.special import lpmv, factorial


def associated_legendre_normalized(l_max, m_order, x):
    x = np.atleast_1d(x)
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("associated_legendre_normalized: |x| 必须 <= 1")

    plm = np.zeros((l_max + 1,) + x.shape, dtype=np.float64)
    m = abs(m_order)


    for l in range(m, l_max + 1):

        val = lpmv(m, l, x)

        if m == 0:
            norm = 1.0
        else:

            norm = np.sqrt(2.0 * factorial(l - m) / factorial(l + m))
        plm[l] = norm * val

    return plm


def spherical_harmonic_basis(l, m, theta, phi):
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    if np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("spherical_harmonic_basis: theta 必须在 [0, π] 内")
    if np.any((phi < 0) | (phi > 2 * np.pi)):
        raise ValueError("spherical_harmonic_basis: phi 必须在 [0, 2π] 内")

    m_abs = abs(m)
    x = np.cos(theta)


    plm = associated_legendre_normalized(l, m_abs, x)


    norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) *
                   factorial(l - m_abs) / factorial(l + m_abs))


    c = norm * plm[l] * np.cos(m * phi)
    s = norm * plm[l] * np.sin(m * phi)

    if m < 0:
        c = -c
        s = -s

    return c, s


def velocity_spectral_decomposition(psi_coeffs, chi_coeffs, l_max, theta_grid, phi_grid, earth_radius=6.371e6):
    theta = np.atleast_1d(theta_grid)
    phi = np.atleast_1d(phi_grid)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    u = np.zeros_like(THETA, dtype=np.float64)
    v = np.zeros_like(THETA, dtype=np.float64)
    a = earth_radius

    sin_theta = np.sin(THETA)
    sin_theta = np.where(np.abs(sin_theta) < 1e-12, 1e-12, sin_theta)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            key = (l, m)
            psi_lm = psi_coeffs.get(key, 0.0)
            chi_lm = chi_coeffs.get(key, 0.0)

            if abs(psi_lm) < 1e-15 and abs(chi_lm) < 1e-15:
                continue


            dtheta = 1e-6
            c_p, s_p = spherical_harmonic_basis(l, m, THETA + dtheta, PHI)
            c_m, s_m = spherical_harmonic_basis(l, m, THETA - dtheta, PHI)
            dY_dtheta = (complex(c_p, s_p) - complex(c_m, s_m)) / (2 * dtheta)

            dphi = 1e-6
            c_p, s_p = spherical_harmonic_basis(l, m, THETA, PHI + dphi)
            c_m, s_m = spherical_harmonic_basis(l, m, THETA, PHI - dphi)
            dY_dphi = (complex(c_p, s_p) - complex(c_m, s_m)) / (2 * dphi)

            Y = complex(*spherical_harmonic_basis(l, m, THETA, PHI))

            psi_val = psi_lm * Y
            dpsi_dtheta = psi_lm * dY_dtheta
            dpsi_dphi = psi_lm * dY_dphi

            chi_val = chi_lm * Y
            dchi_dtheta = chi_lm * dY_dtheta
            dchi_dphi = chi_lm * dY_dphi


            u += (-1.0 / (a * sin_theta) * dpsi_dphi.imag -
                  1.0 / a * dchi_dtheta.real)
            v += (1.0 / (a * sin_theta) * dchi_dphi.imag -
                  1.0 / a * dpsi_dtheta.real)

    return u, v
