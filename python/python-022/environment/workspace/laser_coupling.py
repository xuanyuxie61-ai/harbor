
import numpy as np
from quadrature_utils import gauss_legendre_rule


C_LIGHT = 2.99792458e8
E_CHARGE = 1.602176634e-19
M_E = 9.10938356e-31
EPSILON_0 = 8.854187817e-12


def laser_intensity_profile(r, r_beam, I_0, profile_type='supergaussian'):
    if r_beam <= 0.0:
        return 0.0
    if r < 0.0:
        r = -r

    if profile_type == 'gaussian':
        m = 2.0
    elif profile_type == 'supergaussian':
        m = 4.0
    else:
        m = 2.0

    return I_0 * np.exp(-2.0 * (r / r_beam)**m)


def critical_density(wavelength):
    omega = 2.0 * np.pi * C_LIGHT / wavelength
    n_c = M_E * EPSILON_0 * omega**2 / E_CHARGE**2
    return n_c


def inverse_bremsstrahlung_coeff(n_e, n_i, Z, T_e, wavelength):
    T_e = float(max(T_e, 1.0))
    n_e = float(max(n_e, 0.0))
    n_i = float(max(n_i, 1e10))
    if n_e <= 0.0:
        return 0.0

    n_c = critical_density(wavelength)
    if n_e >= n_c:
        return 1e4


    ln_lambda = max(1.0, 23.5 - 0.5 * np.log(max(n_e * 1e-6, 1e-300)) + 1.5 * np.log(T_e))
    T_e_norm = 3.0 * T_e / M_E
    nu_ei = (4.0 * np.pi * n_i * Z**2 * E_CHARGE**4 * ln_lambda
             / (3.0 * (4.0 * np.pi * EPSILON_0)**2 * M_E**2 * max(T_e_norm, 1e-30)**1.5))

    omega = 2.0 * np.pi * C_LIGHT / wavelength


    alpha_ib = (n_e**2 * Z * E_CHARGE**2 * nu_ei
                / (n_c * M_E * C_LIGHT * EPSILON_0 * max(omega, 1e-30)**2 + 1e-30))

    return float(min(max(alpha_ib, 0.0), 1e6))


def laser_envelope_propagation(E, z, r, n_e_profile, n_c, k_0):
    nr = len(r)
    dr = r[1] - r[0] if nr > 1 else 1.0


    Im1 = np.array([1] + list(range(nr - 2)) + [nr - 2])
    I = np.arange(nr)
    Ip1 = np.array([1] + list(range(2, nr)) + [nr - 2])

    laplacian_E = (E[Ip1] - 2.0 * E[I] + E[Im1]) / dr**2

    for i in range(nr):
        if r[i] > 1e-15:
            laplacian_E[i] += (E[Ip1[i]] - E[Im1[i]]) / (2.0 * r[i] * dr)


    eta = np.zeros(nr)
    for i in range(nr):
        n_e = n_e_profile(r[i])
        eta[i] = k_0**2 * (1.0 - n_e / n_c)
        if n_e >= n_c:
            eta[i] = 0.0


    dEdz = 1j / (2.0 * k_0) * (laplacian_E + eta * E)
    return dEdz


def integrate_laser_deposition(radius_points, z_path, n_e_func, n_i_func, Z_func, T_e_func,
                                wavelength, I_0, r_beam, n_quad=16):
    n_r = len(radius_points)
    deposition = np.zeros(n_r)


    z_nodes, z_weights = gauss_legendre_rule(n_quad, 0.0, z_path)

    n_c = critical_density(wavelength)

    for i, r in enumerate(radius_points):
        I_r = laser_intensity_profile(r, r_beam, I_0)
        if I_r <= 0.0:
            continue

        n_e = n_e_func(r)
        n_i = n_i_func(r)
        Z = Z_func(r)
        T_e = T_e_func(r)

        alpha = inverse_bremsstrahlung_coeff(n_e, n_i, Z, T_e, wavelength)




        deposited_power = 0.0
        for zj, wj in zip(z_nodes, z_weights):
            I_z = I_r * np.exp(-alpha * zj)
            deposited_power += wj * alpha * I_z

        deposition[i] = deposited_power

    return deposition


def compute_absorbed_fraction(rho, T, Z_bar, wavelength, r_beam, shell_thickness,
                               n_quad_r=16, n_quad_z=16):
    m_u = 1.66053906660e-27
    n_i = rho / (2.5 * m_u)
    n_e = Z_bar * n_i

    n_c = critical_density(wavelength)
    if n_e >= 0.99 * n_c:
        return 1.0

    alpha = inverse_bremsstrahlung_coeff(n_e, n_i, Z_bar, T, wavelength)


    path_length = 2.0 * np.sqrt(2.0 * r_beam * shell_thickness)
    if path_length <= 0.0:
        path_length = shell_thickness

    absorbed_frac = 1.0 - np.exp(-alpha * path_length)
    return np.clip(absorbed_frac, 0.0, 1.0)
