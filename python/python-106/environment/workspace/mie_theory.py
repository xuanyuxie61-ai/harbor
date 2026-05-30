
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def drude_permittivity(omega, omega_p=9.0e15, gamma=1.0e14, eps_inf=9.0):



    raise NotImplementedError("Hole 1: Drude permittivity formula is missing.")


def riccati_bessel_functions(lmax, z):
    n = np.arange(0, lmax + 1)
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    djn = spherical_jn(n, z, derivative=True)
    dyn = spherical_yn(n, z, derivative=True)

    psi = z * jn
    dpsi = jn + z * djn
    chi = -z * yn
    dchi = -yn - z * dyn

    xi = psi - 1j * chi
    dxi = dpsi - 1j * dchi
    return psi, dpsi, xi, dxi


def mie_coefficients(lmax, x, m):
    if x <= 0:
        raise ValueError("Size parameter x must be positive.")
    mx = m * x

    psi_x, dpsi_x, xi_x, dxi_x = riccati_bessel_functions(lmax, x)
    psi_mx, dpsi_mx, _, _ = riccati_bessel_functions(lmax, mx)

    a_l = np.zeros(lmax, dtype=complex)
    b_l = np.zeros(lmax, dtype=complex)

    for l in range(1, lmax + 1):

        num_a = m * psi_mx[l] * dpsi_x[l] - psi_x[l] * dpsi_mx[l]
        den_a = m * psi_mx[l] * dxi_x[l] - xi_x[l] * dpsi_mx[l]
        if abs(den_a) < 1e-30:
            den_a = 1e-30
        a_l[l - 1] = num_a / den_a


        num_b = psi_mx[l] * dpsi_x[l] - m * psi_x[l] * dpsi_mx[l]
        den_b = psi_mx[l] * dxi_x[l] - m * xi_x[l] * dpsi_mx[l]
        if abs(den_b) < 1e-30:
            den_b = 1e-30
        b_l[l - 1] = num_b / den_b

    return a_l, b_l


def mie_cross_sections(omega, a, eps_medium=1.0,
                       omega_p=9.0e15, gamma=1.0e14, eps_inf=9.0,
                       lmax=None):
    c = 2.99792458e8
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0):
        raise ValueError("omega must be positive.")
    if a <= 0:
        raise ValueError("Radius a must be positive.")

    k_medium = omega * np.sqrt(eps_medium) / c
    x = k_medium * a
    eps_metal = drude_permittivity(omega, omega_p, gamma, eps_inf)
    m = np.sqrt(eps_metal / eps_medium)

    scalar_input = (omega.ndim == 0)
    if scalar_input:
        x = np.array([x])
        m = np.array([m])
        omega = np.array([omega])

    nfreq = omega.size
    sigma_ext = np.zeros(nfreq)
    sigma_sca = np.zeros(nfreq)

    for i in range(nfreq):
        xi = float(x[i])
        mi = m[i]
        if lmax is None:
            lmax_i = max(2, int(np.ceil(xi + 4.0 * xi ** (1.0 / 3.0))))
        else:
            lmax_i = lmax

        a_l, b_l = mie_coefficients(lmax_i, xi, mi)
        l_arr = np.arange(1, lmax_i + 1)
        prefactor = (2.0 * np.pi / (k_medium[i] ** 2))
        sigma_ext[i] = prefactor * np.sum((2.0 * l_arr + 1.0) * np.real(a_l + b_l))
        sigma_sca[i] = prefactor * np.sum((2.0 * l_arr + 1.0) * (np.abs(a_l) ** 2 + np.abs(b_l) ** 2))

    if scalar_input:
        return float(sigma_ext[0]), float(sigma_sca[0])
    return sigma_ext, sigma_sca


def generate_sphere_surface_grid(n_theta=32, n_phi=32, radius=20.0e-9):
    if n_theta < 2 or n_phi < 2:
        raise ValueError("Grid resolution must be at least 2 in each direction.")
    if radius <= 0:
        raise ValueError("Radius must be positive.")



    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    x = radius * np.sin(THETA) * np.cos(PHI)
    y = radius * np.sin(THETA) * np.sin(PHI)
    z = radius * np.cos(THETA)


    dtheta = np.pi / (n_theta - 1) if n_theta > 1 else np.pi
    dphi = 2.0 * np.pi / (n_phi - 1) if n_phi > 1 else 2.0 * np.pi
    area_element = (radius ** 2) * np.sin(THETA) * dtheta * dphi

    return x, y, z, theta, phi, area_element
