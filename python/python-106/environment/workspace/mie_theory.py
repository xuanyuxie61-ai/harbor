"""
mie_theory.py
==============
Mie scattering theory for metallic nanospheres in the plasmonic regime.

Core physics:
-------------
For a sphere of radius a and complex permittivity ε(ω) embedded in a
medium ε_m, the Mie coefficients are:

    a_l = [ m ψ_l(mx) ψ_l'(x) − ψ_l(x) ψ_l'(mx) ] /
          [ m ψ_l(mx) ξ_l'(x) − ξ_l(x) ψ_l'(mx) ]

    b_l = [ ψ_l(mx) ψ_l'(x) − m ψ_l(x) ψ_l'(mx) ] /
          [ ψ_l(mx) ξ_l'(x) − m ξ_l(x) ψ_l'(mx) ]

where x = k_m a, m = n_sphere / n_medium, ψ_l and ξ_l are the
Riccati-Bessel functions.

The extinction cross section is:
    σ_ext(ω) = (2π / k_m²) Σ_{l=1}^{∞} (2l+1) Re[ a_l(ω) + b_l(ω) ]

The scattering cross section is:
    σ_sca(ω) = (2π / k_m²) Σ_{l=1}^{∞} (2l+1) (|a_l|² + |b_l|²)

Drude model for noble-metal permittivity:
    ε(ω) = ε_∞ − ω_p² / (ω² + i γ ω)

where ω_p is the bulk plasma frequency and γ is the electron collision rate.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def drude_permittivity(omega, omega_p=9.0e15, gamma=1.0e14, eps_inf=9.0):
    """
    Compute the Drude dielectric function for a noble metal.

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency (rad/s).
    omega_p : float
        Bulk plasma frequency (rad/s).  Default 9.0e15 (Au-like).
    gamma : float
        Electron collision rate (rad/s). Default 1.0e14.
    eps_inf : float
        High-frequency dielectric constant. Default 9.0.

    Returns
    -------
    eps : complex ndarray
        Complex permittivity ε(ω).
    """
    # TODO: Implement the Drude dielectric function for a noble metal.
    # The Drude model is:  ε(ω) = ε_∞ − ω_p² / (ω² + i γ ω)
    # Validate input (omega must be strictly positive) and return complex permittivity.
    raise NotImplementedError("Hole 1: Drude permittivity formula is missing.")


def riccati_bessel_functions(lmax, z):
    """
    Compute Riccati-Bessel functions ψ_l(z) and ξ_l(z) = ψ_l(z) − i χ_l(z)
    and their first derivatives for l = 0..lmax.

    ψ_l(z) = z j_l(z),    χ_l(z) = −z y_l(z)
    ξ_l(z) = ψ_l(z) − i χ_l(z)

    We use scipy.special.riccati_jn and riccati_yn.

    Parameters
    ----------
    lmax : int
        Maximum order.
    z : complex
        Argument.

    Returns
    -------
    psi : ndarray, shape (lmax+1,)
    dpsi : ndarray, shape (lmax+1,)
    xi : ndarray, shape (lmax+1,)
    dxi : ndarray, shape (lmax+1,)
    """
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
    """
    Compute the Mie coefficients a_l and b_l for a single sphere.

    Parameters
    ----------
    lmax : int
        Maximum multipole order.
    x : float
        Size parameter x = k_medium * a.
    m : complex
        Relative refractive index m = n_sphere / n_medium.

    Returns
    -------
    a_l : ndarray, shape (lmax,)
    b_l : ndarray, shape (lmax,)
    """
    if x <= 0:
        raise ValueError("Size parameter x must be positive.")
    mx = m * x

    psi_x, dpsi_x, xi_x, dxi_x = riccati_bessel_functions(lmax, x)
    psi_mx, dpsi_mx, _, _ = riccati_bessel_functions(lmax, mx)

    a_l = np.zeros(lmax, dtype=complex)
    b_l = np.zeros(lmax, dtype=complex)

    for l in range(1, lmax + 1):
        # a_l numerator and denominator
        num_a = m * psi_mx[l] * dpsi_x[l] - psi_x[l] * dpsi_mx[l]
        den_a = m * psi_mx[l] * dxi_x[l] - xi_x[l] * dpsi_mx[l]
        if abs(den_a) < 1e-30:
            den_a = 1e-30
        a_l[l - 1] = num_a / den_a

        # b_l numerator and denominator
        num_b = psi_mx[l] * dpsi_x[l] - m * psi_x[l] * dpsi_mx[l]
        den_b = psi_mx[l] * dxi_x[l] - m * xi_x[l] * dpsi_mx[l]
        if abs(den_b) < 1e-30:
            den_b = 1e-30
        b_l[l - 1] = num_b / den_b

    return a_l, b_l


def mie_cross_sections(omega, a, eps_medium=1.0,
                       omega_p=9.0e15, gamma=1.0e14, eps_inf=9.0,
                       lmax=None):
    """
    Compute Mie extinction and scattering cross sections for a metallic nanosphere.

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency (rad/s).
    a : float
        Nanosphere radius (m).
    eps_medium : float
        Permittivity of surrounding medium.  Default 1.0 (vacuum).
    omega_p, gamma, eps_inf : float
        Drude parameters.
    lmax : int or None
        Maximum multipole order.  If None, auto-set to int(2 + x + 4*x**(1/3)).

    Returns
    -------
    sigma_ext : float or ndarray
        Extinction cross section (m²).
    sigma_sca : float or ndarray
        Scattering cross section (m²).
    """
    c = 2.99792458e8  # speed of light (m/s)
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
    """
    Generate a spherical surface grid for nanoparticle discretization.
    Based on tensor-grid generation for θ ∈ [0, π], φ ∈ [0, 2π).

    Parameters
    ----------
    n_theta, n_phi : int
        Number of grid points in polar and azimuthal directions.
    radius : float
        Sphere radius in meters.

    Returns
    -------
    x, y, z : ndarray, shape (n_theta, n_phi)
        Cartesian coordinates of surface points.
    theta, phi : ndarray
        Angular coordinates.
    area_element : ndarray
        Differential surface area at each grid point.
    """
    if n_theta < 2 or n_phi < 2:
        raise ValueError("Grid resolution must be at least 2 in each direction.")
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    # Use Gauss-Legendre nodes for θ to avoid clustering at poles
    # For simplicity and direct mapping to the earth-sphere seed, uniform grid
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    x = radius * np.sin(THETA) * np.cos(PHI)
    y = radius * np.sin(THETA) * np.sin(PHI)
    z = radius * np.cos(THETA)

    # Surface element dA = r² sinθ dθ dφ
    dtheta = np.pi / (n_theta - 1) if n_theta > 1 else np.pi
    dphi = 2.0 * np.pi / (n_phi - 1) if n_phi > 1 else 2.0 * np.pi
    area_element = (radius ** 2) * np.sin(THETA) * dtheta * dphi

    return x, y, z, theta, phi, area_element
