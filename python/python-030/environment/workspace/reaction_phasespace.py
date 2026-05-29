# -*- coding: utf-8 -*-
"""
reaction_phasespace.py
======================
Phase-space and reaction-cross-section integrals for radioactive-beam
induced reactions, inspired by the **unit-disk monomial integrals**
of *disk01_integrals*.

Physical model
--------------
For a peripheral reaction such as neutron transfer or Coulomb breakup,
the differential cross section in the impact-parameter plane can be
written as

.. math::
    \frac{d\sigma}{d\Omega}
    = \frac{\mu^2}{(2\pi\hbar^2)^2}\,
      \frac{k_f}{k_i}\,|T_{fi}|^2 \;.

In the eikonal / semiclassical approximation the total cross section
for a one-neutron transfer reaction :math:`A(a,b)B` is

.. math::
    \sigma_{\text{tr}} = 2\pi\int_{b_{\min}}^{\infty}
    b\,P_{\text{tr}}(b)\,db \;,

where :math:`P_{\text{tr}}(b)` is the transfer probability at impact
parameter :math:`b`.

For a sudden approximation with a Gaussian overlap,

.. math::
    P_{\text{tr}}(b) \approx P_0\,
    \exp\!\left(-\frac{(b-R_{\text{gr}})^2}{\sigma_b^2}\right) \;.

The module also computes **unit-disk integrals** that appear in angular-
momentum coupling (Clebsch-Gordan weighting over the :math:`m`-subspace
sphere), analogous to :math:`\int_{x^2+y^2\le 1} x^{e_1} y^{e_2}\,dxdy`.

Disk integrals
--------------
The general monomial integral on the unit disk is

.. math::
    I_{e_1,e_2} = \int_{x^2+y^2\le 1} x^{e_1} y^{e_2}\,dxdy
    = \frac{2\,\Gamma\!\left(\frac{e_1+1}{2}\right)
           \Gamma\!\left(\frac{e_2+1}{2}\right)}
          {(e_1+e_2+2)\,\Gamma\!\left(\frac{e_1+e_2+2}{2}\right)}

when both :math:`e_1` and :math:`e_2` are even; otherwise the integral
vanishes by symmetry.
"""

import numpy as np
from scipy.special import gamma


def disk_monomial_integral(e1, e2):
    r"""
    Integral of :math:`x^{e_1} y^{e_2}` over the unit disk.

    Parameters
    ----------
    e1, e2 : int
        Non-negative exponents.

    Returns
    -------
    val : float
        Integral value.
    """
    if e1 < 0 or e2 < 0:
        raise ValueError("Exponents must be non-negative.")
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    num = 2.0 * gamma(0.5 * (e1 + 1)) * gamma(0.5 * (e2 + 1))
    den = (e1 + e2 + 2) * gamma(0.5 * (e1 + e2 + 2))
    return num / den


def disk_gaussian_integral(sigma, n_theta=128, n_r=64):
    r"""
    Numerically integrate a 2-D Gaussian over the unit disk.

    .. math::
        I = \int_{x^2+y^2\le 1}
        \exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)dxdy

    Parameters
    ----------
    sigma : float
        Width parameter.
    n_theta, n_r : int
        Angular and radial grid sizes.

    Returns
    -------
    val : float
        Integral value.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    r = np.linspace(0.0, 1.0, n_r)
    dtheta = 2.0 * np.pi / n_theta
    dr = 1.0 / (n_r - 1) if n_r > 1 else 1.0
    val = 0.0
    for ti in theta:
        for ri in r[1:]:
            x = ri * np.cos(ti)
            y = ri * np.sin(ti)
            val += np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) * ri * dr * dtheta
    return val


def transfer_probability(b, R_grazing, sigma_b, P0=0.1):
    r"""
    Gaussian model for peripheral transfer probability.

    .. math::
        P_{\text{tr}}(b) = P_0\,
        \exp\!\left(-\frac{(b - R_{\text{grazing}})^2}{\sigma_b^2}\right)

    Parameters
    ----------
    b : float or ndarray
        Impact parameter in fm.
    R_grazing : float
        Grazing distance in fm.
    sigma_b : float
        Width of the transfer window in fm.
    P0 : float
        Peak probability.

    Returns
    -------
    P : float or ndarray
        Transfer probability.
    """
    return P0 * np.exp(-((b - R_grazing) ** 2) / (sigma_b ** 2))


def transfer_cross_section(R_grazing, sigma_b, P0=0.1, b_min=None, b_max=30.0):
    r"""
    Total transfer cross section in the semiclassical approximation.

    .. math::
        \sigma = 2\pi\int_{b_{\min}}^{b_{\max}} b\,P_{\text{tr}}(b)\,db

    Parameters
    ----------
    R_grazing, sigma_b, P0 : float
        Transfer probability parameters.
    b_min : float, optional
        Minimum impact parameter.  Defaults to :math:`R_{\text{grazing}} - 3\sigma_b`.
    b_max : float
        Maximum impact parameter in fm.

    Returns
    -------
    sigma : float
        Cross section in fm².
    """
    if b_min is None:
        b_min = max(0.0, R_grazing - 3.0 * sigma_b)
    # Gaussian integral over b with linear weight: analytical
    # ∫ b exp(-(b-Rg)^2/σ^2) db = σ^2/2 [exp(-(b-Rg)^2/σ^2)] + Rg σ √π/2 [erf(...)]
    # Use numerical quadrature for robustness
    n = 2000
    b_vals = np.linspace(b_min, b_max, n)
    db = b_vals[1] - b_vals[0]
    P_vals = transfer_probability(b_vals, R_grazing, sigma_b, P0)
    integrand = 2.0 * np.pi * b_vals * P_vals
    return np.trapz(integrand, b_vals)


def coulomb_breakup_cross_section(E_beam, Z_p, Z_t, A_p, A_t,
                                  E_bind, n_points=1000):
    r"""
    Semiclassical Coulomb breakup cross section for a halo nucleus.

    In the equivalent-photon method (Weizsäcker-Williams):

    .. math::
        \sigma_{\text{CU}} = \int_{\omega_{\min}}^{\omega_{\max}}
        n_{\gamma}(\omega)\,\sigma_{\gamma}(\omega)\,d\omega

    where the photon number spectrum for a point charge is

    .. math::
        n_{\gamma}(\omega) = \frac{2\alpha Z_p^2}{\pi\omega}
        \Bigl[\xi K_0(\xi) K_1(\xi)
        - \frac{\xi^2}{2}\bigl(K_1^2(\xi) - K_0^2(\xi)\bigr)\Bigr] \;,

    with :math:`\xi = \omega b_{\min}/(\gamma v)`.

    Parameters
    ----------
    E_beam : float
        Beam energy per nucleon in MeV.
    Z_p, Z_t : int
        Projectile and target charges.
    A_p, A_t : int
        Projectile and target mass numbers.
    E_bind : float
        Binding energy of the removed nucleon in MeV.
    n_points : int
        Integration grid size.

    Returns
    -------
    sigma_cu : float
        Cross section in fm².
    """
    from constants import FINE_STRUCTURE, HBAR_C
    # Reduced mass and velocity
    mu = (A_p * A_t) / (A_p + A_t) * 938.0  # MeV/c²
    v = np.sqrt(2.0 * E_beam / mu) * HBAR_C  # fm/fs (non-relativistic approx)
    if v <= 0:
        return 0.0
    b_min = 1.2 * (A_p ** (1.0 / 3.0) + A_t ** (1.0 / 3.0))

    # Photon energy range
    omega_min = E_bind
    omega_max = 10.0 * E_bind
    omega = np.linspace(omega_min, omega_max, n_points)
    domega = omega[1] - omega[0]

    # Approximate B(E1) strength function: narrow resonance at E_bind
    # σ_γ(ω) ≈ (16π/9ħc) E_bind Γ / [(ω-E_bind)^2 + Γ^2/4]
    Gamma = 0.5  # MeV, width
    sigma_gamma = (16.0 * np.pi / (9.0 * HBAR_C)) * E_bind * Gamma / (
        (omega - E_bind) ** 2 + (Gamma / 2.0) ** 2)

    # Equivalent-photon spectrum (simplified)
    xi = omega * b_min / v
    xi = np.where(xi < 1e-6, 1e-6, xi)
    from scipy.special import kv
    n_gamma = (2.0 * FINE_STRUCTURE * Z_p ** 2 / (np.pi * omega)) * (
        xi * kv(0, xi) * kv(1, xi)
        - 0.5 * xi ** 2 * (kv(1, xi) ** 2 - kv(0, xi) ** 2))
    n_gamma = np.where(n_gamma < 0, 0.0, n_gamma)

    integrand = n_gamma * sigma_gamma
    return np.trapz(integrand, omega)


def angular_momentum_coupling_weight(j1, j2, J, M):
    r"""
    Normalised weight for coupling two angular momenta to total :math:`(J,M)`.

    The weight is proportional to the number of :math:`(m_1,m_2)` pairs
    satisfying :math:`m_1+m_2=M` and the triangle condition.
    It is expressed as an integral over the unit disk:

    .. math::
        W_{j_1,j_2}^{JM} = \frac{1}{\pi}\int_{0}^{2\pi}
        \Theta\bigl(J-|m_1(\phi)+m_2(\phi)|\bigr)\,d\phi

    where :math:`\Theta` is the Heaviside step and the :math:`m_i` are
    sampled uniformly on their allowed intervals.

    Parameters
    ----------
    j1, j2 : float
        Angular momenta (half-integers allowed).
    J : float
        Total angular momentum.
    M : float
        Projection.

    Returns
    -------
    weight : float
        Normalised coupling weight in [0,1].
    """
    if abs(M) > J or J < abs(j1 - j2) or J > j1 + j2:
        return 0.0
    m1_vals = np.arange(-j1, j1 + 0.5, 1.0)
    count = 0
    total = 0
    for m1 in m1_vals:
        m2 = M - m1
        if abs(m2) <= j2 + 1e-6:
            total += 1
            if abs(m1 + m2 - M) < 1e-6:
                count += 1
    return count / max(total, 1)
