# -*- coding: utf-8 -*-
"""
nuclear_potential.py
====================
Construction of self-consistent nuclear mean-field potentials.

Inspired by the grid-discretization concepts of **mario** (pixel-grid
mapping) and **usa_box_plot** (structured rectangular tiling), this
module builds deformed nuclear potentials on a structured spatial mesh.
The multi-peak landscape of **peaks_movie** is abstracted into the
multi-component nuclear potential (central + spin-orbit + pairing).

Physical model
--------------
The single-particle potential for a deformed nucleus is

.. math::
    V(\mathbf{r}) = V_{\text{WS}}(r_{\text{eq}})
    + V_{\text{so}}(r_{\text{eq}})\,\hat{\mathbf{l}}\!\cdot\!\hat{\mathbf{s}}
    + V_{\text{Coul}}(r)\;,

where the equivalent radius for a quadrupole-deformed surface is

.. math::
    R(\theta,\phi) = R_0\Bigl[1
    + \beta_2\,Y_{20}(\theta,\phi)
    + \beta_3\,Y_{30}(\theta,\phi)
    + \beta_4\,Y_{40}(\theta,\phi)\Bigr] \;.

The Woods-Saxon form factor is

.. math::
    f_{\text{WS}}(x) = \frac{1}{1 + \exp\!\left(\dfrac{x - R}{a}\right)} \;.

The spin-orbit term is

.. math::
    V_{\text{so}}(r) = V_{\text{so}}^{0}\,
    \frac{1}{r}\frac{df_{\text{WS}}}{dr}
    \;\hat{\mathbf{l}}\!\cdot\!\hat{\mathbf{s}} \;.

Coulomb potential for uniformly charged sphere:

.. math::
    V_{\text{Coul}}(r) = \begin{cases}
    \dfrac{Ze^2}{2R_c}\left(3 - \dfrac{r^2}{R_c^2}\right), & r \le R_c \\[6pt]
    \dfrac{Ze^2}{r}, & r > R_c
    \end{cases}

with :math:`R_c = 1.2\,A^{1/3}` fm and :math:`e^2 = 1.44` MeV·fm.
"""

import numpy as np
from constants import (
    WS_V0, WS_R0, WS_A, WS_VSO, WS_RSO, WS_ASO,
    HBAR_C, FINE_STRUCTURE
)


def spherical_harmonic_y20(theta):
    r"""
    Real spherical harmonic :math:`Y_{20}(\theta)`.

    .. math::
        Y_{20}(\theta) = \sqrt{\frac{5}{16\pi}}\,(3\cos^2\theta - 1)

    Parameters
    ----------
    theta : array_like
        Polar angle in radians.

    Returns
    -------
    y20 : ndarray
        Values of :math:`Y_{20}`.
    """
    return np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * np.cos(theta) ** 2 - 1.0)


def spherical_harmonic_y30(theta):
    r"""
    Real spherical harmonic :math:`Y_{30}(\theta)`.

    .. math::
        Y_{30}(\theta) = \sqrt{\frac{7}{16\pi}}\,(5\cos^3\theta - 3\cos\theta)

    Parameters
    ----------
    theta : array_like
        Polar angle in radians.

    Returns
    -------
    y30 : ndarray
        Values of :math:`Y_{30}`.
    """
    c = np.cos(theta)
    return np.sqrt(7.0 / (16.0 * np.pi)) * (5.0 * c ** 3 - 3.0 * c)


def spherical_harmonic_y40(theta):
    r"""
    Real spherical harmonic :math:`Y_{40}(\theta)`.

    .. math::
        Y_{40}(\theta) = \sqrt{\frac{9}{256\pi}}\,(35\cos^4\theta
        - 30\cos^2\theta + 3)

    Parameters
    ----------
    theta : array_like
        Polar angle in radians.

    Returns
    -------
    y40 : ndarray
        Values of :math:`Y_{40}`.
    """
    c = np.cos(theta)
    return np.sqrt(9.0 / (256.0 * np.pi)) * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0)


def deformed_radius(theta, A, beta2=0.0, beta3=0.0, beta4=0.0):
    r"""
    Equivalent radius of a deformed nuclear surface.

    .. math::
        R(\theta) = R_0\Bigl[1 + \beta_2 Y_{20}
        + \beta_3 Y_{30} + \beta_4 Y_{40}\Bigr]

    Parameters
    ----------
    theta : array_like
        Polar angles.
    A : int
        Mass number.
    beta2, beta3, beta4 : float
        Quadrupole, octupole, hexadecapole deformation parameters.

    Returns
    -------
    R : ndarray
        Deformed radius in fm.
    """
    R0 = WS_R0 * (A ** (1.0 / 3.0))
    y20 = spherical_harmonic_y20(theta)
    y30 = spherical_harmonic_y30(theta)
    y40 = spherical_harmonic_y40(theta)
    return R0 * (1.0 + beta2 * y20 + beta3 * y30 + beta4 * y40)


def woods_saxon(r, V0, R, a):
    r"""
    Woods-Saxon form factor.

    .. math::
        f(r) = \dfrac{V_0}{1 + \exp\!\left(\dfrac{r - R}{a}\right)}

    Parameters
    ----------
    r : array_like
        Radial coordinate in fm.
    V0 : float
        Potential depth in MeV.
    R : float
        Radius parameter in fm.
    a : float
        Diffuseness in fm.

    Returns
    -------
    V : ndarray
        Potential values in MeV.
    """
    r = np.asarray(r, dtype=float)
    # Avoid overflow in exp for large negative arguments
    arg = (r - R) / a
    # Clamp argument to prevent overflow
    arg = np.clip(arg, -500.0, 500.0)
    return V0 / (1.0 + np.exp(arg))


def woods_saxon_derivative(r, V0, R, a):
    r"""
    Radial derivative of the Woods-Saxon form factor.

    .. math::
        \frac{df}{dr} = -\frac{V_0}{a}
        \frac{\exp\!\left(\dfrac{r-R}{a}\right)}
             {\left[1 + \exp\!\left(\dfrac{r-R}{a}\right)\right]^2}

    Parameters
    ----------
    r : array_like
        Radial coordinate in fm.
    V0 : float
        Potential depth in MeV.
    R : float
        Radius parameter in fm.
    a : float
        Diffuseness in fm.

    Returns
    -------
    dVdr : ndarray
        Derivative in MeV/fm.
    """
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    arg = np.clip(arg, -500.0, 500.0)
    e = np.exp(arg)
    return -(V0 / a) * e / ((1.0 + e) ** 2)


def spin_orbit_potential(r, l, s, A, Vso0=None, Rso=None, aso=None):
    r"""
    Spin-orbit potential for a spherical nucleus.

    .. math::
        V_{\text{so}}(r) = V_{\text{so}}^{0}\,
        \frac{1}{r}\frac{df_{\text{WS}}}{dr}\,\mathbf{l}\!\cdot\!\mathbf{s}

    with :math:`\mathbf{l}\!\cdot\!\mathbf{s} = \frac{1}{2}[j(j+1)-l(l+1)-s(s+1)]`.

    Parameters
    ----------
    r : array_like
        Radial coordinate in fm.
    l : int
        Orbital angular momentum quantum number.
    s : float
        Spin (1/2 for nucleons).
    A : int
        Mass number.
    Vso0, Rso, aso : float, optional
        Spin-orbit parameters.  Defaults from constants module.

    Returns
    -------
    Vso : ndarray
        Spin-orbit potential in MeV.
    """
    if Vso0 is None:
        Vso0 = WS_VSO
    if Rso is None:
        Rso = WS_RSO * (A ** (1.0 / 3.0))
    if aso is None:
        aso = WS_ASO

    r = np.asarray(r, dtype=float)
    ls = 0.5 * (l * (l + 1.0))
    # For spin-1/2: j = l ± 1/2, ls = 0.5*l for j=l+1/2, -0.5*(l+1) for j=l-1/2
    # Here we return the radial shape only; caller multiplies by ls
    dfd = woods_saxon_derivative(r, 1.0, Rso, aso)
    # Spin-orbit is surface-peaked; suppress unphysical core divergence
    r_safe = np.where(r < 0.2, 1.0, r)
    Vso = Vso0 * (1.0 / r_safe) * dfd * ls
    Vso = np.where(r < 0.2, 0.0, Vso)
    return Vso


def coulomb_potential(r, Z, A):
    r"""
    Coulomb potential of a uniformly charged sphere.

    .. math::
        V_{\text{Coul}}(r) = \begin{cases}
        \dfrac{Ze^2}{2R_c}\left(3 - \dfrac{r^2}{R_c^2}\right), & r \le R_c \\[6pt]
        \dfrac{Ze^2}{r}, & r > R_c
        \end{cases}

    Parameters
    ----------
    r : array_like
        Radial coordinate in fm.
    Z : int
        Proton number.
    A : int
        Mass number.

    Returns
    -------
    Vc : ndarray
        Coulomb potential in MeV.
    """
    r = np.asarray(r, dtype=float)
    Rc = 1.2 * (A ** (1.0 / 3.0))
    e2 = 1.439964  # MeV·fm
    Vc = np.empty_like(r)
    inside = r <= Rc
    outside = ~inside
    Vc[inside] = (Z * e2 / (2.0 * Rc)) * (3.0 - (r[inside] / Rc) ** 2)
    Vc[outside] = Z * e2 / r[outside]
    return Vc


def build_mean_field_potential(r, Z, N, beta2=0.0, beta3=0.0, beta4=0.0,
                               return_components=False):
    r"""
    Build the total spherical mean-field potential for protons or neutrons.

    For a spherical nucleus the deformed radius is angle-averaged:

    .. math::
        \langle R(\theta)\rangle = R_0\left(1 + \beta_4 Y_{40}^{\text{avg}}
        + \dots \right)

    but for simplicity we use the spherical equivalent
    :math:`R_{\text{eq}} = R_0(1 + \delta_{\text{def}})` with
    :math:`\delta_{\text{def}} = 0.05\beta_2^2` as a compactness correction.

    Parameters
    ----------
    r : array_like
        Radial mesh in fm.
    Z, N : int
        Proton and neutron numbers.
    beta2, beta3, beta4 : float
        Deformation parameters.
    return_components : bool
        If True, also return dict of individual components.

    Returns
    -------
    V : ndarray
        Total mean-field potential in MeV.
    components : dict, optional
        Dictionary with keys 'central', 'spin_orbit', 'coulomb'.
    """
    A = Z + N
    R0 = WS_R0 * (A ** (1.0 / 3.0))
    # Deformation-induced compactness correction (volume conservation)
    delta_def = 0.05 * beta2 ** 2 + 0.02 * beta3 ** 2 + 0.01 * beta4 ** 2
    Req = R0 * (1.0 - delta_def)

    V_central = woods_saxon(r, WS_V0, Req, WS_A)
    V_coul = coulomb_potential(r, Z, A)

    # Spin-orbit averaged over j = l ± 1/2 (we take l=2, d-shell as representative)
    l_repr = 2
    V_so = spin_orbit_potential(r, l_repr, 0.5, A)

    V_total = V_central + V_so + V_coul

    if return_components:
        return V_total, {
            'central': V_central,
            'spin_orbit': V_so,
            'coulomb': V_coul
        }
    return V_total


def build_neutron_potential(r, N, Z, beta2=0.0, beta3=0.0, beta4=0.0):
    r"""
    Neutron mean-field potential (no Coulomb).

    .. math::
        V_n(r) = V_{\text{WS}}(r) + V_{\text{so}}(r)

    Parameters
    ----------
    r : array_like
        Radial mesh in fm.
    N, Z : int
        Neutron and proton numbers.
    beta2, beta3, beta4 : float
        Deformation parameters.

    Returns
    -------
    Vn : ndarray
        Neutron potential in MeV.
    """
    # HOLE 2: Implement the neutron mean-field potential.
    # Key physics:
    #   1. Mass number A = Z + N.
    #   2. Spherical radius parameter R0 = WS_R0 * A^(1/3).
    #   3. Deformation compactness correction (volume conservation):
    #        delta_def = 0.05*beta2^2 + 0.02*beta3^2 + 0.01*beta4^2
    #      Equivalent radius Req = R0 * (1 - delta_def).
    #   4. Central potential: V_central = woods_saxon(r, WS_V0, Req, WS_A).
    #   5. Spin-orbit potential: V_so = spin_orbit_potential(r, l=2, s=0.5, A).
    #   6. Neutrons have no Coulomb term.
    #   7. Return V_central + V_so.
    raise NotImplementedError("HOLE 2: build_neutron_potential is not implemented.")


def build_proton_potential(r, Z, N, beta2=0.0, beta3=0.0, beta4=0.0):
    r"""
    Proton mean-field potential (includes Coulomb).

    .. math::
        V_p(r) = V_{\text{WS}}(r) + V_{\text{so}}(r) + V_{\text{Coul}}(r)

    Parameters
    ----------
    r : array_like
        Radial mesh in fm.
    Z, N : int
        Proton and neutron numbers.
    beta2, beta3, beta4 : float
        Deformation parameters.

    Returns
    -------
    Vp : ndarray
        Proton potential in MeV.
    """
    A = Z + N
    R0 = WS_R0 * (A ** (1.0 / 3.0))
    delta_def = 0.05 * beta2 ** 2 + 0.02 * beta3 ** 2 + 0.01 * beta4 ** 2
    Req = R0 * (1.0 - delta_def)
    V_central = woods_saxon(r, WS_V0, Req, WS_A)
    V_so = spin_orbit_potential(r, 2, 0.5, A)
    V_coul = coulomb_potential(r, Z, A)
    return V_central + V_so + V_coul
