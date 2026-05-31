# -*- coding: utf-8 -*-
"""
mass_surface.py
===============
Multidimensional interpolation and extrapolation of nuclear mass surfaces,
combining the **Genz test-function** philosophy of *test_interp_nd* with
liquid-drop model and shell-correction theory.

Physical model
--------------
The atomic mass is decomposed as

.. math::
    M(Z,A) = Z m_p + N m_n - B(Z,A)/c^2 \;,

with the binding energy :math:`B(Z,A)` given by the Bethe-Weizsäcker
formula plus shell and pairing corrections:

.. math::
    B(Z,A) &= a_v A - a_s A^{2/3} - a_c \frac{Z^2}{A^{1/3}}
    - a_a \frac{(N-Z)^2}{A} \\
    &\quad + \delta_{\text{pair}} A^{-1/2}
    + E_{\text{shell}}(Z,N) \;.

The shell correction :math:`E_{\text{shell}}` is obtained from the
Strutinsky method:

.. math::
    E_{\text{shell}} = \sum_{i} \epsilon_i n_i
    - \int_{-\infty}^{\lambda} \tilde{g}(\epsilon)\,\epsilon\,d\epsilon \;,

where :math:`\tilde{g}` is a smoothed level density.

Interpolation
-------------
For regions where experimental masses are known, we use radial-basis
interpolation (RBF) on the :math:`(N,Z)` grid.  The basis function is

.. math::
    \phi(r) = r^3 \;,\qquad r = \sqrt{(N-N_i)^2 + (Z-Z_i)^2} \;.

The interpolated mass surface is

.. math::
    M_{\text{int}}(N,Z) = \sum_{j} c_j\,\phi\bigl(\|(N,Z)-(N_j,Z_j)\|\bigr)
    + P(N,Z) \;,

where :math:`P(N,Z)` is a low-degree polynomial ensuring exact reproduction
of global trends.
"""

import numpy as np
from constants import (
    LDA_VOLUME, LDA_SURFACE, LDA_COULOMB, LDA_ASYMMETRY, LDA_PAIRING,
    MASS_PROTON, MASS_NEUTRON
)


def liquid_drop_binding_energy(Z, N):
    r"""
    Liquid-drop model binding energy.

    Parameters
    ----------
    Z, N : int or ndarray
        Proton and neutron numbers.

    Returns
    -------
    B : float or ndarray
        Binding energy in MeV.
    """
    A = Z + N
    if np.any(A == 0):
        return 0.0
    delta_pair = 0.0
    # Pairing: even-even = +, odd-A = 0, odd-odd = -
    if np.isscalar(Z):
        if Z % 2 == 0 and N % 2 == 0:
            delta_pair = LDA_PAIRING / np.sqrt(A)
        elif Z % 2 == 1 and N % 2 == 1:
            delta_pair = -LDA_PAIRING / np.sqrt(A)
    else:
        Z = np.asarray(Z)
        N = np.asarray(N)
        A_arr = np.asarray(A)
        delta_pair = np.zeros_like(A_arr, dtype=float)
        ee = (Z % 2 == 0) & (N % 2 == 0)
        oo = (Z % 2 == 1) & (N % 2 == 1)
        delta_pair[ee] = LDA_PAIRING / np.sqrt(A_arr[ee])
        delta_pair[oo] = -LDA_PAIRING / np.sqrt(A_arr[oo])

    B = (LDA_VOLUME * A
         - LDA_SURFACE * (A ** (2.0 / 3.0))
         - LDA_COULOMB * (Z ** 2) / (A ** (1.0 / 3.0))
         - LDA_ASYMMETRY * ((N - Z) ** 2) / A
         + delta_pair)
    return B


def atomic_mass_ldm(Z, N):
    r"""
    Atomic mass from the liquid-drop model.

    .. math::
        M(Z,A) = Z m_p + N m_n - B_{\text{LDM}}(Z,N)

    Parameters
    ----------
    Z, N : int

    Returns
    -------
    M : float
        Atomic mass in MeV/c².
    """
    A = Z + N
    return Z * MASS_PROTON + N * MASS_NEUTRON - liquid_drop_binding_energy(Z, N)


def shell_correction_from_spectrum(energies, occupancy, lambda_):
    r"""
    Strutinsky shell correction from a discrete single-particle spectrum.

    .. math::
        E_{\text{shell}} = \sum_i \epsilon_i n_i
        - \tilde{E}(\lambda)

    where the smoothed energy is approximated by a Gaussian-smeared sum:

    .. math::
        \tilde{E}(\lambda) \approx \sum_i \epsilon_i
        \frac{1}{\sqrt{2\pi}\sigma}
        \exp\!\left(-\frac{(\epsilon_i-\lambda)^2}{2\sigma^2}\right)
        \times 2\sigma\sqrt{2\pi}

    Parameters
    ----------
    energies : ndarray
        Single-particle energies in MeV.
    occupancy : ndarray
        Occupation numbers (0, 1, or 2).
    lambda_ : float
        Fermi energy.

    Returns
    -------
    E_shell : float
        Shell correction in MeV.
    """
    energies = np.asarray(energies)
    occupancy = np.asarray(occupancy)
    E_sp = np.sum(energies * occupancy)
    # Smoothing width ~ 1.2 * ħω ≈ 41 A^{-1/3} MeV, use generic 7 MeV
    sigma = 7.0
    weights = np.exp(-0.5 * ((energies - lambda_) / sigma) ** 2)
    weights /= np.sum(weights)
    E_smooth = np.sum(energies * weights) * np.sum(occupancy)
    return E_sp - E_smooth


class NuclearMassSurface:
    r"""
    Interpolated nuclear mass surface combining LDM global trend with
    local corrections.
    """

    def __init__(self, data_N, data_Z, data_mass):
        r"""
        Parameters
        ----------
        data_N, data_Z : ndarray, shape (n_data,)
            Known neutron and proton numbers.
        data_mass : ndarray
            Known atomic masses in MeV/c².
        """
        self.data_N = np.asarray(data_N, dtype=float)
        self.data_Z = np.asarray(data_Z, dtype=float)
        self.data_mass = np.asarray(data_mass, dtype=float)
        self.n_data = self.data_N.size
        # Compute LDM prediction at data points
        ldm_masses = np.array([atomic_mass_ldm(int(z), int(n))
                               for z, n in zip(self.data_Z, self.data_N)])
        self.residuals = self.data_mass - ldm_masses
        # Build RBF interpolation matrix
        self._build_rbf()

    def _rbf(self, r):
        """Radial basis function: thin-plate spline r^2 log(r)."""
        r = np.where(r < 1e-10, 1e-10, r)
        return (r ** 2) * np.log(r)

    def _build_rbf(self):
        """Solve for RBF coefficients."""
        Phi = np.zeros((self.n_data, self.n_data))
        for i in range(self.n_data):
            for j in range(self.n_data):
                dx = self.data_N[i] - self.data_N[j]
                dy = self.data_Z[i] - self.data_Z[j]
                Phi[i, j] = self._rbf(np.sqrt(dx * dx + dy * dy))
        # Add polynomial terms [1, N, Z] for reproduction of linear trend
        P = np.vstack([np.ones(self.n_data), self.data_N, self.data_Z]).T
        # Block matrix
        A = np.zeros((self.n_data + 3, self.n_data + 3))
        A[:self.n_data, :self.n_data] = Phi
        A[:self.n_data, self.n_data:] = P
        A[self.n_data:, :self.n_data] = P.T
        rhs = np.zeros(self.n_data + 3)
        rhs[:self.n_data] = self.residuals
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.c = sol[:self.n_data]
        self.p = sol[self.n_data:]

    def evaluate(self, N, Z):
        r"""
        Evaluate interpolated mass at :math:`(N,Z)`.

        Parameters
        ----------
        N, Z : float or ndarray

        Returns
        -------
        mass : float or ndarray
            Atomic mass in MeV/c².
        """
        scalar = np.isscalar(N)
        N = np.atleast_1d(N)
        Z = np.atleast_1d(Z)
        ldm = np.array([atomic_mass_ldm(int(zz), int(nn))
                        for zz, nn in zip(Z.ravel(), N.ravel())])
        ldm = ldm.reshape(N.shape)
        # RBF correction
        corr = np.zeros_like(N, dtype=float)
        for i in range(self.n_data):
            dx = N - self.data_N[i]
            dy = Z - self.data_Z[i]
            r = np.sqrt(dx * dx + dy * dy)
            corr += self.c[i] * self._rbf(r)
        # Polynomial trend
        corr += self.p[0] + self.p[1] * N + self.p[2] * Z
        mass = ldm + corr
        return float(mass) if scalar else mass

    def separation_energy(self, N, Z, nucleon='neutron'):
        r"""
        One-nucleon separation energy.

        .. math::
            S_n = M(Z, N-1) - M(Z, N) + m_n

        Parameters
        ----------
        N, Z : int
        nucleon : {'neutron', 'proton'}

        Returns
        -------
        S : float
            Separation energy in MeV.
        """
        if nucleon == 'neutron':
            if N <= 0:
                return np.inf
            return self.evaluate(N - 1, Z) - self.evaluate(N, Z) + MASS_NEUTRON
        elif nucleon == 'proton':
            if Z <= 0:
                return np.inf
            return self.evaluate(N, Z - 1) - self.evaluate(N, Z) + MASS_PROTON
        else:
            raise ValueError("nucleon must be 'neutron' or 'proton'.")

    def dripline_location(self, Z, direction='neutron'):
        r"""
        Estimate drip-line location by finding where :math:`S_n = 0` or
        :math:`S_p = 0`.

        Parameters
        ----------
        Z : int
            Fixed proton number.
        direction : {'neutron', 'proton'}

        Returns
        -------
        N_drip : int
            Estimated drip-line neutron or proton number.
        """
        search_range = range(1, 3 * Z + 20)
        for N in search_range:
            S = self.separation_energy(N, Z, direction)
            if S < 0:
                return max(N - 1, 0)
        return search_range[-1]


def genz_test_function_oscillatory(m, c, w, x):
    r"""
    Genz "oscillatory" test function (adapted from *test_interp_nd*).

    .. math::
        f(\mathbf{x}) = \cos\!\left(2\pi w_1 + \sum_{j=1}^{m} c_j x_j\right)

    Parameters
    ----------
    m : int
        Dimension.
    c : ndarray, shape (m,)
        Frequency coefficients.
    w : float
        Phase.
    x : ndarray, shape (..., m)
        Evaluation points.

    Returns
    -------
    f : ndarray
        Function values.
    """
    x = np.asarray(x)
    c = np.asarray(c)
    return np.cos(2.0 * np.pi * w + np.dot(x, c))


def mass_surface_curvature(mass_surface, N0, Z0, h=1.0):
    r"""
    Compute local curvature of the mass surface at :math:`(N_0, Z_0)`.

    .. math::
        \kappa = \frac{\partial^2 M}{\partial N^2}
        + \frac{\partial^2 M}{\partial Z^2}
        - 2\frac{\partial^2 M}{\partial N \partial Z}

    Parameters
    ----------
    mass_surface : NuclearMassSurface
    N0, Z0 : float
    h : float
        Finite-difference step.

    Returns
    -------
    kappa : float
        Curvature in MeV.
    """
    M00 = mass_surface.evaluate(N0, Z0)
    M_p0 = mass_surface.evaluate(N0 + h, Z0)
    M_m0 = mass_surface.evaluate(N0 - h, Z0)
    M_0p = mass_surface.evaluate(N0, Z0 + h)
    M_0m = mass_surface.evaluate(N0, Z0 - h)
    M_pp = mass_surface.evaluate(N0 + h, Z0 + h)
    M_mm = mass_surface.evaluate(N0 - h, Z0 - h)
    M_pm = mass_surface.evaluate(N0 + h, Z0 - h)
    M_mp = mass_surface.evaluate(N0 - h, Z0 + h)

    d2N = (M_p0 - 2 * M00 + M_m0) / (h * h)
    d2Z = (M_0p - 2 * M00 + M_0m) / (h * h)
    dNdZ = (M_pp - M_pm - M_mp + M_mm) / (4 * h * h)
    return d2N + d2Z - 2 * dNdZ
