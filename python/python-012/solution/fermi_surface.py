"""
Fermi Surface Geometry and Anisotropic Transport
================================================
Models the Fermi surface of topological insulator surface states,
including:
- Circular Fermi surface (isotropic Dirac cone)
- Ellipsoidal Fermi surface (anisotropic systems)
- Hexagonal warping (Bi2Te3-like)
- 3D impurity scattering geometry and chord distributions

The Fermi surface is defined by E(k) = E_F. For the massive Dirac model:
    k_F = sqrt(E_F^2 - Delta^2) / (hbar v_F)

For anisotropic systems (ellipsoid, project 332):
    (k_x / a)^2 + (k_y / b)^2 = k_F^2

The Fermi surface area determines the DOS:
    N(E_F) = A_FS / (4*pi^2 * hbar v_F)

Scattering geometry uses chord-length distributions from
projects 178 (circle), 230 (cube), and 567 (hypersphere).
"""

import numpy as np
from dirac_surface import DiracSurfaceHamiltonian

HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19


class FermiSurface:
    """
    Fermi surface geometry for TI surface states.
    """

    def __init__(self, hamiltonian=None, E_F=0.15):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian
        self.E_F = E_F * EV_TO_J

    def fermi_wavevector(self):
        """
        Compute the Fermi wavevector k_F.

        For the massive Dirac cone:
            k_F = sqrt(E_F^2 - Delta^2) / (hbar v_F)

        Returns
        -------
        k_F : float
            Fermi wavevector in 1/m.
        """
        E_sq = self.E_F ** 2
        Delta_sq = self.H.Delta ** 2
        if E_sq <= Delta_sq:
            return 0.0
        k_F = np.sqrt(E_sq - Delta_sq) / (self.H.hbar * self.H.v_F)
        return k_F

    def fermi_velocity(self):
        """
        Compute the Fermi velocity magnitude.

        v_F* = dE/dk = hbar v_F^2 k_F / E_F

        Returns
        -------
        v_F_star : float
            Fermi velocity in m/s.
        """
        k_F = self.fermi_wavevector()
        if k_F < 1e-20:
            return 0.0
        E_F_abs = abs(self.E_F)
        if E_F_abs < 1e-30:
            return 0.0
        v_F_star = (self.H.hbar * self.H.v_F ** 2 * k_F) / E_F_abs
        return v_F_star

    def fermi_surface_area(self):
        """
        Compute the Fermi surface area (circumference in 2D).

        For a circular FS: A = 2*pi*k_F.

        Returns
        -------
        area : float
            Fermi surface circumference in 1/m.
        """
        k_F = self.fermi_wavevector()
        return 2.0 * np.pi * k_F

    def density_of_states(self):
        """
        Compute the DOS at the Fermi level per unit area.

        N(E_F) = k_F / (2*pi*hbar*v_F*) = |E_F| / (2*pi*hbar^2*v_F^2)

        Returns
        -------
        dos : float
            DOS in J^{-1} m^{-2}.
        """
        E_F_abs = abs(self.E_F)
        dos = E_F_abs / (2.0 * np.pi * (self.H.hbar * self.H.v_F) ** 2)
        return dos

    def sample_momentum(self, n_samples=1):
        """
        Sample random momentum vectors uniformly on the Fermi surface.

        Inspired by circle_unit_sample (project 178).

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        kx, ky : ndarray
            Momentum components.
        """
        k_F = self.fermi_wavevector()
        theta = 2.0 * np.pi * np.random.rand(n_samples)
        kx = k_F * np.cos(theta)
        ky = k_F * np.sin(theta)
        return kx, ky

    def sample_momentum_ellipsoid(self, a_ratio=1.0, b_ratio=1.2, n_samples=1):
        """
        Sample random momentum on an ellipsoidal Fermi surface.

        (k_x / a)^2 + (k_y / b)^2 = k_F^2

        Inspired by ellipsoid_area (project 332).

        Parameters
        ----------
        a_ratio, b_ratio : float
            Anisotropy ratios.
        n_samples : int

        Returns
        -------
        kx, ky : ndarray
        """
        k_F = self.fermi_wavevector()
        a = k_F * a_ratio
        b = k_F * b_ratio
        theta = 2.0 * np.pi * np.random.rand(n_samples)
        kx = a * np.cos(theta)
        ky = b * np.sin(theta)
        return kx, ky

    def fermi_surface_points(self, n_theta=360):
        """
        Generate points on the Fermi surface.

        Parameters
        ----------
        n_theta : int

        Returns
        -------
        kx, ky : ndarray
        """
        k_F = self.fermi_wavevector()
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        kx = k_F * np.cos(theta)
        ky = k_F * np.sin(theta)
        return kx, ky

    def warped_fermi_surface(self, lambda_w=30.0, n_theta=360):
        """
        Compute the warped Fermi surface for Bi2Te3-like hexagonal warping.

        The energy dispersion with warping:
            E(k) = sqrt((hbar v_F k)^2 + (Delta + lambda_w * cos(3*phi))^2)

        For a given E_F, k_F(phi) is found by solving E(k_F, phi) = E_F.

        Parameters
        ----------
        lambda_w : float
            Warping amplitude in eV·nm^3.
        n_theta : int

        Returns
        -------
        kx, ky : ndarray
        """
        lambda_w_SI = lambda_w * EV_TO_J * (1e-9 ** 3)
        phi = np.linspace(0.0, 2.0 * np.pi, n_theta)

        k_F = np.zeros_like(phi)
        for i, p in enumerate(phi):
            # E_F^2 = (hbar v_F k)^2 + (Delta + lambda_w cos(3phi))^2
            gap_term = self.H.Delta + lambda_w_SI * np.cos(3.0 * p)
            k_sq = max(0.0, self.E_F ** 2 - gap_term ** 2)
            k_F[i] = np.sqrt(k_sq) / (self.H.hbar * self.H.v_F)

        kx = k_F * np.cos(phi)
        ky = k_F * np.sin(phi)
        return kx, ky

    def chord_length_distribution_circle(self, n_samples=50000):
        """
        Compute the chord length distribution for pairs of random points
        on the circular Fermi surface.

        Inspired by circle_distance_pdf (project 178).

        For a circle of radius R, the chord length PDF is:
            pdf(d) = (1 / pi) * 1 / sqrt(1 - 0.25 * d^2 / R^2)

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        distances : ndarray
        hist, bins : ndarray
        """
        k_F = self.fermi_wavevector()
        theta1 = 2.0 * np.pi * np.random.rand(n_samples)
        theta2 = 2.0 * np.pi * np.random.rand(n_samples)

        x1 = k_F * np.cos(theta1)
        y1 = k_F * np.sin(theta1)
        x2 = k_F * np.cos(theta2)
        y2 = k_F * np.sin(theta2)

        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distances

    def chord_length_distribution_cube_projected(self, n_samples=50000):
        """
        Compute the 2D projected chord distribution from a 3D scattering
        cube geometry (project 230).

        This models the distribution of momentum transfer vectors
        q = k' - k projected onto the surface plane when impurities
        are distributed in a 3D cube of size L.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        distances : ndarray
        """
        L = 1.0  # Normalized cube size
        x1 = np.random.uniform(-L / 2, L / 2, n_samples)
        y1 = np.random.uniform(-L / 2, L / 2, n_samples)
        z1 = np.random.uniform(-L / 2, L / 2, n_samples)
        x2 = np.random.uniform(-L / 2, L / 2, n_samples)
        y2 = np.random.uniform(-L / 2, L / 2, n_samples)
        z2 = np.random.uniform(-L / 2, L / 2, n_samples)

        # Projected distance in xy-plane
        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distances

    def impurity_scattering_wavevector_distribution(self, n_samples=50000):
        """
        Generate the distribution of scattering wavevectors q = k' - k
        on the Fermi surface.

        Returns
        -------
        qx, qy : ndarray
        q_magnitude : ndarray
        """
        k_F = self.fermi_wavevector()
        theta1 = 2.0 * np.pi * np.random.rand(n_samples)
        theta2 = 2.0 * np.pi * np.random.rand(n_samples)

        kx1 = k_F * np.cos(theta1)
        ky1 = k_F * np.sin(theta1)
        kx2 = k_F * np.cos(theta2)
        ky2 = k_F * np.sin(theta2)

        qx = kx2 - kx1
        qy = ky2 - ky1
        q_mag = np.sqrt(qx ** 2 + qy ** 2)
        return qx, qy, q_mag

    def spin_texture_on_fs(self, n_theta=360):
        """
        Compute the spin texture (S_x, S_y, S_z) on the Fermi surface.

        Returns
        -------
        theta : ndarray
        Sx, Sy, Sz : ndarray
        """
        kx, ky = self.fermi_surface_points(n_theta)
        Sx, Sy, Sz = self.H.spin_texture(kx, ky, band='upper')
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        return theta, Sx, Sy, Sz

    def carrier_density(self):
        """
        Compute the carrier density n = k_F^2 / (4*pi).

        Returns
        -------
        n : float
            Carrier density in m^{-2}.
        """
        k_F = self.fermi_wavevector()
        return k_F ** 2 / (4.0 * np.pi)

    def cyclotron_mass(self):
        """
        Compute the cyclotron effective mass.

        m_c = (hbar^2 / 2*pi) * dA_FS / dE

        For the Dirac cone:
            m_c = hbar * k_F / v_F = E_F / v_F^2

        Returns
        -------
        m_c : float
            Effective mass in kg.
        """
        E_F_abs = abs(self.E_F)
        if self.H.v_F < 1e-10:
            return float('inf')
        m_c = E_F_abs / (self.H.v_F ** 2)
        return m_c

    def cyclotron_frequency(self, B=1.0):
        """
        Compute the cyclotron frequency omega_c = eB / m_c.

        Parameters
        ----------
        B : float
            Magnetic field in Tesla.

        Returns
        -------
        omega_c : float
            Cyclotron frequency in rad/s.
        """
        m_c = self.cyclotron_mass()
        if m_c < 1e-40:
            return 0.0
        e_charge = 1.602176634e-19
        omega_c = e_charge * B / m_c
        return omega_c
