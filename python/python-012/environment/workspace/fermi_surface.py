
import numpy as np
from dirac_surface import DiracSurfaceHamiltonian

HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19


class FermiSurface:

    def __init__(self, hamiltonian=None, E_F=0.15):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian
        self.E_F = E_F * EV_TO_J

    def fermi_wavevector(self):
        E_sq = self.E_F ** 2
        Delta_sq = self.H.Delta ** 2
        if E_sq <= Delta_sq:
            return 0.0
        k_F = np.sqrt(E_sq - Delta_sq) / (self.H.hbar * self.H.v_F)
        return k_F

    def fermi_velocity(self):
        k_F = self.fermi_wavevector()
        if k_F < 1e-20:
            return 0.0
        E_F_abs = abs(self.E_F)
        if E_F_abs < 1e-30:
            return 0.0
        v_F_star = (self.H.hbar * self.H.v_F ** 2 * k_F) / E_F_abs
        return v_F_star

    def fermi_surface_area(self):
        k_F = self.fermi_wavevector()
        return 2.0 * np.pi * k_F

    def density_of_states(self):
        E_F_abs = abs(self.E_F)
        dos = E_F_abs / (2.0 * np.pi * (self.H.hbar * self.H.v_F) ** 2)
        return dos

    def sample_momentum(self, n_samples=1):
        k_F = self.fermi_wavevector()
        theta = 2.0 * np.pi * np.random.rand(n_samples)
        kx = k_F * np.cos(theta)
        ky = k_F * np.sin(theta)
        return kx, ky

    def sample_momentum_ellipsoid(self, a_ratio=1.0, b_ratio=1.2, n_samples=1):
        k_F = self.fermi_wavevector()
        a = k_F * a_ratio
        b = k_F * b_ratio
        theta = 2.0 * np.pi * np.random.rand(n_samples)
        kx = a * np.cos(theta)
        ky = b * np.sin(theta)
        return kx, ky

    def fermi_surface_points(self, n_theta=360):
        k_F = self.fermi_wavevector()
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        kx = k_F * np.cos(theta)
        ky = k_F * np.sin(theta)
        return kx, ky

    def warped_fermi_surface(self, lambda_w=30.0, n_theta=360):
        lambda_w_SI = lambda_w * EV_TO_J * (1e-9 ** 3)
        phi = np.linspace(0.0, 2.0 * np.pi, n_theta)

        k_F = np.zeros_like(phi)
        for i, p in enumerate(phi):

            gap_term = self.H.Delta + lambda_w_SI * np.cos(3.0 * p)
            k_sq = max(0.0, self.E_F ** 2 - gap_term ** 2)
            k_F[i] = np.sqrt(k_sq) / (self.H.hbar * self.H.v_F)

        kx = k_F * np.cos(phi)
        ky = k_F * np.sin(phi)
        return kx, ky

    def chord_length_distribution_circle(self, n_samples=50000):
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
        L = 1.0
        x1 = np.random.uniform(-L / 2, L / 2, n_samples)
        y1 = np.random.uniform(-L / 2, L / 2, n_samples)
        z1 = np.random.uniform(-L / 2, L / 2, n_samples)
        x2 = np.random.uniform(-L / 2, L / 2, n_samples)
        y2 = np.random.uniform(-L / 2, L / 2, n_samples)
        z2 = np.random.uniform(-L / 2, L / 2, n_samples)


        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distances

    def impurity_scattering_wavevector_distribution(self, n_samples=50000):
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
        kx, ky = self.fermi_surface_points(n_theta)
        Sx, Sy, Sz = self.H.spin_texture(kx, ky, band='upper')
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        return theta, Sx, Sy, Sz

    def carrier_density(self):
        k_F = self.fermi_wavevector()
        return k_F ** 2 / (4.0 * np.pi)

    def cyclotron_mass(self):
        E_F_abs = abs(self.E_F)
        if self.H.v_F < 1e-10:
            return float('inf')
        m_c = E_F_abs / (self.H.v_F ** 2)
        return m_c

    def cyclotron_frequency(self, B=1.0):
        m_c = self.cyclotron_mass()
        if m_c < 1e-40:
            return 0.0
        e_charge = 1.602176634e-19
        omega_c = e_charge * B / m_c
        return omega_c
