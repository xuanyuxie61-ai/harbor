
import numpy as np
from dirac_surface import DiracSurfaceHamiltonian


HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
EV_TO_J = 1.602176634e-19
ME = 9.10938356e-31


class DisorderScattering:

    def __init__(self, hamiltonian=None, n_imp=1e15, V0=0.5,
                 disorder_type='delta', screening_length=1.0):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian
        self.n_imp = n_imp
        self.V0_eV = V0
        self.disorder_type = disorder_type
        self.screening_length = screening_length * 1e-9

    def density_of_states(self, E, n_k=800, k_max=2e10):

        E_abs = abs(E)
        Delta_abs = abs(self.H.Delta)
        if E_abs < Delta_abs:
            return 0.0
        dos = E_abs / (2.0 * np.pi * (self.H.hbar * self.H.v_F) ** 2)
        return dos

    def spin_overlap_factor(self, kx1, ky1, kx2, ky2):
        k1_sq = kx1 ** 2 + ky1 ** 2
        k2_sq = kx2 ** 2 + ky2 ** 2
        k_dot_kp = kx1 * kx2 + ky1 * ky2

        E1 = np.sqrt((self.H.hbar * self.H.v_F * np.sqrt(k1_sq)) ** 2
                     + self.H.Delta ** 2)
        E2 = np.sqrt((self.H.hbar * self.H.v_F * np.sqrt(k2_sq)) ** 2
                     + self.H.Delta ** 2)

        if E1 < 1e-40 or E2 < 1e-40:
            return 1.0

        numerator = self.H.Delta ** 2 + (self.H.hbar * self.H.v_F) ** 2 * k_dot_kp
        overlap = 0.5 * (1.0 + numerator / (E1 * E2))
        return max(0.0, min(1.0, overlap))

    def born_scattering_rate(self, E, n_k=200, k_max=2e10):
        Delta_abs = abs(self.H.Delta)
        if abs(E) < Delta_abs:
            return 0.0


        k_f = np.sqrt(max(0.0, E ** 2 - self.H.Delta ** 2)) / (self.H.hbar * self.H.v_F)
        if k_f < 1e-20:
            return 0.0


        if self.disorder_type == 'delta':
            V0_J = self.V0_eV * EV_TO_J * (1e-9 ** 2)
            V_sq = V0_J ** 2
        else:

            V0_J = self.V0_eV * EV_TO_J
            kappa = 1.0 / self.screening_length
            V_sq = (V0_J / (k_f + kappa)) ** 2


        theta_vals = np.linspace(0.0, 2.0 * np.pi, n_k)
        d_theta = theta_vals[1] - theta_vals[0]

        kx1, ky1 = k_f, 0.0
        integrand = np.zeros_like(theta_vals)
        for i, th in enumerate(theta_vals):
            kx2 = k_f * np.cos(th)
            ky2 = k_f * np.sin(th)
            overlap = self.spin_overlap_factor(kx1, ky1, kx2, ky2)
            integrand[i] = overlap * (1.0 - np.cos(th))

        angular_integral = np.sum(integrand) * d_theta






        prefactor = self.n_imp / self.H.hbar
        geometric = k_f / (2.0 * np.pi)
        rate = prefactor * V_sq * geometric * angular_integral
        return rate

    def self_energy_born(self, E, n_k=400, k_max=2e10):
        k_vals = np.linspace(0.0, k_max, n_k)
        dk = k_vals[1] - k_vals[0]


        if self.disorder_type == 'delta':
            V0_J = self.V0_eV * EV_TO_J * (1e-9 ** 2)
        else:
            V0_J = self.V0_eV * EV_TO_J

        E_k = np.sqrt((self.H.hbar * self.H.v_F * k_vals) ** 2 + self.H.Delta ** 2)


        eta = max(1e-24, abs(E) * 1e-6)
        G0 = 1.0 / (E - E_k + 1.0j * eta)


        weight = 2.0 * np.pi * k_vals * dk / ((2.0 * np.pi) ** 2)

        if self.disorder_type == 'delta':
            V_sq = V0_J ** 2 * np.ones_like(k_vals)
        else:
            kappa = 1.0 / self.screening_length
            V_sq = (V0_J / (k_vals + kappa)) ** 2

        integrand = V_sq * G0 * weight
        sigma = self.n_imp * np.sum(integrand)
        return np.real(sigma), np.imag(sigma)

    def transport_scattering_time(self, E, n_k=200):
        rate = self.born_scattering_rate(E, n_k=n_k)
        if rate < 1e-30:
            return 1e30
        return 1.0 / rate

    def mean_free_path(self, E):
        tau_tr = self.transport_scattering_time(E)
        return self.H.v_F * tau_tr

    def diffusivity(self, E):
        tau_tr = self.transport_scattering_time(E)
        return 0.5 * (self.H.v_F ** 2) * tau_tr

    def self_consistent_tmatrix(self, E, V0_range=None, tol=1e-8, max_iter=100):
        if V0_range is None:
            V0_range = (self.V0_eV * 0.1, self.V0_eV * 10.0)


        n_k = 300
        k_max = 2e10
        k_vals = np.linspace(0.0, k_max, n_k)
        dk = k_vals[1] - k_vals[0]
        E_k = np.sqrt((self.H.hbar * self.H.v_F * k_vals) ** 2 + self.H.Delta ** 2)
        eta = max(1e-24, abs(E) * 1e-6)
        G0 = np.sum(2.0 * np.pi * k_vals * dk / ((2.0 * np.pi) ** 2)
                    * 1.0 / (E - E_k + 1.0j * eta))


        def t_matrix(V):
            V_J = V * EV_TO_J * (1e-9 ** 2)
            return V_J / (1.0 - V_J * G0)




        T_eff = t_matrix(self.V0_eV)
        return T_eff

    def skew_scattering_rate(self, E, n_k=200):
        Delta_abs = abs(self.H.Delta)
        if abs(E) < Delta_abs:
            return 0.0

        k_f = np.sqrt(max(0.0, E ** 2 - self.H.Delta ** 2)) / (self.H.hbar * self.H.v_F)
        if k_f < 1e-20:
            return 0.0

        theta_vals = np.linspace(0.0, 2.0 * np.pi, n_k)
        d_theta = theta_vals[1] - theta_vals[0]

        kx1, ky1 = k_f, 0.0
        integrand = np.zeros_like(theta_vals)
        for i, th in enumerate(theta_vals):
            kx2 = k_f * np.cos(th)
            ky2 = k_f * np.sin(th)

            skew_factor = np.sin(th)
            overlap = self.spin_overlap_factor(kx1, ky1, kx2, ky2)
            integrand[i] = overlap * skew_factor

        angular_integral = np.sum(integrand) * d_theta

        if self.disorder_type == 'delta':
            V0_J = self.V0_eV * EV_TO_J * (1e-9 ** 2)
            V_sq = V0_J ** 2
        else:
            V0_J = self.V0_eV * EV_TO_J
            kappa = 1.0 / self.screening_length
            V_sq = (V0_J / (k_f + kappa)) ** 2

        prefactor = self.n_imp / self.H.hbar
        geometric = k_f / (2.0 * np.pi)
        skew_rate = prefactor * V_sq * geometric * angular_integral
        return skew_rate
