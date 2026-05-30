
import numpy as np
from dirac_surface import DiracSurfaceHamiltonian
from disorder_scattering import DisorderScattering


E_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23


class KuboConductivity:

    def __init__(self, hamiltonian=None, disorder=None):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        if disorder is None:
            disorder = DisorderScattering(hamiltonian)
        self.H = hamiltonian
        self.dis = disorder

    def dc_conductivity_drude(self, E_F, T=0.0, n_k=400, k_max=2e10):
        E_F_J = E_F * 1.602176634e-19

        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)


        if T < 1e-6:


            eta = 1e-23
            fd_deriv = (1.0 / np.pi) * eta / ((E_plus - E_F_J) ** 2 + eta ** 2)
        else:
            beta = 1.0 / (K_B * T)
            f = 1.0 / (1.0 + np.exp(beta * (E_plus - E_F_J)))
            fd_deriv = beta * f * (1.0 - f)


        v_x = self.H.hbar * (self.H.v_F ** 2) * KX / np.where(
            np.abs(E_plus) < 1e-40, 1e-40, E_plus
        )


        tau = np.array([
            [self.dis.transport_scattering_time(E_plus[i, j])
             for j in range(n_k)] for i in range(n_k)
        ])

        integrand = tau * (v_x ** 2) * fd_deriv
        integral = np.sum(integrand) * dkx * dky

        sigma_xx = (E_CHARGE ** 2 / self.H.hbar ** 2) * integral
        return sigma_xx

    def dc_conductivity_semicalassical(self, E_F, T=0.0):
        E_F_J = E_F * 1.602176634e-19
        tau_tr = self.dis.transport_scattering_time(E_F_J)
        sigma_xx = (E_CHARGE ** 2 / (2.0 * np.pi * HBAR)) * (2.0 * E_F_J * tau_tr / HBAR)
        return sigma_xx

    def intrinsic_anomalous_hall(self, E_F, n_k=400, k_max=2e10):
        E_F_J = E_F * 1.602176634e-19
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)


        f_lower = np.where(E_minus < E_F_J, 1.0, 0.0)
        f_upper = np.where(E_plus < E_F_J, 1.0, 0.0)








        raise NotImplementedError("HOLE 2: intrinsic_anomalous_hall integration not implemented")

    def skew_scattering_hall(self, E_F, n_k=200):
        E_F_J = E_F * 1.602176634e-19
        rate = self.dis.born_scattering_rate(E_F_J, n_k=n_k)
        skew_rate = self.dis.skew_scattering_rate(E_F_J, n_k=n_k)
        sigma_xx = self.dc_conductivity_semicalassical(E_F)

        if abs(rate) < 1e-30:
            return 0.0
        ratio = skew_rate / rate
        sigma_xy_skew = ratio * sigma_xx
        return sigma_xy_skew

    def total_hall_conductivity(self, E_F, n_k=400, k_max=2e10):
        sigma_int = self.intrinsic_anomalous_hall(E_F, n_k=n_k, k_max=k_max)
        sigma_skew = self.skew_scattering_hall(E_F, n_k=n_k)


        E_F_J = E_F * 1.602176634e-19
        if abs(E_F_J) > abs(self.H.Delta):
            k_F = np.sqrt(E_F_J ** 2 - self.H.Delta ** 2) / (self.H.hbar * self.H.v_F)
            l_so = self.dis.mean_free_path(E_F_J)
            if k_F * l_so > 1e-10:
                sigma_sj = - (E_CHARGE ** 2 / H_PLANCK) * (1.0 / (k_F * l_so))
            else:
                sigma_sj = 0.0
        else:
            sigma_sj = 0.0

        sigma_total = sigma_int + sigma_skew + sigma_sj
        return sigma_total, sigma_int, sigma_skew, sigma_sj

    def spin_hall_conductivity(self, E_F, n_k=400, k_max=2e10):
        E_F_J = E_F * 1.602176634e-19
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)


        eta = 1e-23
        fd_deriv = (1.0 / np.pi) * eta / ((E_plus - E_F_J) ** 2 + eta ** 2)


        Sx, Sy, Sz = self.H.spin_texture(KX, KY, band='upper')


        v_x = self.H.hbar * (self.H.v_F ** 2) * KX / np.where(
            np.abs(E_plus) < 1e-40, 1e-40, E_plus
        )

        integrand = Sz * v_x * fd_deriv
        integral = np.sum(integrand) * dkx * dky

        sigma_spin = (E_CHARGE / (8.0 * np.pi)) * integral
        return sigma_spin

    def thermoelectric_coefficients(self, E_F, T, n_k=300, k_max=2e10):
        dE = 0.001
        sigma1 = self.dc_conductivity_semicalassical(E_F + dE)
        sigma2 = self.dc_conductivity_semicalassical(E_F - dE)
        d_sigma_dE = (sigma1 - sigma2) / (2.0 * dE * 1.602176634e-19)

        sigma = self.dc_conductivity_semicalassical(E_F)

        if abs(sigma) < 1e-30:
            S = 0.0
        else:
            S = - (np.pi ** 2 / 3.0) * (K_B ** 2 * T / E_CHARGE) * (d_sigma_dE / sigma)

        L = (np.pi ** 2 / 3.0) * (K_B / E_CHARGE) ** 2 * sigma * T
        return S, L


H_PLANCK = 6.62607015e-34
