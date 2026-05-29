"""
Kubo-Greenwood Conductivity Formulas for TI Surface States
==========================================================
Implements the linear response conductivity tensor using the
Kubo-Greenwood formula and the Boltzmann equation.

Kubo-Greenwood formula:
    sigma_{ab}(omega) = (e^2 * hbar / V) * sum_{k,n,m} [
        (f(E_{nk}) - f(E_{mk})) / (E_{mk} - E_{nk})
        * <nk|v_a|mk> <mk|v_b|nk> / (hbar*omega + E_{nk} - E_{mk} + i*eta)
    ]

For DC conductivity (omega = 0) in the relaxation time approximation:
    sigma_{xx} = e^2 * int d^2k/(2pi)^2 * (tau(E_k) / hbar^2) * (dE/dk_x)^2 * (-df/dE)

For the anomalous Hall conductivity from Berry curvature:
    sigma_{xy}^{AH} = - (e^2 / h) * int d^2k/(2pi) * Omega_z(k) * f(E_k)

For the spin Hall conductivity:
    sigma_{xy}^{SH} = (e / 4*pi) * int d^2k * (S_z * v_x)

The total Hall conductivity combines intrinsic (Berry), skew scattering,
and side-jump contributions:
    sigma_{xy} = sigma_{xy}^{intrinsic} + sigma_{xy}^{skew} + sigma_{xy}^{side-jump}
"""

import numpy as np
from dirac_surface import DiracSurfaceHamiltonian
from disorder_scattering import DisorderScattering

# Physical constants
E_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23


class KuboConductivity:
    """
    Computes transport coefficients for TI surface states.
    """

    def __init__(self, hamiltonian=None, disorder=None):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        if disorder is None:
            disorder = DisorderScattering(hamiltonian)
        self.H = hamiltonian
        self.dis = disorder

    def dc_conductivity_drude(self, E_F, T=0.0, n_k=400, k_max=2e10):
        """
        Compute the longitudinal DC conductivity in the Drude-Boltzmann
        relaxation time approximation:

            sigma_{xx} = (e^2 / hbar^2) * int d^2k/(2pi)^2 * tau(E_k)
                         * (dE/dk_x)^2 * (-df/dE)

        For a 2D Dirac cone at T=0:
            sigma_{xx} = (e^2 / hbar^2) * (tau(E_F) / (2*pi)) * (hbar v_F)^2 * k_F
                       = (e^2 / h) * (v_F * k_F * tau(E_F) / hbar)

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        T : float
            Temperature in K.
        n_k : int
        k_max : float

        Returns
        -------
        sigma_xx : float
            Longitudinal conductivity in S (per layer).
        """
        E_F_J = E_F * 1.602176634e-19

        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)

        # Fermi-Dirac and derivative
        if T < 1e-6:
            # At T=0, -df/dE = delta(E - E_F)
            # Approximate with a narrow Lorentzian
            eta = 1e-23
            fd_deriv = (1.0 / np.pi) * eta / ((E_plus - E_F_J) ** 2 + eta ** 2)
        else:
            beta = 1.0 / (K_B * T)
            f = 1.0 / (1.0 + np.exp(beta * (E_plus - E_F_J)))
            fd_deriv = beta * f * (1.0 - f)

        # Velocity: v_x = dE/dk_x = hbar v_F^2 k_x / E
        v_x = self.H.hbar * (self.H.v_F ** 2) * KX / np.where(
            np.abs(E_plus) < 1e-40, 1e-40, E_plus
        )

        # Scattering time
        tau = np.array([
            [self.dis.transport_scattering_time(E_plus[i, j])
             for j in range(n_k)] for i in range(n_k)
        ])

        integrand = tau * (v_x ** 2) * fd_deriv
        integral = np.sum(integrand) * dkx * dky

        sigma_xx = (E_CHARGE ** 2 / self.H.hbar ** 2) * integral
        return sigma_xx

    def dc_conductivity_semicalassical(self, E_F, T=0.0):
        """
        Semiclassical formula for 2D Dirac fermions:

            sigma_{xx} = (e^2 / h) * (2 * E_F * tau_tr / hbar)

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        T : float

        Returns
        -------
        sigma_xx : float
            Conductivity in S (per layer).
        """
        E_F_J = E_F * 1.602176634e-19
        tau_tr = self.dis.transport_scattering_time(E_F_J)
        sigma_xx = (E_CHARGE ** 2 / (2.0 * np.pi * HBAR)) * (2.0 * E_F_J * tau_tr / HBAR)
        return sigma_xx

    def intrinsic_anomalous_hall(self, E_F, n_k=400, k_max=2e10):
        """
        Intrinsic anomalous Hall conductivity from Berry curvature.

        sigma_{xy}^{int} = - (e^2 / h) * (1 / 2*pi) * int d^2k Omega(k) f(E_k)

        For the massive Dirac cone with Fermi level in the gap:
            sigma_{xy}^{int} = (e^2 / 2h) * sign(Delta)

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        n_k : int
        k_max : float

        Returns
        -------
        sigma_xy : float
            Anomalous Hall conductivity in S (per layer).
        """
        E_F_J = E_F * 1.602176634e-19
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)

        # Occupation at T=0
        f_lower = np.where(E_minus < E_F_J, 1.0, 0.0)
        f_upper = np.where(E_plus < E_F_J, 1.0, 0.0)

        # TODO HOLE 2: Compute intrinsic anomalous Hall conductivity from Berry curvature.
        # Steps:
        #   1. Create BerryCurvatureCalculator(self.H)
        #   2. Get Omega_lower and Omega_upper on the (KX, KY) mesh
        #   3. Integrate Omega(k) * f(E_k) over k-space
        #   4. Convert from e^2/h units to SI (S) using E_CHARGE^2 / H_PLANCK
        #   5. Apply the correct sign and 1/(2*pi) normalization.
        raise NotImplementedError("HOLE 2: intrinsic_anomalous_hall integration not implemented")

    def skew_scattering_hall(self, E_F, n_k=200):
        """
        Skew scattering contribution to the Hall conductivity:

            sigma_{xy}^{skew} = (e^2 / h) * (tau_skew / tau) * sigma_{xx}

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        n_k : int

        Returns
        -------
        sigma_xy_skew : float
            Skew scattering Hall conductivity in S.
        """
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
        """
        Total Hall conductivity combining intrinsic, skew, and side-jump.

            sigma_{xy}^{total} = sigma_{xy}^{int} + sigma_{xy}^{skew} + sigma_{xy}^{sj}

        The side-jump contribution for Dirac fermions is approximately:
            sigma_{xy}^{sj} ≈ - (e^2 / h) * (k_F * l_so)^{-1}

        where l_so is the spin-orbit scattering length.

        Parameters
        ----------
        E_F : float
        n_k : int
        k_max : float

        Returns
        -------
        sigma_xy_total : float
        sigma_xy_intrinsic : float
        sigma_xy_skew : float
        sigma_xy_sj : float
        """
        sigma_int = self.intrinsic_anomalous_hall(E_F, n_k=n_k, k_max=k_max)
        sigma_skew = self.skew_scattering_hall(E_F, n_k=n_k)

        # Side-jump estimate
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
        """
        Compute the spin Hall conductivity for TI surface states.

        For a single Dirac cone with spin-momentum locking:
            sigma_{xy}^z = (e / 8*pi) * int d^2k S_z(k) v_x(k) (-df/dE)

        For the ideal Dirac cone, the spin Hall conductivity is:
            sigma_{xy}^{spin} = (e / 8*pi) * (hbar v_F) * k_F

        Parameters
        ----------
        E_F : float
        n_k : int
        k_max : float

        Returns
        -------
        sigma_spin : float
            Spin Hall conductivity in S.
        """
        E_F_J = E_F * 1.602176634e-19
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)

        # T=0 approximation
        eta = 1e-23
        fd_deriv = (1.0 / np.pi) * eta / ((E_plus - E_F_J) ** 2 + eta ** 2)

        # Spin texture
        Sx, Sy, Sz = self.H.spin_texture(KX, KY, band='upper')

        # Velocity
        v_x = self.H.hbar * (self.H.v_F ** 2) * KX / np.where(
            np.abs(E_plus) < 1e-40, 1e-40, E_plus
        )

        integrand = Sz * v_x * fd_deriv
        integral = np.sum(integrand) * dkx * dky

        sigma_spin = (E_CHARGE / (8.0 * np.pi)) * integral
        return sigma_spin

    def thermoelectric_coefficients(self, E_F, T, n_k=300, k_max=2e10):
        """
        Compute thermoelectric coefficients using the Mott formula:

            S = - (pi^2 / 3) * (k_B^2 T / e) * (d ln(sigma) / dE)_{E_F}

            L = (pi^2 / 3) * (k_B / e)^2 * sigma * T

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        T : float
            Temperature in K.
        n_k : int
        k_max : float

        Returns
        -------
        S : float
            Seebeck coefficient in V/K.
        L : float
            Lorenz number in W·Omega/K^2.
        """
        dE = 0.001  # eV
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
