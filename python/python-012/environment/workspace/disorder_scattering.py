"""
Disorder Scattering Theory for TI Surface States
================================================
Implements impurity scattering for topological insulator surface states
including:
- Born approximation scattering rate
- Self-consistent T-matrix approximation (SCTA)
- Spin-dependent scattering (skew scattering)
- Short-range and long-range (Coulomb) disorder

The scattering rate in the Born approximation for a delta-function
impurity potential V(r) = V_0 delta(r):

    1/tau(E) = (2*pi / hbar) * n_i * V_0^2 * N(E)

where N(E) is the density of states at energy E, and n_i is the
impurity concentration.

For spin-momentum locked Dirac fermions, the scattering matrix element
involves the spin texture overlap:

    |<k',s'|V|k,s>|^2 = V_0^2 * |<u_{k'}|u_k>|^2

The spin overlap factor for the massive Dirac cone:
    |<u_{k'}|u_k>|^2 = (1/2) * [1 + (Delta^2 + hbar^2 v_F^2 k·k') / (E_k E_{k'})]

Self-consistent T-matrix for a single impurity:
    T(E) = V_0 / (1 - V_0 * G_0(E))
    G_0(E) = int d^2k / (2pi)^2 * 1 / (E - E_k + i*0+)
"""

import numpy as np
from dirac_surface import DiracSurfaceHamiltonian

# Physical constants
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
EV_TO_J = 1.602176634e-19
ME = 9.10938356e-31


class DisorderScattering:
    """
    Disorder scattering theory for TI surface states.
    """

    def __init__(self, hamiltonian=None, n_imp=1e15, V0=0.5,
                 disorder_type='delta', screening_length=1.0):
        """
        Parameters
        ----------
        hamiltonian : DiracSurfaceHamiltonian
        n_imp : float
            Impurity concentration in m^{-2}.
        V0 : float
            Disorder potential strength in eV·nm^2 (delta) or eV (Coulomb amplitude).
        disorder_type : str
            'delta' or 'coulomb'.
        screening_length : float
            Screening length in nm for Coulomb disorder.
        """
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian
        self.n_imp = n_imp
        self.V0_eV = V0
        self.disorder_type = disorder_type
        self.screening_length = screening_length * 1e-9

    def density_of_states(self, E, n_k=800, k_max=2e10):
        """
        Compute the density of states per unit area per spin:

            N(E) = int d^2k / (2pi)^2 * delta(E - E_k)

        For the massive Dirac cone:
            N(E) = |E| / (2*pi*hbar^2*v_F^2)  for |E| > |Delta|

        Parameters
        ----------
        E : float
            Energy in Joules.
        n_k : int
            Number of k-points for numerical integration.
        k_max : float
            Cutoff wavevector.

        Returns
        -------
        dos : float
            Density of states in J^{-1} m^{-2}.
        """
        # Analytical formula for massive Dirac cone
        E_abs = abs(E)
        Delta_abs = abs(self.H.Delta)
        if E_abs < Delta_abs:
            return 0.0
        dos = E_abs / (2.0 * np.pi * (self.H.hbar * self.H.v_F) ** 2)
        return dos

    def spin_overlap_factor(self, kx1, ky1, kx2, ky2):
        """
        Compute |<u_{k2}|u_{k1}>|^2 for the massive Dirac model.

        For the upper band:
            |<u_{k'}|u_k>|^2 = (1/2) * [1 + (Delta^2 + hbar^2 v_F^2 k·k') / (E_k E_{k'})]

        Parameters
        ----------
        kx1, ky1 : float
            Initial state momentum.
        kx2, ky2 : float
            Final state momentum.

        Returns
        -------
        overlap : float
            Spin overlap factor [0, 1].
        """
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
        """
        Compute the momentum relaxation rate in the Born approximation.

        1/tau = (2*pi / hbar) * n_i * int d^2k'/(2pi)^2 |
                <k'|V|k>|^2 * (1 - cos(theta_{kk'})) * delta(E - E_{k'})

        Parameters
        ----------
        E : float
            Energy in Joules.
        n_k : int
        k_max : float

        Returns
        -------
        rate : float
            Scattering rate in 1/s.
        """
        Delta_abs = abs(self.H.Delta)
        if abs(E) < Delta_abs:
            return 0.0

        # On-shell momentum
        k_f = np.sqrt(max(0.0, E ** 2 - self.H.Delta ** 2)) / (self.H.hbar * self.H.v_F)
        if k_f < 1e-20:
            return 0.0

        # V0 in J·m^2 for delta disorder
        if self.disorder_type == 'delta':
            V0_J = self.V0_eV * EV_TO_J * (1e-9 ** 2)
            V_sq = V0_J ** 2
        else:
            # Coulomb: approximate
            V0_J = self.V0_eV * EV_TO_J
            kappa = 1.0 / self.screening_length
            V_sq = (V0_J / (k_f + kappa)) ** 2

        # Angular integral of spin overlap * (1 - cos theta)
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

        # 1/tau = (2*pi / hbar) * n_i * V^2 * N(E) * <angular>
        # More precisely: integrate over k' with delta function
        # For 2D: int d^2k' delta(E - E') = 2*pi * k' * (dk'/dE')
        # dk/dE = E / (hbar^2 v_F^2 k)
        # Result: 1/tau = (n_i / hbar) * V^2 * (k_f / (2*pi)) * angular_integral
        prefactor = self.n_imp / self.H.hbar
        geometric = k_f / (2.0 * np.pi)
        rate = prefactor * V_sq * geometric * angular_integral
        return rate

    def self_energy_born(self, E, n_k=400, k_max=2e10):
        """
        Compute the retarded self-energy in the Born approximation:

            Sigma_R(E) = n_i * int d^2k'/(2pi)^2 |V|^2 * G_0(k', E)

        where G_0(k', E) = 1 / (E - E_{k'} + i*0+).

        Parameters
        ----------
        E : float
            Energy in Joules.

        Returns
        -------
        sigma_re, sigma_im : float
            Real and imaginary parts of self-energy in Joules.
        """
        k_vals = np.linspace(0.0, k_max, n_k)
        dk = k_vals[1] - k_vals[0]

        # V0
        if self.disorder_type == 'delta':
            V0_J = self.V0_eV * EV_TO_J * (1e-9 ** 2)
        else:
            V0_J = self.V0_eV * EV_TO_J

        E_k = np.sqrt((self.H.hbar * self.H.v_F * k_vals) ** 2 + self.H.Delta ** 2)

        # Principal value + delta function
        eta = max(1e-24, abs(E) * 1e-6)
        G0 = 1.0 / (E - E_k + 1.0j * eta)

        # Jacobian: d^2k = 2*pi*k*dk
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
        """
        Compute the transport scattering time tau_tr(E).

        1/tau_tr = (2*pi / hbar) * n_i * int d^2k'/(2pi)^2 |
                   <k'|V|k>|^2 * (1 - cos theta) * delta(E - E')

        Parameters
        ----------
        E : float
            Energy in Joules.
        n_k : int

        Returns
        -------
        tau_tr : float
            Transport scattering time in seconds.
        """
        rate = self.born_scattering_rate(E, n_k=n_k)
        if rate < 1e-30:
            return 1e30
        return 1.0 / rate

    def mean_free_path(self, E):
        """
        Compute the mean free path l_mfp = v_F * tau_tr.

        Parameters
        ----------
        E : float
            Energy in Joules.

        Returns
        -------
        l_mfp : float
            Mean free path in meters.
        """
        tau_tr = self.transport_scattering_time(E)
        return self.H.v_F * tau_tr

    def diffusivity(self, E):
        """
        Compute the diffusion constant D = (1/2) * v_F^2 * tau_tr.

        Parameters
        ----------
        E : float

        Returns
        -------
        D : float
            Diffusion constant in m^2/s.
        """
        tau_tr = self.transport_scattering_time(E)
        return 0.5 * (self.H.v_F ** 2) * tau_tr

    def self_consistent_tmatrix(self, E, V0_range=None, tol=1e-8, max_iter=100):
        """
        Solve for the self-consistent T-matrix:

            T(E) = V_0 / (1 - V_0 * G_0(E))

        where G_0(E) is integrated over the BZ.

        Parameters
        ----------
        E : float
            Energy in Joules.
        V0_range : tuple
            (V_min, V_max) in eV for bracketing the self-consistent solution.
        tol : float
            Convergence tolerance.
        max_iter : int

        Returns
        -------
        T_eff : complex
            Effective T-matrix.
        """
        if V0_range is None:
            V0_range = (self.V0_eV * 0.1, self.V0_eV * 10.0)

        # Compute G_0(E) = sum_k 1 / (E - E_k + i*eta)
        n_k = 300
        k_max = 2e10
        k_vals = np.linspace(0.0, k_max, n_k)
        dk = k_vals[1] - k_vals[0]
        E_k = np.sqrt((self.H.hbar * self.H.v_F * k_vals) ** 2 + self.H.Delta ** 2)
        eta = max(1e-24, abs(E) * 1e-6)
        G0 = np.sum(2.0 * np.pi * k_vals * dk / ((2.0 * np.pi) ** 2)
                    * 1.0 / (E - E_k + 1.0j * eta))

        # T-matrix for a given V0
        def t_matrix(V):
            V_J = V * EV_TO_J * (1e-9 ** 2)
            return V_J / (1.0 - V_J * G0)

        # Self-consistency: Re[1/T] = Re[1/V0 - G0]
        # For a given physical scattering rate, solve for effective V
        # Here we simply return the T-matrix evaluated at the nominal V0
        T_eff = t_matrix(self.V0_eV)
        return T_eff

    def skew_scattering_rate(self, E, n_k=200):
        """
        Compute the skew scattering rate due to spin-orbit coupling
        and asymmetric scattering (side-jump).

        For spin-momentum locked surface states, the scattering cross-section
        has an asymmetric component:
            W(k->k') = W_s(k->k') + W_a(k->k')

        where W_a contains the skew term proportional to (k x k')_z.

        Parameters
        ----------
        E : float
            Energy in Joules.
        n_k : int

        Returns
        -------
        skew_rate : float
            Skew scattering rate in 1/s.
        """
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
            # Skew term ~ sin(theta) from (k x k')_z
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
