"""
plasma_parameters.py
====================
Fundamental plasma physics constants and dimensionless parameters for
solar flare magnetic reconnection simulations.

Physical context:
  - Solar corona: T ~ 1-10 MK, n_e ~ 1e9-1e10 cm^{-3}, B ~ 10-100 G
  - Lundquist number S = L V_A / eta ~ 1e12-1e14 (corona)
  - Plasma beta = 2 mu0 p / B^2 ~ 0.01-0.1 (corona, low-beta)
  - Alfvén speed V_A = B / sqrt(mu0 rho) ~ 1e3-1e4 km/s
  - Ion inertial length d_i = c/omega_pi ~ 10-100 m
  - Ion sound gyroradius rho_s = c_s / omega_ci ~ 1-10 m

Dimensionless normalization used in this code:
  Length  -> L_ref  (half-current-sheet thickness)
  Magnetic field -> B_ref (upstream reconnecting field)
  Time -> tau_A = L_ref / V_A  (Alfvén time)
  Density -> rho_ref (upstream mass density)
  Velocity -> V_A = B_ref / sqrt(mu0 rho_ref)

Derived from seed projects:
  - 905_pram (PRAM grid/tiling): provides structured parameter grid
  - 777_monomial_value: monomial scaling of dimensionless groups
"""

import numpy as np


# ============================================================
# SI Physical Constants
# ============================================================
MU_0 = 4.0e-7 * np.pi          # vacuum permeability [H/m]
MU_0_INV = 1.0 / MU_0
C_LIGHT = 2.998e8              # speed of light [m/s]
K_B = 1.381e-23                # Boltzmann constant [J/K]
M_PROTON = 1.673e-27           # proton mass [kg]
M_ELECTRON = 9.109e-31         # electron mass [kg]
E_CHARGE = 1.602e-19           # elementary charge [C]
EPSILON_0 = 8.854e-12          # vacuum permittivity [F/m]


class SolarCoronaPlasma:
    """
    Encapsulates the plasma parameters for a solar flare current sheet.

    The Harris equilibrium is characterized by:
        B_x(z) = B_0 tanh(z / L_cs)
        rho(z) = rho_0 / cosh^2(z / L_cs) + rho_bg
        p(z) = p_0 / cosh^2(z / L_cs) + p_bg

    where L_cs is the half-thickness of the current sheet.
    """

    def __init__(
        self,
        B0=20.0,               # upstream reconnecting field [Gauss]
        L_cs=5.0e6,            # current sheet half-thickness [m]
        n0=1.0e15,             # upstream number density [m^{-3}]
        T0=5.0e6,              # upstream temperature [K]
        eta_spitzer=1.0e-4,    # Spitzer resistivity [Ohm m] (can be anomalous)
        ion_mass=None,         # ion mass [kg], default proton
        gamma_ad=5.0 / 3.0,    # adiabatic index (5/3 for ideal monoatomic)
        d_i_fraction=0.01,     # d_i / L_cs ratio (Hall MHD onset)
    ):
        # Convert B from Gauss to Tesla (1 G = 1e-4 T)
        self.B0 = B0 * 1.0e-4
        self.L_cs = L_cs
        self.n0 = n0
        self.T0 = T0
        self.eta_spitzer = eta_spitzer
        self.gamma_ad = gamma_ad
        self.d_i_fraction = d_i_fraction

        # Derived ion mass
        self.m_i = ion_mass if ion_mass is not None else M_PROTON

        # Upstream mass density (fully ionized hydrogen)
        self.rho0 = n0 * self.m_i

        # Thermal pressure p = n k_B T (electron + ion contribution ~ 2 n k_B T)
        self.p0 = 2.0 * n0 * K_B * T0

        # Alfvén speed: V_A = B0 / sqrt(mu0 * rho0)
        self.V_A = self.B0 / np.sqrt(MU_0 * self.rho0)

        # Sound speed: c_s = sqrt(gamma * p / rho)
        self.c_s = np.sqrt(gamma_ad * self.p0 / self.rho0)

        # Plasma beta = 2 mu0 p / B^2
        self.beta = 2.0 * MU_0 * self.p0 / (self.B0 ** 2)

        # Lundquist number: S = mu0 * L * V_A / eta
        self.lundquist = MU_0 * L_cs * self.V_A / eta_spitzer

        # Magnetic Reynolds number: Rm = S
        self.Rm = self.lundquist

        # Lundquist-based Sweet-Parker reconnection rate: R_SP ~ S^{-1/2}
        self.reconnection_rate_sp = self.lundquist ** (-0.5)

        # Petschek rate (logarithmic correction)
        lnS = np.log(max(self.lundquist, 1.0))
        self.reconnection_rate_pet = np.pi / (8.0 * lnS) if lnS > 0 else 0.0

        # Alfvén time: tau_A = L_cs / V_A
        self.tau_A = L_cs / self.V_A

        # Ion inertial length: d_i = c / omega_pi
        omega_pi = np.sqrt(n0 * E_CHARGE ** 2 / (EPSILON_0 * self.m_i))
        self.d_i = C_LIGHT / omega_pi
        self.d_i_normalized = self.d_i / L_cs

        # Ion cyclotron frequency and gyroradius
        omega_ci = E_CHARGE * self.B0 / self.m_i
        self.omega_ci = omega_ci
        self.rho_s = self.c_s / omega_ci if omega_ci > 0 else 0.0

        # Dimensionless resistivity (normalized)
        self.eta_normalized = eta_spitzer / (MU_0 * self.V_A * L_cs)

        # Hall parameter (d_i / L_cs)
        self.hall_param = self.d_i_normalized

        # Guide field (default zero for anti-parallel reconnection)
        self.B_guide = 0.0

    def harris_B(self, z_normalized):
        """
        Harris current sheet magnetic field profile:
            B_x(z) = B0 * tanh(z / L_cs)
        In normalized units: B_x(z_n) = tanh(z_n)
        """
        return np.tanh(z_normalized)

    def harris_J(self, z_normalized):
        """
        Current density from Ampere's law (normalized):
            J_y(z) = (1/mu0) dB_x/dz
            J_y_norm = sech^2(z_n)
        """
        sech = 1.0 / np.cosh(z_normalized)
        return sech ** 2

    def harris_rho(self, z_normalized, rho_bg_frac=0.2):
        """
        Harris density profile:
            rho(z) = rho0 / cosh^2(z/L) + rho_bg
        Normalized: rho_n(z_n) = 1/cosh^2(z_n) + rho_bg_frac
        """
        sech = 1.0 / np.cosh(z_normalized)
        return sech ** 2 + rho_bg_frac

    def harris_pressure(self, z_normalized, p_bg_frac=0.2):
        """
        Pressure profile in pressure balance:
            p_total + B^2/(2 mu0) = const
            p(z) = p0 / cosh^2(z/L) + p_bg
        """
        sech = 1.0 / np.cosh(z_normalized)
        return sech ** 2 + p_bg_frac

    def get_parameter_vector(self):
        """
        Return the key dimensionless parameters as a vector.
        Used for optimization (maps to 907_praxis).
        """
        return np.array([
            self.beta,
            self.eta_normalized,
            self.d_i_normalized,
            self.gamma_ad,
            self.reconnection_rate_sp,
        ])

    def summary(self):
        """Print a physical summary of the plasma configuration."""
        lines = [
            "=" * 60,
            "SOLAR CORONA PLASMA PARAMETERS",
            "=" * 60,
            f"  B0 (upstream)         = {self.B0 * 1e4:.2f} G",
            f"  L_cs (half-thickness) = {self.L_cs:.2e} m",
            f"  n0 (density)          = {self.n0:.2e} m^-3",
            f"  T0 (temperature)      = {self.T0:.2e} K",
            f"  V_A (Alfvén speed)    = {self.V_A:.2e} m/s",
            f"  c_s (sound speed)     = {self.c_s:.2e} m/s",
            f"  beta (plasma)         = {self.beta:.4f}",
            f"  S (Lundquist)         = {self.lundquist:.2e}",
            f"  tau_A (Alfvén time)   = {self.tau_A:.4e} s",
            f"  eta (normalized)      = {self.eta_normalized:.2e}",
            f"  d_i/L_cs (Hall)       = {self.d_i_normalized:.4e}",
            f"  R_SP (Sweet-Parker)   = {self.reconnection_rate_sp:.4e}",
            f"  R_Pet (Petschek)      = {self.reconnection_rate_pet:.4e}",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_parameter_sweep(n_eta=5, n_beta=3):
    """
    Build a 2D parameter sweep in (eta, beta) space.
    Maps the PRAM grid/tiling idea (905_pram) to structured parameter grids.

    Returns:
        list of (eta_norm, beta, lundquist) tuples
    """
    eta_vals = np.logspace(-5, -2, n_eta)
    beta_vals = np.linspace(0.01, 0.5, n_beta)

    grid = []
    for eta in eta_vals:
        for beta in beta_vals:
            # From beta and eta, compute Lundquist via scaling
            # S ~ 1 / eta_norm for normalized system
            S = 1.0 / max(eta, 1e-15)
            grid.append((eta, beta, S))
    return grid
