"""
dusty_plasma_physics_constants.py
==================================
Fundamental physical constants and dimensionless parameters for
dusty plasma crystal simulations.

Physical context:
    In a dusty plasma, micron-sized solid grains are suspended in a
    low-temperature plasma. The grains acquire a large negative charge
    Q_d = -Z_d * e through electron/ion collection (OML theory).
    The interaction between grains is modeled by the Yukawa (screened
    Coulomb) potential:
        phi(r) = (Q_d^2 / (4*pi*eps0*r)) * exp(-r/lambda_D)
    where lambda_D is the plasma Debye length.

    The coupling parameter Gamma determines crystal formation:
        Gamma = (Q_d^2 / (4*pi*eps0*a)) * exp(-a/lambda_D) / (k_B * T_d)
    where a is the Wigner-Seitz radius and T_d is the dust temperature.
    Crystallization occurs for Gamma > Gamma_c ~ 137 (2D hexagonal lattice).

References:
    - Shukla & Mamun, "Introduction to Dusty Plasma Physics" (IoP, 2002)
    - Morfill et al., "Crystallization of dusty plasmas", Phys. Rev. Lett. 83, 1598 (1999)
    - Hamaguchi & Farouki, "Thermodynamic properties of strongly coupled dusty plasmas", Phys. Rev. E 56, 4671 (1997)
"""

import numpy as np
from typing import Dict, Any

# ============================================================
# SI Fundamental Constants (CODATA 2018)
# ============================================================
ELEMENTARY_CHARGE = 1.602176634e-19        # e  [C]
ELECTRON_MASS = 9.1093837015e-31           # m_e [kg]
PROTON_MASS = 1.67262192369e-27            # m_p [kg]
BOLTZMANN_CONSTANT = 1.380649e-23          # k_B [J/K]
VACUUM_PERMITTIVITY = 8.8541878128e-12     # eps_0 [F/m]
VACUUM_PERMEABILITY = 1.25663706212e-6     # mu_0 [H/m]
PLANCK_CONSTANT = 6.62607015e-34           # h [J*s]
SPEED_OF_LIGHT = 2.99792458e8              # c [m/s]
AVOGADRO_NUMBER = 6.02214076e23            # N_A [1/mol]
PI = np.pi                                 # pi


class DustyPlasmaRegime:
    """
    Defines a specific dusty plasma experimental regime.

    Typical parameter ranges for laboratory RF discharge dusty plasmas:
        - Electron density:     n_e ~ 1e14 - 1e16 m^-3
        - Electron temperature: T_e ~ 1 - 5 eV
        - Ion density:          n_i ~ 1e14 - 1e16 m^-3
        - Ion temperature:      T_i ~ 0.025 - 0.05 eV
        - Dust radius:          r_d ~ 0.5 - 5 um
        - Dust charge number:   Z_d ~ 1e3 - 1e5
        - Dust density:         n_d ~ 1e8 - 1e12 m^-3
        - Screening parameter:  kappa = a/lambda_D ~ 0.5 - 5.0
    """

    def __init__(
        self,
        n_e: float = 1.0e15,
        T_e_eV: float = 2.5,
        T_i_eV: float = 0.03,
        T_d_eV: float = 0.025,
        r_d_um: float = 2.0,
        n_d: float = 5.0e10,
        dust_density_kg_m3: float = 2500.0,
        ion_mass_amu: float = 40.0,  # Argon
    ):
        """
        Parameters
        ----------
        n_e : electron number density [m^-3]
        T_e_eV : electron temperature [eV]
        T_i_eV : ion temperature [eV]
        T_d_eV : dust grain temperature (kinetic) [eV]
        r_d_um : dust grain radius [micrometers]
        n_d : dust number density [m^-3]
        dust_density_kg_m3 : mass density of dust material [kg/m^3]
        ion_mass_amu : ion mass [atomic mass units]
        """
        # --- Plasma parameters ---
        self.n_e = n_e
        self.T_e_eV = T_e_eV
        self.T_e_J = T_e_eV * ELEMENTARY_CHARGE
        self.T_i_eV = T_i_eV
        self.T_i_J = T_i_eV * ELEMENTARY_CHARGE
        self.T_d_eV = T_d_eV
        self.T_d_J = T_d_eV * ELEMENTARY_CHARGE

        # --- Ion parameters ---
        self.m_i = ion_mass_amu * PROTON_MASS
        self.n_i = n_e  # Quasi-neutrality (zeroth order)

        # --- Dust grain parameters ---
        self.r_d = r_d_um * 1.0e-6  # Convert to meters
        self.n_d = n_d
        self.rho_d = dust_density_kg_m3
        self.m_d = (4.0 / 3.0) * PI * self.r_d**3 * self.rho_d

        # --- Debye length ---
        # Electron Debye length: lambda_De = sqrt(eps0 * T_e / (n_e * e^2))
        self.lambda_De = np.sqrt(
            VACUUM_PERMITTIVITY * self.T_e_J / (self.n_e * ELEMENTARY_CHARGE**2)
        )
        # Ion Debye length: lambda_Di = sqrt(eps0 * T_i / (n_i * e^2))
        self.lambda_Di = np.sqrt(
            VACUUM_PERMITTIVITY * self.T_i_J / (self.n_i * ELEMENTARY_CHARGE**2)
        )
        # Total Debye length: 1/lambda_D^2 = 1/lambda_De^2 + 1/lambda_Di^2
        self.lambda_D = 1.0 / np.sqrt(
            1.0 / self.lambda_De**2 + 1.0 / self.lambda_Di**2
        )

        # --- Dust charge from OML (Orbital Motion Limit) theory ---
        # The floating potential satisfies:
        #   exp(e*phi_f/(k_B*T_e)) = sqrt(m_e/m_i) * sqrt(1 - 2*e*phi_f/(k_B*T_i))
        # For T_e >> T_i, the zeta = e*|phi_f|/(k_B*T_e) satisfies:
        #   exp(-zeta) ~ sqrt(m_e/m_i * T_e/T_i) * (1 + zeta*T_e/T_i)
        # Simplified estimate: Z_d ~ 4*pi*eps0*r_d*|phi_f|/e^2
        mass_ratio = ELECTRON_MASS / self.m_i
        temp_ratio = self.T_e_J / self.T_i_J
        # Iterative solution for floating potential parameter zeta
        zeta = self._solve_floating_potential(mass_ratio, temp_ratio)
        self.phi_floating_V = zeta * self.T_e_eV  # floating potential in Volts
        self.Z_d = int(
            max(1.0, 4.0 * PI * VACUUM_PERMITTIVITY * self.r_d * abs(self.phi_floating_V) / ELEMENTARY_CHARGE)
        )
        self.Q_d = -self.Z_d * ELEMENTARY_CHARGE  # dust charge [C]

        # --- Lattice parameters (2D hexagonal) ---
        # Wigner-Seitz radius: pi*a^2*n_d = 1  =>  a = 1/sqrt(pi*n_d)
        self.a_ws = 1.0 / np.sqrt(PI * self.n_d)  # [m]

        # --- Coupling parameter ---
        # Gamma = (Q_d^2/(4*pi*eps0*a_ws)) * exp(-kappa) / (k_B*T_d)
        self.kappa = self.a_ws / self.lambda_D  # screening parameter
        self.Gamma = (
            (self.Q_d**2 / (4.0 * PI * VACUUM_PERMITTIVITY * self.a_ws))
            * np.exp(-self.kappa)
            / self.T_d_J
        )

        # --- Dust plasma frequency ---
        # omega_pd = sqrt(n_d * Q_d^2 / (eps0 * m_d))
        self.omega_pd = np.sqrt(
            self.n_d * self.Q_d**2 / (VACUUM_PERMITTIVITY * self.m_d)
        )

        # --- Dust acoustic speed ---
        # C_DA = sqrt(Z_d * k_B * T_e / m_d)  (for T_e >> T_i)
        self.C_DA = np.sqrt(self.Z_d * self.T_e_J / self.m_d)

        # --- Mach number for typical grain velocity ---
        self.v_thermal_d = np.sqrt(self.T_d_J / self.m_d)
        self.M_DA = self.v_thermal_d / self.C_DA

        # --- Grid resolution ---
        self.dx_characteristic = self.a_ws / 10.0  # ~10 cells per lattice spacing

    def _solve_floating_potential(
        self, mass_ratio: float, temp_ratio: float, tol: float = 1e-12, max_iter: int = 200
    ) -> float:
        """
        Solve the OML floating potential equation via Newton-Raphson:
            f(zeta) = exp(-zeta) - sqrt(mu/s_tau) * (1 + zeta/s_tau) = 0
        where mu = m_e/m_i, s_tau = T_i/T_e, zeta = e|phi_f|/(k_B*T_e).

        Parameters
        ----------
        mass_ratio : m_e/m_i
        temp_ratio : T_e/T_i (inverse of s_tau)
        tol : convergence tolerance
        max_iter : maximum iterations

        Returns
        -------
        zeta : dimensionless floating potential
        """
        s_tau = 1.0 / temp_ratio
        prefactor = np.sqrt(mass_ratio / s_tau)
        # Initial guess
        zeta = 2.0
        for _ in range(max_iter):
            f_val = np.exp(-zeta) - prefactor * (1.0 + zeta / s_tau)
            f_prime = -np.exp(-zeta) - prefactor / s_tau
            dzeta = -f_val / f_prime
            zeta += dzeta
            # Enforce positivity
            zeta = max(zeta, 0.1)
            if abs(dzeta) < tol * abs(zeta):
                break
        return zeta

    def regime_summary(self) -> Dict[str, Any]:
        """Return a dictionary summarizing the regime parameters."""
        return {
            "n_e [m^-3]": f"{self.n_e:.3e}",
            "T_e [eV]": f"{self.T_e_eV:.3f}",
            "T_i [eV]": f"{self.T_i_eV:.4f}",
            "lambda_D [m]": f"{self.lambda_D:.3e}",
            "lambda_De [m]": f"{self.lambda_De:.3e}",
            "r_d [um]": f"{self.r_d * 1e6:.2f}",
            "m_d [kg]": f"{self.m_d:.3e}",
            "Z_d": f"{self.Z_d}",
            "Q_d [C]": f"{self.Q_d:.3e}",
            "a_ws [m]": f"{self.a_ws:.3e}",
            "kappa": f"{self.kappa:.4f}",
            "Gamma": f"{self.Gamma:.2f}",
            "omega_pd [rad/s]": f"{self.omega_pd:.3e}",
            "C_DA [m/s]": f"{self.C_DA:.3e}",
            "M_DA": f"{self.M_DA:.4e}",
            "Crystal regime": (
                "STRONG COUPLING (crystal)" if self.Gamma > 137
                else "WEAK COUPLING (liquid/gas)" if self.Gamma < 10
                else "INTERMEDIATE"
            ),
        }

    def validate_regime(self) -> bool:
        """
        Check physical validity of the regime:
            1. Quasi-neutrality: n_i ~ n_e + Z_d*n_d
            2. Dust is a trace component: Z_d*n_d << n_e
            3. Debye length >> grain radius: lambda_D >> r_d
            4. Inter-grain distance >> grain radius: a_ws >> r_d
            5. Coupling parameter is physical: Gamma > 0
        """
        checks = []
        # Quasi-neutrality check
        ne_required = self.n_i
        ne_actual = self.n_e + self.Z_d * self.n_d
        checks.append(abs(ne_required - ne_actual) / ne_required < 0.05)

        # Trace dust
        checks.append(self.Z_d * self.n_d < 0.1 * self.n_e)

        # Debye length >> grain radius
        checks.append(self.lambda_D > 10.0 * self.r_d)

        # Inter-grain distance >> grain radius
        checks.append(self.a_ws > 5.0 * self.r_d)

        # Positive coupling
        checks.append(self.Gamma > 0)

        return all(checks)
