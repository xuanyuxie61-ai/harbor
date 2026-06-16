# -*- coding: utf-8 -*-
"""
physical_constants.py
=====================
Fundamental physical constants and material parameters for the
MoS2/WSe2 二维异质结能带对齐计算.

All quantities in SI units unless otherwise noted.
Energy is also frequently expressed in electron-volts (eV).

Constants from CODATA 2018 recommended values.
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
#  Fundamental physical constants  (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34          # reduced Planck constant  [J·s]
HBAR_EV = HBAR / 1.602176634e-19  # reduced Planck constant  [eV·s]
Q_E = 1.602176634e-19           # elementary charge         [C]
M0 = 9.1093837015e-31           # electron rest mass        [kg]
EPS0 = 8.8541878128e-12         # vacuum permittivity       [F/m]
KB = 1.380649e-23               # Boltzmann constant        [J/K]
KB_EV = KB / Q_E                # Boltzmann constant        [eV/K]
A0_BOHR = 5.29177210903e-11     # Bohr radius               [m]
HARTREE = 27.211386245988       # Hartree energy            [eV]
RYDBERG = 13.605693122994       # Rydberg energy            [eV]

# ---------------------------------------------------------------------------
#  2D material parameters for transition-metal dichalcogenides (TMDCs)
#  Values drawn from first-principles literature (e.g. Kormányos et al.,
#  2D Mater. 2, 022001 (2015); Roldan et al., 2D Mater. 4, 023002 (2017)).
# ---------------------------------------------------------------------------
class MaterialParameters:
    """
    Parameters for monolayer TMDCs. Effective masses in units of m0.
    Band edges referenced to vacuum.

    Conduction-band minimum (CBM) and valence-band maximum (VBM) are
    located at the K (and K') points of the hexagonal Brillouin zone.

    For MoS2:  Eg ≈ 1.90 eV (monolayer, direct gap at K).
    For WSe2:  Eg ≈ 1.65 eV (monolayer, direct gap at K).

    The electron/hole effective masses follow the massive Dirac model:
        H = ℏ^2/(2 m*) (k_x^2 + k_y^2) σ_z + Δ/2 σ_z + α λ_SO τ σ_z s_z (S ± i η)
    """

    _registry = {
        "MoS2": {
            "a_latt": 3.16e-10,           # lattice constant [m]
            "d_layer": 6.15e-10,          # monolayer thickness (X-M-X) [m]
            "eps_in": 6.6,                # in-plane dielectric (2D effective ~ d_layer * eps_3D)
            "eps_out": 4.5,               # out-of-plane dielectric
            "eps_2d": 7.3,                # effective 2D dielectric from RPA
            "m_e": 0.47,                  # electron eff. mass at K  [m0]
            "m_h": 0.56,                  # heavy-hole eff. mass at K [m0]
            "m_so": 0.22,                 # spin-orbit split-off band eff. mass
            "Eg": 1.90,                   # quasiparticle gap [eV]
            "delta_so_c": 0.075,          # SO splitting in CB [eV]
            "delta_so_v": 0.145,          # SO splitting in VB [eV]
            "chi_e": 4.00,                # electron affinity (Anderson) [eV]
            "Ip": 5.90,                   # ionization potential [eV]
            "lambda_c": 0.037,            # CB SO coupling [eV]
            "lambda_v": 0.110,            # VB SO coupling [eV]
            "a_def": -3.1,                # deformation potential CB [eV]
            "a_def_v": -1.5,              # deformation potential VB [eV]
            "mu_sub": 1.0,                # out-of-plane dipole correction [eV]
        },
        "WSe2": {
            "a_latt": 3.28e-10,
            "d_layer": 6.50e-10,
            "eps_in": 12.0,
            "eps_out": 7.5,
            "eps_2d": 11.6,
            "m_e": 0.36,
            "m_h": 0.40,
            "m_so": 0.20,
            "Eg": 1.65,
            "delta_so_c": 0.410,
            "delta_so_v": 0.460,
            "chi_e": 4.00,
            "Ip": 5.65,
            "lambda_c": 0.290,
            "lambda_v": 0.460,
            "a_def": -3.8,
            "a_def_v": -1.8,
            "mu_sub": 0.85,
        },
    }

    def __init__(self, name):
        if name not in self._registry:
            raise ValueError(
                f"Unknown material '{name}'. "
                f"Available: {list(self._registry.keys())}"
            )
        self.name = name
        for k, v in self._registry[name].items():
            setattr(self, k, v)

    # ----- derived quantities -----
    def eff_mass_kg(self, carrier="electron"):
        """Return effective mass in kg."""
        key = "m_e" if carrier == "electron" else "m_h"
        return getattr(self, key) * M0

    def vacuum_level(self):
        """Vacuum level relative to VBM."""
        return self.Ip

    def cbm(self):
        """Conduction band minimum energy [eV] (referenced to 0 at VBM)."""
        return self.Eg

    def vbm(self):
        return 0.0

    def kdos_mass_reduced(self):
        """Reduced mass for density of states: 1/m_r = 1/m_e + 1/m_h."""
        return self.m_e * self.m_h / (self.m_e + self.m_h)

    def exciton_binding_energy(self):
        """Rydberg-like exciton binding energy (1s state), rough estimate:
           E_b ≈ μ e^4 / (2 ℏ^2 (4 π ε)^2) in 2D with Keldysh screening.
        """
        mu_kg = self.kdos_mass_reduced() * M0
        eps_eff = self.eps_2d * EPS0
        # 2D-hydrogenic with Keldysh cutoff r0 ≈ a_latt/2
        r0 = 0.5 * self.a_latt
        # Use simplified 3D-like formula (upper bound)
        E_b = (mu_kg * Q_E**4) / (2.0 * HBAR**2 * (4.0 * math.pi * eps_eff) ** 2)
        return E_b / Q_E   # convert J -> eV

    def strain_shift(self, strain, carrier="electron"):
        """Hydrostatic-strain-induced shift of band edge:
            ΔE_cb = a_def * (ε_xx + ε_yy + C_cb/C_vb * ε_zz)
        Here we assume biaxial in-plane strain ε_xx = ε_yy = strain,
        and Poisson ratio ν ≈ 0.25 gives ε_zz = -2ν/(1-ν) * strain.
        """
        nu_poisson = 0.25
        ratio = -2.0 * nu_poisson / (1.0 - nu_poisson)
        if carrier == "electron":
            return self.a_def * strain * (2.0 + ratio)
        return self.a_def_v * strain * (2.0 + ratio)


# ---------------------------------------------------------------------------
#  Heterojunction interface parameters
# ---------------------------------------------------------------------------
def anderson_band_alignment(mat_a, mat_b):
    """
    Anderson's rule (electron affinity rule) for type-II heterojunction:
        ΔEc = χ_B - χ_A        (CB offset)
        ΔEv = (Ip_A + Eg_A) - (Ip_B + Eg_B)   (VB offset)
    With interface dipole correction δ_int:
        ΔEc_eff = ΔEc + δ_int
    """
    delta_ec = mat_b.chi_e - mat_a.chi_e
    delta_ev = (mat_a.Ip + mat_a.Eg) - (mat_b.Ip + mat_b.Eg)
    # Empirical interface dipole ~ 0.1 eV for MoS2/WSe2
    delta_int = 0.10
    return {
        "delta_ec": delta_ec,
        "delta_ev": delta_ev,
        "delta_ec_eff": delta_ec + delta_int,
        "delta_ev_eff": delta_ev - delta_int,
        "interface_dipole": delta_int,
    }


def keldysh_screening_length(mat):
    """In-plane screening length r0 in the Keldysh potential:
        V(r) = -(π e^2)/(2 ε_2d r0) [H0(r/r0) - Y0(r/r0)]
    Here r0 = 2π χ_2d / ε_2d  (χ_2d = in-plane polarizability).
    """
    chi_2d = (mat.eps_in - 1.0) * mat.d_layer / (4.0 * math.pi)
    return 2.0 * math.pi * chi_2d / mat.eps_2d
