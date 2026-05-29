"""
electrochemistry.py
================================================================================
Pseudo-2D Doyle-Fuller-Newman (DFN) electrochemical model for lithium-ion
batteries, including solid-phase diffusion, electrolyte transport, charge
conservation, and Butler-Volmer kinetics.

Injects core algorithms from:
  - 979_r8gb          (banded matrix solver for the coupled linear systems)
  - 999_r8sto         (Toeplitz solver for radial diffusion operator)
  - 470_gl_fast_rule  (Gauss-Legendre quadrature for reaction rate integration)
  - 209_conte_deboor  (spline interpolation for OCP, root-finding for BV inverse)

Scientific role:
  This is the core electrochemical engine.  It solves the 1D macroscopic
  cell equations coupled with microscopic radial solid diffusion:

    Solid diffusion (spherical particle, radial coordinate r):
      dC_s/dt = (1/r^2) * d/dr( D_s(T) * r^2 * dC_s/dr )
      BC: dC_s/dr|_{r=0} = 0,   -D_s * dC_s/dr|_{r=R} = j_BV / (a_s * F)

    Electrolyte concentration (macroscopic x):
      epsilon_e * dC_e/dt = d/dx( D_e_eff(T) * dC_e/dx )
                          + (1-t_+) * a_s * j_BV / F

    Charge conservation (solid potential phi_s):
      d/dx( sigma_s_eff * dphi_s/dx ) = a_s * F * j_BV

    Charge conservation (electrolyte potential phi_e):
      d/dx( kappa_eff(T) * dphi_e/dx ) + d/dx( kappa_D_eff * d(ln C_e)/dx )
                                        = -a_s * F * j_BV

    Butler-Volmer kinetics:
      j_BV = j_0 * [ exp( alpha_a * F * eta / (R*T) )
                   - exp( -alpha_c * F * eta / (R*T) ) ]
      eta = phi_s - phi_e - U_ocp(C_s_surf) - I*R_film

  Temperature dependence of all transport properties is handled via cubic
  splines (numerical_toolkit.TemperatureDependentProperty).
================================================================================
"""

import numpy as np
from typing import Tuple
from banded_linear_algebra import BandedMatrix, SymmetricToeplitzSolver
from numerical_toolkit import TemperatureDependentProperty, muller_root
from quadrature_special import gauss_legendre_nodes_weights


# ==============================================================================
# Physical constants
# ==============================================================================
FARADAY = 96485.33212      # C/mol
R_GAS = 8.314462618        # J/(mol*K)


# ==============================================================================
# Temperature-dependent material properties (spline-interpolated)
# ==============================================================================

def make_default_diffusivity_spline() -> TemperatureDependentProperty:
    """Solid diffusivity D_s [m^2/s] vs T [K] for LiCoO2 / graphite."""
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    D = np.array([1.0e-15, 3.0e-15, 1.0e-14, 2.5e-14, 5.0e-14, 8.0e-14])
    return TemperatureDependentProperty(T, D)


def make_default_kappa_electrolyte_spline() -> TemperatureDependentProperty:
    """Electrolyte conductivity kappa [S/m] vs T [K]."""
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    k = np.array([0.2, 0.4, 0.8, 1.2, 1.8, 2.4])
    return TemperatureDependentProperty(T, k)


def make_default_diffusivity_electrolyte_spline() -> TemperatureDependentProperty:
    """Electrolyte diffusivity D_e [m^2/s] vs T [K]."""
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    D = np.array([1.0e-11, 2.0e-11, 4.0e-11, 7.0e-11, 1.1e-10, 1.5e-10])
    return TemperatureDependentProperty(T, D)


# ==============================================================================
# Open-circuit potential (OCP) models
# ==============================================================================

def ocp_graphite(sto: float) -> float:
    """
    Open-circuit potential for graphite negative electrode [V vs Li/Li+].
    Simplified empirical fit giving ~0.1 V at sto=0.5, ~0.05 V at sto=0.8.
    """
    sto = np.clip(sto, 0.01, 0.99)
    return 0.05 + 0.2 * (1.0 - sto) + 0.05 * np.sin(3.0 * np.pi * sto)


def ocp_lco(sto: float) -> float:
    """
    Open-circuit potential for LiCoO2 positive electrode [V vs Li/Li+].
    Simplified empirical fit giving ~3.9 V at sto=0.5, ~4.1 V at sto=0.3.
    """
    sto = np.clip(sto, 0.1, 0.99)
    return 3.8 + 0.4 * (1.0 - sto) + 0.1 * np.sin(2.0 * np.pi * sto)


def d_ocp_dT_graphite(sto: float) -> float:
    """Entropy coefficient dU_ocp/dT for graphite [V/K]."""
    return -0.0002 * (1.0 - sto)


def d_ocp_dT_lco(sto: float) -> float:
    """Entropy coefficient dU_ocp/dT for LCO [V/K]."""
    return 0.0001 * sto


# ==============================================================================
# Butler-Volmer kinetics
# ==============================================================================

def butler_volmer_flux(eta: float, j0: float, T: float,
                       alpha_a: float = 0.5, alpha_c: float = 0.5) -> float:
    """
    Butler-Volmer reaction flux [A/m^2].
    j = j0 * [ exp(alpha_a * F * eta / (R*T)) - exp(-alpha_c * F * eta / (R*T)) ]
    """
    RT_F = R_GAS * T / FARADAY
    exp_a = np.exp(alpha_a * eta / RT_F)
    exp_c = np.exp(-alpha_c * eta / RT_F)
    return j0 * (exp_a - exp_c)


def butler_volmer_inverse(j_target: float, j0: float, T: float,
                          alpha_a: float = 0.5, alpha_c: float = 0.5) -> float:
    """
    Solve j_target = j0*(exp(a*eta) - exp(-c*eta)) for eta using Muller's method.
    """
    def residual(eta):
        return butler_volmer_flux(eta, j0, T, alpha_a, alpha_c) - j_target
    # Initial bracket
    eta = muller_root(residual, -0.5, 0.0, 0.5, tol=1e-12, max_iter=30)
    return eta


def exchange_current_density(C_e: float, C_s_surf: float, C_s_max: float,
                             k_ref: float, T: float, E_act: float = 5000.0) -> float:
    """
    Arrhenius temperature-dependent exchange current density:
    j0 = k_ref * sqrt(C_e * C_s_surf * (C_s_max - C_s_surf))
          * exp( -E_act / R * (1/T - 1/T_ref) )
    """
    T_ref = 298.15
    conc_term = np.sqrt(max(C_e, 1e-6) * max(C_s_surf, 1e-6)
                        * max(C_s_max - C_s_surf, 1e-6))
    arrhenius = np.exp(-E_act / R_GAS * (1.0 / T - 1.0 / T_ref))
    return k_ref * conc_term * arrhenius


# ==============================================================================
# Solid-phase diffusion (radial, finite difference)
# ==============================================================================

class SolidDiffusionSolver:
    """
    Finite-difference solver for radial diffusion in spherical particles.
    Uses a banded matrix for the implicit diffusion operator.
    """

    def __init__(self, R: float, n_r: int, D_s: float):
        self.R = R
        self.n_r = n_r
        self.D_s = D_s
        self.dr = R / n_r
        self.r = np.linspace(0.5 * self.dr, R - 0.5 * self.dr, n_r)
        # C_i at cell centers
        self.C = np.full(n_r, 0.5 * 30555.0)  # initial concentration (mol/m^3)

    def _build_diffusion_matrix(self, dt: float) -> BandedMatrix:
        """
        Build implicit diffusion operator for spherical coordinates:
        (1/r^2) d/dr(r^2 D dC/dr).
        Using cell-centered finite differences with banded storage.
        """
        n = self.n_r
        D = self.D_s
        dr = self.dr
        bm = BandedMatrix(n, 1, 1)
        for i in range(n):
            r_plus = self.r[i] + 0.5 * dr
            r_minus = max(self.r[i] - 0.5 * dr, 1e-12)
            r_c = self.r[i]
            # Fluxes at half faces
            if i == 0:
                # No-flux at center: j_{-1/2} = 0
                a_center = 1.0 + dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                a_right = -dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i, a_center)
                bm.set_entry(i, i + 1, a_right)
            elif i == n - 1:
                # Surface boundary handled via RHS modification
                a_left = -dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                a_center = 1.0 + dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i - 1, a_left)
                bm.set_entry(i, i, a_center)
            else:
                a_left = -dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                a_center = 1.0 + dt * D * (r_plus ** 2 + r_minus ** 2) / (r_c ** 2 * dr ** 2)
                a_right = -dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i - 1, a_left)
                bm.set_entry(i, i, a_center)
                bm.set_entry(i, i + 1, a_right)
        return bm

    def step(self, dt: float, j_flux: float, C_s_max: float = 30555.0) -> np.ndarray:
        """
        Advance solid diffusion by one time step with surface flux boundary
        condition: -D_s * dC/dr = j_flux / F.
        """
        bm = self._build_diffusion_matrix(dt)
        rhs = self.C.copy()
        # Surface BC: add flux source to last cell
        # Finite volume: dC_n/dt += (R^2 / r_c^2) * (j_flux/F) / dr
        n = self.n_r
        r_c = self.r[-1]
        R = self.R
        flux_source = dt * (R ** 2 / (r_c ** 2 + 1e-18)) * (j_flux / FARADAY) / self.dr
        rhs[-1] -= flux_source
        info = bm.plu_factor()
        if info != 0:
            # Fallback: use numpy dense solve
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(max(0, i - 1), min(n, i + 2)):
                    A[i, j] = bm.get_entry(i, j)
            self.C = np.linalg.solve(A, rhs)
        else:
            self.C = bm.solve(rhs)
        # Enforce physical bounds
        self.C = np.clip(self.C, 0.0, C_s_max)
        return self.C.copy()

    def surface_concentration(self) -> float:
        return float(self.C[-1])

    def average_concentration(self) -> float:
        # Volume-weighted average: integrate C(r) * r^2 dr / (R^3/3)
        weights = self.r ** 2
        return float(np.sum(self.C * weights) / np.sum(weights))


# ==============================================================================
# 1D macroscopic electrochemical solver (finite volume)
# ==============================================================================

class MacroscopicElectrochemicalSolver:
    """
    1D finite-volume solver for electrolyte concentration and potentials
    across the cell sandwich: neg electrode | separator | pos electrode.
    """

    def __init__(self,
                 L_neg: float = 50e-6,
                 L_sep: float = 25e-6,
                 L_pos: float = 50e-6,
                 n_neg: int = 20,
                 n_sep: int = 10,
                 n_pos: int = 20,
                 T0: float = 298.15):
        self.L_neg = L_neg
        self.L_sep = L_sep
        self.L_pos = L_pos
        self.n_neg = n_neg
        self.n_sep = n_sep
        self.n_pos = n_pos
        self.n_total = n_neg + n_sep + n_pos
        self.T = T0

        # Build x-grid (cell centers)
        x_neg = np.linspace(0.5 * L_neg / n_neg, L_neg - 0.5 * L_neg / n_neg, n_neg)
        x_sep = np.linspace(L_neg + 0.5 * L_sep / n_sep, L_neg + L_sep - 0.5 * L_sep / n_sep, n_sep)
        x_pos = np.linspace(L_neg + L_sep + 0.5 * L_pos / n_pos, L_neg + L_sep + L_pos - 0.5 * L_pos / n_pos, n_pos)
        self.x = np.concatenate([x_neg, x_sep, x_pos])

        # Region indices
        self.regions = np.array(["neg"] * n_neg + ["sep"] * n_sep + ["pos"] * n_pos)

        # Parameters (default for LCO/graphite)
        self.epsilon_e = np.where(self.regions == "neg", 0.385,
                         np.where(self.regions == "sep", 0.724, 0.485))
        self.sigma_s_eff = np.where(self.regions == "neg", 100.0,
                           np.where(self.regions == "sep", 0.0, 10.0))
        self.a_s = np.where(self.regions == "neg", 885000.0,
                   np.where(self.regions == "sep", 0.0, 173000.0))
        self.C_s_max = np.where(self.regions == "neg", 30555.0, 51554.0)

        # Diffusivity splines
        self.D_s_spline = make_default_diffusivity_spline()
        self.kappa_spline = make_default_kappa_electrolyte_spline()
        self.D_e_spline = make_default_diffusivity_electrolyte_spline()

        # Solid diffusion solvers (one per x-location)
        R_neg = 2e-6
        R_pos = 2e-6
        D_s_neg = self.D_s_spline.eval(self.T)
        D_s_pos = self.D_s_spline.eval(self.T)
        self.solid_neg = SolidDiffusionSolver(R_neg, 15, D_s_neg)
        self.solid_pos = SolidDiffusionSolver(R_pos, 15, D_s_pos)

        # Initialize solid concentrations at 50% SOC
        self.solid_neg.C = np.full(self.solid_neg.n_r, 0.5 * 30555.0)
        self.solid_pos.C = np.full(self.solid_pos.n_r, 0.5 * 51554.0)

        # Particle solvers per cell
        self.solid_solvers = []
        for r in self.regions:
            if r == "neg":
                s = SolidDiffusionSolver(R_neg, 10, D_s_neg)
                s.C = np.full(s.n_r, 0.5 * 30555.0)
                self.solid_solvers.append(s)
            elif r == "pos":
                s = SolidDiffusionSolver(R_pos, 10, D_s_pos)
                s.C = np.full(s.n_r, 0.5 * 51554.0)
                self.solid_solvers.append(s)
            else:
                self.solid_solvers.append(None)

        # State variables
        self.C_e = np.full(self.n_total, 1000.0)  # mol/m^3
        self.phi_s = np.zeros(self.n_total)
        self.phi_e = np.zeros(self.n_total)

    def update_temperature(self, T_new: float):
        """Update all temperature-dependent properties."""
        self.T = T_new
        D_s = self.D_s_spline.eval(T_new)
        for solver in self.solid_solvers:
            if solver is not None:
                solver.D_s = D_s
        self.solid_neg.D_s = D_s
        self.solid_pos.D_s = D_s

    def _build_electrolyte_matrix(self, dt: float) -> np.ndarray:
        """Build implicit matrix for electrolyte concentration update."""
        n = self.n_total
        A = np.zeros((n, n))
        D_e = self.D_e_spline.eval(self.T)
        dx_arr = np.diff(self.x)
        dx_arr = np.concatenate([[dx_arr[0]], dx_arr, [dx_arr[-1]]])

        for i in range(n):
            eps = self.epsilon_e[i]
            A[i, i] = eps
            if i > 0:
                dx_avg = 0.5 * (dx_arr[i] + dx_arr[i - 1])
                A[i, i] += dt * D_e / dx_avg ** 2
                A[i, i - 1] = -dt * D_e / dx_avg ** 2
            if i < n - 1:
                dx_avg = 0.5 * (dx_arr[i] + dx_arr[i + 1])
                A[i, i] += dt * D_e / dx_avg ** 2
                A[i, i + 1] = -dt * D_e / dx_avg ** 2
        return A

    def solve_electrolyte(self, dt: float, j_BV: np.ndarray, t_plus: float = 0.38) -> np.ndarray:
        """
        Solve electrolyte concentration equation for one step.
        epsilon_e * dC_e/dt = D_e * d2C_e/dx2 + (1-t_plus)*a_s*j_BV/F
        """
        A = self._build_electrolyte_matrix(dt)
        rhs = self.epsilon_e * self.C_e + dt * (1.0 - t_plus) * self.a_s * j_BV / FARADAY
        # No-flux BC at current collectors (dC_e/dx = 0)
        # Enforced naturally by omitting off-diagonal at boundaries
        self.C_e = np.linalg.solve(A + 1e-12 * np.eye(len(A)), rhs)
        self.C_e = np.clip(self.C_e, 10.0, 5000.0)
        return self.C_e.copy()

    def solve_charge_conservation(self, I_app: float) -> np.ndarray:
        """
        Solve solid and electrolyte potential fields for applied current I_app [A/m^2].
        Uses a simplified 1D finite-volume charge conservation with Butler-Volmer.
        """
        n = self.n_total
        kappa = self.kappa_spline.eval(self.T)
        phi_s = np.zeros(n)
        phi_e = np.zeros(n)
        j_BV = np.zeros(n)

        # Solid potential: linear in each electrode with Ohmic drop
        # Neg electrode: phi_s starts at 0 (reference) and drops
        neg_mask = self.regions == "neg"
        pos_mask = self.regions == "pos"
        if np.any(neg_mask):
            x_neg = self.x[neg_mask]
            phi_s[neg_mask] = -I_app * (x_neg - x_neg[0]) / max(self.sigma_s_eff[neg_mask][0], 1e-6)
        if np.any(pos_mask):
            x_pos = self.x[pos_mask]
            phi_s[pos_mask] = I_app * (x_pos - x_pos[-1]) / max(self.sigma_s_eff[pos_mask][0], 1e-6)

        # Electrolyte potential: linear drop across entire cell
        phi_e = -I_app * self.x / max(kappa, 1e-6)

        # Compute Butler-Volmer flux consistent with applied current
        # For uniform current distribution: j_BV = I_app / (a_s * L_elec)
        for i in range(n):
            if self.regions[i] == "sep":
                continue
            cs_surf = self.solid_solvers[i].surface_concentration() if self.solid_solvers[i] is not None else self.C_s_max[i] * 0.5
            ce_local = self.C_e[i]
            j0 = exchange_current_density(ce_local, cs_surf, self.C_s_max[i], 10.0, self.T)
            sto = cs_surf / self.C_s_max[i]
            ocp = ocp_graphite(sto) if self.regions[i] == "neg" else ocp_lco(sto)
            # Local overpotential solved from current balance
            # j_BV = I_app / (a_s * dx) for this cell
            dx = self.x[1] - self.x[0] if n > 1 else 1e-5
            j_target = I_app / max(self.a_s[i] * dx, 1e-6)
            # Solve BV for eta that gives j_target
            try:
                eta = butler_volmer_inverse(j_target, j0, self.T)
            except Exception:
                eta = 0.01
            j_BV[i] = j_target
            # Update phi_s to be consistent with eta
            phi_s[i] = eta + phi_e[i] + ocp

        self.phi_s = phi_s
        self.phi_e = phi_e
        return j_BV

    def step(self, dt: float, I_app: float, T_local: float = None) -> dict:
        """
        Full electrochemical time step.
        Returns dict with key state variables.
        """
        if T_local is not None:
            self.update_temperature(T_local)

        # Solve charge conservation (gives j_BV)
        j_BV = self.solve_charge_conservation(I_app)

        # Update solid diffusion
        for i in range(self.n_total):
            if self.solid_solvers[i] is not None:
                self.solid_solvers[i].step(dt, j_BV[i], self.C_s_max[i])

        # Update electrolyte
        self.solve_electrolyte(dt, j_BV)

        # Compute cell voltage from OCPs and overpotentials
        # Simplified: V = U_ocp,pos - U_ocp,neg + I*R_int + eta_total
        neg_conc = self.solid_solvers[0].surface_concentration() if self.solid_solvers[0] is not None else 0.5 * 30555.0
        pos_conc = self.solid_solvers[-1].surface_concentration() if self.solid_solvers[-1] is not None else 0.5 * 51554.0
        sto_neg = np.clip(neg_conc / 30555.0, 0.01, 0.99)
        sto_pos = np.clip(pos_conc / 51554.0, 0.35, 0.99)
        U_neg = ocp_graphite(sto_neg)
        U_pos = ocp_lco(sto_pos)
        # TODO: Hole 1 - 计算电池总电压（电化学-热耦合核心）
        # 需要结合正负极开路电势、欧姆压降、过电势计算总电压
        # V_cell = U_pos - U_neg + I_app * R_int + eta_total
        # 注意：此处的内阻模型需与 thermal_fem.py 中的热源模型保持一致
        V_cell = 0.0  # placeholder

        return {
            "voltage": float(V_cell),
            "j_BV": j_BV.copy(),
            "C_e": self.C_e.copy(),
            "phi_s": self.phi_s.copy(),
            "phi_e": self.phi_e.copy(),
            "solid_surface_conc": np.array([
                s.surface_concentration() if s is not None else 0.0
                for s in self.solid_solvers
            ])
        }
