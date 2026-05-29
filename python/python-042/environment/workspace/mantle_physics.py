"""
mantle_physics.py

Physical models and constitutive relations for mantle convection.

Scientific formulas:
- Temperature-dependent viscosity (Arrhenius law):
    η(T) = η₀ exp[ E*(T_ref − T) / (T_ref * T) ]
    where E* = E / (R_gas) is the activation energy divided by gas constant.
- Density (Boussinesq approximation):
    ρ(T) = ρ₀ [1 − α (T − T_ref)]
    where α is thermal expansivity.
- Thermal diffusivity:
    κ = k / (ρ₀ C_p)
- Rayleigh number:
    Ra = (ρ₀ g α ΔT D³) / (η κ)
    where D is mantle thickness, ΔT is temperature contrast.
- Critical Rayleigh number for spherical shell:
    Ra_c ≈ 657.5 (for rigid boundaries, isoviscous)
- Nusselt number:
    Nu = (convective heat transfer) / (conductive heat transfer)
- Stokes equations for creeping flow (infinite Prandtl number):
    ∇·σ + ρ g = 0
    ∇·u = 0
    where σ = −p I + η (∇u + ∇u^T)
- Energy equation:
    ∂T/∂t + u·∇T = κ ∇²T + H / (ρ₀ C_p)
    where H is radiogenic heat production [W/m³].
- Gravitational acceleration (depth-dependent):
    g(r) = G M(r) / r²
    where M(r) is mass enclosed within radius r.
"""

import numpy as np
from typing import Tuple, Optional


class MantleConstants:
    """Standard physical constants for Earth's mantle."""
    R_surf = 6371.0e3      # m
    R_cmb = 3480.0e3       # m
    g_surf = 9.81          # m/s²
    rho0 = 3300.0          # kg/m³ (reference density)
    alpha = 3.0e-5         # K⁻¹ (thermal expansivity)
    kappa = 1.0e-6         # m²/s (thermal diffusivity)
    Cp = 1200.0            # J/(kg·K) (specific heat)
    eta0 = 1.0e21          # Pa·s (reference viscosity)
    E_activation = 3.0e5   # J/mol (activation energy)
    R_gas = 8.314          # J/(mol·K)
    T_surf = 300.0         # K
    T_cmb = 3000.0         # K
    H_radio = 5.0e-12      # W/m³ (radiogenic heat production)
    G_grav = 6.67430e-11   # m³/(kg·s²)


class ViscosityModel:
    """
    Temperature- and depth-dependent viscosity models.
    """
    def __init__(self, eta0: float = MantleConstants.eta0,
                 E_act: float = MantleConstants.E_activation,
                 R_gas: float = MantleConstants.R_gas,
                 T_ref: float = 1600.0):
        self.eta0 = eta0
        self.E_act = E_act
        self.R_gas = R_gas
        self.T_ref = T_ref

    def arrhenius(self, T: np.ndarray) -> np.ndarray:
        """
        Arrhenius viscosity law:
            η(T) = η₀ exp[ E*(1/T − 1/T_ref) / R ]
        Boundary handling: viscosity capped at [η₀/100, 100*η₀].
        """
        T = np.asarray(T, dtype=float)
        T = np.clip(T, 500.0, 4000.0)  # physical bounds
        exponent = (self.E_act / self.R_gas) * (1.0 / T - 1.0 / self.T_ref)
        eta = self.eta0 * np.exp(exponent)
        return np.clip(eta, self.eta0 / 100.0, self.eta0 * 100.0)

    def frank_kamenetskii(self, T: np.ndarray) -> np.ndarray:
        """
        Frank-Kamenetskii viscosity approximation (linearized Arrhenius):
            η(T) ≈ η₀ exp[ −γ (T − T_ref) / T_ref ]
            where γ = E* / (R T_ref)
        """
        T = np.asarray(T, dtype=float)
        gamma = self.E_act / (self.R_gas * self.T_ref)
        eta = self.eta0 * np.exp(-gamma * (T - self.T_ref) / self.T_ref)
        return np.clip(eta, self.eta0 / 100.0, self.eta0 * 100.0)


class DensityModel:
    """
    Density variations via the Boussinesq approximation.
    """
    def __init__(self, rho0: float = MantleConstants.rho0,
                 alpha: float = MantleConstants.alpha,
                 T_ref: float = 1600.0):
        self.rho0 = rho0
        self.alpha = alpha
        self.T_ref = T_ref

    def thermal_density(self, T: np.ndarray) -> np.ndarray:
        """
        ρ(T) = ρ₀ [1 − α (T − T_ref)]
        """
        T = np.asarray(T, dtype=float)
        return self.rho0 * (1.0 - self.alpha * (T - self.T_ref))

    def buoyancy(self, T: np.ndarray) -> np.ndarray:
        """
        Relative buoyancy: Δρ = ρ₀ α (T − T_ref)
        Positive when T > T_ref (hot material rises).
        """
        T = np.asarray(T, dtype=float)
        return self.rho0 * self.alpha * (T - self.T_ref)


class DimensionlessNumbers:
    """
    Computation of dimensionless numbers governing mantle convection.
    """
    @staticmethod
    def rayleigh_number(D: float, delta_T: float, rho0: float = MantleConstants.rho0,
                        g: float = MantleConstants.g_surf,
                        alpha: float = MantleConstants.alpha,
                        eta: float = MantleConstants.eta0,
                        kappa: float = MantleConstants.kappa) -> float:
        """
        Ra = (ρ₀ g α ΔT D³) / (η κ)
        """
        if D <= 0 or eta <= 0 or kappa <= 0:
            raise ValueError("Physical parameters must be positive")
        return (rho0 * g * alpha * delta_T * D ** 3) / (eta * kappa)

    @staticmethod
    def nusselt_number(q_conv: float, q_cond: float) -> float:
        """
        Nu = q_conv / q_cond
        Boundary handling: returns 1.0 if q_cond ≈ 0.
        """
        if abs(q_cond) < 1e-30:
            return 1.0
        return q_conv / q_cond

    @staticmethod
    def prandtl_number(eta: float, rho0: float = MantleConstants.rho0,
                       kappa: float = MantleConstants.kappa) -> float:
        """
        Pr = ν / κ = η / (ρ₀ κ)
        For mantle, Pr → ∞ (infinite Prandtl number approximation).
        """
        if rho0 <= 0 or kappa <= 0:
            raise ValueError("rho0 and kappa must be positive")
        return eta / (rho0 * kappa)

    @staticmethod
    def peclet_number(U: float, D: float, kappa: float = MantleConstants.kappa) -> float:
        """
        Pe = U D / κ
        """
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        return U * D / kappa


class StokesPhysics:
    """
    Stokes flow equations for mantle convection.

    In the infinite Prandtl number limit, momentum balance is:
        ∇·[−p I + η (∇u + ∇u^T)] + ρ g r̂ = 0
    with incompressibility constraint:
        ∇·u = 0

    In 2D polar (r, θ) coordinates for an annular sector:
        ∂_r σ_rr + (1/r) ∂_θ σ_rθ + (σ_rr − σ_θθ)/r − ρ g = 0
        ∂_r σ_rθ + (1/r) ∂_θ σ_θθ + (2 σ_rθ)/r = 0
    where:
        σ_rr = −p + 2 η ∂_r u_r
        σ_θθ = −p + 2 η ( (1/r) ∂_θ u_θ + u_r/r )
        σ_rθ = η ( (1/r) ∂_θ u_r + ∂_r u_θ − u_θ/r )
    """
    def __init__(self, viscosity_model: ViscosityModel,
                 density_model: DensityModel):
        self.viscosity = viscosity_model
        self.density = density_model

    def streamfunction_vorticity_relation(self, psi: np.ndarray,
                                          r_grid: np.ndarray,
                                          theta_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For 2D axisymmetric or sector flow, velocity components from
        streamfunction ψ(r, θ):
            u_r = (1/r) ∂ψ/∂θ
            u_θ = −∂ψ/∂r
        """
        dr = float(np.mean(np.diff(r_grid[:, 0])))
        dtheta = float(np.mean(np.diff(theta_grid[0, :])))
        # Central differences
        dpsi_dtheta = np.zeros_like(psi)
        dpsi_dtheta[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dtheta)
        dpsi_dtheta[:, 0] = (psi[:, 1] - psi[:, 0]) / dtheta
        dpsi_dtheta[:, -1] = (psi[:, -1] - psi[:, -2]) / dtheta

        dpsi_dr = np.zeros_like(psi)
        dpsi_dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dr)
        dpsi_dr[0, :] = (psi[1, :] - psi[0, :]) / dr
        dpsi_dr[-1, :] = (psi[-1, :] - psi[-2, :]) / dr

        u_r = (1.0 / r_grid) * dpsi_dtheta
        u_theta = -dpsi_dr
        return u_r, u_theta


class ThermalPhysics:
    """
    Thermal energy conservation for mantle convection.

    Energy equation in 2D polar coordinates:
        ∂T/∂t + u_r ∂T/∂r + (u_θ/r) ∂T/∂θ =
            κ [ (1/r) ∂/∂r(r ∂T/∂r) + (1/r²) ∂²T/∂θ² ] + H/(ρ₀ C_p)
    """
    def __init__(self, kappa: float = MantleConstants.kappa,
                 H: float = MantleConstants.H_radio,
                 rho0: float = MantleConstants.rho0,
                 Cp: float = MantleConstants.Cp):
        self.kappa = kappa
        self.H = H
        self.rho0 = rho0
        self.Cp = Cp

    def laplacian_polar(self, T: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²T in polar coordinates on a structured grid.
        r and theta are 2D meshgrid arrays.
        """
        nr, ntheta = T.shape
        if nr < 3 or ntheta < 3:
            raise ValueError("Grid must be at least 3x3 for Laplacian")
        dr = float(np.mean(np.diff(r[:, 0])))
        dtheta = float(np.mean(np.diff(theta[0, :])))
        # ∂T/∂r
        dTdr = np.zeros_like(T)
        dTdr[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dr)
        dTdr[0, :] = (T[1, :] - T[0, :]) / dr
        dTdr[-1, :] = (T[-1, :] - T[-2, :]) / dr
        # ∂²T/∂r²
        d2Tdr2 = np.zeros_like(T)
        d2Tdr2[1:-1, :] = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / (dr ** 2)
        d2Tdr2[0, :] = d2Tdr2[1, :]
        d2Tdr2[-1, :] = d2Tdr2[-2, :]
        # ∂²T/∂θ²
        d2Tdtheta2 = np.zeros_like(T)
        d2Tdtheta2[:, 1:-1] = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / (dtheta ** 2)
        d2Tdtheta2[:, 0] = d2Tdtheta2[:, 1]
        d2Tdtheta2[:, -1] = d2Tdtheta2[:, -2]
        # <HOLE 1: 极坐标下拉普拉斯算子公式 ∇²T = (1/r) ∂/∂r(r ∂T/∂r) + (1/r²) ∂²T/∂θ²>
        raise NotImplementedError("HOLE 1: 需要实现极坐标下拉普拉斯算子公式")

    def advection_term(self, T: np.ndarray, u_r: np.ndarray, u_theta: np.ndarray,
                       r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute u·∇T = u_r ∂T/∂r + (u_θ/r) ∂T/∂θ.
        """
        dr = float(np.mean(np.diff(r[:, 0])))
        dtheta = float(np.mean(np.diff(theta[0, :])))
        dTdr = np.zeros_like(T)
        dTdr[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dr)
        dTdr[0, :] = (T[1, :] - T[0, :]) / dr
        dTdr[-1, :] = (T[-1, :] - T[-2, :]) / dr

        dTdtheta = np.zeros_like(T)
        dTdtheta[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dtheta)
        dTdtheta[:, 0] = (T[:, 1] - T[:, 0]) / dtheta
        dTdtheta[:, -1] = (T[:, -1] - T[:, -2]) / dtheta

        return u_r * dTdr + (u_theta / r) * dTdtheta

    def heat_production_term(self) -> float:
        """H / (ρ₀ C_p) in K/s."""
        return self.H / (self.rho0 * self.Cp)
