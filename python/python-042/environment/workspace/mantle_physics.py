
import numpy as np
from typing import Tuple, Optional


class MantleConstants:
    R_surf = 6371.0e3
    R_cmb = 3480.0e3
    g_surf = 9.81
    rho0 = 3300.0
    alpha = 3.0e-5
    kappa = 1.0e-6
    Cp = 1200.0
    eta0 = 1.0e21
    E_activation = 3.0e5
    R_gas = 8.314
    T_surf = 300.0
    T_cmb = 3000.0
    H_radio = 5.0e-12
    G_grav = 6.67430e-11


class ViscosityModel:
    def __init__(self, eta0: float = MantleConstants.eta0,
                 E_act: float = MantleConstants.E_activation,
                 R_gas: float = MantleConstants.R_gas,
                 T_ref: float = 1600.0):
        self.eta0 = eta0
        self.E_act = E_act
        self.R_gas = R_gas
        self.T_ref = T_ref

    def arrhenius(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        T = np.clip(T, 500.0, 4000.0)
        exponent = (self.E_act / self.R_gas) * (1.0 / T - 1.0 / self.T_ref)
        eta = self.eta0 * np.exp(exponent)
        return np.clip(eta, self.eta0 / 100.0, self.eta0 * 100.0)

    def frank_kamenetskii(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        gamma = self.E_act / (self.R_gas * self.T_ref)
        eta = self.eta0 * np.exp(-gamma * (T - self.T_ref) / self.T_ref)
        return np.clip(eta, self.eta0 / 100.0, self.eta0 * 100.0)


class DensityModel:
    def __init__(self, rho0: float = MantleConstants.rho0,
                 alpha: float = MantleConstants.alpha,
                 T_ref: float = 1600.0):
        self.rho0 = rho0
        self.alpha = alpha
        self.T_ref = T_ref

    def thermal_density(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.rho0 * (1.0 - self.alpha * (T - self.T_ref))

    def buoyancy(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.rho0 * self.alpha * (T - self.T_ref)


class DimensionlessNumbers:
    @staticmethod
    def rayleigh_number(D: float, delta_T: float, rho0: float = MantleConstants.rho0,
                        g: float = MantleConstants.g_surf,
                        alpha: float = MantleConstants.alpha,
                        eta: float = MantleConstants.eta0,
                        kappa: float = MantleConstants.kappa) -> float:
        if D <= 0 or eta <= 0 or kappa <= 0:
            raise ValueError("Physical parameters must be positive")
        return (rho0 * g * alpha * delta_T * D ** 3) / (eta * kappa)

    @staticmethod
    def nusselt_number(q_conv: float, q_cond: float) -> float:
        if abs(q_cond) < 1e-30:
            return 1.0
        return q_conv / q_cond

    @staticmethod
    def prandtl_number(eta: float, rho0: float = MantleConstants.rho0,
                       kappa: float = MantleConstants.kappa) -> float:
        if rho0 <= 0 or kappa <= 0:
            raise ValueError("rho0 and kappa must be positive")
        return eta / (rho0 * kappa)

    @staticmethod
    def peclet_number(U: float, D: float, kappa: float = MantleConstants.kappa) -> float:
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        return U * D / kappa


class StokesPhysics:
    def __init__(self, viscosity_model: ViscosityModel,
                 density_model: DensityModel):
        self.viscosity = viscosity_model
        self.density = density_model

    def streamfunction_vorticity_relation(self, psi: np.ndarray,
                                          r_grid: np.ndarray,
                                          theta_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dr = float(np.mean(np.diff(r_grid[:, 0])))
        dtheta = float(np.mean(np.diff(theta_grid[0, :])))

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
    def __init__(self, kappa: float = MantleConstants.kappa,
                 H: float = MantleConstants.H_radio,
                 rho0: float = MantleConstants.rho0,
                 Cp: float = MantleConstants.Cp):
        self.kappa = kappa
        self.H = H
        self.rho0 = rho0
        self.Cp = Cp

    def laplacian_polar(self, T: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        nr, ntheta = T.shape
        if nr < 3 or ntheta < 3:
            raise ValueError("Grid must be at least 3x3 for Laplacian")
        dr = float(np.mean(np.diff(r[:, 0])))
        dtheta = float(np.mean(np.diff(theta[0, :])))

        dTdr = np.zeros_like(T)
        dTdr[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dr)
        dTdr[0, :] = (T[1, :] - T[0, :]) / dr
        dTdr[-1, :] = (T[-1, :] - T[-2, :]) / dr

        d2Tdr2 = np.zeros_like(T)
        d2Tdr2[1:-1, :] = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / (dr ** 2)
        d2Tdr2[0, :] = d2Tdr2[1, :]
        d2Tdr2[-1, :] = d2Tdr2[-2, :]

        d2Tdtheta2 = np.zeros_like(T)
        d2Tdtheta2[:, 1:-1] = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / (dtheta ** 2)
        d2Tdtheta2[:, 0] = d2Tdtheta2[:, 1]
        d2Tdtheta2[:, -1] = d2Tdtheta2[:, -2]

        raise NotImplementedError("HOLE 1: 需要实现极坐标下拉普拉斯算子公式")

    def advection_term(self, T: np.ndarray, u_r: np.ndarray, u_theta: np.ndarray,
                       r: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
        return self.H / (self.rho0 * self.Cp)
