
import numpy as np
from typing import Tuple, Optional
from spectral_basis import GramSchmidt, TrigonometricBasis, LagrangeInterpolation, SpectralExpansion
from quadrature_engine import GaussLegendre
from mantle_physics import MantleConstants, ViscosityModel, DensityModel, StokesPhysics


class StokesSolver:
    def __init__(self, R_inner: float = 0.5, R_outer: float = 1.0,
                 n_radial: int = 12, n_angular: int = 16,
                 viscosity_model: Optional[ViscosityModel] = None):
        if R_inner <= 0 or R_outer <= R_inner:
            raise ValueError("Require 0 < R_inner < R_outer")
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.viscosity = viscosity_model if viscosity_model else ViscosityModel()
        self._build_basis()

    def _build_basis(self):

        self.r_nodes = np.linspace(self.R_inner, self.R_outer, self.n_radial)
        self.r_eval = np.linspace(self.R_inner, self.R_outer, 50)

        self.theta_nodes = np.linspace(0.0, 2.0 * np.pi, self.n_angular, endpoint=False)
        self.theta_eval = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)

        V_r = np.zeros((len(self.r_eval), self.n_radial), dtype=float)
        for j in range(self.n_radial):

            xi = 2.0 * (self.r_eval - self.R_inner) / (self.R_outer - self.R_inner) - 1.0
            V_r[:, j] = np.polynomial.legendre.legval(xi, [0.0] * j + [1.0])
        self.radial_basis = GramSchmidt.classical(V_r)

        self.angular_basis = SpectralExpansion(n_radial=self.n_radial, n_angular=self.n_angular).build_angular_basis(self.theta_eval)

        self.quad_x, self.quad_w = GaussLegendre.compute(16)

    def compute_velocity_from_streamfunction(self, psi_coeffs: np.ndarray,
                                             T_field: np.ndarray,
                                             r_grid: np.ndarray,
                                             theta_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nr, ntheta = T_field.shape
        if nr < 3 or ntheta < 3:
            raise ValueError("T_field must be at least 3x3")

        T_mean = float(np.mean(T_field))


        Ra_eff = 1.0e5
        rhs = -Ra_eff * (T_field - T_mean)

        psi = np.zeros((nr, ntheta), dtype=float)
        dr = float(np.mean(np.diff(r_grid[:, 0])))
        dtheta = float(np.mean(np.diff(theta_grid[0, :])))
        for _ in range(500):
            psi_new = psi.copy()
            for i in range(1, nr - 1):
                for j in range(1, ntheta - 1):
                    r_i = r_grid[i, j]
                    c_r = 1.0 / (dr ** 2)
                    c_theta = 1.0 / ((r_i * dtheta) ** 2)
                    c_center = 2.0 * c_r + 2.0 * c_theta

                    psi_new[i, j] = (
                        c_r * (psi[i + 1, j] + psi[i - 1, j])
                        + c_theta * (psi[i, j + 1] + psi[i, j - 1])
                        - rhs[i, j]
                    ) / c_center

            psi_new[0, :] = 0.0
            psi_new[-1, :] = 0.0
            psi_new[:, 0] = 0.0
            psi_new[:, -1] = 0.0
            if np.max(np.abs(psi_new - psi)) < 1e-8:
                psi = psi_new
                break
            psi = psi_new

        stokes = StokesPhysics(self.viscosity, DensityModel())
        u_r, u_theta = stokes.streamfunction_vorticity_relation(psi, r_grid, theta_grid)
        return u_r, u_theta

    def spectral_project_temperature(self, T_field: np.ndarray,
                                     r_grid: np.ndarray,
                                     theta_grid: np.ndarray) -> np.ndarray:
        nr, ntheta = T_field.shape

        C = np.zeros((self.n_radial, self.n_angular), dtype=float)
        for kr in range(self.n_radial):
            for ka in range(self.n_angular):

                radial_vals = np.interp(self.r_eval, r_grid[:, 0], self.radial_basis[:, kr])


                if nr == len(self.r_eval) and ntheta == len(self.theta_eval):
                    integrand = T_field * np.outer(self.radial_basis[:, kr],
                                                    self.angular_basis[:, ka])
                    C[kr, ka] = np.mean(integrand)
                else:
                    C[kr, ka] = 0.0
        return C
