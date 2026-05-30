# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional
from quadrature_engine import Vandermonde2DQuadrature, GaussLegendreTensor


class PhotothermalCoupler:

    def __init__(self,
                 helmholtz_solver,
                 thermal_solver,
                 dn_dT: float = 1.86e-4,
                 n0: float = 3.47,
                 alpha_abs: float = 1.0e-3,
                 max_iter: int = 30,
                 tol: float = 1e-8):
        self.helmholtz = helmholtz_solver
        self.thermal = thermal_solver
        self.dn_dT = dn_dT
        self.n0 = n0
        self.alpha_abs = alpha_abs
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def compute_heat_source(self, E: np.ndarray) -> np.ndarray:
        intensity = np.abs(E) ** 2
        return self.alpha_abs * intensity

    def update_refractive_index(self, T: np.ndarray, n_current: np.ndarray) -> np.ndarray:





        raise NotImplementedError("Hole 3: 请补全折射率更新公式")

    def integrate_heat_source(self, Q: np.ndarray, method: str = "trapezoidal") -> float:
        ny, nx = Q.shape
        hx = self.helmholtz.hx
        hy = self.helmholtz.hy

        if method == "trapezoidal":

            total = 0.0
            for j in range(ny):
                for i in range(nx):
                    weight = 1.0
                    if i == 0 or i == nx - 1:
                        weight *= 0.5
                    if j == 0 or j == ny - 1:
                        weight *= 0.5
                    total += weight * Q[j, i]
            total *= hx * hy
            return total
        elif method == "gauss_tensor":

            Lx = self.helmholtz.Lx
            Ly = self.helmholtz.Ly

            return float(np.sum(Q) * hx * hy)
        else:
            raise ValueError(f"未知积分方法: {method}")

    def integrate_vandermonde_2d(self, Q: np.ndarray, total_degree: int = 3) -> float:
        ny, nx = Q.shape
        if nx * ny < (total_degree + 1) * (total_degree + 2) // 2:

            return self.integrate_heat_source(Q, "trapezoidal")


        n_needed = (total_degree + 1) * (total_degree + 2) // 2
        step = max(1, int(np.sqrt(nx * ny / n_needed)))
        xs = []
        ys = []
        vals = []
        hx = self.helmholtz.hx
        hy = self.helmholtz.hy
        for j in range(0, ny, step):
            for i in range(0, nx, step):
                xs.append(i * hx)
                ys.append(j * hy)
                vals.append(Q[j, i])
        xs = np.array(xs[:n_needed])
        ys = np.array(ys[:n_needed])
        vals = np.array(vals[:n_needed])

        try:
            w = Vandermonde2DQuadrature.compute_weights(
                xs, ys, total_degree,
                rect_a=0.0, rect_b=self.helmholtz.Lx,
                rect_c=0.0, rect_d=self.helmholtz.Ly
            )
            return float(np.dot(w, vals))
        except Exception:

            return self.integrate_heat_source(Q, "trapezoidal")

    def self_consistent_solve(self,
                               source_mask: np.ndarray,
                               source_amplitude: complex = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        n_profile = np.full((self.helmholtz.ny, self.helmholtz.nx), self.n0, dtype=float)
        E = None
        T = None


        ny, nx = self.helmholtz.ny, self.helmholtz.nx
        if source_mask.ndim == 1 or source_mask.dtype == bool:

            gauss_source = np.zeros((ny, nx), dtype=complex)
            cx, cy = nx // 2, ny // 2
            for j in range(ny):
                for i in range(nx):
                    dx = (i - cx) * self.helmholtz.hx
                    dy = (j - cy) * self.helmholtz.hy
                    r2 = dx**2 + dy**2
                    gauss_source[j, i] = source_amplitude * np.exp(-r2 / (2.0 * (0.3e-6)**2))
            source_rhs = gauss_source
        else:
            source_rhs = source_mask * source_amplitude

        for it in range(1, self.max_iter + 1):

            self.helmholtz.n_profile = n_profile.copy()

            self.helmholtz._band_solver = None
            E = (self.helmholtz.solve_for_rhs(source_rhs.real)
                 + 1j * self.helmholtz.solve_for_rhs(source_rhs.imag))


            Q = self.compute_heat_source(E)


            T = self.thermal.solve_steady_state(Q, bc_type="robin")


            n_new = self.update_refractive_index(T, n_profile)


            delta_n = np.linalg.norm(n_new - n_profile) / (np.linalg.norm(n_profile) + 1e-30)
            self.history.append({
                "iter": it,
                "delta_n": delta_n,
                "max_T": float(np.max(T)),
                "mean_T": float(np.mean(T)),
            })
            n_profile = n_new

            if delta_n < self.tol:
                return E, T, n_profile, it

        return E, T, n_profile, self.max_iter

    def compute_thermal_shift(self, n_profile_uncoupled: np.ndarray,
                               n_profile_coupled: np.ndarray,
                               lambda0_nm: float = 1550.0) -> float:
        n_unc = np.mean(n_profile_uncoupled)
        n_coup = np.mean(n_profile_coupled)
        if abs(n_unc) < 1e-30:
            return 0.0
        return lambda0_nm * (n_coup - n_unc) / n_unc

    def wgm_mode_sphere_integral(self, E: np.ndarray, R_sphere: float) -> float:
        from quadrature_engine import SphereQuadrature
        sq = SphereQuadrature()
        n_samples = 500
        pts = sq.uniform_sample(n_samples)


        return float(np.mean(np.abs(E) ** 2) * 4.0 * np.pi * R_sphere ** 2)
