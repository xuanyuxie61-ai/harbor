# -*- coding: utf-8 -*-

import numpy as np
from helmholtz_fd import BandedMatrixSolver
from typing import Optional


class ThermalSolver:

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 kappa: float = 1.4e2,
                 h_conv: float = 10.0,
                 T_ambient: float = 300.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / (nx - 1)
        self.hy = Ly / (ny - 1)
        self.kappa = kappa
        self.h_conv = h_conv
        self.T_ambient = T_ambient
        self._N = nx * ny
        self._band_solver = None

    def _build_system(self, bc_type: str = "robin") -> BandedMatrixSolver:
        n = self._N
        ml = self.nx
        mu = self.nx
        solver = BandedMatrixSolver(n, ml, mu)
        solver._A_band = np.zeros((solver.lda, n), dtype=float)
        hx2 = self.hx ** 2
        hy2 = self.hy ** 2
        k = self.kappa

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                is_boundary = (i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1)

                if bc_type == "dirichlet" and is_boundary:
                    solver._A_band[mu, idx] = 1.0
                    continue









                raise NotImplementedError("Hole 2: 请补全热传导矩阵构造")
        solver._factorized = False
        return solver

    def solve_steady_state(self, Q_source: np.ndarray,
                           bc_type: str = "robin") -> np.ndarray:
        if Q_source.shape != (self.ny, self.nx):
            raise ValueError("Q_source 形状与网格不匹配")

        rhs = Q_source.flatten().copy()

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                is_boundary = (i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1)
                if bc_type == "dirichlet" and is_boundary:
                    rhs[idx] = self.T_ambient
                elif bc_type == "robin" and is_boundary:
                    rhs[idx] += self.h_conv * self.T_ambient / self.hx if (i == 0 or i == self.nx - 1) else self.h_conv * self.T_ambient / self.hy

                    rhs[idx] = self.h_conv * self.T_ambient / self.hx

        if self._band_solver is None:
            self._band_solver = self._build_system(bc_type)
            self._band_solver.factorize_np()

        T_flat = self._band_solver.solve(rhs)
        return T_flat.reshape(self.ny, self.nx)

    def compute_absorbed_heat(self, intensity: np.ndarray,
                              alpha_abs: float = 1.0e-3) -> np.ndarray:
        if intensity.shape != (self.ny, self.nx):
            raise ValueError("intensity 形状与网格不匹配")
        return alpha_abs * intensity

    def compute_thermal_lens(self, T: np.ndarray, dn_dT: float = 1.86e-4) -> np.ndarray:
        return dn_dT * (T - self.T_ambient)
