# -*- coding: utf-8 -*-
"""
thermal_fd.py
热传导方程有限差分求解器

核心公式与物理背景
------------------
1. 稳态热传导方程（含热源）
   ∇·(κ ∇T) + Q(x,y) = 0
   其中 κ 为热导率 [W/(m·K)]，Q 为体积热源密度 [W/m³]。

2. 光吸收热源
   对于微腔中的光场 E，单位体积吸收功率：
       Q_abs = 0.5 · ω · ε₀ · Im(ε) · |E|²
             = 0.5 · k₀ · c · n · α · |E|² / (2π)
   其中 α 为材料吸收系数 [m⁻¹]，c 为光速。
   简化形式：Q_abs = α · I，I 为光强。

3. 变分形式与有限差分离散
   对均匀网格，κ 常数时：
       κ · (∂²T/∂x² + ∂²T/∂y²) + Q = 0
   5 点 stencil：
       -κ·(2/hx²+2/hy²)·T_{i,j}
       + κ/hx²·(T_{i+1,j}+T_{i-1,j})
       + κ/hy²·(T_{i,j+1}+T_{i,j-1}) + Q_{i,j} = 0

4. 对流边界条件（Robin）
   -κ · ∂T/∂n = h_conv · (T - T_ambient)
   其中 h_conv 为对流换热系数 [W/(m²·K)]。

融合来源
--------
- 647_laplacian : 拉普拉斯差分算子（5点 stencil）
- 973_r8cb      : 带状矩阵求解稳态热传导线性系统
"""

import numpy as np
from helmholtz_fd import BandedMatrixSolver
from typing import Optional


class ThermalSolver:
    """
    微腔截面的稳态热传导有限差分求解器。
    支持 Dirichlet、Neumann 和 Robin 边界条件。
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 kappa: float = 1.4e2,        # Si 热导率 ~140 W/(m·K)
                 h_conv: float = 10.0,        # 自然对流系数 [W/(m²·K)]
                 T_ambient: float = 300.0):   # 环境温度 [K]
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
        """
        构建热传导离散系统矩阵。
        bc_type: "dirichlet" | "neumann" | "robin"
        """
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

                # TODO(Hole 2): 实现热传导方程的离散矩阵构造
                # 提示：
                #   1. 内部点使用标准 5 点 stencil：center = -2κ/hx² - 2κ/hy²
                #   2. Robin 边界（对流换热）需要一阶近似修正：
                #      -κ·∂T/∂n = h_conv·(T - T_ambient)
                #      例如左边界 i=0：center += -κ/hx² - h_conv/hx，右邻接 = κ/hx²
                #   3. 邻接系数在内部点为 κ/hx²（左右）和 κ/hy²（上下）
                #   4. 最后将 center 写入 solver._A_band[mu, idx]
                raise NotImplementedError("Hole 2: 请补全热传导矩阵构造")
        solver._factorized = False
        return solver

    def solve_steady_state(self, Q_source: np.ndarray,
                           bc_type: str = "robin") -> np.ndarray:
        """
        求解稳态温度场 T，给定热源分布 Q_source [W/m³]。

        参数
        ----
        Q_source : np.ndarray, shape (ny, nx)
            体积热源密度
        bc_type : str
            边界条件类型

        返回
        ----
        T : np.ndarray, shape (ny, nx)
            温度场 [K]
        """
        if Q_source.shape != (self.ny, self.nx):
            raise ValueError("Q_source 形状与网格不匹配")

        rhs = Q_source.flatten().copy()
        # 边界条件修正右端项
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                is_boundary = (i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1)
                if bc_type == "dirichlet" and is_boundary:
                    rhs[idx] = self.T_ambient
                elif bc_type == "robin" and is_boundary:
                    rhs[idx] += self.h_conv * self.T_ambient / self.hx if (i == 0 or i == self.nx - 1) else self.h_conv * self.T_ambient / self.hy
                    # 简化：统一用 hx
                    rhs[idx] = self.h_conv * self.T_ambient / self.hx

        if self._band_solver is None:
            self._band_solver = self._build_system(bc_type)
            self._band_solver.factorize_np()

        T_flat = self._band_solver.solve(rhs)
        return T_flat.reshape(self.ny, self.nx)

    def compute_absorbed_heat(self, intensity: np.ndarray,
                              alpha_abs: float = 1.0e-3) -> np.ndarray:
        """
        由光强分布计算体积热源。
        Q_abs = α_abs · I  [W/m³]
        其中 α_abs 为有效吸收系数 [m⁻¹]。
        """
        if intensity.shape != (self.ny, self.nx):
            raise ValueError("intensity 形状与网格不匹配")
        return alpha_abs * intensity

    def compute_thermal_lens(self, T: np.ndarray, dn_dT: float = 1.86e-4) -> np.ndarray:
        """
        由温度场计算热致折射率变化：
            Δn(x,y) = (dn/dT) · (T(x,y) - T_ambient)
        对 Si，dn/dT ≈ 1.86×10⁻⁴ K⁻¹（@1550 nm）。
        """
        return dn_dT * (T - self.T_ambient)
