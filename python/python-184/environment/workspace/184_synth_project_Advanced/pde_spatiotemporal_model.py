"""
PDE-Based Spatiotemporal Smoothing and Forecasting
==================================================
源自种子项目：
  - 390_fem1d_heat_explicit (1D heat equation via explicit FEM)
  - 1368_tumor_pde (Reaction-diffusion PDE model)

将时间序列视为定义在一维空间（时间轴）上的标量场 u(t)，
通过 PDE 演化实现：
1. 热方程平滑：过滤高频噪声
2. 反应-扩散方程：捕捉非线性增长/衰减模式

1. 热方程（1D FEM explicit）
   u_t - κ u_xx = f(x,t),   x ∈ [0, T], t > 0
   空间离散：分段线性 hat 函数 {φ_i}，单元 [x_i, x_{i+1}]
   质量矩阵 M_{ij} = ∫ φ_i φ_j dx
   刚度矩阵 K_{ij} = ∫ φ_i' φ_j' dx
   显式 Euler：u^{n+1} = u^n + dt M^{-1} (-K u^n + b^n)
   CFL 稳定性条件：dt <= h^2 / (2 κ)

2. 反应-扩散方程（用于非线性时间序列演化）
   ∂u/∂t = D ∂²u/∂x² + R(u)
   其中 R(u) 为反应项，例如 logistic 增长：
       R(u) = ρ u (1 - u/K) - μ u / (c_s + u)
   该形式源于肿瘤 PDE 中的增殖-消耗动力学。

数值方法：
- 空间：有限元法（FEM），线性基函数，梯形法则数值积分
- 时间：显式 Euler / 改进 Euler（Heun）
"""

import numpy as np
from typing import Callable


class PDE1DHeatExplicit:
    """
    一维热方程显式 FEM 求解器，用于时间序列平滑。
    """

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def solve(self, x_grid: np.ndarray, u0: np.ndarray,
              dt: float, n_steps: int,
              source: Callable | None = None) -> np.ndarray:
        """
        显式 FEM 求解热方程。

        Parameters
        ----------
        x_grid : np.ndarray, shape (N,)
            空间（时间轴）网格，不必等距。
        u0 : np.ndarray, shape (N,)
            初始场（原始时间序列）。
        dt : float
            时间步长。
        n_steps : int
            演化步数。
        source : callable, optional
            源项 f(x, t, u)。

        Returns
        -------
        u : np.ndarray, shape (N,)
            平滑后的场。
        """
        n = len(x_grid)
        if len(u0) != n:
            raise ValueError("u0 length must match x_grid length.")
        if n < 3:
            raise ValueError("Grid must have at least 3 points.")

        # 计算网格尺寸（变网格处理）
        h = np.diff(x_grid)
        h_min = h.min()

        # CFL 条件检查
        cfl_limit = h_min ** 2 / (2.0 * self.kappa)
        if dt > cfl_limit:
            # 自动调整 dt 以满足稳定性
            dt = cfl_limit * 0.9
            n_steps = max(int(n_steps * cfl_limit / dt), 1)

        # 组装质量矩阵 M（ lumped 质量矩阵简化）
        M_diag = np.zeros(n)
        M_diag[0] = h[0] / 2.0
        M_diag[-1] = h[-1] / 2.0
        for i in range(1, n - 1):
            M_diag[i] = (h[i - 1] + h[i]) / 2.0

        # 组装刚度矩阵 K（三对角，使用变网格公式）
        K_diag = np.zeros(n)
        K_off = np.zeros(n - 1)
        for i in range(n - 1):
            inv_h = 1.0 / h[i]
            K_off[i] = -self.kappa * inv_h
            K_diag[i] += self.kappa * inv_h
            K_diag[i + 1] += self.kappa * inv_h

        u = u0.copy()
        for step in range(n_steps):
            # 计算 -K u（不含源项）
            Ku = np.zeros(n)
            Ku[0] = K_diag[0] * u[0] + K_off[0] * u[1]
            for i in range(1, n - 1):
                Ku[i] = K_off[i - 1] * u[i - 1] + K_diag[i] * u[i] + K_off[i] * u[i + 1]
            Ku[-1] = K_off[-1] * u[-2] + K_diag[-1] * u[-1]

            rhs = -Ku
            if source is not None:
                rhs += source(x_grid, step * dt, u)

            # 显式更新：u_new = u + dt M^{-1} rhs
            du = dt * rhs / M_diag
            u = u + du

        return u


class ReactionDiffusion1D:
    """
    一维反应-扩散方程求解器，用于非线性动力学时间序列建模。
    方程：u_t = D u_xx + ρ u (1 - u/K) - μ u / (c_s + u)
    最后两项分别对应 logistic 增殖与 Michaelis-Menten 消耗。
    """

    def __init__(self, D: float = 0.1, rho: float = 1.0, K: float = 1.0,
                 mu: float = 0.5, c_s: float = 0.1):
        self.D = D
        self.rho = rho
        self.K = K
        self.mu = mu
        self.c_s = c_s

    def reaction(self, u: np.ndarray) -> np.ndarray:
        """反应项 R(u) = ρ u (1 - u/K) - μ u / (c_s + u)"""
        logistic = self.rho * u * (1.0 - u / self.K)
        # Michaelis-Menten 消耗，避免除零
        consumption = self.mu * u / (self.c_s + np.abs(u) + 1e-12)
        return logistic - consumption

    def solve(self, x_grid: np.ndarray, u0: np.ndarray,
              dt: float, n_steps: int, scheme: str = "heun") -> np.ndarray:
        """
        求解反应-扩散方程。
        scheme: "euler" 或 "heun"（改进 Euler，二阶精度）
        """
        n = len(x_grid)
        if len(u0) != n:
            raise ValueError("u0 length mismatch.")
        h = np.diff(x_grid)
        h_min = h.min()

        # 扩散项 CFL
        cfl_diff = h_min ** 2 / (2.0 * self.D) if self.D > 0 else np.inf
        # 反应项稳定性（logistic 最大增长率约 rho）
        cfl_react = 1.0 / abs(self.rho) if self.rho != 0 else np.inf
        dt_limit = min(cfl_diff, cfl_react)
        if dt > dt_limit:
            dt = dt_limit * 0.5

        def laplacian(u):
            """中心差分 Laplacian，变网格处理"""
            Lu = np.zeros(n)
            # 内部点
            for i in range(1, n - 1):
                hp = x_grid[i + 1] - x_grid[i]
                hm = x_grid[i] - x_grid[i - 1]
                # 变网格二阶导数：2/(h+ + h-) * [(u_{i+1}-u_i)/h+ - (u_i-u_{i-1})/hm]
                Lu[i] = 2.0 / (hp + hm) * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm)
            # Neumann 边界：保持边界值不变（零通量）
            Lu[0] = Lu[1]
            Lu[-1] = Lu[-2]
            return Lu

        u = u0.copy()
        if scheme == "euler":
            for _ in range(n_steps):
                Lu = laplacian(u)
                R = self.reaction(u)
                u = u + dt * (self.D * Lu + R)
        elif scheme == "heun":
            for _ in range(n_steps):
                Lu = laplacian(u)
                R = self.reaction(u)
                k1 = dt * (self.D * Lu + R)
                u_temp = u + k1
                Lu2 = laplacian(u_temp)
                R2 = self.reaction(u_temp)
                k2 = dt * (self.D * Lu2 + R2)
                u = u + 0.5 * (k1 + k2)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        return u
