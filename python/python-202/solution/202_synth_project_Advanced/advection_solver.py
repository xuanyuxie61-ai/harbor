"""
随机对流扩散方程求解器 (Stochastic Advection-Diffusion Solver)
================================================================
求解随机对流扩散方程:
  ∂u/∂t + c(ω) ∂u/∂x = ν ∂²u/∂x²,  x ∈ (0, L), t ∈ (0, T]

其中 c(ω) 是随机对流速度, ν 是扩散系数。

初始条件: u(x,0) = u₀(x) = sin(2πx/L) (光滑周期初始条件)
边界条件: u(0,t) = u(L,t) = 0 (Dirichlet)

离散化方案:

  1. FTCS (Forward-Time Centered-Space) - 不稳定!
     u_i^{n+1} = u_i^n - c Δt/(2h) (u_{i+1}^n - u_{i-1}^n) + ν Δt/h² (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
     稳定性条件: ν Δt/h² ≤ 1/2 AND |c|Δt/h ≤ √(2νΔt/h²)

  2. 迎风差分 (Upwind) - 条件稳定
     对于 c > 0:
     u_i^{n+1} = u_i^n - c Δt/h (u_i^n - u_{i-1}^n) + ν Δt/h² (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
     稳定性: CFL = |c|Δt/h ≤ 1 AND ν Δt/h² ≤ 1/2

  3. Crank-Nicolson (隐式, 无条件稳定)
     对流项显式 + 扩散项半隐式

数值色散与耗散:
  迎风差分的修正方程:
    u_t + c u_x = ν u_xx + (c h/2)(1 - CFL) u_xx + O(h²)
  数值粘性: ν_num = c h/2 (1 - CFL)
  当 CFL → 1, 数值粘性 → 0 (但色散增大)

Péclet 数:
  Pe = |c| L / ν
  Pe >> 1: 对流主导 (需要迎风或稳定化)
  Pe << 1: 扩散主导 (中心差分可用)
  Pe ~ 1: 中间区域
"""

import numpy as np
from typing import Optional, Tuple


class AdvectionDiffusionSolver1D:
    """
    1D 随机对流扩散方程求解器。

    求解: ∂u/∂t + c ∂u/∂x = ν ∂²u/∂x²
    使用迎风差分 (空间) + Forward Euler (时间)

    CFL 条件 (Courant-Friedrichs-Lewy):
      CFL = |c| Δt / h ≤ 1  (对流稳定性)
      D = ν Δt / h² ≤ 1/2  (扩散稳定性)
    """

    def __init__(
        self,
        n_spatial: int = 101,
        domain_length: float = 1.0,
        diffusion_coeff: float = 0.01,
        time_end: float = 1.0,
        cfl_number: float = 0.5
    ):
        """
        参数:
            n_spatial: 空间网格点数
            domain_length: 域长度 L
            diffusion_coeff: 扩散系数 ν
            time_end: 终止时间 T
            cfl_number: CFL 数 (用于确定时间步长)
        """
        self.n_spatial = n_spatial
        self.L = domain_length
        self.nu = diffusion_coeff
        self.T = time_end
        self.cfl = cfl_number

        # 空间网格
        self.x = np.linspace(0.0, self.L, self.n_spatial)
        self.h = self.L / (self.n_spatial - 1)

        # 初始条件
        self.u0 = self._initial_condition()

    def _initial_condition(self) -> np.ndarray:
        """
        初始条件:
          u₀(x) = sin(2πx/L) × exp(-x²/(2σ²))

        高斯调制正弦波: 局部化波动, 便于观察对流和扩散效应。
        """
        sigma = self.L / 6.0
        return np.sin(2.0 * np.pi * self.x / self.L) * np.exp(
            -(self.x - self.L / 2.0) ** 2 / (2.0 * sigma ** 2)
        )

    def compute_timestep(self, velocity: float) -> Tuple[float, int]:
        """
        根据 CFL 条件计算时间步长:
          Δt = CFL × h / |c|  (对流限制)
          Δt = min(Δt, h²/(2ν))  (扩散限制)

        参数:
            velocity: 对流速度 c

        返回:
            dt: 时间步长
            n_steps: 总步数
        """
        # 对流 CFL 限制
        if abs(velocity) > 1e-15:
            dt_adv = self.cfl * self.h / abs(velocity)
        else:
            dt_adv = self.T

        # 扩散限制
        if self.nu > 1e-15:
            dt_diff = 0.5 * self.h ** 2 / self.nu
        else:
            dt_diff = self.T

        dt = min(dt_adv, dt_diff)
        # 安全因子
        dt *= 0.9

        n_steps = max(1, int(np.ceil(self.T / dt)))
        dt = self.T / n_steps  # 调整使恰好到达 T

        return dt, n_steps

    def peclet_number(self, velocity: float) -> float:
        """
        网格 Péclet 数:
          Pe_h = |c| h / (2ν)

        Pe_h ≤ 1: 中心差分稳定
        Pe_h > 1: 需要迎风稳定化
        """
        if self.nu < 1e-15:
            return float('inf')
        return abs(velocity) * self.h / (2.0 * self.nu)

    def solve(self, velocity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解对流扩散方程 (确定性问题, 给定速度)。

        迎风差分格式 (c > 0):
          u_i^{n+1} = u_i^n - c Δt/h (u_i^n - u_{i-1}^n)
                      + ν Δt/h² (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

        迎风差分格式 (c < 0):
          u_i^{n+1} = u_i^n - c Δt/h (u_{i+1}^n - u_i^n)
                      + ν Δt/h² (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

        参数:
            velocity: 对流速度 c

        返回:
            u_final: 终止时刻的解, shape (N_spatial,)
            time_history: 时间记录 (每隔一定步数保存)
        """
        dt, n_steps = self.compute_timestep(velocity)
        u = self.u0.copy()

        r_adv = abs(velocity) * dt / self.h  # 对流数
        r_diff = self.nu * dt / self.h ** 2  # 扩散数

        # 稳定性检查
        if r_diff > 0.5:
            # 自动调整 (理论上不应发生)
            r_diff = 0.49

        save_interval = max(1, n_steps // 20)
        time_history = [0.0]
        u_snapshots = [u.copy()]

        for step in range(n_steps):
            u_new = u.copy()

            if velocity >= 0:
                # 右传播: 使用左迎风
                for i in range(1, self.n_spatial - 1):
                    u_new[i] = (u[i]
                                - r_adv * (u[i] - u[i - 1])
                                + r_diff * (u[i + 1] - 2.0 * u[i] + u[i - 1]))
            else:
                # 左传播: 使用右迎风
                for i in range(1, self.n_spatial - 1):
                    u_new[i] = (u[i]
                                - r_adv * (u[i + 1] - u[i])
                                + r_diff * (u[i + 1] - 2.0 * u[i] + u[i - 1]))

            # 边界条件 (Dirichlet = 0)
            u_new[0] = 0.0
            u_new[-1] = 0.0

            u = u_new

            if (step + 1) % save_interval == 0:
                time_history.append((step + 1) * dt)
                u_snapshots.append(u.copy())

        return u, np.array(time_history)

    def compute_l2_norm(self, u: np.ndarray) -> float:
        """
        L² 范数:
          ||u||₂ = √(h Σ u_i²)
        """
        return np.sqrt(self.h * np.sum(u ** 2))

    def compute_h1_seminorm(self, u: np.ndarray) -> float:
        """
        H¹ 半范数:
          |u|₁ = √(h Σ ((u_{i+1}-u_i)/h)²)
        """
        du = np.diff(u) / self.h
        return np.sqrt(self.h * np.sum(du ** 2))

    def mass_conservation_check(self, u: np.ndarray) -> float:
        """
        质量守恒检查:
          M(t) = h Σ u_i

        对于 Dirichlet 边界条件, 质量不一定守恒 (有通量通过边界)。
        但对于周期边界条件, 应严格守恒。
        """
        return self.h * np.sum(u)


def solve_stochastic_advection_diffusion(
    solver: AdvectionDiffusionSolver1D,
    velocity_samples: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机对流扩散方程的集成求解。

    参数:
        velocity_samples: 速度样本, shape (N_samples,)

    返回:
        final_solutions: shape (N_samples, N_spatial)
        l2_norms: shape (N_samples,)
    """
    N = velocity_samples.size
    final_solutions = np.zeros((N, solver.n_spatial))
    l2_norms = np.zeros(N)

    for i in range(N):
        u_final, _ = solver.solve(velocity_samples[i])
        final_solutions[i] = u_final
        l2_norms[i] = solver.compute_l2_norm(u_final)

    return final_solutions, l2_norms
