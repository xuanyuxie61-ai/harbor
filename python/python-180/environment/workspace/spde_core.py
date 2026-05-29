"""
spde_core.py
随机偏微分方程核心求解器

科学背景:
  求解一维随机 Fisher-KPP 反应-扩散-对流方程:

      dU = [ epsilon * d^2U/dx^2 - v * dU/dx + r * U * (1 - U/K) ] dt
           + sigma_noise * U * (1 - U/K) dW(t,x)

  其中 dW(t,x) 为 Q-维纳过程，已截断到有限维。

  空间离散后得到 SDE 系统:
      dU_i = [ D_i(U) - A_i(U) + R_i(U) ] dt + G_i(U) dW_i(t)

  漂移项 f(U) 包含线性扩散部分 (可用隐式处理提升稳定性) 和
  非线性对流-反应部分 (显式处理)。

  对于乘性噪声 G(U) = sigma_noise * U * (1 - U/K):
      Milstein 修正项涉及 G'(U) = sigma_noise * (1 - 2U/K)

  能量估计:
      E[ ||U(t)||^2 ] <= C * exp( (2r + sigma_noise^2) t )
  要求数值格式保持该指数增长上界，否则为不稳定。
"""

import numpy as np
from typing import Callable, Optional, Tuple
from spatial_operators import SpatialDiscretization1D
from stochastic_rk import (
    StochasticIntegrator,
    sde_euler_maruyama_step,
    sde_srk_platen_step,
    sde_milstein_step,
    stiff_sde_semiimplicit_step,
    adaptive_rk12_sde_step,
)
from wiener_process import QWienerProcess
from numerical_utils import apply_dirichlet_bc, apply_neumann_bc_rhs


class SPDESolver1D:
    """
    一维 SPDE 求解器。
    """

    def __init__(self,
                 spatial: SpatialDiscretization1D,
                 wiener: QWienerProcess,
                 integrator: StochasticIntegrator,
                 sigma_noise: float = 0.1,
                 dirichlet_bc: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 neumann_bc: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.spatial = spatial
        self.wiener = wiener
        self.integrator = integrator
        self.sigma_noise = sigma_noise
        self.dirichlet_bc = dirichlet_bc
        self.neumann_bc = neumann_bc

        # 预计算 FEM 线性刚度矩阵 (用于半隐式)
        self.M, self.K = spatial.assemble_fem_matrices()
        # 隐式矩阵: I + theta * dt * M^{-1} K
        # 这里简化为直接使用 K 作为线性漂移矩阵 (假设 lumped mass)
        self.A_implicit = -np.diag(1.0 / self.M) @ self.K

    def _drift(self, u: np.ndarray) -> np.ndarray:
        """
        完整漂移: f(u) = diffusion - advection + reaction
        """
        return self.spatial.full_rhs_deterministic(u, scheme="auto")

    def _drift_split_nonlinear(self, u: np.ndarray) -> np.ndarray:
        """
        非线性漂移部分（用于半隐式，线性部分单独处理）。
        f_nonlin = -advection + reaction
        """
        return -self.spatial.advection_operator(u, scheme="auto") + self.spatial.reaction_operator(u)

    def _diffusion(self, u: np.ndarray) -> np.ndarray:
        """
        扩散系数 g(u) = sigma_noise * u * (1 - u/K)
        """
        K = self.spatial.K
        u_clip = np.clip(u, 0.0, None)
        return self.sigma_noise * u_clip * (1.0 - u_clip / K)

    def _diffusion_jacobian_diag(self, u: np.ndarray) -> np.ndarray:
        """
        扩散系数对角 Jacobian: dg/du = sigma_noise * (1 - 2u/K)
        """
        # TODO: 实现扩散系数的对角 Jacobian
        pass

    def _apply_bc_to_rhs(self, b: np.ndarray, t: float) -> np.ndarray:
        """
        将边界条件施加到右端项。
        """
        if self.neumann_bc is not None:
            nodes, flux = self.neumann_bc
            b = apply_neumann_bc_rhs(b, self.spatial.dx.mean(), nodes, flux)
        return b

    def solve(self,
              u0: np.ndarray,
              t_span: Tuple[float, float],
              store_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 SPDE。

        输入:
            u0: 初始条件
            t_span: (t0, tf)
            store_every: 每隔多少步存储一次

        输出:
            t_array: 存储的时间点
            u_history: 解历史，shape (n_steps, nx)
        """
        t0, tf = t_span
        if tf <= t0:
            raise ValueError("tf must be > t0")

        t = t0
        u = u0.copy()
        nx = self.spatial.nx

        # 预估步数
        est_steps = int((tf - t0) / self.integrator.dt) + 10
        t_list = [t]
        u_list = [u.copy()]

        step_count = 0
        while t < tf - 1e-14:
            dt = min(self.integrator.dt, tf - t)
            if dt <= 1e-14:
                break

            dW = self.wiener.increment(dt)

            # 根据方法选择函数封装
            if self.integrator.method == "semiimplicit":
                # 线性部分: A_implicit, 非线性部分: _drift_split_nonlinear
                y_new = stiff_sde_semiimplicit_step(
                    u, self.A_implicit, self._drift_split_nonlinear,
                    self._diffusion, dt, dW
                )
            elif self.integrator.method == "milstein":
                # TODO: 调用 Milstein 单步方法，注意传入正确的导数函数
                pass
            elif self.integrator.method == "adaptive_rk12":
                y_new, h_new, accepted = adaptive_rk12_sde_step(
                    u, self._drift, self._diffusion, dt, dW, self.integrator.tol
                )
                self.integrator.dt = h_new
                if not accepted:
                    continue
            elif self.integrator.method == "srk_platen":
                y_new = sde_srk_platen_step(
                    u, self._drift, self._diffusion, dt, dW
                )
            else:
                y_new = sde_euler_maruyama_step(
                    u, self._drift, self._diffusion, dt, dW
                )

            # 物理约束：截断到 [0, K] (Fisher-KPP 解的不变性)
            y_new = np.clip(y_new, 0.0, self.spatial.K * 1.5)

            # Dirichlet BC 强施加
            if self.dirichlet_bc is not None:
                bc_nodes, bc_vals = self.dirichlet_bc
                y_new[bc_nodes] = bc_vals

            u = y_new
            t += dt
            step_count += 1

            if step_count % store_every == 0:
                t_list.append(t)
                u_list.append(u.copy())

        if t_list[-1] < tf:
            t_list.append(t)
            u_list.append(u.copy())

        return np.array(t_list, dtype=np.float64), np.array(u_list, dtype=np.float64)

    def compute_energy(self, u: np.ndarray) -> float:
        """
        L2 能量: E = int u^2 dx ~ sum_i u_i^2 * M_i
        """
        return float(np.sum(u ** 2 * self.M))

    def compute_total_mass(self, u: np.ndarray) -> float:
        """
        总质量: M_total = int u dx ~ sum_i u_i * M_i
        """
        return float(np.sum(u * self.M))


