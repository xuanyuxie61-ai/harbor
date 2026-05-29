"""
Method of Manufactured Solutions (MMS) for PDE Verification
===========================================================
源自种子项目 1172_stokes_2d_exact (Manufactured solutions for Stokes equations)。

方法 of Manufactured Solutions 是验证数值 PDE 求解器的黄金标准：
1. 选取一个光滑的解析函数 u_exact(x,t) 作为"制造解"
2. 将其代入 PDE，计算对应的强迫项 f(x,t)（残差）
3. 用数值方法求解带强迫项的 PDE，比较数值解与制造解
4. 通过网格加密检验收敛阶

本模块将其应用于：
- 1D 热方程的 FEM 求解器验证
- 反应-扩散方程的验证
- 时间序列平滑算法的精度评估

数学：
热方程 u_t - κ u_xx = f
若 u_exact = sin(ω x) exp(-λ t)，则
    f = (-λ + κ ω^2) sin(ω x) exp(-λ t)
在边界 x=0, x=L 上 Dirichlet 条件：u(0,t)=0, u(L,t)=sin(ω L)exp(-λ t)。

收敛阶估计：
    log_2(||e_{2h}|| / ||e_h||) → p
其中 p 为理论收敛阶（显式 Euler 时间一阶，线性 FEM 空间二阶）。
"""

import numpy as np
from typing import Callable, Tuple


class ManufacturedVerification:
    """
    制造解方法验证工具。
    """

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def heat_exact(self, x: np.ndarray, t: float, omega: float = np.pi, lam: float = 1.0) -> np.ndarray:
        """
        制造解：u(x,t) = sin(ω x) exp(-λ t)。
        """
        return np.sin(omega * x) * np.exp(-lam * t)

    def heat_forcing(self, x: np.ndarray, t: float, omega: float = np.pi, lam: float = 1.0) -> np.ndarray:
        """
        对应强迫项：f = u_t - κ u_xx = (-λ + κ ω^2) sin(ω x) exp(-λ t)。
        """
        return (-lam + self.kappa * omega ** 2) * np.sin(omega * x) * np.exp(-lam * t)

    def verify_heat_fem(self, x_grid: np.ndarray, dt: float, n_steps: int) -> dict:
        """
        验证 1D 热方程 FEM 求解器。
        """
        from pde_spatiotemporal_model import PDE1DHeatExplicit
        solver = PDE1DHeatExplicit(kappa=self.kappa)
        omega = np.pi / (x_grid[-1] - x_grid[0])
        lam = 1.0

        u0 = self.heat_exact(x_grid, 0.0, omega, lam)

        def source(x, t, u):
            return self.heat_forcing(x, t, omega, lam)

        u_num = solver.solve(x_grid, u0, dt, n_steps, source=source)
        t_final = dt * n_steps
        u_exact = self.heat_exact(x_grid, t_final, omega, lam)

        # L2 误差（离散近似）
        h = np.diff(x_grid)
        h_avg = np.mean(h)
        err = u_num - u_exact
        l2_err = np.sqrt(np.mean(err ** 2))
        linf_err = np.max(np.abs(err))

        return {
            "l2_error": float(l2_err),
            "linf_error": float(linf_err),
            "relative_l2": float(l2_err / (np.sqrt(np.mean(u_exact ** 2)) + 1e-15)),
            "grid_size": len(x_grid),
            "dt": dt,
            "final_time": t_final
        }

    def convergence_study(self, n_grids: list, dt: float, n_steps: int) -> dict:
        """
        网格加密收敛性研究。
        """
        results = []
        for n in n_grids:
            x_grid = np.linspace(0.0, 1.0, n)
            res = self.verify_heat_fem(x_grid, dt, n_steps)
            results.append(res)

        # 估计收敛阶
        orders = []
        for i in range(1, len(results)):
            h_ratio = results[i - 1]["grid_size"] / results[i]["grid_size"]
            err_ratio = results[i - 1]["l2_error"] / (results[i]["l2_error"] + 1e-15)
            if err_ratio > 0:
                p = np.log(err_ratio) / np.log(h_ratio)
                orders.append(p)

        return {
            "results": results,
            "estimated_spatial_order": float(np.mean(orders)) if orders else 0.0
        }

    def reaction_diffusion_exact(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        反应-扩散方程的制造解：
        u(x,t) = exp(-t) sin(π x) + 0.5
        对应反应项需调整使得 PDE 成立。
        """
        return np.exp(-t) * np.sin(np.pi * x) + 0.5

    def reaction_diffusion_forcing(self, x: np.ndarray, t: float, D: float = 0.1) -> np.ndarray:
        """
        计算反应-扩散方程的制造强迫项。
        u_t = D u_xx + R(u) + f(x,t)
        取 R(u) = 0（纯扩散验证），则 f = u_t - D u_xx。
        """
        u = self.reaction_diffusion_exact(x, t)
        ut = -np.exp(-t) * np.sin(np.pi * x)
        uxx = -(np.pi ** 2) * np.exp(-t) * np.sin(np.pi * x)
        return ut - D * uxx

    def verify_reaction_diffusion(self, x_grid: np.ndarray, dt: float, n_steps: int, D: float = 0.1) -> dict:
        """
        验证反应-扩散求解器（纯扩散模式）。
        """
        from pde_spatiotemporal_model import ReactionDiffusion1D
        solver = ReactionDiffusion1D(D=D, rho=0.0, K=1.0, mu=0.0, c_s=1.0)
        u0 = self.reaction_diffusion_exact(x_grid, 0.0)

        # 修改反应项为纯强迫项
        original_reaction = solver.reaction
        solver.reaction = lambda u: self.reaction_diffusion_forcing(x_grid, 0.0, D) * np.ones_like(u)

        u_num = solver.solve(x_grid, u0, dt, n_steps, scheme="heun")
        t_final = dt * n_steps
        u_exact = self.reaction_diffusion_exact(x_grid, t_final)

        err = u_num - u_exact
        l2_err = np.sqrt(np.mean(err ** 2))

        return {
            "l2_error": float(l2_err),
            "relative_l2": float(l2_err / (np.sqrt(np.mean(u_exact ** 2)) + 1e-15)),
            "verified": l2_err < 0.1  # 宽松阈值
        }
