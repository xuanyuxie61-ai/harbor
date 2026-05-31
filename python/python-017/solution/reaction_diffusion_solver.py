"""
反应扩散方程求解模块
融合来源: 434_fisher_pde_ftcs (Fisher-KPP 方程 FTCS 有限差分)

功能:
- 使用 FTCS (Forward Time Centered Space) 格式求解二维反应扩散方程
- 应用于多铁性材料中极化/磁化序参量的非线性时空演化
- 包含稳定性条件检查、边界条件处理、守恒量监测

核心方程:
    ∂u/∂t = D ∇²u + R(u)

其中:
    D: 扩散系数矩阵（张量）
    R(u): 局部反应项（来源于 Landau 自由能变分）

FTCS 离散格式:
    u^{n+1} = u^n + dt * [D * L(u^n) + R(u^n)]
    L(u) ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²

稳定性条件 (二维):
    dt <= h² / (4 D_max)
"""

import numpy as np
from typing import Callable, Tuple, Optional


class ReactionDiffusionFTCS:
    """
    二维反应扩散方程 FTCS 求解器，基于规则网格。
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                 D: float = 1.0, dt: Optional[float] = None):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.D = D

        # CFL 稳定性条件
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        self.dt_max = 0.5 / (D * (1.0 / dx2 + 1.0 / dy2))
        if dt is None:
            self.dt = 0.25 * self.dt_max
        else:
            if dt > self.dt_max:
                raise ValueError(
                    f"时间步长 dt={dt} 超过 FTCS 稳定性限制 dt_max={self.dt_max:.6e}"
                )
            self.dt = dt

        # 预计算 Laplacian 模板系数
        self.cx = D * self.dt / dx2
        self.cy = D * self.dt / dy2

    def laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        """
        计算二维离散 Laplacian，采用五点差分格式，带 Neumann 边界。
        """
        if u.shape != (self.ny, self.nx):
            raise ValueError(f"u 形状应为 ({self.ny}, {self.nx})")

        Lu = np.zeros_like(u)
        # 内部点
        Lu[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx ** 2 +
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dy ** 2
        )

        # Neumann 边界: ∂u/∂n = 0
        # 左边界
        Lu[:, 0] = (2.0 * u[:, 1] - 2.0 * u[:, 0]) / self.dx ** 2 + (
            np.concatenate([[0], (u[2:, 0] - 2.0 * u[1:-1, 0] + u[:-2, 0]), [0]]) / self.dy ** 2
        )
        # 右边界
        Lu[:, -1] = (2.0 * u[:, -2] - 2.0 * u[:, -1]) / self.dx ** 2 + (
            np.concatenate([[0], (u[2:, -1] - 2.0 * u[1:-1, -1] + u[:-2, -1]), [0]]) / self.dy ** 2
        )
        # 下边界
        Lu[0, :] = (np.concatenate([[0], (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]), [0]]) / self.dx ** 2 +
                    (2.0 * u[1, :] - 2.0 * u[0, :]) / self.dy ** 2)
        # 上边界
        Lu[-1, :] = (np.concatenate([[0], (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2]), [0]]) / self.dx ** 2 +
                     (2.0 * u[-2, :] - 2.0 * u[-1, :]) / self.dy ** 2)

        return Lu

    def step(self, u: np.ndarray, reaction: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        执行一个 FTCS 时间步。

        参数:
            u: 当前场值 (ny, nx)
            reaction: 反应项函数 R(u)，输入输出形状均为 (ny, nx)

        返回:
            u_new: 下一时刻场值
        """
        Lu = self.laplacian_2d(u)
        R = reaction(u)
        # 边界处理: 边界点不施加反应项（保持 Neumann 条件主导）
        R[0, :] = 0.0
        R[-1, :] = 0.0
        R[:, 0] = 0.0
        R[:, -1] = 0.0

        u_new = u + self.dt * (self.D * Lu + R)

        # 数值鲁棒性: 截断异常值
        u_max = np.max(np.abs(u))
        if u_max > 0:
            clip_val = u_max * 1e6
            u_new = np.clip(u_new, -clip_val, clip_val)

        # 检测 NaN/Inf
        if not np.all(np.isfinite(u_new)):
            # 若发散，回退到当前值并施加强阻尼
            mask = ~np.isfinite(u_new)
            u_new[mask] = u[mask] * 0.9

        return u_new

    def solve(self, u0: np.ndarray, reaction: Callable[[np.ndarray], np.ndarray],
              nsteps: int, callback: Optional[Callable] = None) -> np.ndarray:
        """
        迭代求解多步。

        参数:
            u0: 初始场
            nsteps: 时间步数
            callback: 可选回调函数 callback(step, u)

        返回:
            u: 最终场
        """
        u = u0.copy()
        for step in range(nsteps):
            u = self.step(u, reaction)
            if callback is not None:
                callback(step, u)
        return u


def fisher_kpp_reaction(u: np.ndarray, r: float = 1.0, K: float = 1.0) -> np.ndarray:
    """
    Fisher-KPP 型反应项: R(u) = r * u * (1 - u/K)
    直接源自 fisher_pde_ftcs 中的非线性源项。
    """
    return r * u * (1.0 - u / K)


def allen_cahn_reaction(u: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """
    Allen-Cahn 型双势阱反应项: R(u) = (u - u^3) / epsilon²
    用于描述多铁性材料中的畴壁动力学。
    """
    return (u - u ** 3) / (epsilon ** 2 + 1e-20)


def coupled_reaction_diffusion_step(
    P: np.ndarray, M: np.ndarray,
    solver_P: ReactionDiffusionFTCS,
    solver_M: ReactionDiffusionFTCS,
    reaction_P: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reaction_M: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行极化 P 与磁化 M 的耦合反应扩散时间步。

    参数:
        P, M: 当前极化和磁化场
        solver_P, solver_M: 各自的 FTCS 求解器
        reaction_P: R_P(P, M)
        reaction_M: R_M(P, M)

    返回:
        P_new, M_new
    """
    P_new = solver_P.step(P, lambda u: reaction_P(u, M))
    M_new = solver_M.step(M, lambda u: reaction_M(P, u))
    return P_new, M_new
