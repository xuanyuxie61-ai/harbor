"""
transport_solver.py
===================
膜附近离子/脂质密度输运求解模块（源自 seed 354_fd1d_advection_lax）

在纳米颗粒靠近生物膜的过程中，局部离子浓度场会发生显著变化，同时膜表面
的脂质密度也可能因曲率诱导的侧向压力而出现再分布。本模块使用 **Lax 格式**
求解一维对流-扩散方程：

    dc/dt = -v * dc/dx + D * d^2c/dx^2 + S(x,t)

其中：
    - c(x,t) 为离子或脂质密度（mol/L 或 1/nm^2）；
    - v 为对流失速度（nm/ns），由膜曲率驱动的 Marangoni 流近似给出；
    - D 为扩散系数（nm^2/ns）；
    - S(x,t) 为源汇项，描述纳米颗粒吸附导致的局部耗竭。

Lax 格式（FTCS 的稳定化修正）：
    c^{n+1}_i = 0.5*(c^n_{i-1} + c^n_{i+1})
                - (v*dt/(2*dx))*(c^n_{i+1} - c^n_{i-1})
                + (D*dt/dx^2)*(c^n_{i-1} - 2c^n_i + c^n_{i+1})
                + dt * S^n_i

稳定性约束（CFL + 扩散稳定性）：
    dt <= min(dx/|v|, dx^2/(2*D))

边界条件：
    - x = 0（膜表面附近）：反射边界或固定浓度；
    - x = L_max（本体溶液）：固定本体浓度 c_0。
"""

import numpy as np
from typing import Tuple


class AdvectionDiffusionSolver:
    """
    一维对流-扩散方程 Lax 格式求解器。
    """

    def __init__(self, L: float = 20.0, v: float = 0.05, D: float = 0.1,
                 c0: float = 0.1, nx: int = 201):
        """
        Parameters
        ----------
        L : float
            空间域长度（nm）。
        v : float
            对流速度（nm/ns），正方向为远离膜表面。
        D : float
            扩散系数（nm^2/ns）。
        c0 : float
            本体平衡浓度。
        nx : int
            空间格点数。
        """
        self.L = float(L)
        self.v = float(v)
        self.D = float(D)
        self.c0 = float(c0)
        self.nx = int(nx)
        self.dx = L / (nx - 1)
        # 自动选择满足稳定性的时间步长
        dt_adv = self.dx / (abs(v) + 1e-12)
        dt_diff = self.dx ** 2 / (2.0 * D + 1e-12)
        self.dt = 0.4 * min(dt_adv, dt_diff)  # 安全因子 0.4
        self.x = np.linspace(0.0, L, nx)

    def initial_condition(self, depletion_width: float = 2.0) -> np.ndarray:
        """
        构造初始浓度分布：在膜表面附近存在由于预吸附导致的耗竭区。

        解析形式：
            c(x, 0) = c0 * (1 - alpha * exp(-x / lambda_d))
        其中 alpha < 1 为耗竭强度，lambda_d 为耗竭特征宽度。
        """
        alpha = 0.3
        c = self.c0 * (1.0 - alpha * np.exp(-self.x / depletion_width))
        c = np.clip(c, 0.0, self.c0)
        return c

    def source_term(self, c: np.ndarray, t: float,
                    sink_strength: float = 0.01) -> np.ndarray:
        """
        源汇项：模拟纳米颗粒在 x=0 附近的持续吸附导致的局部浓度降低。

        S(x,t) = -k_sink * c(x,t) * exp(-x / delta_sink)
        """
        delta_sink = 1.0
        S = -sink_strength * c * np.exp(-self.x / delta_sink)
        return S

    def step(self, c: np.ndarray) -> np.ndarray:
        """
        执行一个显式时间步进（upwind + FTCS 混合格式）。

        为保证稳定性，采用 upwind 处理对流项（v>0 时向后差分）、
        FTCS 处理扩散项。稳定性条件为：
            v*dt/dx + 2*D*dt/dx^2 <= 1

        离散格式：
            c^{n+1}_i = c^n_i
                        - (v*dt/dx)*(c^n_i - c^n_{i-1})
                        + (D*dt/dx^2)*(c^n_{i-1} - 2c^n_i + c^n_{i+1})
                        + dt * S^n_i
        """
        nx = self.nx
        dx = self.dx
        dt = self.dt
        v = self.v
        D = self.D
        cnew = np.zeros(nx, dtype=np.float64)
        # 内部点
        for i in range(1, nx - 1):
            adv = -v * dt / dx * (c[i] - c[i - 1])
            diff = D * dt / dx ** 2 * (c[i - 1] - 2.0 * c[i] + c[i + 1])
            cnew[i] = c[i] + adv + diff
        # 源汇项
        S = self.source_term(c, 0.0)
        cnew[1:nx - 1] += dt * S[1:nx - 1]
        # 边界条件：Dirichlet
        cnew[0] = self.c0 * 0.7  # 膜表面附近因吸附而略低于本体浓度
        cnew[-1] = self.c0
        # 数值鲁棒性：非负截断
        cnew = np.clip(cnew, 0.0, None)
        return cnew

    def solve(self, n_steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行 n_steps 步时间推进。

        Returns
        -------
        c_final : ndarray
            最终浓度分布。
        history : ndarray, shape (n_snapshots, nx)
            每隔一定步数保存的快照。
        """
        c = self.initial_condition()
        snap_every = max(1, n_steps // 50)
        history = []
        for step in range(n_steps):
            c = self.step(c)
            if step % snap_every == 0:
                history.append(c.copy())
        history = np.array(history)
        return c, history

    def compute_flux(self, c: np.ndarray) -> float:
        """
        计算膜表面（x=0）的净通量：

            J = -D * dc/dx |_{x=0} + v * c(0)

        采用前向差分近似梯度。
        """
        dc_dx = (c[1] - c[0]) / self.dx
        J = -self.D * dc_dx + self.v * c[0]
        return float(J)
