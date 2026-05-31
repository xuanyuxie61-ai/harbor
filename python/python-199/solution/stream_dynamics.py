"""
stream_dynamics.py — 科学数据流在内存-外存边界上的动力学模型
===============================================================
融合来源:
  - 020_artery_pde (动脉血管PDE受迫振动模型)
  - 065_ball_and_stick_display (Lax-Wendroff双曲型PDE格式)

在高性能外排序中，数据在内存缓冲区和外部存储之间周期性交换，
形成类波动的数据流。我们将此过程建模为受迫阻尼振动系统，并用
Lax-Wendroff格式进行离散求解，以预测最优缓冲区大小和I/O调度策略。
"""

import math
from typing import List, Tuple


class ArteryFlowModel:
    """
    动脉血管血流动力学模型，用于模拟数据在内存缓冲区中的脉动传输。

    控制方程（二阶受迫阻尼振动）：
        d²u/dt² + β · du/dt + α · u = γ · x · dp/dx · (a + b·cos(ωt))

    其中：
        u(t): 血管壁位移 ↔ 内存缓冲区占用量
        v(t) = du/dt: 壁速度 ↔ 数据流入/流出速率
        α: 弹性恢复系数 ↔ 内存回收压力
        β: 阻尼系数 ↔ I/O 带宽限制
        γ: 外力耦合系数 ↔ 数据生成速率
        ω: 驱动频率 ↔ 数据产生周期性

    参考: Quarteroni, Numerical Mathematics.
    """

    def __init__(self, alpha: float, beta: float, gamma: float,
                 a: float, b: float, omega: float, x: float = 1.0, dp_dx: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.omega = omega
        self.x = x
        self.dp_dx = dp_dx

    def forcing(self, t: float) -> float:
        """
        周期性驱动力：
            F(t) = γ · x · dp/dx · (a + b·cos(ωt))
        """
        return self.gamma * self.x * self.dp_dx * (self.a + self.b * math.cos(self.omega * t))

    def rhs(self, t: float, state: List[float]) -> List[float]:
        """
        状态向量 state = [u, v]，右端项：
            du/dt = v
            dv/dt = -α·u - β·v + F(t)
        """
        u, v = state
        force = self.forcing(t)
        return [v, -self.alpha * u - self.beta * v + force]

    def analytical_amplitude(self) -> float:
        """
        稳态受迫振动的解析振幅（β > 0 时存在稳态解）。

        设特解形式 u_p(t) = C·cos(ωt) + D·sin(ωt) + E，代入方程得：
            E = γ·x·dp/dx·a / α
            C = γ·x·dp/dx·b·(α - ω²) / [(α - ω²)² + (βω)²]
            D = γ·x·dp/dx·b·βω / [(α - ω²)² + (βω)²]

        振幅 A = sqrt(C² + D²) = γ·x·dp/dx·b / sqrt((α - ω²)² + (βω)²)
        """
        denom = math.sqrt((self.alpha - self.omega ** 2) ** 2 + (self.beta * self.omega) ** 2)
        if denom < 1e-15:
            return 0.0
        return abs(self.gamma * self.x * self.dp_dx * self.b) / denom

    def simulate_euler(self, u0: float, v0: float, t_end: float, n_steps: int) -> Tuple[List[float], List[float], List[float]]:
        """
        使用显式Euler法求解系统：
            y_{n+1} = y_n + h · f(t_n, y_n)

        截断误差为 O(h)。用于快速预测缓冲区占用量的时间演化。
        """
        h = t_end / n_steps
        t_vals = [i * h for i in range(n_steps + 1)]
        u_vals = [u0]
        v_vals = [v0]
        u, v = u0, v0
        for i in range(n_steps):
            t = t_vals[i]
            du, dv = self.rhs(t, [u, v])
            u += h * du
            v += h * dv
            # 边界保护：缓冲区占用非负
            u = max(u, 0.0)
            u_vals.append(u)
            v_vals.append(v)
        return t_vals, u_vals, v_vals


class LaxWendroffBuffer:
    """
    Lax-Wendroff 格式模拟数据在排序管线中的对流-扩散过程。

n    将数据流密度 ρ(x,t) 视为双曲型守恒律的解：
        ∂ρ/∂t + c · ∂ρ/∂x = 0

    其中：
        x ∈ [0, L]: 归一化位置（0=外存，1=内存）
        c: 数据对流速度（I/O带宽决定）
        ρ: 数据密度

    Lax-Wendroff 离散格式（二阶精度，稳定性条件 |c·Δt/Δx| ≤ 1）：
        ρ_j^{n+1} = ρ_j^n
                    - ν/2 · (ρ_{j+1}^n - ρ_{j-1}^n)
                    + ν²/2 · (ρ_{j+1}^n - 2ρ_j^n + ρ_{j-1}^n)
        其中 ν = c·Δt/Δx。
    """

    def __init__(self, nx: int, c: float, dx: float, dt: float):
        self.nx = nx
        self.c = c
        self.dx = dx
        self.dt = dt
        self.nu = c * dt / dx
        if abs(self.nu) > 1.0:
            # 稳定性条件不满足时自动调整
            self.nu = math.copysign(1.0, self.nu) * 0.95
            self.dt = self.nu * dx / c if abs(c) > 1e-15 else dt

    def step(self, rho: List[float]) -> List[float]:
        """
        执行一个时间步的 Lax-Wendroff 更新。

        边界条件：采用周期性边界条件（保证数值守恒）。
        """
        if len(rho) != self.nx:
            raise ValueError(f"rho length {len(rho)} != nx {self.nx}")
        rho_new = [0.0] * self.nx
        nu = self.nu
        nu2 = nu * nu
        nx = self.nx

        for j in range(nx):
            jm = (j - 1) % nx
            jp = (j + 1) % nx
            rho_new[j] = (
                rho[j]
                - 0.5 * nu * (rho[jp] - rho[jm])
                + 0.5 * nu2 * (rho[jp] - 2.0 * rho[j] + rho[jm])
            )
        return rho_new

    def simulate(self, rho0: List[float], n_steps: int) -> List[List[float]]:
        """
        模拟 n_steps 个时间步，返回完整演化历史。
        """
        history = [list(rho0)]
        rho = list(rho0)
        for _ in range(n_steps):
            rho = self.step(rho)
            history.append(list(rho))
        return history

    def compute_courant_number(self) -> float:
        """
        Courant 数 C = |c|·Δt/Δx，稳定性要求 C ≤ 1。
        """
        return abs(self.c) * self.dt / self.dx


def predict_optimal_buffer_size(alpha: float, beta: float, gamma: float,
                                omega: float, safety_factor: float = 1.5) -> float:
    """
    基于动脉模型的稳态振幅预测最优缓冲区大小。

    最优缓冲区应能容纳峰值负载：
        B_opt = safety_factor · ( steady_offset + steady_amplitude )
              = safety_factor · ( γ·x·dp/dx·a/α + A )
    """
    if alpha < 1e-15:
        return 1e6
    steady_offset = gamma / alpha
    model = ArteryFlowModel(alpha, beta, gamma, 1.0, 1.0, omega)
    amp = model.analytical_amplitude()
    return safety_factor * (steady_offset + amp)
