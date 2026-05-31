"""
对流动力学反应-扩散模块 (Convective Reaction-Diffusion Dynamics)

集成种子项目:
- 1134_spiral_pde: Barkley 型反应-扩散系统, 9 点 Laplacian 模板

科学背景:
  中尺度对流系统 (MCS) 中的对流单体 (convective cell) 可以被视为
  一类反应-扩散系统中的激发波 (excitation wave).
  这里将 Barkley 模型改写为适用于湿对流的动力学框架:

  快变量 U 对应对流活动度 (convective activity / updraft strength)
  慢变量 V 对应环境湿度/不稳定度积累 (moisture buildup / CAPE accumulation)

  控制方程:
    ∂U/∂t = D_u ∇²U + (1/ε) * U * (1-U) * (U - (V+β)/α)
    ∂V/∂t = D_v ∇²V + U - V

  其中:
    - U ∈ [0,1]: 归一化对流强度
    - V ∈ [0,1]: 归一化环境湿度/不稳定度
    - ε << 1: 时间尺度分离参数
    - α, β: 激发介质参数
    - D_u, D_v: 扩散系数 (对应动量/水汽的水平扩散)

  在 MCS 预报中, 该模型用于刻画:
    - 对流的触发 (initiation)
    - 冷池传播 (cold pool propagation) — 对应 V 的前沿
    - 螺旋/线状对流组织的形成
"""

import numpy as np
from typing import Tuple


class ConvectionDynamics:
    """
    中尺度对流系统反应-扩散动力学求解器.
    """

    def __init__(self, nx: int = 128, ny: int = 128, dx: float = 2000.0,
                 alpha: float = 0.25, beta: float = 0.001,
                 delta: float = 1e-5, epsilon: float = 0.002,
                 Du: float = 5.0e3, Dv: float = 1.0e2):
        """
        参数:
          nx, ny: 水平格点数
          dx: 水平格距 (m), 默认 2 km
          alpha, beta, delta, epsilon: Barkley 模型标准参数
          Du, Dv: 扩散系数 (m²/s), 经尺度变换后使用
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        # 扩散系数经 dx² 无量纲化 (用于离散 Laplacian)
        self.Du = Du
        self.Dv = Dv

        self.U = np.zeros((ny, nx))
        self.V = np.zeros((ny, nx))
        self._set_initial_condition()

    def _set_initial_condition(self):
        """
        初始条件: 在对流触发线 (x >= 0.6*nx) 上设置高 U,
        在湿度前沿 (y >= 0.6*ny) 上设置高 V.
        模拟 MCS 初期的对流线触发.
        """
        nx, ny = self.nx, self.ny
        # 高斯型初始对流单体
        cx, cy = nx // 3, ny // 2
        for j in range(ny):
            for i in range(nx):
                r2 = ((i - cx) / (nx / 8.0))**2 + ((j - cy) / (ny / 8.0))**2
                self.U[j, i] = 0.8 * np.exp(-r2)
                self.V[j, i] = 0.5 * self.alpha * (1.0 + 0.3 * np.sin(2.0 * np.pi * i / nx))
        # 边界保护
        self.U = np.clip(self.U, 0.0, 1.0)
        self.V = np.clip(self.V, 0.0, 1.0)

    def _laplacian_9point(self, A: np.ndarray) -> np.ndarray:
        """
        9 点 Laplacian 模板 (来自 1134_spiral_pde),
        带有周期性边界条件 (模拟无限域).

        模板:
          (1/(6*dx²)) * [[1, 4, 1], [4, -20, 4], [1, 4, 1]]
        """
        ny, nx = A.shape
        L = np.zeros_like(A)
        coeff = 1.0 / (6.0 * self.dx * self.dy)
        for j in range(ny):
            for i in range(nx):
                jp = (j + 1) % ny
                jm = (j - 1) % ny
                ip = (i + 1) % nx
                im = (i - 1) % nx
                L[j, i] = coeff * (
                    A[jm, im] + 4.0 * A[jm, i] + A[jm, ip]
                    + 4.0 * A[j, im] - 20.0 * A[j, i] + 4.0 * A[j, ip]
                    + A[jp, im] + 4.0 * A[jp, i] + A[jp, ip]
                )
        return L

    def _rhs(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算反应-扩散方程的右端项.
        """
        lapU = self._laplacian_9point(U)
        lapV = self._laplacian_9point(V)

        # 反应项
        reaction_U = (1.0 / self.epsilon) * U * (1.0 - U) * (U - (V + self.beta) / self.alpha)
        reaction_V = U - V

        dUdt = self.Du * lapU + reaction_U
        dVdt = self.Dv * lapV + self.delta * lapV + reaction_V
        return dUdt, dVdt

    def step_rk4(self, dt: float):
        """
        使用四阶 Runge-Kutta 时间积分.
        """
        U0 = self.U.copy()
        V0 = self.V.copy()

        k1_U, k1_V = self._rhs(U0, V0)
        k2_U, k2_V = self._rhs(U0 + 0.5 * dt * k1_U, V0 + 0.5 * dt * k1_V)
        k3_U, k3_V = self._rhs(U0 + 0.5 * dt * k2_U, V0 + 0.5 * dt * k2_V)
        k4_U, k4_V = self._rhs(U0 + dt * k3_U, V0 + dt * k3_V)

        self.U = U0 + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)
        self.V = V0 + (dt / 6.0) * (k1_V + 2.0 * k2_V + 2.0 * k3_V + k4_V)

        # 边界截断与数值保护
        self.U = np.clip(self.U, 0.0, 1.0)
        self.V = np.clip(self.V, 0.0, 1.0)
        # 抑制数值噪声
        if np.any(~np.isfinite(self.U)) or np.any(~np.isfinite(self.V)):
            self.U = np.nan_to_num(self.U, nan=0.0, posinf=1.0, neginf=0.0)
            self.V = np.nan_to_num(self.V, nan=0.0, posinf=1.0, neginf=0.0)

    def integrate(self, dt: float, nsteps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        积分 nsteps 步, 返回最终 (U, V).
        """
        for _ in range(nsteps):
            self.step_rk4(dt)
        return self.U.copy(), self.V.copy()

    def get_convective_intensity(self) -> np.ndarray:
        """
        返回对流强度场 (可用于后续降水估计).
        """
        return self.U.copy()

    def get_moisture_accumulation(self) -> np.ndarray:
        """
        返回环境湿度/不稳定度积累场.
        """
        return self.V.copy()

    def total_convective_energy(self) -> float:
        """
        域积分对流能量指标.
        """
        return float(np.sum(self.U**2) * self.dx * self.dy)
