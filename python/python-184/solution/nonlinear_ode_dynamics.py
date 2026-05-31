"""
Nonlinear Biochemical ODE Dynamics for Time Series Modeling
===========================================================
源自种子项目 091_biochemical_nonlinear_ode (Biochemical reaction network ODEs)。

生化反应网络可以视为一类特殊的非线性动力系统，
其 stoichiometric 形式为：
    dy/dt = S · r(y)
其中 S 为 stoichiometric 矩阵（m×n），r(y) 为反应速率向量（n×1）。

本项目将其扩展为一般非线性 ODE 建模工具，用于：
1. 模拟具有饱和动力学（Michaelis-Menten）的非线性时间序列
2. 通过守恒量检验数值精度
3. 作为基准非线性动力学用于异常检测（真实轨迹 vs 模型轨迹）

核心反应（四物种生化网络）：
    E + S ⇌ ES → E + P
其中：
    y = [E, S, ES, P]^T
    反应速率 r_1 = k_f [E][S]        (结合)
    反应速率 r_2 = k_r [ES]          (解离)
    反应速率 r_3 = k_cat [ES]        (催化)

扩展到更一般的 nonlinear oscillator（扩展 Brusselator / 生化振荡器）：
    dx_1/dt = a - (b+1) x_1 + x_1^2 x_2 + D_1 ∇²x_1
    dx_2/dt = b x_1 - x_1^2 x_2 + D_2 ∇²x_2
这是经典的 Turing 不稳定性模型，可产生时间振荡。
"""

import numpy as np
from typing import Callable


class BiochemicalODE:
    """
    四物种生化反应网络的数值积分。
    """

    def __init__(self, kf: float = 1.0, kr: float = 0.1, kcat: float = 0.5):
        self.kf = kf
        self.kr = kr
        self.kcat = kcat

        # Stoichiometric 矩阵 S (4 species x 3 reactions)
        # 反应: R1: E+S -> ES,  R2: ES -> E+S,  R3: ES -> E+P
        self.S = np.array([
            [-1,  1,  1],   # E
            [-1,  1,  0],   # S
            [ 1, -1, -1],   # ES
            [ 0,  0,  1]    # P
        ])

    def reaction_rates(self, y: np.ndarray) -> np.ndarray:
        """
        计算反应速率 r = [r1, r2, r3]^T。
        引入 Michaelis-Menten 饱和修正：
            r1 = kf * E * S / (1 + alpha * ES)
        其中 alpha 表示产物抑制效应。
        """
        E, S, ES, P = y
        alpha = 0.05
        r1 = self.kf * E * S / (1.0 + alpha * ES)
        r2 = self.kr * ES
        r3 = self.kcat * ES / (1.0 + 0.1 * P)  # 产物抑制
        return np.array([r1, r2, r3])

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """dy/dt = S · r(y)"""
        r = self.reaction_rates(y)
        return self.S @ r

    def conserved_quantities(self, y: np.ndarray) -> np.ndarray:
        """
        守恒量 h = E^T · y，其中 E 的列张成 S 的左零空间。
        对于本系统：
            h1 = E + ES         (总酶守恒)
            h2 = S + ES + P     (总底物守恒)
        """
        E, S, ES, P = y
        h1 = E + ES
        h2 = S + ES + P
        return np.array([h1, h2])

    def integrate_rk4(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        经典四阶 Runge-Kutta 积分。
        """
        if len(y0) != 4:
            raise ValueError("y0 must have length 4.")
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 4))
        y[0] = y0

        h0 = self.conserved_quantities(y0)

        for i in range(n_steps):
            yi = y[i]
            k1 = self.rhs(t[i], yi)
            k2 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k1)
            k3 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k2)
            k4 = self.rhs(t[i] + dt, yi + dt * k3)
            y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 数值守恒修正（投影法）：在守恒流形上投影
            h = self.conserved_quantities(y[i + 1])
            dh = h - h0
            # 简单投影：调整 E 和 S 以恢复守恒
            # 这里使用微小修正，保持物理合理性
            y[i + 1][0] -= dh[0] * 0.5
            y[i + 1][2] += dh[0] * 0.5
            y[i + 1][1] -= dh[1] * 0.3
            y[i + 1][2] -= dh[1] * 0.3
            y[i + 1][3] -= dh[1] * 0.4

            # 非负性约束
            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return t, y


class ExtendedBrusselator:
    """
    扩展 Brusselator 模型，用于模拟具有 Hopf 分岔的非线性振荡时间序列。
    """

    def __init__(self, a: float = 1.0, b: float = 3.0, D1: float = 0.01, D2: float = 0.01):
        self.a = a
        self.b = b
        self.D1 = D1
        self.D2 = D2

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        y = [x1, x2]
        dx1/dt = a - (b+1) x1 + x1^2 x2
        dx2/dt = b x1 - x1^2 x2
        """
        x1, x2 = y
        dx1 = self.a - (self.b + 1.0) * x1 + x1 ** 2 * x2
        dx2 = self.b * x1 - x1 ** 2 * x2
        return np.array([dx1, dx2])

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        """
        雅可比矩阵，用于稳定性分析与 Lyapunov 指数计算。
        J = [[∂f1/∂x1, ∂f1/∂x2],
             [∂f2/∂x1, ∂f2/∂x2]]
        """
        x1, x2 = y
        df1_dx1 = -(self.b + 1.0) + 2.0 * x1 * x2
        df1_dx2 = x1 ** 2
        df2_dx1 = self.b - 2.0 * x1 * x2
        df2_dx2 = -x1 ** 2
        return np.array([[df1_dx1, df1_dx2],
                         [df2_dx1, df2_dx2]])

    def integrate(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """RK4 积分。"""
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = y0
        for i in range(n_steps):
            yi = y[i]
            k1 = self.rhs(t[i], yi)
            k2 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k1)
            k3 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k2)
            k4 = self.rhs(t[i] + dt, yi + dt * k3)
            y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, y

    def lyapunov_exponent_numerical(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> float:
        """
        数值计算最大 Lyapunov 指数。
        λ_max = lim_{t→∞} (1/t) ln(||δx(t)|| / ||δx(0)||)
        这里使用有限时间近似。
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        y = y0.copy()
        delta = np.array([1e-8, 0.0])
        norm_delta0 = np.linalg.norm(delta)
        lyap_sum = 0.0
        count = 0

        for i in range(n_steps):
            # 主轨道
            k1 = self.rhs(t0 + i * dt, y)
            k2 = self.rhs(t0 + i * dt + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t0 + i * dt + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t0 + i * dt + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 扰动轨道
            J = self.jacobian(y)
            # 线性化演化：delta' = J delta
            delta = delta + dt * (J @ delta)
            norm_d = np.linalg.norm(delta)
            if norm_d > 1e-6:
                lyap_sum += np.log(norm_d / norm_delta0)
                delta = delta / norm_d * norm_delta0
                count += 1

        if count == 0:
            return 0.0
        return lyap_sum / (count * dt)
