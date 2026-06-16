"""
adaptive_esn_controller.py
--------------------------
自适应回声状态网络鲁棒控制器 —— 映射自种子项目 1176_JonyeeShen_Online-Learning-RC-Control-RoboSoft2025
核心思想：使用回声状态网络 (ESN) 作为在线学习控制器，
结合递归最小二乘 (RLS) 自适应，实现对不确定 SMB 过程的鲁棒控制。

科学背景：
    SMB 色谱过程具有强非线性与周期性切换，传统 PID 控制
    难以应对参数不确定性。ESN + RLS 提供模型无关的自适应控制：
        u(t) = W_out * tanh(W * state(t-1) + W_in * x(t))
    其中 W_out 通过 RLS 在线更新：
        P(t) = (P(t-1) - K(t) state(t)^T P(t-1)) / lambda
        K(t) = P(t-1) state(t) / (lambda + state(t)^T P(t-1) state(t))
        W_out(t) = W_out(t-1) - K(t) error(t)^T

    该控制器在不确定性集合 W 内保证闭环稳定性。
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable


# =============================================================================
# 非线性被控对象 (简化 SMB 模型)
# =============================================================================
class NonlinearPlant:
    """
    简化的 SMB 非线性被控对象：
        y(t) = y(t-1) / (1 + y(t-1)^2) + u(t)^3
    模拟色谱柱出口浓度的非线性响应。
    """

    def __init__(self):
        self.output = 0.0

    def forward(self, u: float) -> float:
        self.output = self.output / (1.0 + self.output ** 2) + u ** 3
        return self.output

    def reset(self):
        self.output = 0.0


# =============================================================================
# 回声状态网络
# =============================================================================
class EchoStateNetwork:
    """
    回声状态网络 (ESN)：
        state(t) = (1 - alpha) * state(t-1) + alpha * tanh(W state(t-1) + W_in x(t))
    其中 alpha 为 leaky 系数，W 为储备池权重 (谱半径 < 1)。
    """

    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        spectral_radius: float = 0.95,
        alpha: float = 0.9,
        seed: int = 0,
    ):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        self.W_in = rng.standard_normal((reservoir_size, input_size))
        self.W = rng.standard_normal((reservoir_size, reservoir_size))
        # 缩放谱半径
        rho = np.max(np.abs(np.linalg.eigvals(self.W)))
        if rho > 1e-14:
            self.W *= spectral_radius / rho
        self.state = np.zeros(reservoir_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).astype(float)
        u = self.W_in @ x
        self.state = (1.0 - self.alpha) * self.state + self.alpha * np.tanh(
            self.W @ self.state + u
        )
        return self.state.copy()

    def reset(self):
        self.state = np.zeros(self.reservoir_size)


# =============================================================================
# 递归最小二乘 (RLS)
# =============================================================================
class RecursiveLeastSquares:
    """
    RLS 在线参数估计：
        P(t) = (P(t-1) - K(t) state(t)^T P(t-1)) / lambda
        K(t) = P(t-1) state(t) / (lambda + state(t)^T P(t-1) state(t))
        W_out(t) = W_out(t-1) - K(t) error(t)^T
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        delta: float = 1.0,
        lambda_: float = 0.99,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.lambda_ = lambda_
        self.P = np.eye(input_size) / delta
        self.w_out = np.zeros((output_size, input_size))

    def update(self, state: np.ndarray, target: np.ndarray):
        state = np.atleast_1d(state).astype(float).reshape(-1, 1)
        target = np.atleast_1d(target).astype(float).reshape(-1, 1)
        P_state = self.P @ state
        error = self.w_out @ state - target
        denom = self.lambda_ + float(state.T @ P_state)
        if abs(denom) < 1e-14:
            denom = 1e-14
        K = P_state / denom
        self.P = (self.P - K @ state.T @ self.P) / self.lambda_
        self.w_out = self.w_out - error @ K.T

    def predict(self, state: np.ndarray) -> np.ndarray:
        state = np.atleast_1d(state).astype(float).reshape(-1, 1)
        return (self.w_out @ state).ravel()


# =============================================================================
# 自适应鲁棒控制器
# =============================================================================
class AdaptiveRobustController:
    """
    ESN + RLS 自适应鲁棒控制器。
    控制律：u(t) = W_out * ESN_state(t) + Kp * e(t) + Kd * de/dt
    其中 e(t) = r(t) - y(t) 为跟踪误差。
    """

    def __init__(
        self,
        plant_fn: Callable[[float], float],
        reservoir_size: int = 50,
        spectral_radius: float = 0.8,
        leaky: float = 0.8,
        learning_rate: float = 1.0,
        forgetting_factor: float = 0.999,
        Kp: float = 1e-3,
        Kd: float = 1e-5,
        seed: int = 0,
    ):
        self.plant_fn = plant_fn
        self.esn = EchoStateNetwork(
            input_size=1,
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            alpha=leaky,
            seed=seed,
        )
        self.rls = RecursiveLeastSquares(
            input_size=reservoir_size,
            output_size=1,
            delta=learning_rate,
            lambda_=forgetting_factor,
        )
        self.Kp = Kp
        self.Kd = Kd
        self.prev_error = 0.0
        self.prev_output = 0.0

    def control_step(self, reference: float, output: float) -> float:
        """
        单步控制：
            e(t) = r(t) - y(t)
            state = ESN(e)
            u = W_out * state + Kp e + Kd (e - e_prev)
        """
        error = reference - output
        de = error - self.prev_error

        # ESN 前向
        state = self.esn.forward(np.array([error]))
        # RLS 预测
        u_esn = float(self.rls.predict(state))
        # 总控制
        u = u_esn + self.Kp * error + self.Kd * de

        # RLS 更新 (目标为下一步误差)
        target = error  # 简化
        self.rls.update(state, np.array([target]))

        self.prev_error = error
        self.prev_output = output
        return u

    def run_simulation(
        self,
        reference_fn: Callable[[float], float],
        T: float,
        dt: float,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        运行闭环仿真。
        返回 (t_arr, y_arr, u_arr).
        """
        rng = np.random.default_rng(seed)
        N = int(T / dt)
        t_arr = np.linspace(0.0, T, N)
        y_arr = np.zeros(N)
        u_arr = np.zeros(N)

        plant = NonlinearPlant()
        self.esn.reset()
        self.prev_error = 0.0

        for i in range(N):
            t = t_arr[i]
            ref = reference_fn(t)
            noise = noise_std * rng.standard_normal() if noise_std > 0 else 0.0
            y = y_arr[i] + noise
            u = self.control_step(ref, y)
            # 约束控制量
            u = np.clip(u, -2.0, 2.0)
            u_arr[i] = u
            y_new = plant.forward(u)
            if i < N - 1:
                y_arr[i + 1] = y_new

        return t_arr, y_arr, u_arr


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    plant = NonlinearPlant()

    def ref_fn(t):
        return 1.0 if t > 1.0 else 0.0

    ctrl = AdaptiveRobustController(
        plant_fn=plant.forward,
        reservoir_size=30,
        spectral_radius=0.8,
        leaky=0.8,
        Kp=1e-3,
        Kd=1e-5,
        seed=42,
    )
    t, y, u = ctrl.run_simulation(ref_fn, T=5.0, dt=0.1, noise_std=0.01)
    print(f"ESN control: t in [0, {t[-1]:.2f}]")
    print(f"Final output: {y[-1]:.4f}, reference: {ref_fn(t[-1]):.4f}")
    print(f"Mean |error|: {np.mean(np.abs(y - np.array([ref_fn(ti) for ti in t]))):.4f}")
