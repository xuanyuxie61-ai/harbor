"""
dynamical_system.py

非线性动力学环境模型

基于种子项目:
  - 1059_sawtooth_ode: 锯齿波驱动的谐振子
  - 488_grazing_ode: 放牧捕食者-猎物模型 (非线性种群动力学)

科学问题:
  将两种 ODE 系统融合为一个受控非线性振荡网络,
  代表一类具有周期强迫与状态依赖阻尼的物理系统
  (如等离子体约束中的粒子轨道、海洋生态系统的季节强迫).

  状态方程:
      dx_1/dt = x_2 + u_1                           (位置速度耦合)
      dx_2/dt = -ω_0^2 x_1 + S(t) + F_graze(x) + u_2  (受迫振动)
      dx_3/dt = r_1 x_3 (1 - x_3/k) - c_1 x_4 (1 - exp(-d_1 x_3)) + u_3
      dx_4/dt = -a x_4 + c_2 x_4 (1 - exp(-d_2 x_3)) + u_4

  其中 S(t) 为归一化锯齿波:
      S(t) = 2·( (ω_s t / 2π) - floor(ω_s t / 2π + 1/2) )

  F_graze(x) 为状态耦合的放牧型非线性项:
      F_graze(x) = -γ · x_1 · x_3 / (1 + x_3^2)

  控制目标: 使系统状态跟踪参考轨迹 x*(t) 同时最小化控制能量.
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# 锯齿波驱动函数
# ---------------------------------------------------------------------------

def sawtooth_wave(t: float, omega: float = 2.0 * np.pi) -> float:
    """
    归一化锯齿波: S(t) = 2·( frac - 0.5 ), frac = (ωt/2π) mod 1.

    Fourier 展开:
        S(t) = -(2/π) Σ_{n=1}^∞ (-1)^n sin(n ω t) / n
    """
    if not np.isfinite(t):
        return 0.0
    frac = (omega * t / (2.0 * np.pi)) % 1.0
    return 2.0 * (frac - 0.5)


# ---------------------------------------------------------------------------
# 放牧动力学参数与耦合项
# ---------------------------------------------------------------------------

GRAZING_PARAMS = {
    'a': 0.2,      # 捕食者死亡率
    'c1': 0.05,    # 捕食率系数 1
    'c2': 0.05,    # 捕食者增长系数
    'd1': 2.0,     # 功能反应半饱和 1
    'd2': 2.0,     # 功能反应半饱和 2
    'k': 2.0,      # 猎物环境容纳量
    'r1': 0.5,     # 猎物内禀增长率
    'gamma': 0.1,  # 振荡-生态耦合强度
}


def grazing_coupling(x1: float, x3: float, params: dict = None) -> float:
    """
    振荡子与生态模块之间的状态耦合非线性项.

    物理动机:
        在海洋生态-气候耦合系统中,
        浮游生物种群密度 x_3 通过改变海水粘滞性影响振荡幅度 x_1.
    """
    if params is None:
        params = GRAZING_PARAMS
    gamma = params['gamma']
    # 数值鲁棒性: 限制 x3 避免溢出
    x3_clipped = float(np.clip(x3, -100.0, 100.0))
    denom = 1.0 + x3_clipped ** 2
    if not np.isfinite(denom) or denom == 0.0:
        return 0.0
    return -gamma * x1 * x3_clipped / denom


# ---------------------------------------------------------------------------
# 受控非线性动力学系统
# ---------------------------------------------------------------------------

class ControlledNonlinearOscillator:
    """
    受控非线性振荡网络环境.

    状态空间: s = [x_1, x_2, x_3, x_4] ∈ R^4
    动作空间: a = [u_1, u_2, u_3, u_4] ∈ R^4
    """

    def __init__(self, omega0: float = 1.0, omega_s: float = 2.0 * np.pi,
                 dt: float = 0.005, params: dict = None,
                 state_bounds: tuple = (-10.0, 10.0)):
        self.omega0 = omega0
        self.omega_s = omega_s
        self.dt = dt
        self.params = params if params is not None else GRAZING_PARAMS.copy()
        self.state_bounds = state_bounds
        self.state = np.zeros(4)
        self.t = 0.0
        self.step_count = 0

    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        """重置环境到初始状态."""
        if initial_state is None:
            self.state = np.random.randn(4) * 0.1
        else:
            self.state = np.array(initial_state, dtype=float).copy()
        self.t = 0.0
        self.step_count = 0
        return self._get_observation()

    def _dynamics(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        """
        连续时间右端项 f(s, a, t).
        """
        # 数值鲁棒性: 先 clip 状态防止中间计算溢出
        state = np.clip(np.asarray(state, dtype=float), -50.0, 50.0)
        x1, x2, x3, x4 = state
        u1, u2, u3, u4 = action
        p = self.params

        # 振荡子部分 (sawtooth driven harmonic oscillator)
        dx1 = x2 + u1
        dx2 = -(self.omega0 ** 2) * x1 + sawtooth_wave(t, self.omega_s) \
               + grazing_coupling(x1, x3, p) + u2

        # 放牧生态部分 (grazing predator-prey)
        exp_arg1 = np.clip(-p['d1'] * x3, -50.0, 50.0)
        exp_arg2 = np.clip(-p['d2'] * x3, -50.0, 50.0)
        dx3 = p['r1'] * x3 * (1.0 - x3 / p['k']) \
              - p['c1'] * x4 * (1.0 - np.exp(exp_arg1)) + u3
        dx4 = -p['a'] * x4 + p['c2'] * x4 * (1.0 - np.exp(exp_arg2)) + u4

        # 最终导数截断
        deriv = np.array([dx1, dx2, dx3, dx4])
        deriv = np.clip(deriv, -100.0, 100.0)
        return deriv

    def step(self, action: np.ndarray, integrator: str = 'rk4') -> tuple:
        """
        执行一步环境演化.

        返回: (observation, reward, done, info)
        """
        action = np.clip(np.asarray(action, dtype=float), -2.0, 2.0)

        if integrator == 'rk4':
            self.state = self._rk4_step(self.state, action, self.t, self.dt)
        elif integrator == 'euler':
            self.state = self.state + self.dt * self._dynamics(self.state, action, self.t)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # 边界截断
        self.state = np.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        self.t += self.dt
        self.step_count += 1

        obs = self._get_observation()
        reward = self._compute_reward(action)
        done = self._check_done()
        info = {'t': self.t, 'step': self.step_count}
        return obs, reward, done, info

    def _rk4_step(self, state: np.ndarray, action: np.ndarray, t: float, h: float) -> np.ndarray:
        """经典四阶 Runge-Kutta 积分."""
        k1 = self._dynamics(state, action, t)
        k2 = self._dynamics(state + 0.5 * h * k1, action, t + 0.5 * h)
        k3 = self._dynamics(state + 0.5 * h * k2, action, t + 0.5 * h)
        k4 = self._dynamics(state + h * k3, action, t + h)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_observation(self) -> np.ndarray:
        """获取带噪声的观测."""
        noise = np.random.randn(4) * 0.02
        return self.state + noise

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        奖励函数设计 (结合控制理论与特殊函数).

        R(s,a) = -0.5·||s||^2 - 0.1·||a||^2 + 0.5·Si(||s||)·exp(-||a||^2/4)

        其中 Si(x) 为正弦积分, 提供对小幅状态的额外鼓励 (软饱和).
        """
        from special_functions import sine_integral
        s_norm = np.linalg.norm(self.state)
        a_norm = np.linalg.norm(action)
        # 数值鲁棒性检查
        if not np.isfinite(s_norm):
            s_norm = 0.0
        if not np.isfinite(a_norm):
            a_norm = 0.0
        si_term = sine_integral(min(float(s_norm), 50.0))
        reward = -0.5 * s_norm ** 2 - 0.1 * a_norm ** 2 + 0.5 * si_term * np.exp(-a_norm ** 2 / 4.0)
        return float(reward)

    def _check_done(self) -> bool:
        """终止条件."""
        if self.step_count >= 500:
            return True
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            return True
        return False

    def reference_trajectory(self, t: float) -> np.ndarray:
        """
        参考轨迹 x*(t) —— 基于 Bessel 函数的准周期轨道.

        x*_1(t) = J_0(ω_0 t) · cos(ω_s t)
        x*_2(t) = d/dt x*_1(t)
        x*_3(t) = 0.5 + 0.3 sin(0.5 t)
        x*_4(t) = 0.2 + 0.1 cos(0.3 t)
        """
        from scipy.special import jv
        x1 = jv(0, self.omega0 * t) * np.cos(self.omega_s * t)
        # 数值微分近似
        eps = 1.0e-6
        x1_p = jv(0, self.omega0 * (t + eps)) * np.cos(self.omega_s * (t + eps))
        x2 = (x1_p - x1) / eps
        x3 = 0.5 + 0.3 * np.sin(0.5 * t)
        x4 = 0.2 + 0.1 * np.cos(0.3 * t)
        return np.array([x1, x2, x3, x4])
