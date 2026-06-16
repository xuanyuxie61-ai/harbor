# -*- coding: utf-8 -*-
"""
rate_integrator.py
==================

反冲率 ODE 积分器与年调制效应

对应种子项目:
  - 100_blood_pressure_ode: 带周期性的 ODE 系统
  - 1166_Hybrid-Quantum-Classical-Reservoir-Computing: ESN 动态系统 + 种子复现

科学公式
--------
年调制信号:
  dR/dE_R(t) = S_0(E_R) + S_m(E_R) cos(ω(t - t_0))

其中:
  S_0  = 年平均值
  S_m  = 调制振幅 (~2-5% of S_0)
  ω    = 2π / T_year
  t_0  = ~ June 2nd (最大速率)

调制振幅:
  S_m/S_0 ≈ 2 v_orb v_0 / (v_0² + v_E²) ≈ 0.03 - 0.05

年调制相位:
  Δt_max ≈ 152.5 天 (春分后 ~ June 2)

ESN 代理模型 (源自 1166):
  x_{t+1} = (1 - α) x_t + α tanh(W_in u_t + W_esn x_t)
  y_t = W_out x_t
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from recoil_physics import differential_rate, recoil_spectrum
from astro_parameters import StandardHaloModel, get_default_shm


# ---------------------------------------------------------------------------
# 第一部分: 率 ODE (源自 100_blood_pressure_ode)
# ---------------------------------------------------------------------------

class RateODE:
    """
    暗物质反冲率的 ODE 系统。

    类比血压 ODE: dP/dt = -P/(C R) + Q(t)/C
    DM 版本:     dR/dt = -Γ R + S(t)

    其中:
      Γ = 散射率 (DM-核 碰撞频率)
      S(t) = 时变源项 (含年调制)
      R(t) = 反冲率

    周期性驱动:
      S(t) = S_0 + S_m cos(ω(t - t_0))
    """

    def __init__(
        self,
        m_chi: float = 100.0,
        A: int = 131,
        sigma_n: float = 1e-44,
        gamma: float = 1e-3,  # s^{-1}
        shm: Optional[StandardHaloModel] = None,
    ):
        self.m_chi = m_chi
        self.A = A
        self.sigma_n = sigma_n
        self.gamma = gamma
        self.shm = shm or get_default_shm()
        self.omega = 2.0 * np.pi / (365.25 * 86400.0)  # rad/s
        self.t_0 = 152.5 * 86400.0  # 秒

    def source(self, E_R_keV: float, t_s: float) -> float:
        """
        时变源项 S(E_R, t)。

        包含年调制:
          S(E_R, t) = S_0(E_R) * (1 + A_m cos(ω(t - t_0)))

        调制振幅 A_m 近似:
          A_m ≈ 2 v_orb / v_0 * cos(γ)
        """
        # 平均速率
        S_0 = differential_rate(
            E_R_keV, self.m_chi, self.A, self.sigma_n,
            self.shm, day=152.5,
        )
        # 调制振幅 (经验值)
        A_m = 2.0 * self.shm.v_orb / self.shm.v_0 * np.cos(self.shm.gamma_angle)
        return S_0 * (1.0 + A_m * np.cos(self.omega * (t_s - self.t_0)))

    def rhs(self, t_s: float, R: np.ndarray, E_R_grid: np.ndarray) -> np.ndarray:
        """
        ODE 右端项:
          dR_i/dt = -Γ R_i + S(E_i, t)

        R 是各能量格点的率向量。
        """
        S = np.array([self.source(E, t_s) for E in E_R_grid])
        return -self.gamma * R + S

    def integrate_rk4(
        self,
        E_R_grid: np.ndarray,
        t_start: float,
        t_end: float,
        dt: float = 86400.0,
        R0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 积分 ODE 系统。

        返回: (time_array, rate_matrix[time, energy])
        """
        n_E = len(E_R_grid)
        if R0 is None:
            R0 = np.zeros(n_E)
        n_steps = int(np.ceil((t_end - t_start) / dt))
        times = np.linspace(t_start, t_end, n_steps + 1)
        rates = np.zeros((n_steps + 1, n_E))
        rates[0] = R0.copy()

        R = R0.copy()
        for i in range(n_steps):
            t = times[i]
            k1 = self.rhs(t, R, E_R_grid)
            k2 = self.rhs(t + 0.5*dt, R + 0.5*dt*k1, E_R_grid)
            k3 = self.rhs(t + 0.5*dt, R + 0.5*dt*k2, E_R_grid)
            k4 = self.rhs(t + dt, R + dt*k3, E_R_grid)
            R = R + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            # 非负约束
            R = np.maximum(R, 0.0)
            rates[i + 1] = R

        return times, rates


# ---------------------------------------------------------------------------
# 第二部分: 年调制分析
# ---------------------------------------------------------------------------

def annual_modulation_analysis(
    E_R_keV: float,
    m_chi: float,
    A: int,
    sigma_n: float,
    shm: Optional[StandardHaloModel] = None,
    n_days: int = 365,
) -> Dict[str, float]:
    """
    计算年调制信号参数。

    对每个 day ∈ [1, 365]:
      rate(day) = differential_rate(E_R, m_χ, A, σ_n, shm, day)

    拟合:
      rate(day) = S_0 + S_m cos(2π(day - t_0)/365)

    返回: {S_0, S_m, t_0_fit, modulation_fraction}
    """
    shm = shm or get_default_shm()
    days = np.arange(1, n_days + 1)
    rates = np.array([
        differential_rate(E_R_keV, m_chi, A, sigma_n, shm, day=d)
        for d in days
    ])

    # 简单傅里叶分析
    S_0 = np.mean(rates)
    omega = 2.0 * np.pi / n_days
    t_0_approx = 152.5  # 天
    cos_term = np.cos(omega * (days - t_0_approx))
    S_m = 2.0 * np.mean(rates * cos_term)

    mod_fraction = abs(S_m / S_0) if S_0 > 0 else 0.0

    return {
        'S_0': float(S_0),
        'S_m': float(S_m),
        't_0': float(t_0_approx),
        'modulation_fraction': float(mod_fraction),
        'rates': rates,
        'days': days,
    }


# ---------------------------------------------------------------------------
# 第三部分: ESN 代理模型 (源自 1166 Hybrid Reservoir)
# ---------------------------------------------------------------------------

class EchoStateRecoilModel:
    """
    Echo State Network 用于快速近似反冲率计算 (源自 1166)。

    结构:
      x_{t+1} = (1 - α) x_t + α tanh(W_in [u_t; y_t] + W_esn x_t)
      y_{t+1} = W_out x_t

    应用于反冲率预测:
      u_t = (E_R, m_χ, σ_n, ...) 输入参数
      y_t = dR/dE_R 输出率
    """

    def __init__(
        self,
        N_reservoir: int = 100,
        spectral_radius: float = 0.9,
        input_gain: float = 0.1,
        leakage: float = 0.3,
        seed: int = 42,
    ):
        self.N = N_reservoir
        self.rho = spectral_radius
        self.lam = input_gain
        self.alpha = leakage
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化 ESN 权重 (单位谱半径归一化)。"""
        self.W_in = self.rng.uniform(-1, 1, (self.N, 4))
        self.W_in *= self.lam
        self.W_esn = self.rng.uniform(-1, 1, (self.N, self.N))
        # 谱半径归一化
        rho = np.max(np.abs(np.linalg.eigvals(self.W_esn)))
        self.W_esn *= self.rho / max(rho, 1e-10)
        self.W_out = None  # 训练后填充
        self.x = np.zeros(self.N)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """tanh 激活 (带 clip 防止溢出)。"""
        return np.tanh(np.clip(x, -20, 20))

    def _step(self, u: np.ndarray) -> np.ndarray:
        """单步 reservoir 更新。"""
        x_new = (1 - self.alpha) * self.x + self.alpha * self._activation(
            self.W_in @ u + self.W_esn @ self.x
        )
        self.x = x_new
        return x_new

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        washout: int = 50,
    ):
        """
        训练 ESN: 用线性回归确定 W_out。

        inputs: shape (T, d_in)
        targets: shape (T,)
        """
        T = len(targets)
        states = np.zeros((T, self.N))
        self.x = np.zeros(self.N)
        for t in range(T):
            self._step(inputs[t])
            states[t] = self.x.copy()

        # 使用 washout 后的状态
        states_wash = states[washout:]
        targets_wash = targets[washout:]

        # 岭回归
        reg = 1e-6
        A = states_wash.T @ states_wash + reg * np.eye(self.N)
        b = states_wash.T @ targets_wash
        self.W_out = np.linalg.solve(A, b)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """预测 (需要已训练)。"""
        if self.W_out is None:
            raise RuntimeError("ESN 尚未训练")
        T = len(inputs)
        outputs = np.zeros(T)
        self.x = np.zeros(self.N)
        for t in range(T):
            self._step(inputs[t])
            outputs[t] = self.W_out @ self.x
        return outputs


# ---------------------------------------------------------------------------
# 第四部分: 种子复现控制 (源自 1166 BaseSeededClass)
# ---------------------------------------------------------------------------

class SeededExperiment:
    """
    确保实验可复现的种子控制 (源自 1166 BaseSeededClass)。

    固定所有随机源:
      - numpy.random
      - 探测器噪声
      - 蒙特卡罗采样
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # 设置全局种子 (不影响其他模块)
        np.random.seed(seed)

    def generate_noise(self, shape: tuple, sigma: float) -> np.ndarray:
        """生成高斯噪声 (确定性)。"""
        return self.rng.normal(0, sigma, size=shape)

    def poisson_counts(self, rate: np.ndarray) -> np.ndarray:
        """生成泊松计数。"""
        return self.rng.poisson(np.maximum(rate, 0.0)).astype(float)
