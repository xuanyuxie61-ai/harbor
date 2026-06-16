"""
日历转换与多周期耦合模块 (Calendar Conversion & Multi-Period Coupling)
========================================================================
受日历转换系统启发, 实现不同随机周期基之间的转换算法。

日历系统的数学结构:
  不同日历系统通过 Julian Epoch Date (JED) 中间表示互相转换:
    Calendar_A → JED → Calendar_B

  这构成了一个群作用:
    g_{A→B} = g_{JED→B} ∘ g_{A→JED}
    g_{A→B}^{-1} = g_{B→A}
    g_{A→A} = id

类比到随机配置:
  不同时间尺度的随机过程通过 canonical 时间尺度转换:
    Process_A(t_A) → canonical_time → Process_B(t_B)

  例如:
    - 快变过程 (化学反应): t_fast ~ ms
    - 慢变过程 (热传导): t_slow ~ s
    - 耦合: t_canonical = (t_fast + t_slow) / 2

多尺度随机配置:
  将不同时间尺度的随机过程统一到一个配置框架:
    u(t, ω) = u_fast(t_fast, ω_fast) × u_slow(t_slow, ω_slow)

  配置策略:
    - 快变维度: 高精度配置 (更多点)
    - 慢变维度: 低精度配置 (更少点)
    - 耦合: 通过 canonical 表示同步

日期算术在随机过程中的应用:
  1. 周期性边界条件: 解在周期边界上连续
  2. 相位计算: 不同频率的相对相位
  3. 周期平均: 对周期分量求平均得到长期行为
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


class MultiScaleCoupling:
    """
    多尺度随机过程耦合器。

    处理具有不同时间尺度的随机过程:
      fast process: τ = t / ε  (快变)
      slow process: T = t       (慢变)

    耦合模型:
      du/dt = f_fast(u, τ, ω_fast) + f_slow(u, T, ω_slow)

    均匀化理论 (Homogenization):
      当 ε → 0:
        u(t) → ū(t) (均匀化解)
        dū/dt = f̄_slow(ū, T)
      其中 f̄_slow = <f_slow>_{τ} (对快变时间平均)
    """

    def __init__(
        self,
        n_fast_steps: int = 100,
        n_slow_steps: int = 10,
        epsilon: float = 0.01
    ):
        """
        参数:
            n_fast_steps: 快变时间步数 (每个慢变步内)
            n_slow_steps: 慢变时间步数
            epsilon: 尺度分离参数 ε
        """
        self.n_fast = n_fast_steps
        self.n_slow = n_slow_steps
        self.epsilon = epsilon

    def fast_oscillation(
        self, t: np.ndarray, frequency: float, amplitude: float = 1.0
    ) -> np.ndarray:
        """
        快变振荡:
          f(t) = A sin(2π t / ε)

        模拟化学反应中的快速振荡动力学。
        """
        return amplitude * np.sin(2.0 * np.pi * t / self.epsilon * frequency)

    def slow_envelope(
        self, t: np.ndarray, decay_rate: float = 0.5
    ) -> np.ndarray:
        """
        慢变包络:
          g(t) = exp(-decay_rate × t)

        模拟热传导中的缓慢衰减。
        """
        return np.exp(-decay_rate * t)

    def coupled_solution(
        self,
        t: np.ndarray,
        fast_freq: float = 1.0,
        fast_amp: float = 0.1,
        slow_decay: float = 0.5,
        coupling_strength: float = 0.3
    ) -> np.ndarray:
        """
        耦合解:
          u(t) = g(t) [1 + α f(t)]

        其中:
          g(t) = 慢变包络
          f(t) = 快变振荡
          α = 耦合强度

        这模拟了受快速振荡调制的缓慢衰减过程。
        """
        fast = self.fast_oscillation(t, fast_freq, fast_amp)
        slow = self.slow_envelope(t, slow_decay)
        return slow * (1.0 + coupling_strength * fast)

    def homogenized_solution(
        self,
        t: np.ndarray,
        slow_decay: float = 0.5
    ) -> np.ndarray:
        """
        均匀化解 (快变振荡的平均效应):
          ū(t) = exp(-decay_rate × t)

        当 ε → 0, 快变振荡的平均贡献为零 (对正弦波)。
        更一般地, 需要计算周期平均:
          <f(u, t/ε)>_{period}
        """
        return self.slow_envelope(t, slow_decay)

    def phase_error(
        self,
        t: np.ndarray,
        exact: np.ndarray,
        approx: np.ndarray
    ) -> float:
        """
        计算相位误差:
          ε_phase = max_t |u_exact(t) - u_approx(t)|

        用于评估多尺度近似的精度。
        """
        return float(np.max(np.abs(exact - approx)))


class PeriodicBoundaryHandler:
    """
    周期性边界条件处理器。

    在周期性系统中, 解满足:
      u(x + L, t) = u(x, t)

    这等价于在圆 S¹ = ℝ/Lℤ 上求解。

    傅里叶分解:
      u(x, t) = Σ_{k=-∞}^{∞} û_k(t) exp(2πi k x / L)

    功率谱:
      E(k) = |û_k|²

    能量守恒 (Parseval):
      ∫ |u|² dx = L Σ |û_k|²
    """

    def __init__(self, domain_length: float = 1.0):
        self.L = domain_length

    def fourier_coefficients(
        self, u: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """
        计算离散傅里叶系数:
          û_k = (1/N) Σ_j u(x_j) exp(-2πi k j / N)
        """
        N = len(u)
        return np.fft.fft(u) / N

    def reconstruct_from_fourier(
        self, coeffs: np.ndarray, n_modes: int, x: np.ndarray
    ) -> np.ndarray:
        """
        从前 n_modes 个傅里叶模式重构:
          u(x) ≈ Σ_{|k|≤n_modes} û_k exp(2πi k x / L)
        """
        N = len(coeffs)
        result = np.zeros(len(x))
        for k in range(-n_modes, n_modes + 1):
            idx = k % N
            result += np.real(coeffs[idx] * np.exp(2.0j * np.pi * k * x / self.L))
        return result

    def spectral_energy(self, coeffs: np.ndarray) -> np.ndarray:
        """
        计算谱能量:
          E(k) = |û_k|²
        """
        return np.abs(coeffs) ** 2

    def apply_periodic_bc(self, u: np.ndarray) -> np.ndarray:
        """
        强制周期性: u[0] = u[-1]
        """
        u_periodic = u.copy()
        u_periodic[-1] = u_periodic[0]
        return u_periodic
