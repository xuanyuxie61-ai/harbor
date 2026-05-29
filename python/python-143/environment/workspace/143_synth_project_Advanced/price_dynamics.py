"""
price_dynamics.py
=================
多尺度高频价格动力学模型

本模块基于以下种子项目融合:
- 1140_spring_sweep_ode: 弹簧阻尼系统的参数扫描思想 → 均值回归OU过程的参数敏感性分析
- 1164_stiff_ode: 刚性ODE的快速松弛动力学 → 订单簿失衡的瞬态恢复
- 104_boundary_locus: 稳定性区域分析 → 数值离散化方案的稳定性边界
- 1031_rk2: 二阶Runge-Kutta积分器 → 价格SDE路径模拟

核心数学模型:
--------------
1.  Ornstein-Uhlenbeck 均值回归过程:
        dX_t = -κ (X_t - μ) dt + σ dW_t
    其中 κ 为均值回归速率(对应弹簧刚度 k/m), μ 为长期均衡价格,
    σ 为波动率, W_t 为标准布朗运动.

2.  刚性松弛分量 (Stiff Relaxation):
        dY_t = λ (cos(ω t) - Y_t) dt
    描述市场微观结构噪声的快速衰减, λ ≫ 1 为大刚性参数.
    当 λ → ∞ 时, Y_t ≈ cos(ω t) 瞬时追踪.

3.  耦合多尺度系统:
        dS_t = dX_t + α Y_t dt
    其中 α 为微观结构影响系数.

4.  数值离散化稳定性分析:
    对测试方程 y' = ζ y, RK2 方法的放大因子:
        R(z) = 1 + z + z²/2,   z = hζ
    绝对稳定区域:  { z ∈ ℂ : |R(z)| ≤ 1 }
    我们在复平面上分析该区域的边界, 确保高频回测的时间步长 h
    落在稳定区域内, 避免数值爆炸.

5.  离散化误差估计:
    RK2 的局部截断误差为 O(h³), 全局误差 O(h²).
    对 OU 过程, 强收敛阶为 1/2 (Itô 积分), 但采用修正的 Milstein
    型离散可提升至阶 1.
"""

import numpy as np
from typing import Tuple, Callable, Optional


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck 随机过程, 建模高频价格的均值回归特性.
    """

    def __init__(self, kappa: float, mu: float, sigma: float,
                 s0: float, t_max: float, n_steps: int, seed: Optional[int] = None):
        """
        Parameters
        ----------
        kappa : float
            均值回归速率 κ > 0.  物理类比: 弹簧-阻尼系统中的刚度系数.
        mu : float
            长期均衡水平 μ.
        sigma : float
            波动率参数 σ ≥ 0.
        s0 : float
            初始价格 S(0).
        t_max : float
            模拟终止时间 T.
        n_steps : int
            时间离散步数 N.
        seed : int, optional
            随机数种子, 用于可复现实验.
        """
        if kappa <= 0.0:
            raise ValueError("均值回归速率 κ 必须为正.")
        if sigma < 0.0:
            raise ValueError("波动率 σ 必须非负.")
        if n_steps <= 0:
            raise ValueError("步数 N 必须为正整数.")

        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = t_max / n_steps

        if seed is not None:
            np.random.seed(seed)

    def exact_solution(self, t: np.ndarray) -> np.ndarray:
        """
        OU 过程的精确期望与方差 (用于验证数值方法):
            E[S_t] = μ + (S_0 - μ) e^{-κ t}
            Var[S_t] = σ²/(2κ) (1 - e^{-2κ t})
        """
        expectation = self.mu + (self.s0 - self.mu) * np.exp(-self.kappa * t)
        variance = (self.sigma ** 2) / (2.0 * self.kappa) * (
            1.0 - np.exp(-2.0 * self.kappa * t)
        )
        return expectation, variance

    def simulate_rk2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用修正的二阶 Runge-Kutta (Heun) 方法离散化 OU-SDE.

        对漂移项采用确定性 RK2, 对扩散项采用 Euler-Maruyama:
            k1 = -κ (S_n - μ) dt + σ √dt Z_n
            k2 = -κ (S_n + k1 - μ) dt + σ √dt Z_n
            S_{n+1} = S_n + 0.5 (k1 + k2)

        其中 Z_n ~ N(0,1) i.i.d.

        Returns
        -------
        t : np.ndarray, shape (N+1,)
            时间网格.
        s : np.ndarray, shape (N+1,)
            模拟价格路径.
        """
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        s = np.zeros(self.n_steps + 1)
        s[0] = self.s0
        sqrt_dt = np.sqrt(self.dt)

        for n in range(self.n_steps):
            z_n = np.random.normal()
            drift_1 = -self.kappa * (s[n] - self.mu) * self.dt
            diff_1 = self.sigma * sqrt_dt * z_n
            k1 = drift_1 + diff_1

            drift_2 = -self.kappa * (s[n] + k1 - self.mu) * self.dt
            diff_2 = self.sigma * sqrt_dt * z_n
            k2 = drift_2 + diff_2

            s[n + 1] = s[n] + 0.5 * (k1 + k2)

            # 边界保护: 价格不能为负
            if s[n + 1] <= 0.0:
                s[n + 1] = 1e-6

        return t, s

    def simulate_exact_milstein(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用精确离散化 (对 OU 过程存在闭式解) 进行强收敛基准测试.

        OU 过程的精确一步转移:
            S_{n+1} = S_n e^{-κ Δt} + μ (1 - e^{-κ Δt})
                      + σ √((1-e^{-2κ Δt})/(2κ)) Z_n

        这是强阶 1.0 的精确方法, 用于评估 RK2 近似误差.
        """
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        s = np.zeros(self.n_steps + 1)
        s[0] = self.s0

        exp_kdt = np.exp(-self.kappa * self.dt)
        std_factor = self.sigma * np.sqrt(
            (1.0 - np.exp(-2.0 * self.kappa * self.dt)) / (2.0 * self.kappa)
        )

        for n in range(self.n_steps):
            z_n = np.random.normal()
            s[n + 1] = s[n] * exp_kdt + self.mu * (1.0 - exp_kdt) + std_factor * z_n
            if s[n + 1] <= 0.0:
                s[n + 1] = 1e-6

        return t, s


class StiffRelaxation:
    """
    刚性松弛过程, 建模订单簿失衡的快速恢复动力学.
    基于 1164_stiff_ode 的思想.
    """

    def __init__(self, lam: float, omega: float, y0: float,
                 t_max: float, n_steps: int):
        if lam <= 0.0:
            raise ValueError("刚性参数 λ 必须为正.")
        if n_steps <= 0:
            raise ValueError("步数 N 必须为正整数.")

        self.lam = lam
        self.omega = omega
        self.y0 = y0
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = t_max / n_steps

    def derivative(self, t: float, y: float) -> float:
        """
        刚性ODE右端项:
            y' = λ (cos(ω t) - y)
        """
        return self.lam * (np.cos(self.omega * t) - y)

    def solve_rk2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用显式 RK2 求解刚性ODE, 当 hλ 超出稳定性区域时会出现数值不稳定.
        本函数用于演示稳定性边界的重要性.
        """
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        y = np.zeros(self.n_steps + 1)
        y[0] = self.y0

        for n in range(self.n_steps):
            k1 = self.dt * self.derivative(t[n], y[n])
            k2 = self.dt * self.derivative(t[n] + self.dt, y[n] + k1)
            y[n + 1] = y[n] + 0.5 * (k1 + k2)

        return t, y

    def solve_exact(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        线性ODE的精确解 (用于验证):
            y(t) = y_0 e^{-λ t} + λ e^{-λ t} ∫_0^t e^{λ s} cos(ω s) ds
        积分结果为:
            = y_0 e^{-λ t} + λ/(λ²+ω²) (λ cos(ω t) + ω sin(ω t) - λ e^{-λ t})
        """
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        exp_lt = np.exp(-self.lam * t)
        factor = self.lam / (self.lam ** 2 + self.omega ** 2)
        y = (self.y0 * exp_lt
             + factor * (self.lam * np.cos(self.omega * t)
                         + self.omega * np.sin(self.omega * t)
                         - self.lam * exp_lt))
        return t, y


class StabilityAnalysis:
    """
    数值方法的稳定性区域分析, 基于 104_boundary_locus 的思想.
    对高频回测而言, 时间步长 h 的选择必须保证数值稳定性.
    """

    @staticmethod
    def rk2_amplification(z_real: np.ndarray, z_imag: np.ndarray) -> np.ndarray:
        """
        RK2 方法对测试方程 y' = ζ y 的放大因子:
            R(z) = 1 + z + z²/2,   z = h ζ

        Parameters
        ----------
        z_real, z_imag : np.ndarray
            复平面上的网格坐标.

        Returns
        -------
        abs_R : np.ndarray
            |R(z)| 在网格上的值.
        """
        Z = z_real + 1j * z_imag
        R = 1.0 + Z + 0.5 * Z ** 2
        return np.abs(R)

    @staticmethod
    def is_stable(z_real: float, z_imag: float) -> bool:
        """判断给定点是否在 RK2 的绝对稳定区域内."""
        return StabilityAnalysis.rk2_amplification(
            np.array([[z_real]]), np.array([[z_imag]])
        )[0, 0] <= 1.0 + 1e-10

    @staticmethod
    def maximum_stable_step(lambda_max: float) -> float:
        """
        对实负特征值 ζ = -λ (λ>0), RK2 稳定条件要求:
            |1 - hλ + (hλ)²/2| ≤ 1
        解得实轴上最大稳定步长:
            h_max = 2 / λ
        """
        if lambda_max <= 0.0:
            raise ValueError("lambda_max 必须为正.")
        return 2.0 / lambda_max


class ParameterSweep:
    """
    参数扫描模块, 基于 1140_spring_sweep_ode 的思想.
    在 (κ, σ) 参数平面上扫描 OU 过程的统计特性,
    为策略参数优化提供先验知识.
    """

    def __init__(self, kappa_vals: np.ndarray, sigma_vals: np.ndarray,
                 mu: float = 100.0, s0: float = 100.0,
                 t_max: float = 1.0, n_steps: int = 10000):
        self.kappa_vals = kappa_vals
        self.sigma_vals = sigma_vals
        self.mu = mu
        self.s0 = s0
        self.t_max = t_max
        self.n_steps = n_steps

    def sweep_half_life(self) -> np.ndarray:
        """
        计算均值回归半衰期:
            t_{1/2} = ln(2) / κ
        在参数网格上返回半衰期矩阵.
        """
        k_grid, s_grid = np.meshgrid(self.kappa_vals, self.sigma_vals, indexing='ij')
        half_life = np.log(2.0) / k_grid
        return half_life

    def sweep_stationary_variance(self) -> np.ndarray:
        """
        计算稳态方差:
            Var_∞ = σ² / (2 κ)
        """
        k_grid, s_grid = np.meshgrid(self.kappa_vals, self.sigma_vals, indexing='ij')
        var_inf = s_grid ** 2 / (2.0 * k_grid)
        return var_inf

    def sweep_peak_volatility(self, n_paths: int = 50) -> np.ndarray:
        """
        对每个 (κ, σ) 组合, 模拟 n_paths 条路径,
        计算价格路径的最大振幅 (peak volatility).
        基于 spring_sweep_ode 中寻找峰值的思想.
        """
        m = len(self.kappa_vals)
        n = len(self.sigma_vals)
        peak_vals = np.full((m, n), np.nan)

        for i, kappa in enumerate(self.kappa_vals):
            for j, sigma in enumerate(self.sigma_vals):
                ou = OrnsteinUhlenbeck(
                    kappa=kappa, mu=self.mu, sigma=sigma,
                    s0=self.s0, t_max=self.t_max, n_steps=self.n_steps, seed=42
                )
                peaks = []
                for _ in range(n_paths):
                    t, s = ou.simulate_exact_milstein()
                    peaks.append(np.max(np.abs(s - self.mu)))
                peak_vals[i, j] = np.mean(peaks)

        return peak_vals
