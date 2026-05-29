# -*- coding: utf-8 -*-
"""
damage_ode.py
=============
复合材料损伤演化常微分方程（ODE）系统与疲劳周期分析模块。

源自种子项目：
  - 511_heartbeat_ode（快-慢 stiff ODE 结构）
  - 1387_vanderpol_ode_period（周期渐近估计）

科学背景：
---------
复合材料损伤演化具有多时间尺度特征：
  - 快尺度：应力/应变在冲击载荷下的瞬态响应（微秒级）；
  - 慢尺度：基体微裂纹扩展、纤维/基体界面脱粘（毫秒~秒级）；
  - 极慢尺度：疲劳载荷下的循环累积损伤（小时~年）。

本模块包含三类模型：

1. 相场型损伤演化律（Rate-dependent phase-field damage）：
   损伤变量 d ∈ [0,1] 的演化受驱动力 Y 控制：
     ∂d/∂t = - (1/τ) * ∂Ψ/∂d = (1/τ) * max(0, Y - Y_c)
   其中：
     Ψ(ε, d) = g(d) * Ψ_e(ε) + Ψ_c(d)      — 总自由能
     g(d) = (1 - d)^2 + η                    — 退化函数（η>0 为正则化参数）
     Ψ_e(ε) = (1/2) * E_0 * ε^2             — 弹性应变能密度
     Y = -∂Ψ/∂d = 2(1-d) * Ψ_e              — 热力学损伤驱动力
     Y_c = (G_c / c_0)                       — 临界损伤驱动力（G_c 为断裂韧性）
     τ                                     — 粘性正则化时间

   该方程在 d → 1 时呈现 stiff 结构（类似 heartbeat ODE 的 cubic nullcline），
   因为 ∂d/∂t 中的 (1-d) 导致损伤在接近 1 时演化急剧加速。

2. 快-慢分解损伤模型（Fast-Slow Damage ODE）：
   引入慢变量 s（累积等效应变）和快变量 d（瞬时损伤）：
     ε d' = -(d^3 - a(σ_eq) * d + s)        — 快方程（ε << 1）
     s' = d - γ(σ_eq)                       — 慢方程
   其中 a(σ_eq) 为等效应力相关的恢复力参数，γ 为损伤阈值偏移。
   这直接映射 heartbeat ODE 的快-慢结构到损伤力学：
   当等效应力 σ_eq 超过阈值时，系统在 d-s 相平面上发生极限环振荡，
   对应损伤在稳定传播与弹性恢复之间的交替（如韧性复合材料的 stick-slip 裂纹扩展）。

3. 疲劳损伤累积模型与周期估计（Van der Pol 类比）：
   对于循环载荷 σ(t) = σ_m + σ_a * sin(ωt)，引入疲劳损伤变量 d_f：
     dd_f/dN = C * (Δσ_eq / σ_ref)^m * (1 - d_f)^{-q}
   其中 N 为循环次数，Δσ_eq 为等效应力幅，m 为 S-N 曲线斜率（通常 5~20）。

   利用 Van der Pol 型渐近分析，对大非线性参数 μ（类比高应力幅）
   估计损伤累积至失效的循环次数 N_f：
     N_f ≈ ∫_0^1 (1 - d_f)^q / [C * (Δσ_eq / σ_ref)^m] dd_f
         = 1 / [(q+1) * C * (Δσ_eq / σ_ref)^m]

   对于非对称循环，引入平均应力修正（Goodman 关系）：
     Δσ_eq_eff = Δσ_eq / (1 - σ_m / σ_uts)
     N_f = 1 / [(q+1) * C * (Δσ_eq_eff / σ_ref)^m]
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple


class PhaseFieldDamageModel:
    """
    相场损伤演化模型。
    """

    def __init__(self, E0: float, Gc: float, c0: float = 3.0 / 8.0,
                 eta: float = 1e-6, tau: float = 1e-3):
        """
        Parameters
        ----------
        E0 : float
            无损弹性模量 [Pa].
        Gc : float
            临界能量释放率 [J/m²].
        c0 : float
            相场长度尺度参数相关的常数（c0 = 3/8 为典型值）。
        eta : float
            退化函数正则化参数，防止刚度完全丧失导致数值奇异性。
        tau : float
            粘性正则化时间 [s].
        """
        if E0 <= 0 or Gc <= 0 or tau <= 0:
            raise ValueError("E0, Gc, tau must be positive.")
        self.E0 = E0
        self.Gc = Gc
        self.c0 = c0
        self.eta = eta
        self.tau = tau
        self.Yc = Gc / c0  # 临界损伤驱动力

    def degradation(self, d: np.ndarray) -> np.ndarray:
        """退化函数 g(d) = (1-d)^2 + η。"""
        d = np.clip(d, 0.0, 1.0)
        return (1.0 - d) ** 2 + self.eta

    def elastic_energy(self, epsilon: float) -> float:
        """弹性应变能密度 Ψ_e = 0.5 * E0 * ε²。"""
        return 0.5 * self.E0 * epsilon ** 2

    def damage_driving_force(self, epsilon: float, d: float) -> float:
        """
        热力学损伤驱动力 Y = -∂Ψ/∂d = 2(1-d) * Ψ_e(ε)。
        """
        d_clip = np.clip(d, 0.0, 1.0)
        return 2.0 * (1.0 - d_clip) * self.elastic_energy(epsilon)

    def evolution_rate(self, epsilon: float, d: float) -> float:
        """
        损伤演化率 dd/dt = (1/τ) * max(0, Y - Y_c)。
        包含边界处理：d=0 时只允许增加，d=1 时停止演化。
        """
        if d <= 0.0:
            return max(0.0, (self.damage_driving_force(epsilon, d) - self.Yc) / self.tau)
        elif d >= 1.0:
            return 0.0
        else:
            rate = (self.damage_driving_force(epsilon, d) - self.Yc) / self.tau
            return max(0.0, rate)

    def integrate(self, epsilon_history: Callable[[float], float],
                  d0: float, t_span: Tuple[float, float],
                  num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        数值积分损伤演化方程。

        Returns
        -------
        t : np.ndarray
            时间数组。
        d : np.ndarray
            损伤变量历史。
        """
        def ode_func(t, y):
            d = y[0]
            eps = epsilon_history(t)
            dd_dt = self.evolution_rate(eps, d)
            return [dd_dt]

        sol = solve_ivp(ode_func, t_span, [d0], dense_output=True,
                        max_step=(t_span[1] - t_span[0]) / num_points,
                        method='RK45')
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        d_sol = sol.sol(t_eval)[0]
        # 数值鲁棒性：强制截断到 [0,1]
        d_sol = np.clip(d_sol, 0.0, 1.0)
        return t_eval, d_sol


class FastSlowDamageODE:
    """
    快-慢 stiff 损伤 ODE 系统。
    映射 heartbeat_ode 的快-慢结构到损伤力学。
    """

    def __init__(self, eps_fast: float = 1e-3,
                 sigma_threshold: float = 100e6,
                 sigma_uts: float = 1500e6):
        """
        Parameters
        ----------
        eps_fast : float
            快时间尺度参数 ε << 1。
        sigma_threshold : float
            损伤起始应力阈值 [Pa]。
        sigma_uts : float
            极限拉伸强度 [Pa]。
        """
        if eps_fast <= 0:
            raise ValueError("eps_fast must be positive.")
        self.eps = eps_fast
        self.sigma_th = sigma_threshold
        self.sigma_uts = sigma_uts

    def _a_parameter(self, sigma_eq: float) -> float:
        """
        快方程中的恢复力参数 a(σ_eq)。
        采用双曲正切软化 law：
          a(σ) = a0 * (1 - tanh((σ - σ_th)/σ_ref))
        当 σ_eq 超过阈值时，a 减小，系统趋向不稳定（损伤增长）。
        """
        a0 = 3.0
        sigma_ref = 0.2 * self.sigma_uts
        return a0 * (1.0 - np.tanh((sigma_eq - self.sigma_th) / (sigma_ref + 1e-30)))

    def _gamma_parameter(self, sigma_eq: float) -> float:
        """
        慢方程中的阈值偏移 γ(σ_eq)。
        """
        gamma0 = 0.5
        return gamma0 * (sigma_eq / (self.sigma_uts + 1e-30)) ** 2

    def rhs(self, t: float, state: np.ndarray,
            sigma_eq_func: Callable[[float], float]) -> np.ndarray:
        """
        计算 ODE 右端项。
        state = [d, s]
        """
        d, s = state
        sigma_eq = sigma_eq_func(t)
        a = self._a_parameter(sigma_eq)
        gamma = self._gamma_parameter(sigma_eq)

        # 快方程：ε * dd/dt = -(d^3 - a*d + s)
        dd_dt = -(d ** 3 - a * d + s) / (self.eps + 1e-30)
        # 慢方程：ds/dt = d - γ
        ds_dt = d - gamma

        # 边界处理
        if d <= 0.0 and dd_dt < 0:
            dd_dt = 0.0
        if d >= 1.0 and dd_dt > 0:
            dd_dt = 0.0

        return np.array([dd_dt, ds_dt])

    def integrate(self, sigma_eq_func: Callable[[float], float],
                  d0: float, s0: float,
                  t_span: Tuple[float, float],
                  num_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数值积分快-慢损伤 ODE。

        Returns
        -------
        t, d, s
        """
        def ode_func(t, y):
            return self.rhs(t, y, sigma_eq_func)

        # 使用 stiff 求解器（BDF）处理快尺度
        sol = solve_ivp(ode_func, t_span, [d0, s0],
                        dense_output=True,
                        max_step=(t_span[1] - t_span[0]) / num_points,
                        method='BDF')
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y_sol = sol.sol(t_eval)
        d_sol = np.clip(y_sol[0], 0.0, 1.0)
        s_sol = y_sol[1]
        return t_eval, d_sol, s_sol


class FatigueDamageModel:
    """
    疲劳损伤累积模型，含 Van der Pol 型周期渐近分析。
    """

    def __init__(self, C: float = 5.0e-12, m: float = 9.0, q: float = 2.0,
                 sigma_ref: float = 1.0e6, sigma_uts: float = 1500e6):
        """
        Parameters
        ----------
        C : float
            Basquin 常数。
        m : float
            S-N 曲线斜率（对 CFRP 典型值 8~12）。
        q : float
            损伤演化加速指数。
        sigma_ref : float
            参考应力 [Pa].
        sigma_uts : float
            极限拉伸强度 [Pa]（用于 Goodman 修正）。
        """
        self.C = C
        self.m = m
        self.q = q
        self.sigma_ref = sigma_ref
        self.sigma_uts = sigma_uts

    def cycles_to_failure(self, delta_sigma: float, sigma_mean: float = 0.0) -> float:
        """
        基于 Goodman 修正的疲劳寿命估计。

        公式推导：
          等效应力幅（Goodman 修正）：
            Δσ_eff = Δσ / (1 - σ_mean/σ_uts)
          损伤演化率：
            dd_f/dN = C * (Δσ_eff / σ_ref)^m * (1 - d_f)^{-q}
          积分从 d_f=0 到 d_f=1：
            N_f = ∫_0^1 (1-d_f)^q / [C*(Δσ_eff/σ_ref)^m] dd_f
                = 1 / [(q+1) * C * (Δσ_eff/σ_ref)^m]

        Parameters
        ----------
        delta_sigma : float
            应力幅 [Pa].
        sigma_mean : float
            平均应力 [Pa].

        Returns
        -------
        N_f : float
            失效循环次数。
        """
        if delta_sigma <= 0:
            return np.inf
        # Goodman 修正因子
        goodman_factor = 1.0 - sigma_mean / (self.sigma_uts + 1e-30)
        if goodman_factor <= 0:
            return 0.0  # 平均应力已超限，立即失效
        delta_sigma_eff = delta_sigma / goodman_factor
        exponent = (delta_sigma_eff / self.sigma_ref) ** self.m
        Nf = 1.0 / ((self.q + 1.0) * self.C * exponent)
        return Nf

    def damage_after_cycles(self, N: float, delta_sigma: float,
                            sigma_mean: float = 0.0) -> float:
        """
        给定循环次数后的疲劳损伤量。

        由 dd_f/dN = A * (1-d_f)^{-q}，其中 A = C*(Δσ_eff/σ_ref)^m，
        分离变量积分得：
          (1-d_f)^{q+1} = 1 - (q+1) * A * N
          d_f(N) = 1 - [1 - (q+1) * A * N]^{1/(q+1)}
        """
        if N <= 0:
            return 0.0
        goodman_factor = 1.0 - sigma_mean / (self.sigma_uts + 1e-30)
        if goodman_factor <= 0:
            return 1.0
        delta_sigma_eff = delta_sigma / goodman_factor
        A = self.C * (delta_sigma_eff / self.sigma_ref) ** self.m
        val = 1.0 - (self.q + 1.0) * A * N
        if val <= 0:
            return 1.0
        return 1.0 - val ** (1.0 / (self.q + 1.0))

    def asymptotic_period_large_mu(self, delta_sigma: float) -> float:
        """
        类比 Van der Pol 大 μ 渐近分析：
        在高应力幅（大非线性）下，损伤在一个周期内的有效演化时间尺度。
        对于大 μ（类比大 Δσ/σ_ref），渐近周期 T ≈ (3 - 2 ln 2) * μ，
        映射到疲劳问题：
          T_eff ≈ (3 - 2*ln(2)) * (σ_ref / Δσ)^{m/2}
        该式描述了损伤累积从慢速启动到快速加速过渡的有效时间尺度。
        """
        if delta_sigma <= 0:
            return np.inf
        mu_eff = (self.sigma_ref / delta_sigma) ** (self.m / 2.0)
        return (3.0 - 2.0 * np.log(2.0)) * mu_eff


if __name__ == "__main__":
    # 自测试 1：相场损伤
    pf = PhaseFieldDamageModel(E0=100e9, Gc=500.0, tau=1e-4)
    eps_hist = lambda t: 0.01 * np.sin(2 * np.pi * 100 * t) + 0.005
    t, d = pf.integrate(eps_hist, d0=0.0, t_span=(0.0, 0.01))
    print("Phase-field damage final:", d[-1])

    # 自测试 2：快-慢 ODE
    fs = FastSlowDamageODE(eps_fast=1e-3)
    sigma_eq = lambda t: 200e6 + 100e6 * np.sin(2 * np.pi * 50 * t)
    t2, d2, s2 = fs.integrate(sigma_eq, d0=0.1, s0=0.0, t_span=(0.0, 0.02))
    print("Fast-slow damage final:", d2[-1], "s final:", s2[-1])

    # 自测试 3：疲劳损伤
    fat = FatigueDamageModel(C=1e-11, m=8.0, q=2.5)
    Nf = fat.cycles_to_failure(delta_sigma=200e6, sigma_mean=50e6)
    print("Cycles to failure:", Nf)
    d_fat = fat.damage_after_cycles(N=Nf / 2, delta_sigma=200e6, sigma_mean=50e6)
    print("Damage at half life:", d_fat)
