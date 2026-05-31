"""
rate_state_dynamics.py
断层滑动速率-状态摩擦动力学模块。

融合种子项目:
  - 121_brusselator_ode: 非线性耦合 ODE 系统（Brusselator 反应-扩散动力学）
  - 859_pendulum_double_ode_movie: 耦合非线性 ODE（双摆运动方程）

在 InSAR 形变反演中的应用:
  1. 采用 Dieterich-Ruina 速率-状态摩擦本构定律描述断层滑移演化；
  2. 将断层滑动视为非线性动力学系统，模拟地震周期中的应力加载、
     稳态蠕滑、以及地震成核过程；
  3. 通过 ODE 积分获取时间相关的滑动速率分布，为 InSAR 时序反演提供
     物理约束。

核心方程:
  1. 本构关系 (Rate-and-State Friction):
        τ = σ_n [ μ_0 + a ln(V/V_0) + b ln(V_0 θ / D_c) ]
  2. 状态变量演化 (Aging Law):
        dθ/dt = 1 - V θ / D_c
  3. 力平衡 (弹簧-滑块模型):
        dV/dt = (k (V_pl - V) - τ) / (M/V)
     或简化为应力率方程:
        dτ/dt = k (V_pl - V)

  其中:
    τ:   剪应力
    σ_n: 有效正应力
    V:   滑动速率
    θ:   状态变量
    D_c: 特征滑移距离
    a, b: 速率-状态参数 (a > 0, b > 0)
    μ_0: 参考摩擦系数
    V_0: 参考滑动速率
    k:   有效刚度
    V_pl: 构造加载速率
"""

import numpy as np
from scipy.integrate import solve_ivp
from utils import check_finite, clip_to_range


class RateStateFriction:
    """
    速率-状态摩擦本构模型求解器。
    """

    def __init__(self, a, b, Dc, sigma_n, mu0, V0, k, V_pl,
                 radiation_damping=False):
        """
        参数:
            a: 直接效应参数
            b: 演化效应参数
            Dc: 特征滑移距离 (m)
            sigma_n: 有效正应力 (Pa)
            mu0: 参考摩擦系数
            V0: 参考滑动速率 (m/s)
            k: 有效刚度 (Pa/m)
            V_pl: 构造加载速率 (m/s)
            radiation_damping: 是否包含辐射阻尼
        """
        self.a = a
        self.b = b
        self.Dc = Dc
        self.sigma_n = sigma_n
        self.mu0 = mu0
        self.V0 = V0
        self.k = k
        self.V_pl = V_pl
        self.radiation_damping = radiation_damping
        # 辐射阻尼系数: η = μ / (2 c_s)，其中 c_s 为 S 波速度 (~3 km/s)
        self.eta = 3e9 / (2.0 * 3000.0) if radiation_damping else 0.0

    def friction_coefficient(self, V, theta):
        """
        计算摩擦系数 μ(V, θ)。
            μ = μ_0 + a ln(V/V_0) + b ln(V_0 θ / D_c)
        对 V 做截断避免 ln(0)。
        """
        V_safe = clip_to_range(V, 1e-18, 1e2)
        theta_safe = clip_to_range(theta, 1e-18, 1e10)
        mu = (self.mu0 +
              self.a * np.log(V_safe / self.V0) +
              self.b * np.log(self.V0 * theta_safe / self.Dc))
        return mu

    def shear_stress(self, V, theta):
        """
        剪应力 τ = σ_n * μ(V, θ)。
        """
        return self.sigma_n * self.friction_coefficient(V, theta)

    def dstate_dt(self, V, theta):
        """
        状态变量演化率 (Aging Law):
            dθ/dt = 1 - V θ / D_c
        """
        return 1.0 - V * theta / self.Dc

    def derivatives(self, t, y):
        """
        动力学系统的右端项。
        状态变量 y = [slip, V, theta, tau]^T

        方程组:
            d(slip)/dt = V
            dV/dt      = (dτ/dt - dτ_friction/dt) / (effective_mass)
            dθ/dt      = 1 - V θ / D_c
            dτ/dt      = k (V_pl - V)

        这里采用简化的弹簧-滑块方程:
            dV/dt = [k (V_pl - V) - σ_n dμ/dt] / (σ_n a / V + η)
        其中 dμ/dt = (a/V) dV/dt + (b/θ) dθ/dt，代入后解出 dV/dt。
        """
        slip, V, theta, tau = y
        V = clip_to_range(V, 1e-20, 1e2)
        theta = clip_to_range(theta, 1e-20, 1e10)

        dtheta = self.dstate_dt(V, theta)

        # 应力率由构造加载驱动
        dtau_load = self.k * (self.V_pl - V)

        # 摩擦系数对 V 和 θ 的偏导
        dmu_dV = self.a / V
        dmu_dtheta = self.b / theta

        # 有效阻尼
        damping = self.sigma_n * dmu_dV + self.eta
        if damping < 1e-20:
            damping = 1e-20

        dV = (dtau_load - self.sigma_n * dmu_dtheta * dtheta) / damping
        dV = clip_to_range(dV, -1e6, 1e6)

        dstate = np.array([V, dV, dtheta, dtau_load])
        return dstate

    def solve_ode(self, t_span, y0, t_eval=None, method='RK45'):
        """
        求解速率-状态摩擦 ODE。

        参数:
            t_span: (t0, tf)
            y0: [slip0, V0, theta0, tau0]
            t_eval: 评估时间点
            method: 积分方法

        返回:
            sol: solve_ivp 结果对象
        """
        y0 = np.asarray(y0, dtype=float)
        if y0[1] <= 0 or y0[2] <= 0:
            raise ValueError("rate_state_dynamics: V and theta must be positive")

        sol = solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        if not sol.success:
            raise RuntimeError("RateStateFriction ODE integration failed")
        return sol

    def steady_state_solution(self, V_ss):
        """
        给定稳态滑动速率 V_ss，计算稳态状态变量 θ_ss 和剪应力 τ_ss。

        稳态条件: dθ/dt = 0  =>  θ_ss = D_c / V_ss
        """
        theta_ss = self.Dc / V_ss
        tau_ss = self.shear_stress(V_ss, theta_ss)
        return theta_ss, tau_ss


class MultiSegmentRateState:
    """
    多段断层（多个弹簧-滑块）的速率-状态摩擦系统。
    模拟断层上不同深度/位置的滑移差异。
    """

    def __init__(self, segments_params):
        """
        segments_params: list of dict，每个 dict 包含一段的参数
            {'a':..., 'b':..., 'Dc':..., 'sigma_n':..., 'mu0':...,
             'V0':..., 'k':..., 'V_pl':...}
        """
        self.n_segments = len(segments_params)
        self.segments = [RateStateFriction(**p) for p in segments_params]

    def derivatives_coupled(self, t, y):
        """
        耦合多段系统的右端项。
        y 长度为 4 * n_segments: [slip_i, V_i, theta_i, tau_i] for each segment i。

        段间耦合通过应力相互作用矩阵实现（简化版）：
            dτ_i/dt = k_i (V_pl - V_i) + Σ_j C_ij (V_j - V_i)
        其中 C_ij 为弹性相互作用系数。
        """
        n = self.n_segments
        dydt = np.zeros(4 * n)
        # 简化的均匀耦合
        coupling = 1e8  # Pa/m^2
        V_avg = np.mean(y[1::4])

        for i in range(n):
            seg = self.segments[i]
            slip_i = y[4 * i]
            V_i = y[4 * i + 1]
            theta_i = y[4 * i + 2]
            tau_i = y[4 * i + 3]

            V_i = clip_to_range(V_i, 1e-20, 1e2)
            theta_i = clip_to_range(theta_i, 1e-20, 1e10)

            dtheta = seg.dstate_dt(V_i, theta_i)
            dtau_load = seg.k * (seg.V_pl - V_i) + coupling * (V_avg - V_i)

            dmu_dV = seg.a / V_i
            dmu_dtheta = seg.b / theta_i
            damping = seg.sigma_n * dmu_dV + seg.eta
            if damping < 1e-20:
                damping = 1e-20
            dV = (dtau_load - seg.sigma_n * dmu_dtheta * dtheta) / damping
            dV = clip_to_range(dV, -1e6, 1e6)

            dydt[4 * i] = V_i
            dydt[4 * i + 1] = dV
            dydt[4 * i + 2] = dtheta
            dydt[4 * i + 3] = dtau_load

        return dydt

    def solve_coupled(self, t_span, y0_list, t_eval=None, method='RK45'):
        """
        求解耦合多段系统。
        """
        y0 = np.concatenate(y0_list)
        sol = solve_ivp(
            fun=self.derivatives_coupled,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-7,
            atol=1e-9
        )
        if not sol.success:
            raise RuntimeError("MultiSegmentRateState ODE integration failed")
        return sol
