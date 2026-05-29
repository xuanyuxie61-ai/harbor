"""
rotor_multiphysics.py
================================================================================
电机转子多物理场耦合动力学分析

融合原项目:
  - 495_gyroscope_ode   : 刚体转动动力学ODE（欧拉角、角动量方程）
  - 1089_sling_ode      : 非线性振荡ODE（极限环动力学）
  - 1387_vanderpol_ode_period : 非线性振荡周期估计（Urabe渐近公式）

核心科学内容:
  1. 永磁同步电机转子运动方程:

        J dω/dt = τ_em - τ_load - B_d ω - τ_cog(θ)
        dθ/dt = ω

     其中:
        J       : 转子转动惯量 [kg·m^2]
        ω       : 机械角速度 [rad/s]
        θ       : 转子角位置 [rad]
        τ_em    : 电磁转矩 [N·m]（由 FEM 计算得到）
        τ_load  : 负载转矩 [N·m]
        B_d     : 粘滞阻尼系数 [N·m·s/rad]
        τ_cog   : 齿槽转矩（cogging torque）, 近似为傅里叶级数:
                    τ_cog(θ) = Σ_k T_k sin(N_s k θ)

  2. 考虑转子偏心故障时的附加振动方程（融合 sling_ode 的非线性结构）:

        偏心位移 (u, v) 满足:
            m_r d^2u/dt^2 + c_r du/dt + k_r u = F_u(θ, ω) + m_r e ω^2 cos(θ)
            m_r d^2v/dt^2 + c_r dv/dt + k_r v = F_v(θ, ω) + m_r e ω^2 sin(θ)

     其中 e 为偏心距，F_u, F_v 为不平衡磁拉力（UMP）。

     融合 sling_ode 的极坐标非线性形式:
        令 r = sqrt(u^2+v^2), 则在旋转坐标系中:
            dr/dt = s * r * (r - r_0) + ε ω^2 cos(φ)
        其中 s 为刚度非线性系数，r_0 为平衡半径。

  3. 陀螺效应（高速电机）:

     当电机转速很高时，转子作为陀螺系统，其角动量方程为（融合 gyroscope_ode）:

        dL/dt = τ_ext

     在体坐标系中:
        A1 dω1/dt = (A2 - A3) ω2 ω3 + M1
        A2 dω2/dt = (A3 - A1) ω3 ω1 + M2
        A3 dω3/dt = (A1 - A2) ω1 ω2 + M3

     其中 A1, A2, A3 为绕三个主轴的转动惯量。

  4. 非线性振动周期估计（融合 vanderpol_ode_period 的 Urabe 公式）:

     对于强非线性系统，周期可表示为渐近展开:

        T(μ) ≈ (3 - 2 ln 2) μ + 3 α / μ^{1/3} - (1/3) ln(μ)/μ + C/μ

     其中 μ 为非线性强度参数，α ≈ 2.338107 为 Airy 函数第一个零点。

     在电机应用中，μ 与偏心比 ε/e_max 成正比。
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp


class RotorDynamics:
    """
    永磁同步电机转子机电耦合动力学模型。
    """

    def __init__(
        self,
        J: float = 0.01,          # 转动惯量 [kg·m^2]
        B_d: float = 0.001,       # 粘滞阻尼 [N·m·s/rad]
        tau_load: float = 5.0,    # 负载转矩 [N·m]
        n_slots: int = 6,         # 定子槽数
        cogging_amplitude: float = 0.5,  # 齿槽转矩幅值 [N·m]
    ):
        self.J = float(J)
        self.B_d = float(B_d)
        self.tau_load = float(tau_load)
        self.n_slots = int(n_slots)
        self.cogging_amp = float(cogging_amplitude)

    def cogging_torque(self, theta: float) -> float:
        """齿槽转矩（基波+三次谐波）."""
        return self.cogging_amp * (
            np.sin(self.n_slots * theta)
            + 0.3 * np.sin(2 * self.n_slots * theta)
        )

    def rotor_ode(self, t: float, y: np.ndarray, tau_em_func) -> np.ndarray:
        """
        转子运动 ODE:
            dy/dt = [dθ/dt, dω/dt]^T = [ω, (τ_em - τ_load - B_d ω - τ_cog)/J]^T

        参数:
            tau_em_func : callable(t, y) -> float, 时变电磁转矩
        """
        theta, omega = y
        tau_em = tau_em_func(t, y)
        tau_cog = self.cogging_torque(theta)
        dtheta = omega
        domega = (tau_em - self.tau_load - self.B_d * omega - tau_cog) / self.J
        return np.array([dtheta, domega])

    def simulate(
        self,
        tau_em_func,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 1.0),
        t_eval: np.ndarray = None,
    ) -> dict:
        """数值积分求解转子运动方程."""
        if y0 is None:
            y0 = np.array([0.0, 0.0])  # θ(0)=0, ω(0)=0

        sol = solve_ivp(
            lambda t, y: self.rotor_ode(t, y, tau_em_func),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.01,
        )

        if not sol.success:
            raise RuntimeError("转子动力学ODE求解失败")

        return {
            "t": sol.t,
            "theta": sol.y[0],
            "omega": sol.y[1],
            "sol": sol,
        }


class EccentricVibration:
    """
    转子偏心故障引起的非线性振动模型（融合 sling_ode 思想）。

    在极坐标 (r, φ) 下，径向运动方程:
        d^2r/dt^2 = -ω_n^2 (r - r_0) + ε ω^2 cos(φ - θ)
                    + s (r - r_0)^3  （非线性刚度项）

    其中:
        ω_n = sqrt(k_r / m_r)   为固有频率
        s   为非线性刚度系数
        ε   为偏心距
        θ   为转子转角
    """

    def __init__(
        self,
        m_r: float = 2.0,       # 转子质量 [kg]
        k_r: float = 1.0e6,     # 支承刚度 [N/m]
        c_r: float = 100.0,     # 支承阻尼 [N·s/m]
        r0: float = 0.0,        # 平衡位置 [m]
        s_nl: float = 1.0e8,    # 非线性刚度系数 [N/m^3]
        epsilon: float = 0.1e-3,  # 偏心距 [m]
    ):
        self.m_r = float(m_r)
        self.k_r = float(k_r)
        self.c_r = float(c_r)
        self.r0 = float(r0)
        self.s_nl = float(s_nl)
        self.epsilon = float(epsilon)
        self.omega_n = np.sqrt(k_r / m_r)

    def vibration_ode(self, t: float, y: np.ndarray, theta_func, omega_func) -> np.ndarray:
        """
        状态向量 y = [u, v, du/dt, dv/dt]^T（笛卡尔坐标）.

        融合 sling_ode 的非线性结构:
            du/dt = p
            dv/dt = q
            dp/dt = -(k_r/m) (u - r0 cosθ) - (c_r/m) p + (ε/m) ω^2 cosθ + (s_nl/m) (u - r0)^3
            dq/dt = -(k_r/m) (v - r0 sinθ) - (c_r/m) q + (ε/m) ω^2 sinθ + (s_nl/m) (v - r0)^3
        """
        u, v, p, q = y
        theta = theta_func(t)
        omega = omega_func(t)

        # 径向位移
        r = np.sqrt(u * u + v * v)
        r_safe = max(r, 1.0e-12)

        # 非线性恢复力（软弹簧特性）
        nl_force_u = self.s_nl * (r - self.r0) ** 3 * (u / r_safe)
        nl_force_v = self.s_nl * (r - self.r0) ** 3 * (v / r_safe)

        # 不平衡激振力
        unbalance_u = self.epsilon * omega * omega * np.cos(theta)
        unbalance_v = self.epsilon * omega * omega * np.sin(theta)

        dp = (
            -(self.k_r / self.m_r) * u
            - (self.c_r / self.m_r) * p
            + unbalance_u
            + nl_force_u
        )
        dq = (
            -(self.k_r / self.m_r) * v
            - (self.c_r / self.m_r) * q
            + unbalance_v
            + nl_force_v
        )

        return np.array([p, q, dp, dq])

    def simulate(
        self,
        theta_func,
        omega_func,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 0.5),
        t_eval: np.ndarray = None,
    ) -> dict:
        """数值积分求解偏心振动方程."""
        if y0 is None:
            y0 = np.array([self.epsilon, 0.0, 0.0, 0.0])

        sol = solve_ivp(
            lambda t, y: self.vibration_ode(t, y, theta_func, omega_func),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.005,
        )

        if not sol.success:
            raise RuntimeError("偏心振动ODE求解失败")

        u = sol.y[0]
        v = sol.y[1]
        displacement = np.sqrt(u * u + v * v)

        return {
            "t": sol.t,
            "u": u,
            "v": v,
            "displacement": displacement,
            "sol": sol,
        }


class GyroscopicEffects:
    """
    高速电机转子的陀螺效应分析（融合 gyroscope_ode 的核心方程）。

    欧拉方程（体坐标系）:
        A1 dω1/dt = (A2 - A3) ω2 ω3 + M1
        A2 dω2/dt = (A3 - A1) ω3 ω1 + M2
        A3 dω3/dt = (A1 - A2) ω1 ω2 + M3

    其中 (ω1, ω2, ω3) 为体坐标系中的角速度分量，
    (M1, M2, M3) 为外力矩分量。

    对于圆柱形转子:
        A1 = A2 = J_t   （横向转动惯量）
        A3 = J_a        （轴向转动惯量）
    """

    def __init__(
        self,
        A1: float = 0.005,   # 绕x轴转动惯量 [kg·m^2]
        A2: float = 0.005,   # 绕y轴转动惯量 [kg·m^2]
        A3: float = 0.015,   # 绕z轴转动惯量 [kg·m^2]
        m_unbalance: float = 0.01,  # 不平衡质量 [kg]
    ):
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.A3 = float(A3)
        self.m_unb = float(m_unbalance)

    def gyro_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        状态向量 y = [ψ, θ, φ, ω1, ω2, ω3]^T.

        ψ : 进动角（precession）
        θ : 章动角（nutation）
        φ : 自转角（spin）
        """
        psi, theta, phi, omega1, omega2, omega3 = y

        # 重力/不平衡力矩（简化模型）
        M1 = -self.m_unb * self.A1 * np.sin(theta) * np.cos(phi)
        M2 = self.m_unb * self.A2 * np.sin(theta) * np.sin(phi)
        M3 = 0.0

        # 欧拉角速率（3-1-3 旋转序列）
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # 边界保护
        sin_theta_safe = sin_theta if abs(sin_theta) > 1.0e-10 else 1.0e-10 * np.sign(sin_theta + 1.0e-20)

        dpsi = (omega1 * sin_phi + omega2 * cos_phi) / sin_theta_safe
        dtheta = omega1 * cos_phi - omega2 * sin_phi
        dphi = omega3 - cos_theta * dpsi

        domega1 = ((self.A2 - self.A3) * omega2 * omega3 + M1) / self.A1
        domega2 = ((self.A3 - self.A1) * omega3 * omega1 + M2) / self.A2
        domega3 = ((self.A1 - self.A2) * omega1 * omega2 + M3) / self.A3

        return np.array([dpsi, dtheta, dphi, domega1, domega2, domega3])

    def simulate(
        self,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 0.5),
        t_eval: np.ndarray = None,
    ) -> dict:
        """数值积分求解陀螺方程."""
        if y0 is None:
            y0 = np.array([0.25, 0.4, 0.1, 100.0, 50.0, 314.0])  # ω3 ≈ 3000 rpm

        sol = solve_ivp(
            self.gyro_ode,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.005,
        )

        if not sol.success:
            raise RuntimeError("陀螺效应ODE求解失败")

        return {
            "t": sol.t,
            "psi": sol.y[0],
            "theta": sol.y[1],
            "phi": sol.y[2],
            "omega1": sol.y[3],
            "omega2": sol.y[4],
            "omega3": sol.y[5],
            "sol": sol,
        }


class NonlinearPeriodEstimator:
    """
    非线性振动周期估计器（融合 vanderpol_ode_period 的 Urabe 渐近公式）。

    对于广义 Van der Pol 型非线性振荡器:
        d^2x/dt^2 + μ (x^2 - 1) dx/dt + ω_0^2 x = 0

    当 μ >> 1 时，周期由 Urabe 公式给出:

        T(μ) ≈ (3 - 2 ln 2) μ
              + 3 α / μ^{1/3}
              - (1/3) ln(μ) / μ
              + (3 ln 2 - ln 3 - 1.5 + b0 - 2d) / μ

    其中:
        α  = 2.338107  （Airy 函数首个零点）
        b0 = 0.1723
        d  = 0.4889

    在电机偏心故障中，将 μ 映射为等效非线性强度:
        μ_eff = ε / δ_gap

    其中 ε 为偏心距，δ_gap 为标称气隙长度。
    """

    ALPHA = 2.338107
    B0 = 0.1723
    D = 0.4889

    @classmethod
    def urabe_period(cls, mu: float) -> float:
        """
        计算 Urabe 周期估计.

        参数:
            mu : 非线性强度参数（μ ≥ 0）
        """
        if mu <= 0.0:
            # 线性极限: T = 2π/ω_0, 取 ω_0 = 1
            return 2.0 * np.pi

        term1 = (3.0 - 2.0 * np.log(2.0)) * mu
        term2 = 3.0 * cls.ALPHA / (mu ** (1.0 / 3.0))
        term3 = -(1.0 / 3.0) * np.log(mu) / mu
        term4 = (3.0 * np.log(2.0) - np.log(3.0) - 1.5 + cls.B0 - 2.0 * cls.D) / mu

        return term1 + term2 + term3 + term4

    @classmethod
    def estimate_motor_fault_period(
        cls, eccentricity: float, nominal_airgap: float, omega_0: float = 1.0
    ) -> float:
        """
        估计电机偏心故障引起的振动周期.

        参数:
            eccentricity      : 偏心距 [m]
            nominal_airgap    : 标称气隙 [m]
            omega_0           : 线性固有频率 [rad/s]
        """
        if nominal_airgap <= 1.0e-12:
            raise ValueError("气隙必须为正")
        mu_eff = eccentricity / nominal_airgap
        T_dimless = cls.urabe_period(mu_eff)
        # 量纲还原: T = T_dimless / omega_0
        return T_dimless / omega_0

    @classmethod
    def cartwright_period(cls, mu: float) -> float:
        """Cartwright 周期估计（低阶近似）."""
        if mu <= 0.0:
            return 2.0 * np.pi
        return (3.0 - 2.0 * np.log(2.0)) * mu + 2.0 * np.pi / (mu ** (1.0 / 3.0))

    @classmethod
    def grimshaw_period(cls, mu: float) -> float:
        """Grimshaw 周期估计."""
        if mu <= 0.0:
            return 2.0 * np.pi
        alpha = 2.338
        return (3.0 - 2.0 * np.log(2.0)) * mu + 2.0 * alpha / (mu ** (1.0 / 3.0))
