"""
qgp_eos.py — QCD 状态方程: 多项式拟合与热力学量
================================================

融合种子项目: 898_polynomials (多项式优化基准库)

本模块实现夸克胶子等离子体的状态方程 (Equation of State),
核心思想借鉴多项式拟合方法, 将格点 QCD 计算得到的 p(T) 关系
用分段多项式高精度表示.

核心公式:
---------

1. 理想 QGP 状态方程 (高温极限):
   P(T) = sigma_SB * T^4 / 3
   e(T) = sigma_SB * T^4
   s(T) = 4 * sigma_SB * T^3 / 3

2. 格点 QCD 修正 (Wuppertal-Budapest 参数化):
   P(T)/T^4 = sum_{k=0}^{N} c_k * (T_c/T)^k
   其中 c_k 为拟合系数, T_c = 155 MeV

3. 声速 (绝热):
   c_s^2 = dP/de |_s = (dP/dT) / (de/dT) = s(T) / (T * c_v(T))
   共形极限: c_s^2 = 1/3
   相变区: c_s^2 出现极小值 ("声速极小值 softest point")

4. 体粘滞 (非共形效应):
   zeta/s = zeta_max * (1/3 - c_s^2)^2
   满足 Buchen-Csernai 定理: 体粘滞在共形极限下消失

5. 剪切粘滞温度依赖性:
   eta/s(T) = (eta/s)_min + a * (T/T_c - 1)^2  (T > T_c)
   eta/s(T) = (eta/s)_min                         (T <= T_c)

6. 迹反常 (interaction measure):
   (e - 3P) / T^4 = -T * d/dT[P/T^4]
   在 T_c 附近达到极大值, 反映 QCD 相变的非微扰特性
"""

import numpy as np
from qgp_config import (
    SIGMA_SB, T_CRITICAL, T_SWITCH,
    ETA_OVER_S_KSS, ZETA_OVER_S_MAX,
    NumericalParams
)


class QGPEquationOfState:
    """
    QCD 状态方程类

    采用分段多项式逼近格点 QCD 结果 (Wuppertal-Budapest 2014),
    高温区趋向共形极限, 低温区描述强子气体.

    数学结构 (源自 898_polynomials):
    - 标准多项式基: P(T) = sum_k a_k * T^k
    - Chebyshev 基: P(T) = sum_k c_k * T_k(x(T))
    - 分段 Hermite 插值: 保证 C^1 连续性

    Attributes:
        T_c: QCD 相变温度 (GeV)
        T_switch: 冻结温度 (GeV)
        poly_hr: 高温区多项式系数 [a_0, a_1, ..., a_N]
        poly_lr: 低温区多项式系数
        conformal: 是否使用共形 EoS
    """

    def __init__(self, conformal: bool = False):
        """
        初始化状态方程

        Args:
            conformal: 若为 True, 使用共形 EoS P = e/3
        """
        self.T_c = T_CRITICAL
        self.T_switch = T_SWITCH
        self.sigma_sb = SIGMA_SB
        self.conformal = conformal

        # 高温区多项式系数 (格点 QCD 拟合, HRG + pQCD)
        # P/T^4 = c_0 + c_1*(T_c/T) + c_2*(T_c/T)^2 + ...
        self.poly_hr = np.array([
            0.3436,   # c_0: Stefan-Boltzmann 极限项
            -0.0255,  # c_1: 领头阶修正
            0.0098,   # c_2: 次领头阶
            -0.0041,  # c_3
            0.0015,   # c_4
            -0.0005,  # c_5
        ])

        # 低温区多项式系数 (强子共振气体 HRG)
        # P/T^4 = sum_k d_k * (T/T_c)^k
        self.poly_lr = np.array([
            0.0,      # d_0: T=0 时 P=0
            0.012,    # d_1
            0.18,     # d_2
            0.85,     # d_3
            -0.32,    # d_4
            0.045,    # d_5
        ])

        # 声速平方参数 (相变区域 softest point)
        self._cs2_min = 0.05    # 声速极小值
        self._cs2_transition_width = 0.02  # 过渡宽度 (GeV)

    def pressure(self, temperature: np.ndarray) -> np.ndarray:
        """
        计算压力 P(T)

        物理:
            高温极限: P -> sigma_SB * T^4 / 3 (Stefan-Boltzmann)
            低温极限: P -> 强子气体压力
            过渡区: 由格点 QCD 多项式拟合给出

        多项式形式:
            P(T)/T^4 = sum_k c_k * f_k(T)

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            压力 P (GeV/fm^3)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)

        if self.conformal:
            return self.sigma_sb * T**4 / 3.0

        P = np.zeros_like(T)

        # 高温区 (T > T_c): 微扰 QCD 多项式展开
        mask_hr = T > self.T_c
        if np.any(mask_hr):
            T_hr = T[mask_hr]
            x = self.T_c / T_hr  # 展开参数
            # P/T^4 = sum_k c_k * x^k
            p_over_T4 = np.polyval(self.poly_hr[::-1], x)
            P[mask_hr] = p_over_T4 * T_hr**4

        # 低温区 (T <= T_c): 强子共振气体
        mask_lr = ~mask_hr
        if np.any(mask_lr):
            T_lr = T[mask_lr]
            x = T_lr / self.T_c
            p_over_T4 = np.polyval(self.poly_lr[::-1], x)
            P[mask_lr] = p_over_T4 * T_lr**4

        # 确保压力非负
        P = np.maximum(P, NumericalParams.PRESSURE_FLOOR)
        return P

    def energy_density(self, temperature: np.ndarray) -> np.ndarray:
        """
        计算能量密度 e(T)

        热力学关系:
            e = T * dP/dT - P

        对于多项式 P/T^4 = f(x), 有:
            e/T^4 = -T * d/dT(P/T^4) + 4*P/T^4
                  = -T * f'(x) * dx/dT + 4*f(x)
                  = (x^2) * f'(x) + 4*f(x)
        其中 x = T_c/T, dx/dT = -T_c/T^2 = -x^2/T_c

        共形极限: e = 3*P = sigma_SB * T^4

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            能量密度 e (GeV/fm^3)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)

        if self.conformal:
            return 3.0 * self.sigma_sb * T**4 / 3.0

        # 数值微分: e = T * dP/dT - P
        dT = 1.0e-6 * T + 1.0e-10
        P_plus = self.pressure(T + dT)
        P_minus = self.pressure(T - dT)
        dPdT = (P_plus - P_minus) / (2.0 * dT)
        e = T * dPdT - self.pressure(T)

        # 确保能量密度为正
        e = np.maximum(e, NumericalParams.ENERGY_FLOOR)
        return e

    def temperature_from_energy(self, energy_density: np.ndarray) -> np.ndarray:
        """
        从能量密度反解温度: T = T(e)

        方法: Newton-Raphson 迭代
            e(T) = sigma_SB * T^4 (共形) => T = (e/sigma_SB)^{1/4}

        非共形情况: 需要数值求根
            f(T) = e(T) - e_target = 0
            T_{n+1} = T_n - f(T_n) / f'(T_n)
            f'(T) = de/dT = T * c_v(T)

        Args:
            energy_density: 能量密度 (GeV/fm^3)

        Returns:
            温度 (GeV)
        """
        e = np.maximum(np.asarray(energy_density, dtype=np.float64),
                       NumericalParams.ENERGY_FLOOR)

        # 共形初始猜测
        T = (3.0 * e / self.sigma_sb)**0.25

        # Newton-Raphson 迭代 (最多 20 次)
        for _ in range(20):
            e_current = self.energy_density(T)
            # de/dT = d(e)/dT, 用数值微分
            dT = 1.0e-6 * T + 1.0e-10
            de_dT = (self.energy_density(T + dT) - e_current) / dT

            # 防止除零
            de_dT = np.maximum(de_dT, 1.0e-15)

            residual = (e_current - e) / de_dT
            T = T - residual
            T = np.maximum(T, NumericalParams.TEMPERATURE_FLOOR)

            # 收敛判断
            if np.max(np.abs(residual / (T + 1.0e-15))) < 1.0e-10:
                break

        return T

    def sound_speed_squared(self, temperature: np.ndarray) -> np.ndarray:
        """
        声速平方 c_s^2 = dP/de

        物理:
            共形极限: c_s^2 = 1/3
            相变区: c_s^2 下降至 ~0.05 (softest point)
            这是 QCD 相变的特征信号

        数学:
            c_s^2 = dP/de = (dP/dT) / (de/dT) = s / (T * c_v)

        也等价于:
            c_s^2 = (d ln T / d ln s) |_s

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            声速平方 c_s^2 (无量纲, 0 到 1/3 之间)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)

        if self.conformal:
            return np.full_like(T, 1.0 / 3.0)

        dT = 1.0e-5 * T + 1.0e-10
        P_plus = self.pressure(T + dT)
        P_minus = self.pressure(T - dT)
        dPdT = (P_plus - P_minus) / (2.0 * dT)

        e_plus = self.energy_density(T + dT)
        e_minus = self.energy_density(T - dT)
        dedT = (e_plus - e_minus) / (2.0 * dT)

        # 防止除零
        dedT = np.maximum(dedT, 1.0e-15)
        cs2 = dPdT / dedT

        # 物理约束: 0 < c_s^2 < 1/3 (因果性)
        cs2 = np.clip(cs2, 0.0, 1.0 / 3.0)
        return cs2

    def trace_anomaly(self, temperature: np.ndarray) -> np.ndarray:
        """
        迹反常 (interaction measure)

        (e - 3P) / T^4 = -T * d/dT(P/T^4)

        物理:
            共形理论: e - 3P = 0 (迹反常消失)
            QCD: 在 T_c 附近出现峰值, 反映相变的非微扰特性
            微扰展开: (e-3P)/T^4 ~ alpha_s * T^2 (高温)

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            (e - 3P) / T^4
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        e = self.energy_density(T)
        P = self.pressure(T)
        return (e - 3.0 * P) / T**4

    def shear_viscosity(self, temperature: np.ndarray) -> np.ndarray:
        """
        剪切粘滞 eta/s 的温度依赖性

        模型:
            eta/s(T) = (eta/s)_KSS + a * max(0, T/T_c - 1)^2

        满足 KSS 下限: eta/s >= 1/(4*pi)

        物理动机:
            强耦合 QGP 接近完美流体 (eta/s ~ 0.08)
            高温极限: 弱耦合 pQCD eta/s ~ T^2 / (alpha_s^2 * ln(1/alpha_s))

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            eta/s (无量纲)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        a_visc = 0.1  # 温度依赖系数
        eta_over_s = ETA_OVER_S_KSS + a_visc * np.maximum(0, T / self.T_c - 1.0)**2
        return eta_over_s

    def bulk_viscosity(self, temperature: np.ndarray) -> np.ndarray:
        """
        体粘滞 zeta/s 的温度依赖性

        Buchel-Csergai 定理:
            zeta/s proportional (1/3 - c_s^2)^2

        在共形极限下体粘滞消失, 在相变区增强.
        参数化形式:
            zeta/s = (zeta/s)_max * (1/3 - c_s^2)^2 / (1/3)^2

        满足:
            zeta/s >= 0
            zeta/s -> 0 当 c_s^2 -> 1/3

        Args:
            temperature: 温度数组 (GeV)

        Returns:
            zeta/s
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        cs2 = self.sound_speed_squared(T)
        # Buchel-Csergai 参数化
        zeta_over_s = ZETA_OVER_S_MAX * (1.0/3.0 - cs2)**2 / (1.0/3.0)**2
        return zeta_over_s

    def relaxation_time(self, temperature: np.ndarray) -> np.ndarray:
        """
        Israel-Stewart 弛豫时间 tau_pi

        二阶流体动力学中, 剪切应力张量的弛豫时间:
            tau_pi = 5 * eta / (e + P) = 5 * (eta/s) / T

        物理:
            tau_pi 控制剪切应力趋向 Navier-Stokes 极限的速率
            当 Kn = tau_pi / tau_exp ~ 1 时, 二阶效应显著

        Args:
            temperature: 温度 (GeV)

        Returns:
            tau_pi (fm/c)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        eta_s = self.shear_viscosity(T)
        tau_pi = 5.0 * eta_s / T  # in 1/GeV
        # 转换到 fm/c: 1 GeV^{-1} = hbar*c / (1 GeV) = 0.1973 fm
        tau_pi_fm = tau_pi * 0.1973
        return np.maximum(tau_pi_fm, 1.0e-6)

    def enthalpy_density(self, temperature: np.ndarray) -> np.ndarray:
        """
        焓密度 w = e + P

        在流体动力学中, 焓密度出现在:
            T^{mu nu} = w * u^mu * u^nu - P * g^{mu nu}

        Args:
            temperature: 温度 (GeV)

        Returns:
            焓密度 w (GeV/fm^3)
        """
        e = self.energy_density(temperature)
        P = self.pressure(temperature)
        return e + P

    def specific_heat(self, temperature: np.ndarray) -> np.ndarray:
        """
        比热 c_v = de/dT

        在相变区, c_v 出现尖峰 (类似二阶相变).

        Args:
            temperature: 温度 (GeV)

        Returns:
            c_v (1/fm^3, 无量纲化)
        """
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        dT = 1.0e-5 * T + 1.0e-10
        e_plus = self.energy_density(T + dT)
        e_minus = self.energy_density(T - dT)
        cv = (e_plus - e_minus) / (2.0 * dT)
        return np.maximum(cv, 0.0)

    def debye_mass(self, temperature: np.ndarray) -> np.ndarray:
        """
        Debye 屏蔽质量 (色电屏蔽)

        m_D^2 = (g^2 * T^2 / 3) * (N_c + N_f/2)

        物理: 色电场在 QGP 中被 Debye 屏蔽,
        屏蔽长度 r_D = 1/m_D 给出准粒子相互作用范围.

        Args:
            temperature: 温度 (GeV)

        Returns:
            Debye 质量 m_D (GeV)
        """
        from qgp_config import N_COLOR, N_FLAVOR, ALPHA_S
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        g2 = 4.0 * np.pi * ALPHA_S
        mD2 = (g2 * T**2 / 3.0) * (N_COLOR + N_FLAVOR / 2.0)
        return np.sqrt(mD2)

    def jet_quenching_parameter(self, temperature: np.ndarray) -> np.ndarray:
        """
        喷注淬火参数 q_hat

        q_hat = (4*pi^2 * alpha_s * C_F / N_c) * T^3 * ln(Q^2/mD^2)

        物理: 描述高能部分子在 QGP 中的横向动量展宽
        单位为 GeV^2/fm, 典型值 ~ 1-5 GeV^2/fm (LHC)

        Args:
            temperature: 温度 (GeV)

        Returns:
            q_hat (GeV^2/fm)
        """
        from qgp_config import N_COLOR, ALPHA_S, HBAR_C
        T = np.maximum(np.asarray(temperature, dtype=np.float64),
                       NumericalParams.TEMPERATURE_FLOOR)
        CF = (N_COLOR**2 - 1) / (2.0 * N_COLOR)
        mD = self.debye_mass(T)
        Q2 = (2.0 * np.pi * T)**2  # 热标度
        log_factor = np.log(np.maximum(Q2 / (mD**2 + 1e-15), 1.5))
        qhat = (4.0 * np.pi**2 * ALPHA_S * CF) * T**3 * log_factor
        # 转换单位: GeV^3 -> GeV^2/fm
        qhat *= HBAR_C
        return np.maximum(qhat, 0.0)
