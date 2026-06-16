"""
qgp_conservation.py — 守恒律与守恒量: 能动张量与重子流
========================================================

融合种子项目: 208_conservation_ode (守恒 ODE 系统)

本模块实现相对论流体力学的守恒律方程, 将物理守恒量
编码为离散形式, 保持能量-动量守恒的数值不变量.

守恒律方程 (源自 208_conservation_ode 思想):
-------------------------------------------

相对论流体力学的核心是能量-动量张量守恒:

    partial_mu T^{mu nu} = 0

和重子流守恒:

    partial_mu N^mu = 0

在 Milne 坐标系 (tau, x, y, eta_s) 中, 对于 boost-invariant 流:

    d tau D + d_x (v_x * D_tilde) + d_y (v_y * D_tilde) = -D / tau
    d tau S_x + d_x (T_xx_tilde) + d_y (T_xy_tilde) = T^{eta eta} * tau - T_xx_tilde / tau  (几何源项)

其中守恒变量:
    D = gamma * n_B * tau  (修正重子密度)
    S_x = tau * T^{0x} = tau * w * gamma^2 * v_x  (x 动量密度)
    S_y = tau * T^{0y} = tau * w * gamma^2 * v_y  (y 动量密度)

守恒量 (源自 208_conservation_ode 的 conserved quantity 概念):
    H_total = integral (T^{00}) dx dy = 常数 (在封闭系统中)
    B_total = integral (n_B) dx dy = 常数 (重子数守恒)

理想流体能动张量:
    T^{mu nu} = (e + P) * u^mu * u^nu - P * g^{mu nu}
    = w * u^mu * u^nu - P * g^{mu nu}

其中 w = e + P 为焓密度, u^mu = gamma*(1, vx, vy, 0) 为四速度.

非理想修正 (Israel-Stewart 二阶理论):
    T^{mu nu} = T^{mu nu}_ideal + pi^{mu nu}
    pi^{mu nu} = 2*eta*sigma^{mu nu} + Delta^{mu nu}*zeta*theta
    (剪切应力 + 体粘滞)
"""

import numpy as np
from qgp_config import NumericalParams, lorentz_factor
from qgp_eos import QGPEquationOfState
from qgp_grid import QGPGrid


class ConservedVariables:
    """
    守恒变量管理

    管理三组守恒量 (D, Sx, Sy) 及其与原始变量 (e, vx, vy) 的变换.

    守恒变量定义 (Milne 坐标, boost-invariant):
        D = gamma * n_B * tau     (修正重子密度)
        S_x = w * gamma^2 * v_x * tau   (x-动量密度)
        S_y = w * gamma^2 * v_y * tau   (y-动量密度)

    原始变量:
        e: 固有能量密度 (GeV/fm^3)
        vx, vy: 横向流速 (自然单位 c=1)
        n_B: 重子密度

    变换关系:
        w = e + P(e)  (焓密度, 通过 EoS)
        gamma = 1 / sqrt(1 - vx^2 - vy^2)  (Lorentz 因子)
        S = sqrt(Sx^2 + Sy^2)  (动量密度模)
    """

    def __init__(self, eos: QGPEquationOfState):
        """
        Args:
            eos: QCD 状态方程
        """
        self.eos = eos

    def primitive_to_conserved(self, e: np.ndarray, vx: np.ndarray,
                                vy: np.ndarray, nB: np.ndarray,
                                tau: float) -> tuple:
        """
        原始变量 -> 守恒变量

        变换公式:
            w = e + P(e)
            gamma = 1/sqrt(1 - vx^2 - vy^2)
            D = gamma * nB * tau
            Sx = w * gamma^2 * vx * tau
            Sy = w * gamma^2 * vy * tau

        因果性约束:
            vx^2 + vy^2 < 1 (流速不能超过光速)
            如果违反, 截断到 v_max = 0.99

        Args:
            e: 能量密度 (含鬼单元)
            vx, vy: 流速
            nB: 重子密度
            tau: 当前固有时间 (fm/c)

        Returns:
            (D, Sx, Sy) 守恒变量 (含鬼单元)
        """
        # 因果性限制
        v2 = vx**2 + vy**2
        v2_safe = np.minimum(v2, 0.99**2)
        scale = np.ones_like(v2)
        mask = v2 > 0.99**2
        if np.any(mask):
            scale[mask] = 0.99 / np.sqrt(v2_safe[mask])
            vx = vx * scale
            vy = vy * scale

        gamma = lorentz_factor(vx, vy) if np.isscalar(vx) else \
                1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))

        P = self.eos.pressure(e)
        w = e + P  # 焓密度

        D = gamma * nB * tau
        Sx = w * gamma**2 * vx * tau
        Sy = w * gamma**2 * vy * tau

        return D, Sx, Sy

    def conserved_to_primitive(self, D: np.ndarray, Sx: np.ndarray,
                                Sy: np.ndarray, tau: float) -> tuple:
        """
        守恒变量 -> 原始变量 (Newton-Raphson 迭代)

        逆变换需要求解非线性方程:
            S^2 = (Sx^2 + Sy^2)
            S^2 = w^2 * gamma^4 * v^2 * tau^2
            = w^2 * gamma^2 * (gamma^2 - 1) * tau^2

        定义:
            r = S / (tau * D)  (动量-密度比)
            需要求解 gamma:
            gamma^2 - 1 = r^2 * gamma^2 * w(e) / (gamma * nB)

        迭代:
            1. 初始猜测 gamma = 1
            2. 由 D, gamma 得 nB = D / (gamma * tau)
            3. 由 S, gamma 得 v = S / (w * gamma^2 * tau)
            4. 更新 gamma = 1/sqrt(1-v^2)
            5. 重复至收敛

        物理约束:
            e >= e_floor > 0
            gamma >= 1
            v^2 < 1

        Args:
            D, Sx, Sy: 守恒变量
            tau: 当前固有时间

        Returns:
            (e, vx, vy, nB) 原始变量
        """
        # 防止除零
        D_safe = np.maximum(np.abs(D), 1.0e-15)
        S2 = Sx**2 + Sy**2
        S = np.sqrt(S2)

        # 初始猜测: 非相对论极限 gamma ≈ 1
        gamma = np.ones_like(D)
        e = np.maximum(D_safe / tau, NumericalParams.ENERGY_FLOOR)

        # Newton-Raphson 迭代 (最多 30 次)
        for iteration in range(30):
            P = self.eos.pressure(e)
            w = e + P  # 焓密度
            w = np.maximum(w, 1.0e-15)

            nB = D_safe / (gamma * tau)

            # 速度
            v2 = S2 / (w**2 * gamma**4 * tau**2 + 1.0e-30)
            v2 = np.minimum(v2, 0.99**2)  # 因果性

            # 更新 gamma
            gamma_new = 1.0 / np.sqrt(1.0 - v2)
            gamma_new = np.minimum(gamma_new, 100.0)  # 防止极端 Lorentz 因子

            # 收敛判断
            if np.max(np.abs(gamma_new - gamma)) < 1.0e-10:
                gamma = gamma_new
                break
            gamma = gamma_new

        # 最终原始变量
        P = self.eos.pressure(e)
        w = np.maximum(e + P, 1.0e-15)
        vx = Sx / (w * gamma**2 * tau + 1.0e-30)
        vy = Sy / (w * gamma**2 * tau + 1.0e-30)

        # 因果性最终检查
        v2 = vx**2 + vy**2
        mask = v2 > 0.99**2
        if np.any(mask):
            v_scale = np.ones_like(v2)
            v_scale[mask] = 0.99 / np.sqrt(v2[mask])
            vx *= v_scale
            vy *= v_scale

        nB = D_safe / (gamma * tau)
        e = np.maximum(e, NumericalParams.ENERGY_FLOOR)

        return e, vx, vy, nB

    def compute_flux_x(self, D: np.ndarray, Sx: np.ndarray,
                        Sy: np.ndarray, tau: float) -> tuple:
        """
        x 方向的物理通量 (F_x, F_y, F_D)

        守恒律: d tau D + d_x F_D = ...
        通量:
            F_D = v_x * D_tilde  (重子流)
            F_Sx = T^{xx} * tau  (x-动量流)
            F_Sy = T^{xy} * tau  (xy-剪切流)

        T^{xx} = w * gamma^2 * vx^2 + P
        T^{xy} = w * gamma^2 * vx * vy

        Args:
            D, Sx, Sy: 守恒变量
            tau: 固有时间

        Returns:
            (F_D, F_Sx, F_Sy) x 方向通量
        """
        e, vx, vy, nB = self.conserved_to_primitive(D, Sx, Sy, tau)
        P = self.eos.pressure(e)
        w = e + P
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))

        F_D = vx * D
        F_Sx = (w * gamma**2 * vx**2 + P) * tau
        F_Sy = w * gamma**2 * vx * vy * tau

        return F_D, F_Sx, F_Sy

    def compute_flux_y(self, D: np.ndarray, Sx: np.ndarray,
                        Sy: np.ndarray, tau: float) -> tuple:
        """
        y 方向的物理通量

        G_D = v_y * D
        G_Sx = T^{xy} * tau = w * gamma^2 * vx * vy * tau
        G_Sy = T^{yy} * tau = (w * gamma^2 * vy^2 + P) * tau

        Args:
            D, Sx, Sy: 守恒变量
            tau: 固有时间

        Returns:
            (G_D, G_Sx, G_Sy) y 方向通量
        """
        e, vx, vy, nB = self.conserved_to_primitive(D, Sx, Sy, tau)
        P = self.eos.pressure(e)
        w = e + P
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))

        G_D = vy * D
        G_Sx = w * gamma**2 * vx * vy * tau
        G_Sy = (w * gamma**2 * vy**2 + P) * tau

        return G_D, G_Sx, G_Sy

    def geometric_source(self, D: np.ndarray, Sx: np.ndarray,
                          Sy: np.ndarray, e: np.ndarray,
                          vx: np.ndarray, vy: np.ndarray,
                          tau: float) -> tuple:
        """
        Milne 坐标几何源项

        在 Bjorken 膨胀几何中, 守恒律包含几何源项:
            S_D = -D / tau  (Bjorken 膨胀稀释)
            S_Sx = -Sx / tau  (动量衰减)
            S_Sy = -Sy / tau
            S_energy = T^{eta eta} * tau  (纵向压力做功)

        纵向压力 T^{eta eta}:
            T^{eta eta} = w * (u^eta)^2 + P / tau^2
            对于 boost-invariant 流: u^eta = 0
            => T^{eta eta} = P / tau^2

        物理含义:
            Bjorken 膨胀导致横向能量密度以 1/tau 速率稀释
            纵向压力对横向动力学做负功

        Args:
            D, Sx, Sy: 守恒变量
            e: 能量密度
            vx, vy: 流速
            tau: 固有时间

        Returns:
            (src_D, src_Sx, src_Sy) 几何源项
        """
        P = self.eos.pressure(e)
        w = e + P
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))

        # Bjorken 膨胀源项
        src_D = -D / tau
        src_Sx = -Sx / tau
        src_Sy = -Sy / tau

        # 纵向压力贡献 (T^{eta eta} * tau = P/tau)
        # 这一项驱动横向膨胀
        src_energy = P / tau

        return src_D, src_Sx, src_Sy, src_energy

    def compute_max_characteristic_speed(self, e: np.ndarray,
                                          vx: np.ndarray,
                                          vy: np.ndarray) -> float:
        """
        计算最大特征速度 (用于 CFL 条件和 LF 分裂)

        特征速度 = |v| + c_s  (流速 + 声速)

        对于相对论流体:
            lambda_{pm} = (v_x ± c_s * sqrt(1-v^2)) / (1 ± v_x * c_s)

        全局最大特征速度:
            alpha = max over all cells of max(|lambda_+|, |lambda_-|)

        Args:
            e: 能量密度
            vx, vy: 流速

        Returns:
            最大特征速度 alpha
        """
        cs2 = self.eos.sound_speed_squared(e)
        cs = np.sqrt(np.maximum(cs2, 0.0))

        v_mag = np.sqrt(vx**2 + vy**2)
        v_mag = np.minimum(v_mag, 0.99)

        # 相对论特征速度
        sqrt_1_minus_v2 = np.sqrt(1.0 - v_mag**2)
        lambda_plus = np.abs(v_mag + cs * sqrt_1_minus_v2)
        lambda_minus = np.abs(v_mag - cs * sqrt_1_minus_v2)

        alpha = float(np.maximum(np.max(lambda_plus), np.max(lambda_minus)))
        alpha = max(alpha, cs.flat[0] if cs.size > 0 else 0.1)
        return alpha

    def total_energy(self, e: np.ndarray, vx: np.ndarray,
                      vy: np.ndarray, grid: QGPGrid) -> float:
        """
        计算系统总能量 (守恒量监测)

        E_total = integral T^{00} dx dy
        T^{00} = w * gamma^2 - P

        对于封闭系统, E_total 应为常数.
        数值误差导致的变化反映守恒性.

        Args:
            e: 能量密度 (内部区域)
            vx, vy: 流速
            grid: 计算网格

        Returns:
            总能量 (GeV/fm)
        """
        P = self.eos.pressure(e)
        w = e + P
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))
        T00 = w * gamma**2 - P
        return grid.integrate_2d(T00)

    def total_baryon_number(self, nB: np.ndarray, vx: np.ndarray,
                             vy: np.ndarray, grid: QGPGrid) -> float:
        """
        计算系统总重子数

        B_total = integral gamma * nB dx dy

        Args:
            nB: 重子密度 (固有系)
            vx, vy: 流速
            grid: 计算网格

        Returns:
            总重子数
        """
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))
        return grid.integrate_2d(gamma * nB)

    def entropy_density_current(self, e: np.ndarray, vx: np.ndarray,
                                 vy: np.ndarray) -> np.ndarray:
        """
        熵流密度 s^mu = s * u^mu

        对于理想流体, 熵守恒:
            d_mu (s * u^mu) = 0

        对于粘滞流体, 熵产生:
            d_mu (s * u^mu) = (pi^{mu nu} * sigma_{mu nu}) / T >= 0

        其中:
            s = (e + P) / T (热力学熵密度)
            sigma_{mu nu} = 剪切张量

        熵增原理要求 dS/dt >= 0, 这是数值格式的物理约束.

        Args:
            e: 能量密度
            vx, vy: 流速

        Returns:
            s * gamma (实验室系熵密度)
        """
        P = self.eos.pressure(e)
        T = self.eos.temperature_from_energy(e)
        w = e + P
        s = w / np.maximum(T, 1.0e-10)
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))
        return s * gamma
