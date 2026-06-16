"""
qgp_exact_solutions.py — 精确解与基准测试: Bjorken 流与 Gubser 流
=====================================================================

融合种子项目: 018_arenstorf_ode (周期性轨道精确解验证)

本模块提供 QGP 流体动力学方程的精确/半精确解,
用于验证数值格式的精度和收敛性.
类比 018_arenstorf_ode 中使用 Jacobi 常数验证轨道积分器.

Bjorken 纵向膨胀流 (1+1D):
----------------------------
假设: boost-invariant, 横向均匀
    解仅依赖于 tau = sqrt(t^2 - z^2)

    e(tau) = e_0 * (tau_0/tau)^{1+c_s^{-2}}
    共形 EoS: e ~ tau^{-4/3}
    T(tau) = T_0 * (tau_0/tau)^{1/3}

    精确解用于验证:
    - 纵向膨胀的 1/tau 稀释
    - 能量守恒 (包括膨胀功)
    - 温度标度律

Gubser 流 (2+1D 共形解):
-------------------------
Gubser (2010) 给出了共形流体动力学的精确解:
    在 de Sitter 坐标 (rho, theta, phi, eta_s) 中:

    e(rho) = e_0 * (cosh rho)^{-4} * f(tanh rho)

    其中 rho 为无量纲时间变量:
        tanh rho = (tau^2 - R^2) / (tau^2 + R^2)
        R = sqrt(x^2 + y^2) 横向半径

    简化: 在横向均匀极限, Gubser 流退化为:
        e(tau) = e_0 * (tau_0/tau)^{4/3} * h(tau)

    其中 h(tau) 为超几何函数修正.

粘滞修正 Bjorken 流:
    对 Navier-Stokes 修正:
        e(tau) = e_Bjorken * [1 + 2*eta/(s*T_0*tau_0) *
                   ((tau_0/tau)^{2/3} - 1)]

    对 Israel-Stewart 二阶修正:
        Phi = pi^{eta eta} 满足弛豫方程:
        tau_pi * dPhi/dtau + Phi = -4*eta/(3*tau) + ...

守恒量验证 (源自 208_conservation_ode):
    Bjorken 流中, 守恒量为:
        H(tau) = tau * e(tau) + integral_0^tau P(tau') dtau'
    应为常数.
"""

import numpy as np
from qgp_config import NumericalParams, SIGMA_SB, HBAR_C
from qgp_eos import QGPEquationOfState
from qgp_grid import QGPGrid


class BjorkenFlow:
    """
    Bjorken 纵向膨胀流精确解

    1+1 维 boost-invariant 流体动力学:
        de/dtau = -(e + P) / tau

    共形 EoS (P = e/3):
        e(tau) = e_0 * (tau_0/tau)^{4/3}
        T(tau) = T_0 * (tau_0/tau)^{1/3}
        P(tau) = e(tau) / 3
    """

    def __init__(self, eos: QGPEquationOfState):
        self.eos = eos

    def energy_density(self, tau: float, tau_0: float,
                        e_0: float) -> float:
        """
        Bjorken 能量密度精确解 (共形 EoS)

        e(tau) = e_0 * (tau_0/tau)^{4/3}

        推导:
            de/dtau = -(e + P)/tau = -(4/3)*e/tau
            => de/e = -(4/3) * dtau/tau
            => ln(e) = -(4/3)*ln(tau) + const
            => e = C * tau^{-4/3}

        Args:
            tau: 固有时间 (fm/c)
            tau_0: 初始时间 (fm/c)
            e_0: 初始能量密度 (GeV/fm^3)

        Returns:
            e(tau) (GeV/fm^3)
        """
        return e_0 * (tau_0 / max(tau, 1.0e-10))**(4.0/3.0)

    def temperature(self, tau: float, tau_0: float,
                     T_0: float) -> float:
        """
        Bjorken 温度 (共形 EoS)

        T(tau) = T_0 * (tau_0/tau)^{1/3}

        来自 e ~ T^4 => T ~ e^{1/4} ~ tau^{-1/3}

        Args:
            tau: 固有时间
            tau_0: 初始时间
            T_0: 初始温度 (GeV)

        Returns:
            T(tau) (GeV)
        """
        return T_0 * (tau_0 / max(tau, 1.0e-10))**(1.0/3.0)

    def pressure(self, tau: float, tau_0: float,
                  e_0: float) -> float:
        """Bjorken 压力 (共形: P = e/3)"""
        return self.energy_density(tau, tau_0, e_0) / 3.0

    def viscous_correction(self, tau: float, tau_0: float,
                            e_0: float, eta_over_s: float) -> float:
        """
        Navier-Stokes 粘滞修正

        e_NS(tau) = e_Bj(tau) * [1 + (4*eta)/(3*s*T_0*tau_0) *
                        ((tau_0/tau)^{2/3} - 1)]

        修正项来自剪切应力 pi^{eta eta} = 4*eta/(3*tau)

        当 eta/s -> 0: 回到理想 Bjorken 流
        当 tau -> tau_0: 修正为零

        Args:
            tau: 固有时间
            tau_0: 初始时间
            e_0: 初始能量密度
            eta_over_s: 剪切粘滞比

        Returns:
            e_viscous(tau)
        """
        e_bj = self.energy_density(tau, tau_0, e_0)
        T_0 = (e_0 / SIGMA_SB)**0.25

        # 修正因子
        ratio = tau_0 / max(tau, 1.0e-10)
        correction = 1.0 + (4.0 * eta_over_s) / (3.0 * T_0 * tau_0) * \
                     (ratio**(2.0/3.0) - 1.0)

        return e_bj * max(correction, 0.1)  # 防止负能量密度

    def conserved_quantity(self, tau: float, tau_0: float,
                            e_0: float) -> float:
        """
        Bjorken 守恒量 (源自 208_conservation_ode)

        H = tau * e + integral_{tau_0}^{tau} P(tau') dtau'

        对于共形 EoS:
            H = tau * e + (1/3) * integral e dtau'
            = tau * e_0 * (tau_0/tau)^{4/3} + e_0*tau_0^{4/3} *
              (1/3) * integral_{tau_0}^{tau} tau'^{-4/3} dtau'
            = tau_0 * e_0 * [1/3 + 2/3 * (tau_0/tau)^{1/3}]

        应近似为常数 (tau_0 * e_0).

        Args:
            tau: 固有时间
            tau_0: 初始时间
            e_0: 初始能量密度

        Returns:
            H(tau)
        """
        e_tau = self.energy_density(tau, tau_0, e_0)
        # 数值积分
        n_pts = 100
        tau_arr = np.linspace(tau_0, tau, n_pts)
        P_arr = np.array([self.pressure(t, tau_0, e_0) for t in tau_arr])
        integral = np.trapz(P_arr, tau_arr)
        return tau * e_tau + integral

    def generate_reference_solution(self, grid: QGPGrid,
                                     tau: float, tau_0: float,
                                     T_0: float) -> tuple:
        """
        在 2D 网格上生成 Bjorken 参考解

        假设横向均匀: e 只依赖 tau, 不依赖 (x, y)

        Args:
            grid: 计算网格
            tau: 当前固有时间
            tau_0: 初始时间
            T_0: 初始温度 (GeV)

        Returns:
            (e_ref, vx_ref, vy_ref, T_ref) 内部区域
        """
        ny, nx = grid.ny, grid.nx
        e_0 = SIGMA_SB * T_0**4

        e_ref = np.full((ny, nx), self.energy_density(tau, tau_0, e_0))
        T_ref = np.full((ny, nx), self.temperature(tau, tau_0, T_0))
        vx_ref = np.zeros((ny, nx))
        vy_ref = np.zeros((ny, nx))

        return e_ref, vx_ref, vy_ref, T_ref


class GubserFlow:
    """
    Gubser 精确解 (共形流体力学 2+1D)

    Gubser (2010) 精确解描述了横向膨胀的共形流体:

    在 de Sitter 坐标中:
        ds^2 = d_rho^2 + cosh^2(rho) * (d_theta^2 + sin^2(theta)*d_phi^2)
               + sinh^2(rho) * d_eta^2

    解:
        T(rho) = T_0 / cosh(rho) * [1/(1 + (sinh(rho))^2)]^{1/3}

    坐标变换:
        tanh(rho) = (tau^2 - R^2)/(tau^2 + R^2)
        其中 R = sqrt(x^2 + y^2)

    简化形式 (横向均匀极限):
        e(tau) ~ tau^{-4/3} (与 Bjorken 相同)
"""

    def __init__(self, eos: QGPEquationOfState):
        self.eos = eos

    def gubser_rho(self, tau: float, r: float) -> float:
        """
        Gubser 坐标 rho(tau, r)

        tanh(rho) = (tau^2 - r^2) / (tau^2 + r^2)

        Args:
            tau: 固有时间 (fm/c)
            r: 横向半径 (fm)

        Returns:
            rho (无量纲)
        """
        tau2 = tau**2
        r2 = r**2
        denom = tau2 + r2
        if denom < 1.0e-15:
            return 0.0
        tanh_rho = (tau2 - r2) / denom
        tanh_rho = np.clip(tanh_rho, -0.999, 0.999)
        return np.arctanh(tanh_rho)

    def temperature_gubser(self, tau: float, r: float,
                            T_0: float, tau_0: float) -> float:
        """
        Gubser 温度分布

        T = T_0 * (tau_0/tau)^{1/3} * F(rho)

        其中 F(rho) 为横向修正函数:
            F(rho) = [cosh(rho)]^{-1} * [1 + tanh^2(rho)]^{-1/6}

        当 r -> 0 (rho -> 0): F -> 1, 退化为 Bjorken
        当 r -> infinity: F -> 0, 温度衰减更快

        Args:
            tau: 固有时间
            r: 横向半径
            T_0: 初始温度
            tau_0: 初始时间

        Returns:
            T(tau, r)
        """
        rho = self.gubser_rho(tau, r)
        rho_0 = self.gubser_rho(tau_0, r)

        # Bjorken 部分
        T_bj = T_0 * (tau_0 / max(tau, 1.0e-10))**(1.0/3.0)

        # Gubser 横向修正
        cosh_rho = np.cosh(rho)
        tanh_rho = np.tanh(rho)
        F_rho = 1.0 / (cosh_rho * (1.0 + tanh_rho**2)**(1.0/6.0) + 1.0e-10)

        return T_bj * F_rho

    def generate_reference(self, grid: QGPGrid, tau: float,
                            tau_0: float, T_0: float) -> tuple:
        """
        在 2D 网格上生成 Gubser 参考解

        Args:
            grid: 计算网格
            tau: 当前时间
            tau_0: 初始时间
            T_0: 初始温度

        Returns:
            (e_ref, T_ref) 内部区域
        """
        ny, nx = grid.ny, grid.nx
        X = grid.X_int
        Y = grid.Y_int
        R = np.sqrt(X**2 + Y**2)

        T_ref = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                T_ref[i, j] = self.temperature_gubser(
                    tau, R[i, j], T_0, tau_0)

        e_ref = SIGMA_SB * T_ref**4
        return e_ref, T_ref


def compute_error_norm(numerical: np.ndarray, exact: np.ndarray,
                        norm_type: str = 'L2') -> float:
    """
    计算数值解与精确解之间的误差范数

    范数类型:
    - L1: ||e||_1 = sum |e_i| / N
    - L2: ||e||_2 = sqrt(sum e_i^2 / N)
    - Linf: ||e||_inf = max |e_i|
    - L2_rel: ||e||_2 / ||exact||_2 (相对误差)

    Args:
        numerical: 数值解
        exact: 精确解
        norm_type: 范数类型

    Returns:
        误差值
    """
    error = numerical - exact

    if norm_type == 'L1':
        return float(np.mean(np.abs(error)))
    elif norm_type == 'L2':
        return float(np.sqrt(np.mean(error**2)))
    elif norm_type == 'Linf':
        return float(np.max(np.abs(error)))
    elif norm_type == 'L2_rel':
        exact_norm = np.sqrt(np.mean(exact**2))
        if exact_norm < 1.0e-15:
            return 0.0
        return float(np.sqrt(np.mean(error**2)) / exact_norm)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def convergence_order(errors: list, dx_values: list) -> float:
    """
    从误差-网格间距数据计算收敛阶

    error ~ C * dx^p
    => log(error) = log(C) + p * log(dx)
    => p = slope of log-log plot

    Args:
        errors: 不同分辨率下的误差列表
        dx_values: 对应的网格间距列表

    Returns:
        收敛阶 p
    """
    log_e = np.log(np.array(errors) + 1.0e-30)
    log_dx = np.log(np.array(dx_values) + 1.0e-30)

    if len(log_e) < 2:
        return 0.0

    coeffs = np.polyfit(log_dx, log_e, 1)
    return float(coeffs[0])
