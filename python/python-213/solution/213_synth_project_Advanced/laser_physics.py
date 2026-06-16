"""
laser_physics.py - 超辐射激光物理模型模块
=========================================

融合种子项目:
  - 1199_hrish573_mhz-linewidth-laser-interview: mHz 线宽超辐射激光模型

核心数学: 激光物理中的凸优化问题

坏腔超辐射激光模型 (Bad-cavity superradiant laser):
  该系统描述了原子系综与光腔耦合的集体辐射现象.

  宏观变量:
    D: 粒子数反转 (population inversion)
    J_x, J_y, J_z: Bloch 矢量的宏观分量
    alpha: 腔场振幅

  运动方程 (Maxwell-Bloch 方程):
    dJ_x/dt = -Delta * J_y - gamma_perp * J_x
    dJ_y/dt = Delta * J_x - gamma_perp * J_y + g * (alpha^* D + alpha D)
    dD/dt = -2g * (alpha^* J_+ + alpha J_-) - gamma_par * (D - D_0)
    dalpha/dt = -(kappa + i*Delta_c) * alpha + g * J_-

  其中:
    g: 原子-光耦合常数
    kappa: 腔衰减率
    gamma_perp: 横向弛豫率
    gamma_par: 纵向弛豫率
    Delta: 原子失谐
    Delta_c: 腔失谐
    D_0: 泵浦粒子数反转

  在稳态下 (d/dt = 0), 系统变为非线性代数方程组.
  优化目标: 调节泵浦 D_0 和腔参数使线宽最小.

  线宽公式 (Schawlow-Townes 极限):
    Delta_nu = (pi * h * nu * (Delta_nu_c)^2) / P_output

  其中 Delta_nu_c = kappa / (2*pi) 是腔线宽.

  优化问题:
    min_{D_0, kappa} Delta_nu
    s.t. 稳态 Maxwell-Bloch 方程
         D_0_min <= D_0 <= D_0_max
         kappa_min <= kappa <= kappa_max

  这可以转化为凸优化问题 (在适当的参数变换下).
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SuperradiantLaser:
    """
    坏腔超辐射激光模型 (融合 1199).

    简化版 Maxwell-Bloch 方程的稳态求解.

    在超辐射状态下 (g*N >> kappa >> gamma_perp):
      输出线宽远小于 Schawlow-Townes 极限.
      Delta_nu_SR ~ gamma_perp * (kappa / (g*N))^2

    Parameters
    ----------
    N_atoms : int
        原子数
    g : float
        单原子耦合强度 (rad/s)
    kappa : float
        腔衰减率 (rad/s)
    gamma_perp : float
        横向弛豫率 (rad/s)
    gamma_par : float
        纵向弛豫率 (rad/s)
    """

    def __init__(self, N_atoms: int = 1000, g: float = 1.0e4,
                 kappa: float = 1.0e6, gamma_perp: float = 1.0,
                 gamma_par: float = 0.1):
        self.N = N_atoms
        self.g = g
        self.kappa = kappa
        self.gamma_perp = gamma_perp
        self.gamma_par = gamma_par

    def cooperativity(self) -> float:
        """
        计算协作参数 C.

        C = g^2 * N / (kappa * gamma_perp)

        物理意义:
          C >> 1: 强耦合 (超辐射态)
          C << 1: 弱耦合 (单原子辐射)
          C ~ 1: 中间区域

        Returns
        -------
        float
            协作参数
        """
        return self.g**2 * self.N / (self.kappa * self.gamma_perp)

    def steady_state_inversion(self, pump_rate: float) -> float:
        """
        计算稳态粒子数反转.

        在坏腔极限 (kappa >> g*sqrt(N)):
          D_ss = -D_0 * gamma_par / (gamma_par + 4*g^2*N/kappa)

        其中 D_0 = pump_rate / gamma_par 是泵浦决定的反转.

        Parameters
        ----------
        pump_rate : float
            泵浦速率

        Returns
        -------
        float
            稳态反转
        """
        D_0 = pump_rate / self.gamma_par
        effective_decay = self.gamma_par + 4.0 * self.g**2 * self.N / self.kappa
        D_ss = -D_0 * self.gamma_par / effective_decay
        return D_ss

    def superradiant_linewidth(self) -> float:
        """
        计算超辐射线宽.

        Delta_nu_SR = (gamma_perp / (2*pi)) * (kappa / (g*N))^2 * (1 + 1/C)

        当 C >> 1:
          Delta_nu_SR ~ gamma_perp / (2*pi) * (kappa/(g*N))^2

        这远小于普通激光的 Schawlow-Townes 线宽.

        Returns
        -------
        float
            线宽 (Hz)
        """
        C = self.cooperativity()
        ratio = self.kappa / (self.g * self.N)
        linewidth = (self.gamma_perp / (2.0 * np.pi)) * ratio**2 * (1.0 + 1.0 / C)
        return linewidth

    def schawlow_townes_linewidth(self, output_power: float,
                                    nu: float = 5.0e14) -> float:
        """
        Schawlow-Townes 线宽 (基准比较).

        Delta_nu_ST = (pi * h * nu * kappa^2) / (4 * P_output)

        Parameters
        ----------
        output_power : float
            输出功率 (W)
        nu : float
            激光频率 (Hz)

        Returns
        -------
        float
            ST 线宽 (Hz)
        """
        h_planck = 6.626e-34  # Planck 常数
        if output_power < 1.0e-30:
            return np.inf
        delta_nu_st = (np.pi * h_planck * nu * self.kappa**2) / (
            4.0 * output_power)
        return delta_nu_st

    def optimize_for_narrow_linewidth(self, pump_range: Tuple[float, float],
                                       n_samples: int = 50
                                       ) -> Dict[str, float]:
        """
        优化泵浦参数以获得最窄线宽.

        这是一个一维凸优化问题:
          min_{pump} Delta_nu(pump)
          s.t. pump_min <= pump <= pump_max

        使用黄金分割搜索 (斐波那契法的连续版本):
          每次迭代将搜索区间缩小 (sqrt(5)-1)/2 ~ 0.618 倍.

        Parameters
        ----------
        pump_range : tuple
            泵浦速率范围 (min, max)
        n_samples : int
            评估点数

        Returns
        -------
        dict
            最优参数和结果
        """
        a, b = pump_range
        golden = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618

        c = b - golden * (b - a)
        d = a + golden * (b - a)

        for _ in range(n_samples):
            lw_c = self.superradiant_linewidth()  # 简化: 线宽不直接依赖泵浦
            # 在实际问题中, 线宽与稳态反转和输出功率有关
            # 这里使用一个与泵浦相关的有效线宽

            # 有效线宽 (含泵浦效应)
            D_ss_c = self.steady_state_inversion(c)
            D_ss_d = self.steady_state_inversion(d)

            # 简化模型: 线宽 ~ gamma_perp * |D_ss|
            lw_c_eff = self.gamma_perp * abs(D_ss_c) / (2.0 * np.pi)
            lw_d_eff = self.gamma_perp * abs(D_ss_d) / (2.0 * np.pi)

            if lw_c_eff < lw_d_eff:
                b = d
            else:
                a = c

            c = b - golden * (b - a)
            d = a + golden * (b - a)

        optimal_pump = 0.5 * (a + b)
        D_ss_opt = self.steady_state_inversion(optimal_pump)
        lw_opt = self.superradiant_linewidth()

        return {
            'optimal_pump': optimal_pump,
            'steady_state_inversion': D_ss_opt,
            'superradiant_linewidth': lw_opt,
            'cooperativity': self.cooperativity(),
            'search_interval': (a, b)
        }

    def maxwell_bloch_rhs(self, state: np.ndarray, pump_rate: float,
                           detuning: float = 0.0) -> np.ndarray:
        """
        Maxwell-Bloch 方程右端函数.

        状态向量: [J_x, J_y, D, alpha_r, alpha_i]
        其中 alpha = alpha_r + i * alpha_i 是腔场.

        dJ_x/dt = -detuning * J_y - gamma_perp * J_x
        dJ_y/dt = detuning * J_x - gamma_perp * J_y + 2*g*alpha_r*D
        dD/dt = -2*g*(alpha_r*J_y - alpha_i*J_x) - gamma_par*(D - D_0)
        dalpha_r/dt = -kappa*alpha_r + detuning*alpha_i + g*J_x
        dalpha_i/dt = -kappa*alpha_i - detuning*alpha_r + g*J_y

        Parameters
        ----------
        state : ndarray, shape (5,)
            状态向量
        pump_rate : float
            泵浦速率
        detuning : float
            原子-腔失谐

        Returns
        -------
        ndarray
            右端函数值
        """
        J_x, J_y, D, alpha_r, alpha_i = state
        D_0 = pump_rate / self.gamma_par
        g = self.g

        dJx = -detuning * J_y - self.gamma_perp * J_x
        dJy = detuning * J_x - self.gamma_perp * J_y + 2.0 * g * alpha_r * D
        dD = (-2.0 * g * (alpha_r * J_y - alpha_i * J_x)
              - self.gamma_par * (D - D_0))
        dalpha_r = -self.kappa * alpha_r + detuning * alpha_i + g * J_x
        dalpha_i = -self.kappa * alpha_i - detuning * alpha_r + g * J_y

        return np.array([dJx, dJy, dD, dalpha_r, dalpha_i])
