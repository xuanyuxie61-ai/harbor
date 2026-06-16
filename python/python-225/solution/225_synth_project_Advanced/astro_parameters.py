# -*- coding: utf-8 -*-
"""
astro_parameters.py
===================

天体物理参数与浮点机器常数模块

本模块封装暗物质直接探测实验所需的全部天体物理常数、标准晕模型（SHM）参数
以及浮点机器精度常数。机器精度通过 Malcolm–Gentleman–Maroney 算法动态测定
（对应种子项目 705_machar），用于确保在极小截面 σ ~ 1e-46 cm^2 量级的计算中
不发生灾难性对消。

科学公式
--------
标准晕模型速度分布:
    f_SHM(v) = (1 / (π^(3/2) v_0^3)) * exp(-|v + v_E|^2 / v_0^2)
                * Θ(v_esc - |v + v_E|)

归一化系数:
    N_esc = 1 - (1 + (v_esc/v_0)^2) * exp(-(v_esc/v_0)^2)

地球速度:
    v_E(t) = v_sun + v_orb * (cos γ * cos(ω(t - t_0)) + ...)

本地暗物质密度:
    ρ_0 = 0.3 GeV/cm^3 = 0.3 * 1.7827e-24 g/cm^3

Helm 核形状因子:
    F^2(q) = [3 j_1(q R_n) / (q R_n)]^2 * exp(-(q s)^2)
    R_n = sqrt(c^2 s^2 - 5),  c = 1.23 A^(1/3) - 0.60 fm,  s = 0.9 fm
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# 第一部分: Malcolm–Gentleman–Maroney 浮点机器常数测定 (源自 705_machar)
# ---------------------------------------------------------------------------

class MacharConstants:
    """
    通过 Malcolm–Gentleman–Maroney 算法动态确定 IEEE-754 双精度浮点参数。

    该算法依次确定:
      ibeta  - 浮点基数 (通常为 2)
      it     - 尾数位数 (双精度 = 53)
      irnd   - 舍入模式 (0=截断, 2=IEEE 舍入)
      machep - 最大负整数使得 1.0 + ibeta^machep > 1.0
      negep  - 最大负整数使得 1.0 - ibeta^negep < 1.0
      minexp - 最小指数
      maxexp - 最大指数
      eps    - 机器精度 eps = ibeta^machep
      xmin   - 最小正规格化数
      xmax   - 最大正有限数

    参考文献
    --------
    Cody, W. J. "ACM Algorithm 665: MACHAR," ACM TOMS 14(4), 303-311 (1988).
    """

    def __init__(self):
        self.ibeta, self.it, self.irnd, self.ngrd, self.machep, \
        self.negep, self.iexp, self.minexp, self.maxexp, \
        self.eps, self.epsneg, self.xmin, self.xmax = self._machar()

    def _machar(self) -> Tuple[int, ...]:
        """核心 Malcolm 算法，全整数运算避免浮点干扰。"""
        beta = 1
        # 确定基数 ibeta
        while True:
            beta += 1
            betain = beta
            a = beta
            # 找到最小的 a 使得 fl(a+1)-a != 1
            tres = a + 1.0
            temp = tres - a
            if temp != 1.0:
                break
            if beta > 10000:
                break  # 安全阀
        ibeta = betain

        # 确定尾数位数 it
        it = 0
        a = 1.0
        while True:
            it += 1
            a *= ibeta
            temp1 = a + 1.0
            temp2 = temp1 - a
            if temp2 != 1.0:
                break
            if it > 200:
                break

        # 确定舍入模式
        irnd = 0
        a = 0.5 * ibeta
        b = a + 1.0
        temp = b - a
        if temp == 1.0:
            irnd = 1
        b = a + a + 1.0
        temp = b - a
        if temp == 2.0 * a and irnd == 1:
            irnd = 2  # IEEE 风格舍入

        # 确定 machep
        betah = ibeta / 2.0
        a = 1.0
        machep = 0
        while True:
            a *= betah
            temp = 1.0 + a
            if temp != 1.0:
                machep = -(it + 1)  # 初始估计
                break
            machep -= 1
            if machep < -200:
                break
        # 精确定位
        a = 1.0
        for _ in range(-machep):
            a /= ibeta
        while True:
            temp = 1.0 + a
            if temp != 1.0:
                break
            a *= ibeta
            machep += 1
            if machep > 0:
                machep = -52  # 回退到默认
                a = 2.0**(-52)
                break
        eps = a

        # 确定 negep
        a = 1.0
        negep = 0
        while True:
            a *= betah
            temp = 1.0 - a
            if temp != 1.0:
                break
            negep -= 1
            if negep < -200:
                negep = -(it + 3)
                break
        negep_init = negep
        a = 1.0
        for _ in range(-negep):
            a /= ibeta
        while True:
            temp = 1.0 - a
            if temp != 1.0:
                break
            a *= ibeta
            negep += 1
            if negep > 0:
                negep = negep_init
                break
        epsneg = a

        # 确定 minexp / xmin
        minexp = -it
        a = 1.0
        for _ in range(it):
            a /= ibeta
        xmin = a
        while xmin * (1.0 - epsneg) > 0:
            a = xmin
            xmin *= betah
            if xmin <= 0:
                xmin = a
                break
            minexp -= 1
            if minexp < -2000:
                break

        # 确定 maxexp / xmax
        maxexp = 2 - minexp
        a = 1.0
        for _ in range(maxexp):
            a *= ibeta
            if not np.isfinite(a):
                maxexp -= 1
                break
        xmax = a / ibeta  # 最后一个有限值

        ngrd = 0
        iexp = minexp

        return (ibeta, it, irnd, ngrd, machep, negep, iexp,
                minexp, maxexp, eps, epsneg, xmin, xmax)

    def summary(self) -> Dict[str, float]:
        return {
            'ibeta': self.ibeta, 'it': self.it, 'irnd': self.irnd,
            'machep': self.machep, 'eps': self.eps,
            'xmin': self.xmin, 'xmax': self.xmax,
        }


# ---------------------------------------------------------------------------
# 第二部分: 标准晕模型 (SHM) 天体物理参数
# ---------------------------------------------------------------------------

class StandardHaloModel:
    """
    暗物质标准等温晕模型 (SHM) 物理参数封装。

    核心物理:
      - 本地 DM 密度: ρ_0 = 0.3 GeV/cm³
      - 最概然速度: v_0 = 220 km/s (对应圆周速度 Θ_0)
      - 地球轨道速度: v_orb ≈ 29.8 km/s
      - 太阳速度: v_sun ≈ 220 km/s (近似无 v_sun_pec)
      - 逃逸速度: v_esc = 544 km/s (对应 Galactic escape velocity)
      - 核质量: m_N = A * m_p (其中 A 为质量数)

    约化质量:
      μ_N = m_χ m_N / (m_χ + m_N)

    最小可探测速度:
      v_min(E_R) = √(m_N E_R / 2) / μ_N

    量纲约定:
      能量: GeV (或 keV 用于反冲能量)
      速度: km/s (自然单位下 c = 3e5 km/s)
      质量: GeV/c^2
    """

    def __init__(
        self,
        rho_0: float = 0.3,        # GeV/cm^3
        v_0: float = 220.0,        # km/s
        v_esc: float = 544.0,      # km/s
        v_sun: float = 232.0,      # km/s (含太阳本动速度修正)
        v_orb: float = 29.8,       # km/s
        m_proton: float = 0.938272,# GeV/c^2
        year_day: float = 365.25,  # 天/年
        t_0: float = 152.5,        # 春分后 ~ June 2nd
        gamma_angle: float = 0.906,# rad (~ 60.2 deg 赤经方向角)
    ):
        self.rho_0 = rho_0
        self.v_0 = v_0
        self.v_esc = v_esc
        self.v_sun = v_sun
        self.v_orb = v_orb
        self.m_proton = m_proton
        self.year_day = year_day
        self.t_0 = t_0
        self.gamma_angle = gamma_angle

        # N_esc 归一化因子
        x_esc = v_esc / v_0
        self.N_esc = 1.0 - (1.0 + x_esc**2) * np.exp(-x_esc**2)

        # 验证物理合理性
        if self.N_esc < 0.0:
            raise ValueError(
                f"N_esc = {self.N_esc} < 0: v_esc/v_0 参数非物理"
            )
        if rho_0 <= 0:
            raise ValueError("暗物质密度 ρ_0 必须为正")

    def nuclear_mass(self, A: int) -> float:
        """原子核质量 m_N = A * m_p - 结合能修正 (忽略, 近似 m_N ≈ A m_p)。"""
        if A < 1:
            raise ValueError(f"质量数 A 必须 >= 1, 得到 A={A}")
        return A * self.m_proton

    def reduced_mass(self, m_chi: float, m_N: float) -> float:
        """DM-核约化质量 μ_N = m_χ m_N / (m_χ + m_N)。"""
        denom = m_chi + m_N
        if denom == 0.0:
            return 0.0
        return m_chi * m_N / denom

    def v_min(self, E_R_keV: float, m_chi: float, A: int) -> float:
        """
        最小 DM 速度 (km/s), 产生反冲能量 E_R:
            v_min = √(m_N E_R / 2) / μ_N

        单位转换:
          E_R: keV → GeV: 1 keV = 1e-6 GeV
          m_N: GeV, μ_N: GeV
          v_min 返回 km/s (乘以 c = 3e5 km/s 从自然单位恢复)
        """
        if E_R_keV < 0:
            raise ValueError(f"E_R 不能为负: {E_R_keV}")
        if m_chi <= 0 or A < 1:
            return np.inf
        m_N = self.nuclear_mass(A)
        mu = self.reduced_mass(m_chi, m_N)
        if mu == 0:
            return np.inf
        E_R_GeV = E_R_keV * 1.0e-6
        # v_min in natural units (dimensionless = v/c)
        v_min_c = np.sqrt(m_N * E_R_GeV / (2.0 * mu**2))
        # 转换为 km/s
        c_km_s = 2.9979e5
        return v_min_c * c_km_s

    def earth_velocity(self, day: float) -> np.ndarray:
        """
        地球在银河系坐标系中的瞬时速度 (km/s), 3 维向量。

        v_E(t) = v_sun ẑ + v_orb * [cos γ cos(ω(t - t_0)) x̂
                                      + sin γ cos(ω(t - t_0)) ŷ
                                      - sin(ω(t - t_0)) ẑ]
        简化模型: ω = 2π / T_year
        """
        omega = 2.0 * np.pi / self.year_day
        phase = omega * (day - self.t_0)
        v_x = self.v_orb * np.cos(self.gamma_angle) * np.cos(phase)
        v_y = self.v_orb * np.sin(self.gamma_angle) * np.cos(phase)
        v_z = self.v_sun - self.v_orb * np.sin(phase)
        return np.array([v_x, v_y, v_z])


# ---------------------------------------------------------------------------
# 第三部分: 核物理参数库
# ---------------------------------------------------------------------------

class NuclearData:
    """
    常用探测靶核的物理数据。

    包含:
      - 质量数 A, 原子序数 Z
      - 核自旋 J
      - 核半径参数
    常用靶核: Xe (A=131), Ge (A=73), Ar (A=40), I (A=127), Na (A=23)
    """

    # 核素数据库 {名称: (A, Z, J, m_nuclear_GeV)}
    TARGETS: Dict[str, Dict] = {
        'Xe131': {'A': 131, 'Z': 54, 'J': 1.5, 'm_nuc': 122.38},
        'Ge73':  {'A': 73,  'Z': 32, 'J': 0.5, 'm_nuc': 68.01},
        'Ge76':  {'A': 76,  'Z': 32, 'J': 0.0, 'm_nuc': 70.79},
        'Ar40':  {'A': 40,  'Z': 18, 'J': 0.0, 'm_nuc': 37.22},
        'I127':  {'A': 127, 'Z': 53, 'J': 2.5, 'm_nuc': 118.25},
        'Na23':  {'A': 23,  'Z': 11, 'J': 1.5, 'm_nuc': 21.39},
        'F19':   {'A': 19,  'Z': 9,  'J': 0.5, 'm_nuc': 17.66},
        'Si28':  {'A': 28,  'Z': 14, 'J': 0.0, 'm_nuc': 26.12},
    }

    @classmethod
    def get(cls, name: str) -> Dict:
        if name not in cls.TARGETS:
            raise KeyError(
                f"核素 '{name}' 未注册。可选: {list(cls.TARGETS.keys())}"
            )
        return cls.TARGETS[name].copy()

    @classmethod
    def helm_radius(cls, A: int) -> Dict[str, float]:
        """
        Helm 核形状因子几何参数:
          c  = 1.23 * A^(1/3) - 0.60  [fm]
          a  = 0.52 fm (皮肤厚度)
          s  = 0.9 fm
          R_1 = sqrt(c^2 + (7/3)(π a)^2 - 5 s^2)
          R_n = R_1 (用于形状因子)
        所有长度单位: fm
        """
        if A < 1:
            raise ValueError(f"质量数 A 必须 >= 1, 得到 A={A}")
        c = 1.23 * (A ** (1.0 / 3.0)) - 0.60
        a_skin = 0.52
        s = 0.9
        R_1_sq = c**2 + (7.0 / 3.0) * (np.pi * a_skin)**2 - 5.0 * s**2
        # 保护: R_1_sq 应 > 0
        if R_1_sq <= 0:
            # 小 A 极端情况, 使用 fallback
            R_1 = max(0.5, c)
        else:
            R_1 = np.sqrt(R_1_sq)
        return {'c': c, 'a': a_skin, 's': s, 'R_1': R_1, 'R_n': R_1}


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

def get_default_shm() -> StandardHaloModel:
    """返回默认标准晕模型参数。"""
    return StandardHaloModel()


def get_machar() -> MacharConstants:
    """返回当前浮点环境的机器常数。"""
    return MacharConstants()
