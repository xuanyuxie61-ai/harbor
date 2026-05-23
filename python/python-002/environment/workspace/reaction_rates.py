# -*- coding: utf-8 -*-
"""
reaction_rates.py
基于 055_asa310 合成
核反应率计算：pp链、CNO循环、三重α过程、电子屏蔽修正。
引入非中心Beta分布建模核反应分支比的不确定性。
"""

import numpy as np
from typing import Tuple

# 物理常数 (CGS)
NA = 6.02214076e23          # 阿伏伽德罗常数 [mol^-1]
K_B = 1.380649e-16          # 玻尔兹曼常数 [erg/K]
Q_PP = 6.55e18              # pp链每次反应释放能量 [erg/g] (近似)
Q_CNO = 1.53e19             # CNO循环每次反应释放能量 [erg/g] (近似)
Q_3A = 7.275e-5 * 1.602e-6  # 3α反应每次能量 [erg] (7.275 MeV)

# 反应率参数 (NACRE II + 经典值)
# 形式: λ = λ0 * T9^(-2/3) * exp(-τ) * f_screen
# 其中 τ = 3 * (π^2 * m_r * k_B * T / (2 * hbar^2))^(1/3) * (Z1*Z2*e^2)^{2/3}


class NuclearReactionRates:
    """
    恒星核反应率计算类。
    主要反应通道：
      1) p + p -> d + e+ + ν_e       (pp-I 链起始)
      2) d + p -> 3He + γ            (pp-I)
      3) 3He + 3He -> 4He + 2p      (pp-I 终结)
      4) 3He + 4He -> 7Be + γ        (pp-II/III 分支)
      5) 7Be + e- -> 7Li + ν_e       (pp-II)
      6) 7Li + p -> 2 4He            (pp-II)
      7) 7Be + p -> 8B + γ           (pp-III)
      8) 8B -> 8Be* + e+ + ν_e       (pp-III)
      9) 8Be* -> 2 4He               (pp-III)
     10) 12C + p -> 13N + γ          (CNO-I)
     11) 13N -> 13C + e+ + ν_e       (CNO-I β衰变)
     12) 13C + p -> 14N + γ          (CNO-I)
     13) 14N + p -> 15O + γ          (CNO-I 限速步)
     14) 15O -> 15N + e+ + ν_e       (CNO-I β衰变)
     15) 15N + p -> 12C + 4He        (CNO-I 闭合)
     16) 3 4He -> 12C + γ            (3α过程，氦燃烧)
    """

    def __init__(self):
        self.reaction_names = [
            "pp", "pd", "He3He3", "He3He4", "Be7e", "Li7p",
            "Be7p", "B8", "Be8", "C12p", "N13",
            "C13p", "N14p", "O15", "N15p", "triple_alpha"
        ]
        self.num_reactions = len(self.reaction_names)

    @staticmethod
    def _temperature_factor(T9: float, Z1Z2: float, mu: float) -> float:
        """
        计算 Gamow 峰温度因子。
        Gamow 能量: E_G = (π α Z1 Z2)^2 2 m_r c^2
        天体物理 S-因子通常在 Gamow 峰 E0 附近展开：
          E0 = 1.22 * (Z1^2 Z2^2 * A_r * T9^2)^{1/3}  [MeV]
          ΔE0 = 0.2368 * (Z1^2 Z2^2 * A_r * T9^5)^{1/6} [MeV]
        """
        if T9 <= 0:
            return 0.0
        # 约化质量 (以 amu 为单位)
        tau = 3.0 * (np.pi ** 2 * mu * 1.6605e-24 / (2.0 * (1.054e-27) ** 2)) ** (1.0 / 3.0) \
              * (Z1Z2 * 2.307e-19) ** (2.0 / 3.0) * (1.0 / (K_B * T9 * 1e9)) ** (1.0 / 3.0)
        return tau

    @staticmethod
    def screening_factor(rho: float, T: float, X: float, Y: float, Z_metal: float,
                         Z1: int, Z2: int) -> float:
        """
        Salpeter 电子屏蔽因子（弱屏蔽极限）：
          f_screen = exp( H / kT )
          H = 1.88 * Z1 * Z2 * NA^(1/3) * e^2 * (ρ ζ / T7^3)^{1/2}
          ζ = Σ_i (Z_i^2 + Z_i) X_i / A_i / (1 + Z_i)
        弱屏蔽条件: Γ << 1, 其中 Γ = Z^2 e^2 / (a kT), a = (3/(4π n_e))^{1/3}
        """
        if T <= 0 or rho <= 0:
            return 1.0
        T7 = T / 1e7
        # 平均电离电荷平方和
        zeta = (2.0 ** 2 + 2.0) * X / 1.0 + (2.0 ** 2 + 2.0) * Y / 4.0
        zeta += Z_metal * (0.5 * (Z_metal ** 2 + Z_metal)) / (1.0 + Z_metal)
        zeta = max(zeta, 1e-10)
        # Debye-Hückel 屏蔽长度相关
        H = 1.88 * Z1 * Z2 * (NA ** (1.0 / 3.0)) * 2.307e-19
        arg = H * (rho * zeta / (T7 ** 3)) ** 0.5 / (K_B * T)
        # 弱屏蔽极限
        if arg > 10.0:
            arg = 10.0
        return np.exp(arg)

    def pp_chain_rates(self, T: float, rho: float, X: float, Y: float,
                       Z_metal: float) -> np.ndarray:
        """
        计算 pp 链各分支反应率 [s^-1] (每对反应物)。
        公式来源：Iliadis 2007, Nuclear Physics of Stars。
        """
        if T <= 1e5 or rho <= 0 or X <= 0:
            return np.zeros(self.num_reactions, dtype=np.float64)
        T9 = T / 1e9
        T9_inv = 1.0 / T9

        rates = np.zeros(self.num_reactions, dtype=np.float64)

        # pp: p(p,e+ν)d
        # λ_pp = 4.01e-15 * T9^{-2/3} * exp(-3.380/T9^{1/3}) * (1 + 0.123*T9^{1/3} + ...)
        f_pp = self.screening_factor(rho, T, X, Y, Z_metal, 1, 1)
        rates[0] = 4.01e-15 * T9 ** (-2.0 / 3.0) * np.exp(-3.380 * T9 ** (-1.0 / 3.0)) * f_pp
        rates[0] *= (1.0 + 0.123 * T9 ** (1.0 / 3.0) + 1.09 * T9 ** (2.0 / 3.0) + 0.938 * T9)

        # pd: d(p,γ)3He  (快速平衡，d丰度极低)
        rates[1] = 2.24e3 * T9 ** (-2.0 / 3.0) * np.exp(-3.720 * T9 ** (-1.0 / 3.0))
        rates[1] *= (1.0 + 0.112 * T9 ** (1.0 / 3.0) + 1.99 * T9 ** (2.0 / 3.0) + 1.56 * T9)

        # 3He(3He,2p)4He  (pp-I)
        f_33 = self.screening_factor(rho, T, X, Y, Z_metal, 2, 2)
        rates[2] = 6.04e10 * T9 ** (-2.0 / 3.0) * np.exp(-12.276 * T9 ** (-1.0 / 3.0)) * f_33
        rates[2] *= (1.0 + 0.034 * T9 ** (1.0 / 3.0) - 0.522 * T9 ** (2.0 / 3.0)
                     - 0.124 * T9 + 0.353 * T9 ** (4.0 / 3.0) + 0.213 * T9 ** (5.0 / 3.0))

        # 3He(4He,γ)7Be  (pp-II/III 分支)
        f_34 = self.screening_factor(rho, T, X, Y, Z_metal, 2, 2)
        rates[3] = 5.61e6 * T9 ** (-2.0 / 3.0) * np.exp(-12.826 * T9 ** (-1.0 / 3.0)) * f_34
        rates[3] *= (1.0 + 0.015 * T9 ** (1.0 / 3.0) + 0.238 * T9 ** (2.0 / 3.0)
                     + 0.030 * T9 + 0.042 * T9 ** (4.0 / 3.0) + 0.020 * T9 ** (5.0 / 3.0))

        # 7Be(e-,ν)7Li  (电子俘获)
        # λ = 1.34e-10 * T9^{-1/2} * (1 - 0.537*T9^{1/3} + 3.86*T9^{2/3} - 5.48*T9)
        if T9 < 0.01:
            rates[4] = 1.34e-10 * T9 ** (-0.5)
        else:
            rates[4] = 1.34e-10 * T9 ** (-0.5) * (1.0 - 0.537 * T9 ** (1.0 / 3.0)
                                                     + 3.86 * T9 ** (2.0 / 3.0)
                                                     - 5.48 * T9)

        # 7Li(p,α)4He
        f_Li = self.screening_factor(rho, T, X, Y, Z_metal, 1, 3)
        rates[5] = 1.096e9 * T9 ** (-2.0 / 3.0) * np.exp(-8.472 * T9 ** (-1.0 / 3.0)) * f_Li
        rates[5] *= (1.0 + 0.049 * T9 ** (1.0 / 3.0) - 0.134 * T9 ** (2.0 / 3.0)
                     + 0.010 * T9 + 0.019 * T9 ** (4.0 / 3.0))

        # 7Be(p,γ)8B  (pp-III)
        f_Be = self.screening_factor(rho, T, X, Y, Z_metal, 1, 4)
        rates[6] = 2.32e-3 * T9 ** (-2.0 / 3.0) * np.exp(-10.262 * T9 ** (-1.0 / 3.0)) * f_Be
        rates[6] *= (1.0 + 0.049 * T9 ** (1.0 / 3.0) + 0.213 * T9 ** (2.0 / 3.0)
                     + 0.028 * T9 + 0.019 * T9 ** (4.0 / 3.0))

        # 8B -> 8Be* + e+ + ν_e  (β+ 衰变，半衰期约0.77s，与温度无关)
        rates[7] = 0.9  # s^-1 (近似衰变常数)

        # 8Be* -> 2 4He  (瞬间衰变)
        rates[8] = 1e16  # 极快

        return rates

    def cno_cycle_rates(self, T: float, rho: float, X: float, Y: float,
                        Z_metal: float) -> np.ndarray:
        """
        CNO-I 循环反应率。
        限速步是 14N(p,γ)15O。
        """
        if T <= 1e6 or rho <= 0 or X <= 0:
            return np.zeros(self.num_reactions, dtype=np.float64)
        T9 = T / 1e9
        rates = np.zeros(self.num_reactions, dtype=np.float64)

        # 12C(p,γ)13N
        f_C = self.screening_factor(rho, T, X, Y, Z_metal, 1, 6)
        rates[9] = 2.04e7 * T9 ** (-2.0 / 3.0) * np.exp(-13.690 * T9 ** (-1.0 / 3.0)) * f_C
        rates[9] *= (1.0 + 0.03 * T9 ** (1.0 / 3.0) + 0.94 * T9 ** (2.0 / 3.0)
                     + 0.058 * T9 + 0.022 * T9 ** (4.0 / 3.0))

        # 13N -> 13C + e+ + ν_e  (β+ 衰变，半衰期9.965 min)
        rates[10] = 1.16e-3  # s^-1

        # 13C(p,γ)14N
        f_C13 = self.screening_factor(rho, T, X, Y, Z_metal, 1, 6)
        rates[11] = 8.01e7 * T9 ** (-2.0 / 3.0) * np.exp(-13.717 * T9 ** (-1.0 / 3.0)) * f_C13
        rates[11] *= (1.0 + 0.03 * T9 ** (1.0 / 3.0) + 0.94 * T9 ** (2.0 / 3.0)
                      + 0.058 * T9 + 0.022 * T9 ** (4.0 / 3.0))

        # 14N(p,γ)15O  (CNO 限速步)
        f_N = self.screening_factor(rho, T, X, Y, Z_metal, 1, 7)
        rates[12] = 4.90e7 * T9 ** (-2.0 / 3.0) * np.exp(-15.228 * T9 ** (-1.0 / 3.0)) * f_N
        rates[12] *= (1.0 + 0.027 * T9 ** (1.0 / 3.0) - 0.778 * T9 ** (2.0 / 3.0)
                      - 0.149 * T9 + 0.261 * T9 ** (4.0 / 3.0) + 0.127 * T9 ** (5.0 / 3.0))

        # 15O -> 15N + e+ + ν_e  (β+ 衰变，半衰期122.24 s)
        rates[13] = 5.68e-3  # s^-1

        # 15N(p,α)12C
        f_N15 = self.screening_factor(rho, T, X, Y, Z_metal, 1, 7)
        rates[14] = 1.08e12 * T9 ** (-2.0 / 3.0) * np.exp(-15.251 * T9 ** (-1.0 / 3.0)) * f_N15
        rates[14] *= (1.0 + 0.027 * T9 ** (1.0 / 3.0) + 0.162 * T9 ** (2.0 / 3.0)
                      + 0.010 * T9 + 0.006 * T9 ** (4.0 / 3.0))

        return rates

    def triple_alpha_rate(self, T: float, rho: float, Y: float) -> float:
        """
        三重α过程反应率 [cm^6 g^-2 s^-1]（每克平方）。
        实际能源产生率: ε_3α = Y^3 ρ^2 λ_3α / (3! * 4^3) * Q_3α / NA^2
        """
        if T <= 1e7 or rho <= 0 or Y <= 0:
            return 0.0
        T9 = T / 1e9
        # Fushiki & Lamb 1987 / Nomoto 修正
        # λ_3α = 2.79e-8 * T9^{-3} * exp(-4.4027/T9)  (低 T)
        # 或 对 T9 > 0.1:
        f_3a = self.screening_factor(rho, T, 0.0, Y, 0.0, 2, 2)
        if T9 < 0.1:
            rate = 2.79e-8 * T9 ** (-3.0) * np.exp(-4.4027 / T9) * f_3a
        else:
            rate = (2.79e-8 * T9 ** (-3.0) * np.exp(-4.4027 / T9)
                    + 1.36e-7 * T9 ** (-3.0) * np.exp(-13.490 / T9)
                    + 2.60e-8 * T9 ** (-3.0) * np.exp(-15.541 / T9)) * f_3a
        return max(rate, 0.0)

    def branch_ratios(self, T: float, rho: float, X: float, Y: float,
                      Z_metal: float) -> Tuple[float, float, float]:
        """
        计算 pp 链三个分支的比例 (pp-I, pp-II, pp-III)。
        基于 055_asa310 的非中心 Beta 分布思想建模分支比不确定性。
        
        r_I   = λ(3He+3He) / (λ(3He+3He) + λ(3He+4He))
        r_II  = λ(3He+4He) * λ(Be7+e) / ((λ(Be7+e)+λ(Be7+p)) * denom)
        r_III = λ(3He+4He) * λ(Be7+p) / ((λ(Be7+e)+λ(Be7+p)) * denom)
        其中 denom = λ(3He+3He) + λ(3He+4He)
        """
        rates = self.pp_chain_rates(T, rho, X, Y, Z_metal)
        r33 = rates[2]
        r34 = rates[3]
        r7e = rates[4]
        r7p = rates[6]

        denom = r33 + r34
        if denom <= 0:
            return 1.0, 0.0, 0.0

        r_I = r33 / denom
        denom2 = r7e + r7p
        if denom2 <= 0:
            r_II = 0.0
            r_III = 0.0
        else:
            r_II = r34 * r7e / (denom * denom2)
            r_III = r34 * r7p / (denom * denom2)

        # 归一化
        total = r_I + r_II + r_III
        if total > 0:
            r_I /= total
            r_II /= total
            r_III /= total
        return r_I, r_II, r_III

    def energy_generation(self, T: float, rho: float, X: float, Y: float,
                          Z_cno: float, Z_metal: float) -> float:
        """
        总核能源产生率 ε [erg g^-1 s^-1]。
        ε = ε_pp + ε_CNO + ε_3α
        ε_pp  ≈ 2.4e4 * X^2 * ρ * T9^{-2/3} * exp(-3.380/T9^{1/3})  [erg g^-1 s^-1]
        ε_CNO ≈ 8.24e25 * X*Z_CNO * ρ * T9^{-2/3} * exp(-15.228/T9^{1/3})
        ε_3α  ≈ 5.09e11 * Y^3 * ρ^2 * T9^{-3} * exp(-4.4027/T9)    [erg g^-1 s^-1]
        """
        # TODO: Hole 1 - 实现总核能源产生率计算
        # 需考虑 pp 链、CNO 循环、三重α过程的贡献
        # 注意与 nuclear_network.py 中反应率索引的一致性
        raise NotImplementedError("Hole 1: 待实现 energy_generation 核能源产生率公式")
