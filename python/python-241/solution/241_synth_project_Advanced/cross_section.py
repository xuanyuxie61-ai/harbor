"""
cross_section.py
===================================================================
核反应截面计算模块

核心物理公式:
  分波截面:
    sigma_el^l = (pi/k^2) * (2l+1) * |1 - S_l|^2
    sigma_re^l = (pi/k^2) * (2l+1) * (1 - |S_l|^2)
    sigma_tot^l = (2*pi/k^2) * (2l+1) * (1 - Re(S_l))

  总截面 (分波求和):
    sigma_el = sum_l sigma_el^l
    sigma_re = sum_l sigma_re^l  (反应截面)
    sigma_tot = sigma_el + sigma_re

  光学定理:
    sigma_tot = (4*pi/k) * Im(f(0))
    其中 f(0) = (1/(2ik)) * sum_l (2l+1) * (S_l - 1)

  R-matrix 共振截面 (Breit-Wigner):
    sigma_res(E) = (pi/k^2) * (2J+1)/((2J_t+1)(2j+1)) *
                   Gamma_n * Gamma / ((E-E_r)^2 + (Gamma/2)^2)
===================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------- 物理常数 ----------
HBAR_C = 197.3269804
PI = np.pi


class CrossSectionCalculator:
    """
    核反应截面计算器

    输入: S 矩阵元素 S_l 或相移 delta_l
    输出: 弹性、反应、总截面及角分布
    """

    def __init__(self, k: float, l_max: int = 20):
        """
        参数:
            k: 入射波数 [1/fm]
            l_max: 最大角动量量子数
        """
        self.k = k
        self.l_max = l_max
        self.l_values = np.arange(l_max + 1)

    def compute_partial_cross_sections(
        self, S_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        计算各分波截面

        参数:
            S_matrix: S 矩阵元素数组 [S_0, S_1, ..., S_lmax]

        返回:
            sigma_el_l: 弹性分波截面 [fm^2]
            sigma_re_l: 反应分波截面 [fm^2]
            sigma_tot_l: 总分波截面 [fm^2]
        """
        l = self.l_values
        k2 = self.k ** 2
        factor = PI / k2

        # 弹性截面: sigma_el^l = (pi/k^2)(2l+1)|1-S_l|^2
        sigma_el_l = factor * (2 * l + 1) * np.abs(1.0 - S_matrix) ** 2

        # 反应截面: sigma_re^l = (pi/k^2)(2l+1)(1-|S_l|^2)
        sigma_re_l = factor * (2 * l + 1) * (1.0 - np.abs(S_matrix) ** 2)
        sigma_re_l = np.maximum(sigma_re_l, 0.0)  # 物理约束

        # 总截面: sigma_tot^l = (2pi/k^2)(2l+1)(1-Re(S_l))
        sigma_tot_l = 2.0 * factor * (2 * l + 1) * (1.0 - np.real(S_matrix))

        return {
            'sigma_el_l': sigma_el_l,
            'sigma_re_l': sigma_re_l,
            'sigma_tot_l': sigma_tot_l,
        }

    def compute_total_cross_sections(
        self, S_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        计算总截面 (分波求和至 l_max)
        """
        partial = self.compute_partial_cross_sections(S_matrix)

        sigma_el = float(np.sum(partial['sigma_el_l']))
        sigma_re = float(np.sum(partial['sigma_re_l']))
        sigma_tot = float(np.sum(partial['sigma_tot_l']))

        # 光学定理验证
        sigma_tot_optical = self._optical_theorem(S_matrix)

        return {
            'sigma_elastic': sigma_el,
            'sigma_reaction': sigma_re,
            'sigma_total': sigma_tot,
            'sigma_total_optical_theorem': sigma_tot_optical,
            'optical_theorem_check': abs(sigma_tot - sigma_tot_optical) / (
                abs(sigma_tot) + 1e-30),
            'l_max_used': self.l_max,
            'converged': self._check_convergence(partial['sigma_tot_l']),
        }

    def _optical_theorem(self, S_matrix: np.ndarray) -> float:
        """
        光学定理计算总截面:
            sigma_tot = (4pi/k) * Im(f(0))
            f(0) = (1/(2ik)) * sum_l (2l+1)(S_l - 1)
        """
        l = self.l_values
        f_forward = np.sum((2 * l + 1) * (S_matrix - 1.0)) / (2j * self.k)
        return float(4.0 * PI / self.k * np.imag(f_forward))

    def _check_convergence(self, sigma_l: np.ndarray, threshold: float = 1e-4) -> bool:
        """
        检查分波求和收敛性:
            sigma_l_max / sigma_total < threshold
        """
        total = np.sum(sigma_l)
        if total < 1e-30:
            return True
        return abs(sigma_l[-1]) / total < threshold

    def differential_cross_section(
        self,
        S_matrix: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        微分弹性散射截面:
            d(sigma)/dOmega = |f(theta)|^2

        散射振幅:
            f(theta) = (1/(2ik)) * sum_l (2l+1)(S_l - 1) * P_l(cos(theta))

        参数:
            S_matrix: S 矩阵元素
            theta: 散射角数组 [弧度]

        返回:
            d(sigma)/dOmega [fm^2/sr] 对每个角度
        """
        l = self.l_values
        dsigma = np.zeros(len(theta))

        for idx, th in enumerate(theta):
            cos_th = np.cos(th)
            # 勒让德多项式
            Pl = np.array([self._legendre(ll, cos_th) for ll in l])

            # 散射振幅
            f_th = np.sum((2 * l + 1) * (S_matrix - 1.0) * Pl) / (2j * self.k)
            dsigma[idx] = abs(f_th) ** 2

        return dsigma

    def _legendre(self, l: int, x: float) -> float:
        """
        勒让德多项式 P_l(x) 递推计算:
            (l+1)*P_{l+1}(x) = (2l+1)*x*P_l(x) - l*P_{l-1}(x)

        P_0(x) = 1, P_1(x) = x
        """
        if l == 0:
            return 1.0
        if l == 1:
            return x
        P_prev = 1.0
        P_curr = x
        for n in range(1, l):
            P_next = ((2 * n + 1) * x * P_curr - n * P_prev) / (n + 1)
            P_prev = P_curr
            P_curr = P_next
        return P_curr

    def r_matrix_resonance_cross_section(
        self,
        E: np.ndarray,
        E_res: float,
        Gamma_n: float,
        Gamma_total: float,
        l_quantum: int,
        J_target: float = 0.0,
        J_projectile: float = 0.5
    ) -> np.ndarray:
        """
        R 矩阵共振截面 (单级 Breit-Wigner):

        sigma_res(E) = (pi/k^2) * g * Gamma_n * Gamma_total /
                       ((E - E_res)^2 + (Gamma_total/2)^2)

        统计因子:
            g = (2J+1) / ((2J_t+1)(2j_p+1))

        其中 J 为复合核自旋, J_t 靶核自旋, j_p 入射粒子自旋

        参数:
            E: 能量数组 [MeV]
            E_res: 共振能量 [MeV]
            Gamma_n: 中子宽度 [MeV]
            Gamma_total: 总宽度 [MeV]
            l_quantum: 轨道角动量
            J_target: 靶核自旋
            J_projectile: 入射粒子自旋

        返回:
            共振截面数组 [fm^2]
        """
        # 波数 (能量依赖)
        m_red = 938.0 / 2.0  # 约化质量近似
        k_E = np.sqrt(2.0 * m_red * np.maximum(E, 0.0)) / HBAR_C
        k_E = np.maximum(k_E, 1e-10)

        # 统计因子 (对所有可能的 J 求和)
        # J = |l - j_p| 到 l + j_p
        g_total = 0.0
        for j_sign in [-1, 1]:
            J = l_quantum + j_sign * J_projectile
            if J < 0:
                continue
            g = (2 * J + 1) / ((2 * J_target + 1) * (2 * J_projectile + 1))
            g_total += g

        # Breit-Wigner 公式
        numerator = PI / k_E ** 2 * g_total * Gamma_n * Gamma_total
        denominator = (E - E_res) ** 2 + (Gamma_total / 2.0) ** 2

        return numerator / denominator

    def strength_function(
        self, S_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        强度函数计算:
            S_l(E) = (1/(2*pi)) * Gamma_n / D_l
                   = (k*R) / (pi) * (1 - Re(S_l))

        其中 D_l 为能级间距, R 为相互作用半径

        强度函数表征平均共振特性
        """
        l = self.l_values
        # 简化: 假设相互作用半径 R = 1.25 * A^(1/3) ≈ 7 fm (对 Pb-208)
        R_int = 7.0

        S_l = (self.k * R_int) / PI * (1.0 - np.real(S_matrix))

        return {
            'strength_function': S_l,
            'l_values': l,
        }

    def analyzing_power(
        self,
        S_plus: np.ndarray,
        S_minus: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        矢量分析本领 (极化观测量的):
            A(theta) = (1/P) * (d(sigma_up) - d(sigma_down)) /
                       (d(sigma_up) + d(sigma_down))

        其中 S_plus (S_minus) 对应 j = l + 1/2 (j = l - 1/2) 的 S 矩阵

        参数:
            S_plus: j = l + 1/2 的 S 矩阵
            S_minus: j = l - 1/2 的 S 矩阵
            theta: 散射角数组

        返回:
            分析本领 A(theta)
        """
        dsigma_plus = self.differential_cross_section(S_plus, theta)
        dsigma_minus = self.differential_cross_section(S_minus, theta)

        denom = dsigma_plus + dsigma_minus
        denom = np.where(denom > 1e-30, denom, 1e-30)

        return (dsigma_plus - dsigma_minus) / denom
