"""
qgp_freeze_out.py — Cooper-Frye 冻结面与粒子谱积分
=====================================================

融合种子项目: 179_circle_integrals (圆面积分)

本模块实现 Cooper-Frye 冻结面公式, 将流体动力学演化结果
转换为可观测的粒子动量谱.

Cooper-Frye 公式:
    E * dN/d^3p = integral_S f(x, p) * p^mu * d_sigma_mu

其中:
    S: 冻结超曲面 (T = T_switch)
    f(x, p): 相空间分布函数
    d_sigma_mu: 超曲面法向量

对于 freeze-out 在 tau = const 超曲面:
    d_sigma_mu = (tau * r dr dphi d_eta, 0, 0, 0)

粒子分布 (Boltzmann 近似):
    f(x, p) = exp(-(p^mu * u_mu) / T)
            = exp(-gamma * (E_p - p_x*v_x - p_y*v_y) / T)
    其中 E_p = sqrt(p_T^2 + m^2) * cosh(y_p - eta_s)

对于费米子 (Fermi-Dirac):
    f_FD = 1 / (exp(E/T) + 1)

对于玻色子 (Bose-Einstein):
    f_BE = 1 / (exp(E/T) - 1)

圆面积分 (源自 179_circle_integrals):
    方位角积分: integral_0^{2pi} F(phi) dphi
    使用圆上均匀采样或精确积分

    圆上 monomial 积分:
        integral x^a * y^b dOmega
        = 0 if a or b is odd (对称性)
        = 2*pi * Gamma((a+1)/2)*Gamma((b+1)/2) / Gamma((a+b+2)/2) if a,b even
"""

import numpy as np
from typing import Dict, List
from qgp_config import (
    NumericalParams, ParticleSpecies, HBAR_C, T_SWITCH
)
from qgp_grid import QGPGrid
from qgp_eos import QGPEquationOfState


def circle_monomial_integral(exp_a: int, exp_b: int) -> float:
    """
    单位圆上 monomial x^a * y^b 的精确积分 (源自 179_circle_integrals)

    integral_0^{2pi} cos^a(theta) * sin^b(theta) dtheta

    结果:
    - 若 a 或 b 为奇数: 积分为 0 (对称性)
    - 若 a, b 均为偶数:
      = 4 * integral_0^{pi/2} cos^a * sin^b dtheta
      = 2 * Beta((a+1)/2, (b+1)/2)
      = 2 * Gamma((a+1)/2) * Gamma((b+1)/2) / Gamma((a+b)/2 + 1)

    Args:
        exp_a: x 的幂次
        exp_b: y 的幂次

    Returns:
        积分值
    """
    if exp_a % 2 == 1 or exp_b % 2 == 1:
        return 0.0
    from math import gamma
    return 2.0 * gamma((exp_a + 1) / 2.0) * gamma((exp_b + 1) / 2.0) / \
           gamma((exp_a + exp_b) / 2.0 + 1.0)


def circle_sample_uniform(n_points: int) -> tuple:
    """
    单位圆上均匀采样 (源自 179_circle_integrals)

    theta_j = 2*pi*j/N, j = 0, ..., N-1
    x_j = cos(theta_j), y_j = sin(theta_j)

    Args:
        n_points: 采样点数

    Returns:
        (x, y) 坐标数组
    """
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    return np.cos(theta), np.sin(theta)


class CooperFryeFreezeOut:
    """
    Cooper-Frye 冻结面计算

    将流体动力学场转换为粒子动量谱.

    流程:
    1. 找到冻结面 (T = T_switch 的等温面)
    2. 在冻结面上计算粒子发射率
    3. 对动量空间积分得到 dN/(p_T dp_T dphi dy)
    4. 对方位角积分得到 dN/(p_T dp_T dy)
    5. 提取流谐波 v_n(p_T)
    """

    def __init__(self, grid: QGPGrid, eos: QGPEquationOfState):
        """
        Args:
            grid: 计算网格
            eos: 状态方程
        """
        self.grid = grid
        self.eos = eos
        self.T_switch = T_SWITCH

    def find_freeze_surface(self, e: np.ndarray) -> np.ndarray:
        """
        找到冻结面 (T = T_switch 的位置)

        条件: T(x, y) = T_switch
        方法: 温度场的等值线

        返回冻结面的 mask (内部区域):
            mask[i,j] = 1 if |T(x_i, y_j) - T_switch| < dT

        Args:
            e: 能量密度 (内部区域)

        Returns:
            冻结面 mask (ny, nx)
        """
        T = self.eos.temperature_from_energy(e)
        dT = 0.005  # 温度容差 (GeV)
        mask = np.abs(T - self.T_switch) < dT

        # 如果没有找到精确等温面, 使用最接近的区域
        if not np.any(mask):
            T_diff = np.abs(T - self.T_switch)
            min_idx = np.unravel_index(np.argmin(T_diff), T_diff.shape)
            mask[min_idx] = True
            # 扩展邻域
            i, j = min_idx
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < T.shape[0] and 0 <= nj < T.shape[1]:
                        if np.abs(T[ni, nj] - self.T_switch) < 3 * dT:
                            mask[ni, nj] = True

        return mask.astype(np.float64)

    def cooper_frye_spectrum(self, e: np.ndarray, vx: np.ndarray,
                              vy: np.ndarray, species: dict,
                              y_rapidity: float = 0.0) -> dict:
        """
        Cooper-Frye 粒子谱计算

        dN / (p_T dp_T dphi dy) = integral_S
            g / (2*pi)^3 * f(x,p) * p^mu * d_sigma_mu * r dr dphi

        简化: 在 tau=const 面上:
            dN/(p_T dp_T dphi dy) = tau * sum_cells
                g / (2*pi)^3 * exp(-p^mu u_mu / T) * p_T * m_T
                * cosh(y - eta_s) * dx * dy

        其中:
            m_T = sqrt(p_T^2 + m^2) 横向质量
            p^mu u_mu = m_T * gamma * cosh(y-eta_s)
                       - p_T * gamma * (vx*cos(phi) + vy*sin(phi))

        Args:
            e: 能量密度 (内部区域)
            vx, vy: 流速
            species: 粒子种类字典 {'name': str, 'mass': float,
                      'degeneracy': int, 'is_fermion': bool}
            y_rapidity: 粒子快度

        Returns:
            {'pt': array, 'phi': array, 'dN_dpt_dphi': 2d array,
             'dN_dpt': 1d array}
        """
        # 动量网格
        pt_bins = NumericalParams.PT_BINS
        phi_bins = NumericalParams.PHI_BINS
        pt_min = NumericalParams.PT_MIN
        pt_max = NumericalParams.PT_MAX

        pt = np.linspace(pt_min, pt_max, pt_bins)
        phi = np.linspace(0, 2*np.pi, phi_bins, endpoint=False)
        dpt = pt[1] - pt[0] if pt_bins > 1 else 0.1
        dphi = 2*np.pi / phi_bins

        mass = species['mass']
        g = species['degeneracy']

        # 温度场
        T = self.eos.temperature_from_energy(e)
        gamma_flow = 1.0 / np.sqrt(1.0 - np.minimum(vx**2 + vy**2, 0.99**2))

        # 冻结面 mask
        freeze_mask = self.find_freeze_surface(e)

        # 谱计算
        dN = np.zeros((pt_bins, phi_bins))

        for ip, p_T in enumerate(pt):
            m_T = np.sqrt(p_T**2 + mass**2)

            for iphi, phi_p in enumerate(phi):
                px = p_T * np.cos(phi_p)
                py = p_T * np.sin(phi_p)

                # p^mu * u_mu (逐格点)
                # p^mu u_mu = m_T * cosh(y-eta_s) * gamma
                #             - (px*vx + py*vy) * gamma
                # 在 eta_s = 0, y = y_rapidity:
                cosh_y = np.cosh(y_rapidity)
                p_dot_u = m_T * cosh_y * gamma_flow - \
                          (px * vx + py * vy) * gamma_flow

                # 分布函数 (Boltzmann 近似)
                f = np.exp(-p_dot_u / np.maximum(T, 1.0e-6))

                # Cooper-Frye 被积函数
                integrand = g / (2*np.pi)**3 * f * m_T * cosh_y * \
                            p_T * freeze_mask

                # 面积分
                dN[ip, iphi] = self.grid.integrate_2d(integrand)

        # 方位角积分: dN/(p_T dp_T dy)
        dN_dpt = np.sum(dN, axis=1) * dphi

        return {
            'pt': pt,
            'phi': phi,
            'dN_dpt_dphi': dN,
            'dN_dpt': dN_dpt,
            'species_name': species['name'],
            'mass': mass,
        }

    def compute_all_species_spectra(self, e: np.ndarray,
                                     vx: np.ndarray, vy: np.ndarray) -> dict:
        """
        计算所有粒子种类的动量谱

        Args:
            e: 能量密度
            vx, vy: 流速

        Returns:
            {species_name: spectrum_dict}
        """
        spectra = {}
        for sp in ParticleSpecies.ALL_SPECIES:
            spectra[sp['name']] = self.cooper_frye_spectrum(e, vx, vy, sp)
        return spectra

    def inverse_slope_parameter(self, spectrum: dict) -> float:
        """
        提取逆斜率参数 T_eff (有效温度)

        dN/(p_T dp_T) ~ exp(-m_T / T_eff)
        => ln(dN/(p_T dp_T)) = -m_T / T_eff + const

        线性拟合:
            T_eff = -1 / slope

        物理: T_eff 反映 freeze-out 温度和径向流的贡献:
            T_eff = T_kin + m * <v_T>^2

        Args:
            spectrum: 单粒子谱字典

        Returns:
            有效温度 T_eff (GeV)
        """
        pt = spectrum['pt']
        dN_dpt = spectrum['dN_dpt']
        mass = spectrum['mass']

        mT = np.sqrt(pt**2 + mass**2)

        # 过滤有效范围
        mask = dN_dpt > 1.0e-20
        if np.sum(mask) < 3:
            return self.T_switch

        mT_valid = mT[mask]
        log_dN = np.log(dN_dpt[mask])

        # 线性拟合
        try:
            coeffs = np.polyfit(mT_valid, log_dN, 1)
            T_eff = -1.0 / coeffs[0]
            T_eff = max(T_eff, 0.01)  # 物理下限
        except (np.linalg.LinAlgError, ZeroDivisionError):
            T_eff = self.T_switch

        return T_eff

    def mean_pt(self, spectrum: dict) -> float:
        """
        计算平均横动量 <p_T>

        <p_T> = integral p_T * dN/dpt * dpt / integral dN/dpt * dpt

        Args:
            spectrum: 粒子谱

        Returns:
            <p_T> (GeV)
        """
        pt = spectrum['pt']
        dN = spectrum['dN_dpt']
        dpt = pt[1] - pt[0] if len(pt) > 1 else 0.1

        numerator = np.sum(pt**2 * dN) * dpt
        denominator = np.sum(pt * dN) * dpt + 1.0e-30
        return numerator / denominator

    def particle_yield(self, spectrum: dict) -> float:
        """
        计算总粒子产额 dN/dy

        dN/dy = integral dN/(p_T dp_T) * p_T dp_T
              = 2*pi * integral (dN/(p_T dp_T dphi)) * p_T dp_T dphi / (2*pi)

        Args:
            spectrum: 粒子谱

        Returns:
            dN/dy
        """
        pt = spectrum['pt']
        dN_dpt = spectrum['dN_dpt']
        dpt = pt[1] - pt[0] if len(pt) > 1 else 0.1

        return np.sum(dN_dpt * pt) * dpt * 2 * np.pi
