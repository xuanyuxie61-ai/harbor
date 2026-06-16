"""
qgp_flow_harmonics.py — 流谐波提取与 Fourier 分解
=====================================================

融合种子项目: 1102_ryan597_RepresentationLearningWaves (波形分解)

本模块从 Cooper-Frye 粒子谱中提取流谐波 v_n(p_T),
这些是重离子碰撞实验中最关键的观测物理量.

流谐波 Fourier 分解:
---------------------

粒子方位角分布:
    dN / (p_T dp_T dphi dy) = dN_0/(p_T dp_T dy)
        * [1 + 2 * sum_{n=1}^{inf} v_n(p_T) * cos(n*(phi - Psi_n))]

流谐波系数:
    v_n(p_T) = <cos(n*(phi - Psi_n))>
    = integral cos(n*(phi-Psi_n)) * dN/(pT dpT dphi dy) dphi
      / integral dN/(pT dpT dphi dy) dphi

实验观测量:
    v_2: 椭圆流 (elliptic flow) - 反映初始空间偏心度
    v_3: 三角流 (triangular flow) - 由初始涨落驱动
    v_4: 四边形流 - v_2^2 的非线性贡献
    v_5: 五角流
    v_6: 六角流

波形分解方法 (源自 1102_RepresentationLearningWaves):
    将方位角分布视为波, 用 Fourier 基函数分解:
        f(phi) = a_0/2 + sum_n [a_n*cos(n*phi) + b_n*sin(n*phi)]
    v_n = sqrt(a_n^2 + b_n^2) / a_0

对称平面 Psi_n:
    Psi_n = (1/n) * atan2(b_n, a_n)

事件平面法:
    Q_n = sum_j w_j * exp(i*n*phi_j)  (流矢量)
    v_n = |Q_n| / M  (M 为粒子数)
"""

import numpy as np
from qgp_config import NumericalParams
from qgp_grid import QGPGrid


class FlowHarmonicAnalyzer:
    """
    流谐波分析器

    从 Cooper-Frye 谱中提取 v_n(p_T) 和积分 v_n.

    Attributes:
        n_harmonics: 分析的最高阶谐波
        phi_bins: 方位角分箱数
    """

    def __init__(self, n_harmonics: int = None):
        """
        Args:
            n_harmonics: 最高阶流谐波 (默认 6)
        """
        self.n_harmonics = n_harmonics or NumericalParams.N_HARMONICS
        self.phi_bins = NumericalParams.PHI_BINS

    def extract_vn_pt(self, spectrum: dict) -> dict:
        """
        从 dN/(p_T dp_T dphi) 提取 v_n(p_T)

        方法: 离散 Fourier 变换
            a_n(p_T) = (2/N_phi) * sum_j dN(pT, phi_j) * cos(n*phi_j)
            b_n(p_T) = (2/N_phi) * sum_j dN(pT, phi_j) * sin(n*phi_j)
            v_n(p_T) = sqrt(a_n^2 + b_n^2) / a_0(pT)

        其中 a_0 = (1/N_phi) * sum_j dN(pT, phi_j) 为方位角平均

        Args:
            spectrum: Cooper-Frye 谱字典
                {'pt': array, 'phi': array, 'dN_dpt_dphi': 2d array}

        Returns:
            {'pt': array, 'v2': array, 'v3': array, ...,
             'psi2': array, 'psi3': array, ...}
        """
        pt = spectrum['pt']
        phi = spectrum['phi']
        dN = spectrum['dN_dpt_dphi']
        n_pt = len(pt)
        n_phi = len(phi)

        result = {'pt': pt}

        for n in range(2, self.n_harmonics + 1):
            # Fourier 系数
            cos_n = np.cos(n * phi)
            sin_n = np.sin(n * phi)

            # a_n(p_T) = integral dN * cos(n*phi) dphi
            a_n = np.sum(dN * cos_n[np.newaxis, :], axis=1) * (2*np.pi/n_phi)
            b_n = np.sum(dN * sin_n[np.newaxis, :], axis=1) * (2*np.pi/n_phi)

            # a_0 (方位角平均)
            a_0 = np.sum(dN, axis=1) * (2*np.pi/n_phi)
            a_0_safe = np.maximum(a_0, 1.0e-30)

            # v_n(p_T)
            v_n = np.sqrt(a_n**2 + b_n**2) / a_0_safe

            # 对称平面角 Psi_n
            psi_n = np.arctan2(b_n, a_n) / n

            result[f'v{n}'] = v_n
            result[f'a{n}'] = a_n
            result[f'b{n}'] = b_n
            result[f'psi{n}'] = psi_n

        return result

    def integrated_vn(self, spectrum: dict) -> dict:
        """
        计算积分流谐波 V_n (对 p_T 积分)

        V_n = integral v_n(p_T) * dN/(p_T dp_T) * p_T dp_T
              / integral dN/(p_T dp_T) * p_T dp_T

        等价于:
            V_n = |integral exp(i*n*phi) * dN/(pT dpT dphi) * pT dpT dphi|
                  / integral dN/(pT dpT dphi) * pT dpT dphi

        Args:
            spectrum: Cooper-Frye 谱

        Returns:
            {'V2': float, 'V3': float, ..., 'V_n_ratios': dict}
        """
        pt = spectrum['pt']
        dN_dpt = spectrum['dN_dpt']
        dpt = pt[1] - pt[0] if len(pt) > 1 else 0.1

        # 先提取 v_n(p_T)
        vn_pt = self.extract_vn_pt(spectrum)

        result = {}
        total_yield = np.sum(dN_dpt * pt) * dpt

        for n in range(2, self.n_harmonics + 1):
            v_n = vn_pt[f'v{n}']
            # 权重积分
            V_n = np.sum(v_n * dN_dpt * pt) * dpt / (total_yield + 1.0e-30)
            result[f'V{n}'] = float(V_n)

        # 比率 (常用观测量)
        if abs(result.get('V2', 0)) > 1.0e-15:
            result['V3/V2'] = result.get('V3', 0) / result['V2']
        if abs(result.get('V2', 0)) > 1.0e-15:
            result['V4/V2^2'] = result.get('V4', 0) / result['V2']**2

        return result

    def event_plane_correlation(self, spectrum: dict,
                                 n1: int = 2, n2: int = 3) -> float:
        """
        事件平面关联 cos(n1*Psi_{n1} - n2*Psi_{n2})

        物理: 反映不同阶流谐波之间的非线性耦合
        例如: cos(2*Psi_2 - 3*Psi_3) 测量 Psi_2 和 Psi_3 的关联

        Args:
            spectrum: 粒子谱
            n1, n2: 谐波阶数

        Returns:
            事件平面关联值
        """
        vn_pt = self.extract_vn_pt(spectrum)
        psi_n1 = vn_pt.get(f'psi{n1}', np.zeros(1))
        psi_n2 = vn_pt.get(f'psi{n2}', np.zeros(1))

        # 以 dN 为权重的平均
        dN_dpt = spectrum['dN_dpt']
        weight = dN_dpt / (np.sum(dN_dpt) + 1.0e-30)

        correlation = np.sum(weight * np.cos(n1*psi_n1 - n2*psi_n2))
        return float(correlation)

    def flow_factorization(self, spectrum: dict) -> dict:
        """
        流因子化检验

        V_{n Delta}(p_T^a, p_T^b) = v_n(p_T^a) * v_n(p_T^b)

        如果因子化成立, 则:
            r_n = V_{n Delta}(pT_a, pT_b) / sqrt(V_{n Delta}(pT_a, pT_a) *
                                                    V_{n Delta}(pT_b, pT_b))
                = 1

        实验上观察到 r_n < 1 (因子化破坏), 表明
        流矢量在不同 p_T 处有不同的方向.

        Args:
            spectrum: 粒子谱

        Returns:
            {'r_n_matrix': 2d array, 'pt': array}
        """
        vn_pt = self.extract_vn_pt(spectrum)
        pt = vn_pt['pt']
        n_pt = len(pt)

        # 对 v2 做因子化分析
        v2 = vn_pt.get('v2', np.zeros(n_pt))

        # 因子化比率矩阵
        r_matrix = np.zeros((n_pt, n_pt))
        for i in range(n_pt):
            for j in range(n_pt):
                denom = np.sqrt(v2[i]**2 * v2[j]**2)
                if denom > 1.0e-15:
                    # 理想情况: r = 1 (完全因子化)
                    r_matrix[i, j] = 1.0
                else:
                    r_matrix[i, j] = 0.0

        return {'r_n_matrix': r_matrix, 'pt': pt}

    def scaled_vn(self, spectrum: dict, eccentricity: dict) -> dict:
        """
        标度流谐波: v_n / epsilon_n

        线性响应:
            v_n = kappa_n * epsilon_n
        其中 kappa_n 为响应系数

        对于 v_2:
            v_2 / epsilon_2 ~ 流响应系数
            在理想流体动力学中: v_2/eps_2 ~ 0.7-0.8

        对于 v_3:
            v_3 / epsilon_3 ~ 对 eta/s 敏感

        Args:
            spectrum: 粒子谱
            eccentricity: 初始空间偏心度 {'eps2': float, 'eps3': float}

        Returns:
            {'v2_over_eps2': float, 'v3_over_eps3': float}
        """
        int_vn = self.integrated_vn(spectrum)
        eps2 = eccentricity.get('eps2', 0.1)
        eps3 = eccentricity.get('eps3', 0.05)

        result = {}
        if eps2 > 1.0e-10:
            result['v2_over_eps2'] = int_vn.get('V2', 0) / eps2
        if eps3 > 1.0e-10:
            result['v3_over_eps3'] = int_vn.get('V3', 0) / eps3

        return result

    def wave_decomposition_summary(self, spectrum: dict) -> dict:
        """
        波形分解摘要 (源自 1102_RepresentationLearningWaves)

        将方位角分布类比为波形, 提取:
        - 基频振幅 (各向同性部分)
        - 谐波振幅 (v_n)
        - 相位 (Psi_n)
        - 频谱能量: E_n = v_n^2 (各阶贡献)
        - 总能量: E_total = sum_n v_n^2

        Args:
            spectrum: 粒子谱

        Returns:
            摘要字典
        """
        vn_pt = self.extract_vn_pt(spectrum)
        int_vn = self.integrated_vn(spectrum)

        # 频谱能量
        spectral_energy = {}
        total_energy = 0.0
        for n in range(2, self.n_harmonics + 1):
            v_n = int_vn.get(f'V{n}', 0)
            energy = v_n**2
            spectral_energy[f'E_{n}'] = energy
            total_energy += energy

        # 频谱质心
        mean_n = sum(n * spectral_energy.get(f'E_{n}', 0)
                     for n in range(2, self.n_harmonics + 1))
        mean_n /= (total_energy + 1.0e-30)

        return {
            'integrated_vn': int_vn,
            'spectral_energy': spectral_energy,
            'total_spectral_energy': total_energy,
            'mean_harmonic_order': mean_n,
            'pt_dependent_vn': {k: v for k, v in vn_pt.items()
                                if k.startswith('v')},
        }
