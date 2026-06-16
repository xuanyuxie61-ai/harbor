# -*- coding: utf-8 -*-
"""
basis_selector.py — 基函数选择器
===================================
核心科学问题: 根据异质结物理特征自动选择最优基函数
(有限差分/有限元/谱方法).

融合种子项目:
  - 综合各项目的基函数选择逻辑
"""

import numpy as np


class BasisSelector:
    """
    基函数选择器.

    根据问题特征 (势能变化尺度、界面锐度、精度要求)
    自动选择最优离散化方法.
    """

    METHODS = ['fd2', 'fd4', 'fd6', 'fd8', 'fem_t3', 'fem_t6',
               'legendre', 'fourier']

    def __init__(self, z_grid, V_profile, m_star_profile):
        """
        参数
        ----
        z_grid : ndarray
            空间网格
        V_profile : ndarray
            势能剖面 [eV]
        m_star_profile : ndarray
            有效质量剖面 [m0]
        """
        self.z = z_grid
        self.V = V_profile
        self.m_star = m_star_profile
        self.N = len(z_grid)
        self.dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1e-10

        # 分析问题特征
        self._analyze_problem()

    def _analyze_problem(self):
        """分析问题的数值特征."""
        # 势能变化率
        dVdz = np.gradient(self.V, self.dz)
        self.max_dVdz = np.max(np.abs(dVdz))

        # 势能二阶导数
        d2Vdz2 = np.gradient(dVdz, self.dz)
        self.max_d2Vdz2 = np.max(np.abs(d2Vdz2))

        # 有效质量变化率
        dmdz = np.gradient(self.m_star, self.dz)
        self.max_dmdz = np.max(np.abs(dmdz))

        # 界面锐度参数
        V_range = np.max(self.V) - np.min(self.V)
        V_range = max(V_range, 1e-10)
        self.sharpness = self.max_dVdz * self.dz / V_range

        # 质量不连续度
        m_range = np.max(self.m_star) - np.min(self.m_star)
        self.mass_discontinuity = m_range / max(np.mean(self.m_star), 1e-10)

    def recommend_method(self, target_accuracy=1e-6):
        """
        推荐最优方法.

        选择准则:
        - 势能变化平缓 → Fourier (谱方法)
        - 势能变化陡峭但光滑 → 高阶 FD
        - 有效质量不连续 → FEM (处理界面条件)
        - 一般情况 → 4阶 FD
        """
        # 界面很尖锐 → 需要能处理不连续的方法
        if self.sharpness > 0.5 or self.mass_discontinuity > 0.3:
            return 'fd4'  # BDD 格式处理质量不连续

        # 势能很光滑 → 谱方法
        if self.sharpness < 0.05 and self.mass_discontinuity < 0.05:
            return 'legendre'

        # 中等变化 → 高阶差分
        if self.max_d2Vdz2 * self.dz**2 < 0.01:
            return 'fd4'
        elif self.max_d2Vdz2 * self.dz**2 < 0.1:
            return 'fd6'
        else:
            return 'fd4'

    def estimate_error(self, method):
        """
        估计指定方法的截断误差.

        对于 p 阶差分:
            ε ≈ C_p * h^p * max|f^{(p+2)}|

        使用简化估计避免高阶导数的数值噪声.
        """
        h = self.dz
        V_range = max(np.max(np.abs(self.V)) - np.min(np.abs(self.V)), 1e-10)
        # 特征长度尺度
        L_char = (self.z[-1] - self.z[0])

        if method == 'fd2':
            return h**2 / 12.0 * self.max_d2Vdz2
        elif method == 'fd4':
            # 估计: 用 V_range / L^4 作为 4 阶导数量级
            d4V_est = V_range / max(L_char**4, 1e-60)
            return h**4 / 90.0 * max(d4V_est, self.max_d2Vdz2 / L_char**2)
        elif method == 'fd6':
            d6V_est = V_range / max(L_char**6, 1e-80)
            return h**6 / 560.0 * max(d6V_est, self.max_d2Vdz2 / L_char**4)
        elif method == 'fd8':
            d8V_est = V_range / max(L_char**8, 1e-100)
            return h**8 * 2.0 / 3150.0 * max(d8V_est, self.max_d2Vdz2 / L_char**6)
        elif method == 'legendre':
            # 谱方法指数收敛
            N = self.N
            return np.exp(-2 * min(N, 50) * 0.3)
        else:
            return h**2 / 12.0 * self.max_d2Vdz2

    def _estimate_higher_derivative(self, order):
        """
        数值估计高阶导数.
        """
        f = self.V.copy()
        h = self.dz
        for _ in range(order):
            f_new = np.zeros_like(f)
            for i in range(1, len(f) - 1):
                f_new[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / h**2
            f = f_new
        return np.max(np.abs(f))

    def required_grid_points(self, method, target_error):
        """
        计算达到目标精度所需的网格点数.
        """
        # 二分法求解 N
        N_low, N_high = 10, 10000

        for _ in range(50):
            N_mid = (N_low + N_high) // 2
            h_test = (self.z[-1] - self.z[0]) / N_mid

            if method == 'fd2':
                err = h_test**2 / 12.0 * self.max_d2Vdz2
            elif method == 'fd4':
                err = h_test**4 / 90.0 * self.max_d2Vdz2
            else:
                err = h_test**2 * self.max_d2Vdz2

            if err < target_error:
                N_high = N_mid
            else:
                N_low = N_mid

            if N_high - N_low <= 1:
                break

        return N_high

    def convergence_rate(self, method, refinement_ratios=None):
        """
        预测网格加密时的收敛速率.

        Richardson 外推:
            rate = log(err1/err2) / log(r)
        其中 r 是加密比.
        """
        if refinement_ratios is None:
            refinement_ratios = [2, 4, 8]

        rates = []
        for r in refinement_ratios:
            if method.startswith('fd'):
                order = int(method[2:])
                rates.append(order)
            elif method == 'legendre':
                rates.append('exponential')
            elif method.startswith('fem'):
                if 't6' in method:
                    rates.append(4)  # 二次单元
                else:
                    rates.append(2)  # 线性单元
            else:
                rates.append(2)

        return rates
