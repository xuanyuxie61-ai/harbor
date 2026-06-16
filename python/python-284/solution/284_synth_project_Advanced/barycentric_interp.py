# -*- coding: utf-8 -*-
"""
barycentric_interp.py — 重心插值方法
=======================================
核心科学问题: 使用重心插值在高 k 点之间精确重建能带结构 E(k).
Chebyshev 节点上的重心插值具有最优稳定性.

融合种子项目:
  - 072_barycentric_interp_1d: Berrut-Trefethen 重心 Lagrange 插值

数学基础:
  重心 Lagrange 插值公式:
    p(x) = [sum_j w_j/(x-x_j) * f_j] / [sum_j w_j/(x-x_j)]

  其中权重:
    w_j = 1 / prod_{k≠j} (x_j - x_k)

  Chebyshev 节点 (第一类):
    x_j = cos((2j+1)π/(2n+2)), j=0,...,n
  权重简化:
    w_j = (-1)^j * sin((2j+1)π/(2n+2))
"""

import numpy as np


class BarycentricInterpolator:
    """
    重心 Lagrange 插值器.

    用于能带结构的高精度重建.
    """

    def __init__(self, x_nodes, f_values, node_type='chebyshev1'):
        """
        参数
        ----
        x_nodes : ndarray
            插值节点
        f_values : ndarray
            节点处的函数值
        node_type : str
            节点类型 ('chebyshev1', 'chebyshev2', 'equispaced')
        """
        self.x_nodes = np.asarray(x_nodes, dtype=np.float64)
        self.f_values = np.asarray(f_values, dtype=np.float64)
        self.n = len(x_nodes)
        self.node_type = node_type

        # 计算重心权重
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """
        计算重心权重 w_j.

        对于 Chebyshev 节点有解析公式:
          第一类: w_j = (-1)^j * sin((2j+1)π/(2n))
          第二类: w_j = (-1)^j * δ_j, δ_0=δ_n=1/2, 否则=1

        对于等距节点:
          w_j = (-1)^j * C(n, j) = (-1)^j * n! / (j! * (n-j)!)
        """
        n = self.n
        x = self.x_nodes

        if self.node_type == 'chebyshev1':
            w = np.zeros(n)
            for j in range(n):
                w[j] = (-1)**j * np.sin((2 * j + 1) * np.pi / (2 * n))
            return w

        elif self.node_type == 'chebyshev2':
            w = np.zeros(n)
            for j in range(n):
                w[j] = (-1)**j
                if j == 0 or j == n - 1:
                    w[j] *= 0.5
            return w

        elif self.node_type == 'equispaced':
            # 二项式系数权重
            import math
            w = np.zeros(n)
            for j in range(n):
                w[j] = (-1)**j * math.comb(n - 1, j)
            return w

        else:
            # 通用: 直接计算
            w = np.zeros(n)
            for j in range(n):
                prod = 1.0
                for k in range(n):
                    if k != j:
                        prod *= (x[j] - x[k])
                if abs(prod) > 1e-30:
                    w[j] = 1.0 / prod
            return w

    def evaluate(self, x_eval):
        """
        重心插值求值 (Berrut-Trefethen 第二公式):
            p(x) = [sum_j w_j/(x-x_j) * f_j] / [sum_j w_j/(x-x_j)]

        参数
        ----
        x_eval : float or ndarray
            求值点

        返回
        ----
        float or ndarray
            插值结果
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))
        result = np.zeros_like(x_eval)

        for idx, x in enumerate(x_eval):
            # 检查是否在节点上
            diffs = x - self.x_nodes
            exact = np.where(np.abs(diffs) < 1e-14)[0]

            if len(exact) > 0:
                result[idx] = self.f_values[exact[0]]
                continue

            # 重心公式
            terms = self.weights / diffs
            numerator = np.sum(terms * self.f_values)
            denominator = np.sum(terms)

            if abs(denominator) > 1e-30:
                result[idx] = numerator / denominator
            else:
                result[idx] = 0.0

        return result[0] if result.size == 1 else result

    def lebesgue_function(self, x_array):
        """
        Lebesgue 函数 (衡量插值稳定性):
            Λ(x) = sum_j |w_j / (x - x_j)| / |sum_j w_j / (x - x_j)|

        Λ(x) 越大, 插值越不稳定.
        """
        lebesgue = np.zeros_like(x_array)
        for idx, x in enumerate(x_array):
            diffs = x - self.x_nodes
            abs_terms = np.abs(self.weights / np.maximum(np.abs(diffs), 1e-30))
            signed_terms = self.weights / np.maximum(np.abs(diffs), 1e-30) * np.sign(diffs)
            denom = abs(np.sum(signed_terms))
            if denom > 1e-30:
                lebesgue[idx] = np.sum(abs_terms) / denom
            else:
                lebesgue[idx] = 1.0
        return lebesgue


class BandStructureInterpolator:
    """
    能带结构插值器.
    使用重心插值在稀疏 k 点之间重建 E(k).
    """

    def __init__(self, k_points, energies, n_interp=100):
        """
        参数
        ----
        k_points : ndarray
            k 点 [1/m]
        energies : ndarray
            能带能量 [eV]
        n_interp : int
            插值网格密度
        """
        self.k_raw = k_points
        self.E_raw = energies

        # 排序
        idx = np.argsort(self.k_raw)
        self.k_sorted = self.k_raw[idx]
        self.E_sorted = self.E_raw[idx]

        # 创建 Chebyshev 节点 (在 k 范围内)
        k_min, k_max = self.k_sorted[0], self.k_sorted[-1]
        n_nodes = min(len(self.k_sorted), 20)

        # Chebyshev 节点
        j = np.arange(n_nodes)
        self.k_cheb = 0.5 * (k_min + k_max) + 0.5 * (k_max - k_min) * np.cos(
            (2 * j + 1) * np.pi / (2 * n_nodes))
        self.k_cheb = np.sort(self.k_cheb)

        # 在 Chebyshev 节点处插值能带值
        self.E_cheb = np.interp(self.k_cheb, self.k_sorted, self.E_sorted)

        # 创建重心插值器
        self.interp = BarycentricInterpolator(
            self.k_cheb, self.E_cheb, 'chebyshev1')

        # 精细网格
        self.k_fine = np.linspace(k_min, k_max, n_interp)

    def get_fine_band(self):
        """返回精细网格上的能带."""
        E_fine = self.interp.evaluate(self.k_fine)
        return self.k_fine, E_fine

    def group_velocity(self, k_point):
        """
        群速度:
            v_g = (1/ℏ) * dE/dk

        使用重心插值解析求导.
        """
        dk = 1e6  # [1/m], 差分步长
        E_plus = self.interp.evaluate(np.array([k_point + dk]))[0]
        E_minus = self.interp.evaluate(np.array([k_point - dk]))[0]

        from high_order_fd import HBAR, E0
        v_g = (E_plus - E_minus) / (2 * dk) * E0 / HBAR  # [m/s]
        return v_g

    def effective_mass_at(self, k_point):
        """
        有效质量:
            1/m* = (1/ℏ²) * d²E/dk²

        数值差分.
        """
        dk = 1e7  # [1/m]
        E_plus = self.interp.evaluate(np.array([k_point + dk]))[0]
        E_0 = self.interp.evaluate(np.array([k_point]))[0]
        E_minus = self.interp.evaluate(np.array([k_point - dk]))[0]

        d2E_dk2 = (E_plus - 2 * E_0 + E_minus) / dk**2  # [eV·m²]

        from high_order_fd import HBAR, M0, E0
        # m* = ℏ² / (d²E/dk²)
        m_star = HBAR**2 / (abs(d2E_dk2) * E0) / M0
        return m_star
