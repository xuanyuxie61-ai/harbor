# -*- coding: utf-8 -*-
"""
lagrange_nd_hetband.py — 多维 Lagrange 插值用于能带面 E(kx,ky)
================================================================
核心科学问题: 在二维 k 空间上构造能带面 E(kx,ky) 的
多项式逼近, 用于计算有效质量张量和态密度.

融合种子项目:
  - 638_lagrange_nd: 多维 Lagrange 插值 (Sauer-Xu 理论)
"""

import numpy as np


class LagrangeND:
    """
    N 维 Lagrange 插值.

    使用 Sauer-Xu 方法构造部分 Lagrange 基.
    对于 d 维空间和 n 个节点, 构造次数 ≤ r 的多项式空间.
    """

    def __init__(self, points, values, max_degree=None):
        """
        参数
        ----
        points : ndarray, shape (n, d)
            插值节点坐标
        values : ndarray, shape (n,)
            节点函数值
        max_degree : int or None
            最大多项式次数
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.n_pts, self.dim = self.points.shape
        self.n = self.n_pts

        if max_degree is None:
            self.max_degree = self._estimate_degree()
        else:
            self.max_degree = max_degree

        # 构造插值
        self._build_interpolant()

    def _estimate_degree(self):
        """估计合适的多项式次数."""
        # 对于 d 维, r 次多项式有 C(r+d, d) 个系数
        # 需要 n ≤ C(r+d, d)
        d = self.dim
        for r in range(1, 20):
            n_basis = 1
            for k in range(1, r + 1):
                n_basis *= (d + k)
                n_basis //= k
            if n_basis >= self.n:
                return r
        return 5

    def _build_interpolant(self):
        """
        构建 Lagrange 插值 (简化版: 使用 Vandermonde 矩阵).

        对于小规模问题, 直接求解 Vandermonde 系统:
            V · c = f
        其中 V_{ij} = p_j(x_i).
        """
        # 生成单项式指数 (grlex 序)
        self.exponents = self._generate_grlex_exponents()
        n_basis = len(self.exponents)

        # 构建 Vandermonde 矩阵
        V = np.zeros((self.n, n_basis))
        for i in range(self.n):
            for j, exp in enumerate(self.exponents):
                V[i, j] = self._monomial(self.points[i], exp)

        # 最小二乘求解 (可能超定或欠定)
        if self.n >= n_basis:
            self.coeffs, _, _, _ = np.linalg.lstsq(V, self.values, rcond=None)
        else:
            # 欠定: 最小范数解
            self.coeffs = np.linalg.lstsq(V, self.values, rcond=None)[0]

    def _generate_grlex_exponents(self):
        """
        生成 grlex (分次字典序) 排序的单项式指数.
        融合 638_lagrange_nd 的 mono_unrank_grlex 思想.
        """
        d = self.dim
        r = self.max_degree
        exponents = []
        self._grlex_recursive(d, r, [0] * d, 0, exponents)
        return exponents

    def _grlex_recursive(self, d, max_total, current, current_sum, result):
        """递归生成 grlex 排序指数."""
        if d == 1:
            remaining = max_total - current_sum
            for k in range(remaining + 1):
                exp = current[:1] + [k]
                result.append(tuple(exp))
            return

        for k in range(max_total - current_sum + 1):
            self._grlex_recursive(d - 1, max_total,
                                  current + [k], current_sum + k, result)

    def _monomial(self, x, exp):
        """计算单项式 x^exp = x_1^e_1 * x_2^e_2 * ..."""
        val = 1.0
        for i, e in enumerate(exp):
            if e > 0:
                val *= x[i] ** e
        return val

    def evaluate(self, x_eval):
        """
        在点 x_eval 处求值.
        """
        x_eval = np.atleast_2d(np.asarray(x_eval, dtype=np.float64))
        results = np.zeros(len(x_eval))

        for idx, x in enumerate(x_eval):
            val = 0.0
            for j, exp in enumerate(self.exponents):
                val += self.coeffs[j] * self._monomial(x, exp)
            results[idx] = val

        return results


class BandSurfaceInterpolator:
    """
    二维能带面 E(kx, ky) 插值器.
    """

    def __init__(self, kx_points, ky_points, E_values, max_degree=4):
        """
        参数
        ----
        kx_points, ky_points : ndarray
            k 点坐标 (可以是非结构化的)
        E_values : ndarray
            能带能量 [eV]
        max_degree : int
            多项式最大次数
        """
        pts = np.column_stack([kx_points, ky_points])
        self.interp = LagrangeND(pts, E_values, max_degree)
        self.kx_range = (np.min(kx_points), np.max(kx_points))
        self.ky_range = (np.min(ky_points), np.max(ky_points))

    def evaluate_surface(self, kx_grid, ky_grid):
        """
        在网格上计算能带面.
        """
        KX, KY = np.meshgrid(kx_grid, ky_grid, indexing='ij')
        pts = np.column_stack([KX.ravel(), KY.ravel()])
        E = self.interp.evaluate(pts)
        return KX, KY, E.reshape(KX.shape)

    def curvature_tensor(self, k0):
        """
        能带曲率张量 (有效质量张量的倒数):
            C_ij = ∂²E / ∂k_i ∂k_j

        使用多项式解析求导.
        """
        dk = 1e7  # [1/m]
        k0 = np.asarray(k0)

        # 数值 Hessian
        E_pp = self.interp.evaluate(k0 + np.array([dk, dk]))[0]
        E_pm = self.interp.evaluate(k0 + np.array([dk, -dk]))[0]
        E_mp = self.interp.evaluate(k0 + np.array([-dk, dk]))[0]
        E_mm = self.interp.evaluate(k0 + np.array([-dk, -dk]))[0]
        E_p0 = self.interp.evaluate(k0 + np.array([dk, 0]))[0]
        E_m0 = self.interp.evaluate(k0 + np.array([-dk, 0]))[0]
        E_0p = self.interp.evaluate(k0 + np.array([0, dk]))[0]
        E_0m = self.interp.evaluate(k0 + np.array([0, -dk]))[0]
        E_00 = self.interp.evaluate(k0)[0]

        d2E_dkx2 = (E_p0 - 2*E_00 + E_m0) / dk**2
        d2E_dky2 = (E_0p - 2*E_00 + E_0m) / dk**2
        d2E_dkxky = (E_pp - E_pm - E_mp + E_mm) / (4 * dk**2)

        return np.array([[d2E_dkx2, d2E_dkxky],
                         [d2E_dkxky, d2E_dky2]])
