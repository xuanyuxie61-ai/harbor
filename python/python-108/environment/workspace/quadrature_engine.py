# -*- coding: utf-8 -*-
"""
quadrature_engine.py
高维数值积分引擎

核心公式与物理背景
------------------
1. 球面求积 (Sphere Quadrature)
   单位球面 S² 上单项式的精确积分：
   I(e₁,e₂,e₃) = ∮_{S²} x^{e₁} y^{e₂} z^{e₃} dΩ
   若任一 eᵢ 为奇数，则 I = 0 (对称性)。
   否则：
       I = 2·∏_{i=1}^{3} Γ((eᵢ+1)/2) / Γ(½·Σ(eᵢ+1))
   其中 Γ(z) 为 Gamma 函数。

2. 2D Vandermonde 求积权重
   给定 n = (t+1)(t+2)/2 个节点 (xₗ, yₗ)，要求精确积分所有
   次数 ≤ t 的单项式 xⁱ yʲ。构造线性系统：
       V · w = r
   其中 V_{k,l} = xₗ^{i_k} yₗ^{j_k}，
   r_k = [(B^{i_k+1}-A^{i_k+1})/(i_k+1)] · [(D^{j_k+1}-C^{j_k+1})/(j_k+1)]
   解得权重向量 w。

3. 高斯-勒让德节点与权重
   在 [-1,1] 上的 m 点高斯-勒让德规则具有 2m-1 次代数精度：
       ∫_{-1}^{1} f(x) dx ≈ Σ_{k=1}^{m} w_k f(x_k)

融合来源
--------
- 951_quadrature_weights_vandermonde_2d : 2D Vandermonde 求积权重
- 1119_sphere_integrals                 : 球面单项式精确积分与蒙特卡洛采样
"""

import numpy as np
from scipy.special import gamma as Gamma
from typing import Tuple, Optional


class SphereQuadrature:
    """
    单位球面 S² 上的精确积分与蒙特卡洛采样。
    """

    @staticmethod
    def sphere01_area() -> float:
        """返回单位球面面积 A = 4π"""
        return 4.0 * np.pi

    @staticmethod
    def monomial_integral_exact(e: Tuple[int, int, int]) -> float:
        """
        精确计算 ∮_{S²} x^{e₁} y^{e₂} z^{e₃} dΩ。

        公式
        ----
        若 e₁, e₂, e₃ 中任一为奇数，返回 0。
        否则：
            I = 2 · Γ((e₁+1)/2) · Γ((e₂+1)/2) · Γ((e₃+1)/2)
                / Γ((e₁+e₂+e₃+3)/2)
        """
        e1, e2, e3 = e
        if e1 < 0 or e2 < 0 or e3 < 0:
            raise ValueError("指数必须非负")
        if (e1 % 2 == 1) or (e2 % 2 == 1) or (e3 % 2 == 1):
            return 0.0
        num = 2.0 * Gamma(0.5 * (e1 + 1)) * Gamma(0.5 * (e2 + 1)) * Gamma(0.5 * (e3 + 1))
        den = Gamma(0.5 * (e1 + e2 + e3 + 3))
        return float(num / den)

    @staticmethod
    def evaluate_monomial(points: np.ndarray, e: Tuple[int, int, int]) -> np.ndarray:
        """
        在 points (N×3) 上求值单项式 x^{e₁} y^{e₂} z^{e₃}。
        """
        e1, e2, e3 = e
        return (points[:, 0] ** e1) * (points[:, 1] ** e2) * (points[:, 2] ** e3)

    @staticmethod
    def uniform_sample(n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        在单位球面 S² 上均匀采样 n 个点。
        方法：先采样 N(0,I₃) 再单位化。
        理论保证：若 g ~ N(0,I₃)，则 g/||g|| 在 S² 上均匀分布。
        """
        if n <= 0:
            raise ValueError("n 必须 > 0")
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(size=(n, 3))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return g / norms

    def monte_carlo_integral(self, n_samples: int, e: Tuple[int, int, int], seed: Optional[int] = None) -> Tuple[float, float]:
        """
        用蒙特卡洛法估计球面单项式积分，并返回标准误差。

        公式
        ----
        I ≈ A · (1/N) · Σ f(xᵢ)
        σ_I ≈ A · σ_f / √N
        """
        pts = self.uniform_sample(n_samples, seed)
        f_vals = self.evaluate_monomial(pts, e)
        mean_f = np.mean(f_vals)
        std_f = np.std(f_vals, ddof=1)
        A = self.sphere01_area()
        I_est = A * mean_f
        I_err = A * std_f / np.sqrt(n_samples)
        return I_est, I_err


class Vandermonde2DQuadrature:
    """
    基于 2D Vandermonde 矩阵的任意节点求积权重计算。
    """

    @staticmethod
    def compute_weights(nodes_x: np.ndarray,
                        nodes_y: np.ndarray,
                        total_degree: int,
                        rect_a: float = -1.0,
                        rect_b: float = 1.0,
                        rect_c: float = -1.0,
                        rect_d: float = 1.0,
                        rcond: float = 1e-12) -> np.ndarray:
        """
        计算 2D 求积权重，使得对所有 i+j ≤ total_degree 的单项式 xⁱ yʲ 精确积分。

        参数
        ----
        nodes_x, nodes_y : np.ndarray
            节点坐标，长度 n = (t+1)(t+2)/2
        total_degree : int
            总次数 t
        rect_a, rect_b, rect_c, rect_d : float
            矩形积分域 [a,b] × [c,d]
        rcond : float
            伪逆截断阈值（处理病态 Vandermonde 矩阵）

        返回
        ----
        w : np.ndarray
            权重向量，长度 n
        """
        n_expected = (total_degree + 1) * (total_degree + 2) // 2
        n = len(nodes_x)
        if n != n_expected:
            raise ValueError(f"节点数 {n} 与期望 {n_expected} 不符，t={total_degree}")

        # 构造单项式指数列表
        exponents = []
        for i in range(total_degree + 1):
            for j in range(total_degree + 1 - i):
                exponents.append((i, j))

        # 构造 Vandermonde 矩阵 V_{k,l} = x_l^{i_k} y_l^{j_k}
        V = np.zeros((n, n), dtype=float)
        for k, (i, j) in enumerate(exponents):
            V[k, :] = (nodes_x ** i) * (nodes_y ** j)

        # 构造右端项：精确积分值
        rhs = np.zeros(n, dtype=float)
        for k, (i, j) in enumerate(exponents):
            int_x = (rect_b ** (i + 1) - rect_a ** (i + 1)) / (i + 1)
            int_y = (rect_d ** (j + 1) - rect_c ** (j + 1)) / (j + 1)
            rhs[k] = int_x * int_y

        # 求解线性系统（使用伪逆处理病态）
        w = np.linalg.lstsq(V, rhs, rcond=rcond)[0]
        return w

    @staticmethod
    def integrate(values: np.ndarray, weights: np.ndarray) -> float:
        """用给定权重对函数值积分"""
        if len(values) != len(weights):
            raise ValueError("values 与 weights 长度不一致")
        return float(np.dot(weights, values))


class GaussLegendreTensor:
    """
    张量积形式的高斯-勒让德求积，用于矩形域上的二重积分。
    """

    @staticmethod
    def tensor_quad_2d(f, a: float, b: float, c: float, d: float, m: int = 8):
        """
        在 [a,b]×[c,d] 上用 m×m 张量积高斯-勒让德规则积分函数 f(x,y)。

        公式
        ----
        ∫∫ f(x,y) dx dy ≈ Σ_{p=1}^{m} Σ_{q=1}^{m} w_p w_q · f(x_p, y_q) · J
        其中 J = (b-a)(d-c)/4 为雅可比行列式。
        """
        from numpy.polynomial.legendre import leggauss
        xi, wi = leggauss(m)
        # 映射到物理坐标
        x_phys = 0.5 * (b - a) * xi + 0.5 * (b + a)
        y_phys = 0.5 * (d - c) * xi + 0.5 * (d + c)
        J = 0.25 * (b - a) * (d - c)
        total = 0.0
        for p in range(m):
            for q in range(m):
                total += wi[p] * wi[q] * f(x_phys[p], y_phys[q])
        return total * J


