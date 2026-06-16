# -*- coding: utf-8 -*-
"""
random_field.py
===============

随机场生成模块: 基于 Karhunen-Loève (KL) 展开构造随机亥姆霍兹波速场
    a(x, omega) = a0(x) + sum_{k=1}^{K} sqrt(lambda_k) * xi_k(omega) * phi_k(x)

融合的种子项目:
  - 479_gram_polynomial   -> Gram 正交多项式作为 KL 基函数
  - 1095_blip_bio Pore_model -> 多层介质随机建模思想
  - 333_ellipsoid_grid    -> 参数空间椭球置信域采样

科学背景:
  随机亥姆霍兹方程  -Delta u - k^2 a(x,omega) u = f
  中 a(x,omega) 是随机波速平方倒数场, 通过 KL 展开离散化为有限维
  随机变量 xi = (xi_1, ..., xi_K), xi_k iid N(0,1).
"""

import math
from typing import List, Tuple
from middle_square_rng import MiddleSquareRNG
from scientific_formulas import (
    kl_eigenvalue_1d, gram_inner_product_discrete,
    hermite_probabilist, covariance_kernel
)


class GramPolynomialBasis:
    """Gram 正交多项式基 (复刻 seed project 479_gram_polynomial).

    在离散点集 {-1, -1+2/(m-1), ..., 1} 上关于离散内积正交,
    递推关系 (源自 gram_polynomial_evaluate.m):
        p_0 = 1, p_1 = x
        p_{n+1}(x) = x p_n(x) - beta_{n-1} p_{n-1}(x)
    其中 beta_k 来自 Gram 内积结构。

    用于: 把 KL 基从简单 sin 函数提升为 Gram 正交多项式,
    更好地逼近非均匀协方差核的特征函数。
    """

    def __init__(self, m: int):
        """
        Args:
            m: 离散节点数 (必须 >= 2)
        """
        if m < 2:
            raise ValueError("Gram 基至少需要 2 个节点, 收到 m=%d" % m)
        self.m = m
        # 节点: [-1, 1] 上均匀分布
        self.nodes = [-1.0 + 2.0 * i / (m - 1) for i in range(m)]
        # 离散内积权重 (均匀)
        self.weights = [2.0 / m] * m
        self._beta_cache = {}

    def _gram_beta(self, s: int) -> float:
        """Gram beta 系数: beta_s = (s+1) * s * m^2 / (4*m^2 - (2s+1)^2)
        这是 Gram 多项式三项递推的显式公式 (Dahlquist & Bjorck 2008).
        """
        if s in self._beta_cache:
            return self._beta_cache[s]
        num = (s + 1) * s * (self.m ** 2)
        denom = 4.0 * (self.m ** 2) - ((2 * s + 1) ** 2)
        if abs(denom) < 1e-14:
            beta = 0.0
        else:
            beta = num / denom
        self._beta_cache[s] = beta
        return beta

    def evaluate(self, n: int, x_vals: List[float]) -> List[float]:
        """计算 n 阶 Gram 多项式在 x_vals 处的值。
        复刻 gram_polynomial_evaluate.m 的递推逻辑。
        """
        k = len(x_vals)
        if n == 0:
            return [1.0] * k
        pn_minus1 = [1.0] * k
        pn = list(x_vals)
        if n == 1:
            return pn
        for step in range(1, n):
            beta = self._gram_beta(step - 1)
            pnp1 = [x_vals[i] * pn[i] - beta * pn_minus1[i]
                    for i in range(k)]
            pn_minus1 = pn
            pn = pnp1
        return pn

    def evaluate_on_nodes(self, n: int) -> List[float]:
        return self.evaluate(n, self.nodes)


class RandomFieldKL:
    """随机场的 KL 展开: 封装从 Gaussian 系数 xi 到空间场的映射。

    数学形式:
        a(x, xi) = a0(x) + sum_{k=1}^K sqrt(lambda_k) * xi_k * phi_k(x)

    本实现使用一维区间 [0, L] 上的平方指数协方差核:
        C(x1, x2) = sigma^2 * exp(-|x1-x2|^2 / (2*ell^2))
    特征值通过 kl_eigenvalue_1d 的渐近式近似, 基函数为 Gram 正交多项式
    (经 affine 映射到 [0, L])。
    """

    def __init__(self, L: float = 1.0, sigma: float = 0.3,
                 ell: float = 0.2, K: int = 6, m_gram: int = 32,
                 a0_func=None):
        """
        Args:
            L: 空间域长度
            sigma: 随机场标准差 (扰动幅度)
            ell: 相关长度
            K: KL 展开截断阶数
            m_gram: Gram 多项式离散节点数
            a0_func: 基线场 a0(x), 默认为常数 1.0
        """
        if sigma < 0:
            raise ValueError("sigma 必须非负")
        if ell <= 0:
            raise ValueError("相关长度 ell 必须为正")
        if K < 1:
            raise ValueError("KL 截断阶数 K 必须 >= 1")
        self.L = L
        self.sigma = sigma
        self.ell = ell
        self.K = K
        self.m_gram = m_gram
        self.a0_func = a0_func if a0_func is not None else (lambda x: 1.0)

        # KL 特征值 (降序)
        self.eigenvalues = [kl_eigenvalue_1d(k + 1, L, ell)
                            for k in range(K)]
        self.sqrt_eig = [math.sqrt(max(ev, 0.0)) for ev in self.eigenvalues]

        # Gram 基
        self.gram = GramPolynomialBasis(m=m_gram)

    def _basis_k(self, k: int, x: float) -> float:
        """第 k 个 KL 基函数在 x 处的值。
        把 Gram 多项式从 [-1, 1] affinely 映射到 [0, L], 然后归一化。
        """
        # affine 映射: t = 2x/L - 1 in [-1, 1]
        t = 2.0 * x / self.L - 1.0
        # 用 Gram 基的 (k+1) 阶多项式
        g = self.gram.evaluate(k + 1, [t])[0]
        # L^2([0,L]) 归一化 (近似)
        norm = math.sqrt(self.L / (2.0 * (k + 1) + 1.0))
        norm = max(norm, 1e-12)
        return g / norm

    def evaluate(self, x: float, xi: List[float]) -> float:
        """在空间点 x、随机实现 xi 下计算随机场 a(x, omega).

        Args:
            x: 空间坐标 in [0, L]
            xi: 长度 K 的 Gaussian 系数向量

        Returns:
            a(x, omega) = a0(x) + sum_k sqrt(lambda_k) xi_k phi_k(x)

        边界保护: 保证 a(x, omega) >= a_min > 0, 以确保亥姆霍兹问题
        良态 (波速平方不能为零或负)。
        """
        if len(xi) < self.K:
            raise ValueError("xi 长度 %d 不足 KL 阶数 K=%d" %
                             (len(xi), self.K))
        s = self.a0_func(x)
        for k in range(self.K):
            s += self.sqrt_eig[k] * xi[k] * self._basis_k(k, x)
        # 物理下界: 波速平方至少 0.1 (避免奇异)
        return max(s, 0.1)

    def sample_xi(self, rng: MiddleSquareRNG) -> List[float]:
        """从标准高斯采样一个 K 维系数向量。"""
        return rng.next_gaussian_vector(self.K)

    def truncation_error(self) -> float:
        """KL 截断相对误差: sum_{k>K} lambda_k / sum_{k>=1} lambda_k.
        用几何级数近似尾部。
        """
        total = sum(self.eigenvalues)
        if total < 1e-15:
            return 0.0
        captured = sum(self.eigenvalues)
        # 估计尾部: 用最后一个特征值的指数衰减
        tail = self.eigenvalues[-1] * math.exp(-1.0) / (1.0 - math.exp(-1.0))
        return tail / (total + tail + 1e-15)


class EllipsoidConfidenceSampler:
    """椭球置信域采样器 (复刻 seed project 333_ellipsoid_grid).

    在 SAA 中, 有时需要在 Gaussian 随机变量的椭球置信域内
    {xi : sum_k xi_k^2 / r_k^2 <= 1}
    进行确定性采样, 以构造低差异 (low-discrepancy) SAA 样本集,
    替代纯 Monte Carlo, 从而获得 O(1/N) 而非 O(1/sqrt(N)) 的收敛率。

    本实现采用简单的网格截断 + 椭球过滤。
    """

    def __init__(self, radii: List[float], n_per_axis: int = 5):
        """
        Args:
            radii: 各轴半径 (通常 = sqrt(特征值) * 置信系数)
            n_per_axis: 每轴采样数
        """
        self.dim = len(radii)
        self.radii = radii
        self.n_per_axis = max(3, min(n_per_axis, 12))
        # 归一化步长
        self.step = 2.0 / self.n_per_axis

    def generate(self) -> List[List[float]]:
        """生成椭球内部的网格点集 (全组合 + 过滤)."""
        import itertools
        coords_1d = [[-1.0 + self.step * (i + 0.5)
                      for i in range(self.n_per_axis)]] * self.dim
        points = []
        for combo in itertools.product(*coords_1d):
            # 椭球检验: sum (c_k)^2 <= 1
            s = sum(c * c for c in combo)
            if s <= 1.0:
                # 缩放回原始半径
                pt = [combo[k] * self.radii[k] for k in range(self.dim)]
                points.append(pt)
        return points

    def count_points(self) -> int:
        """估计椭球内网格点数 (不真正生成), 用于预算。"""
        import itertools
        coords_1d = [range(self.n_per_axis)] * self.dim
        count = 0
        for combo in itertools.product(*coords_1d):
            s = sum((-1.0 + self.step * (i + 0.5)) ** 2 for i in combo)
            if s <= 1.0:
                count += 1
        return count
