"""
hermite_functionals.py
======================
广义 Hermite 泛函与 Gaussian 加权 VI 的变分结构.

数学背景
--------
广义 Hermite 多项式 H_n^{(α)}(x) 关于权重函数
    w_α(x) = |x|^α · exp(-x²)
正交.

在随机 VI 中, 当随机参数服从 Gaussian 分布时,
用 Hermite 多项式展开解:
    x*(ω) ≈ Σ_{k=0}^M c_k · H_k(ω)

投影到 Hermite 基:
    ⟨x*, H_j⟩_{w} = c_j · ⟨H_j, H_j⟩_w

Gauss-Hermite 求积:
    ∫_{-∞}^{∞} f(x) w_α(x) dx ≈ Σ_{k=1}^N w_k f(x_k)

其中 x_k 为 H_N^{(α)} 的零点, w_k 为对应的求积权重.

多项式精确性:
    N 点 Gauss-Hermite 求积对 2N-1 次多项式精确.

在 VI 中的应用:
    随机 VI 的 Galerkin 投影:
    求 c = (c_0, ..., c_M) 使得
        Σ_j c_j ⟨F(Σ_k c_k H_k(ω)), H_j(ω)⟩_w = 0

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy.special import gamma as gamma_fn, factorial


class GeneralizedHermitePolynomials:
    """
    广义 Hermite 多项式的计算.

    定义:
        H_0^{(α)}(x) = 1
        H_1^{(α)}(x) = x
        H_{n+1}^{(α)}(x) = x · H_n^{(α)}(x) - n · H_{n-1}^{(α)}(x) - (α/2)(1-(-1)^n) H_{n-1}^{(α)}(x)

    当 α = 0 时, 退化为标准 (概率论) Hermite 多项式.
    当 α > 0 时, 关于权重 |x|^α exp(-x²) 正交.

    正交关系:
        ∫_{-∞}^{∞} H_m^{(α)}(x) H_n^{(α)}(x) |x|^α exp(-x²) dx
            = δ_{mn} · γ_n(α)

    其中
        γ_n(α) = n! · √π · 2^{-n} · Γ((α+n+1)/2) / Γ((α+1)/2)
                 · (1 + α · (1-(-1)^n)/(2n))
    """

    @staticmethod
    def evaluate(n_max: int, x: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        计算 H_0, H_1, ..., H_{n_max} 在点 x 处的值.

        Parameters
        ----------
        n_max : int
            最高阶数
        x : ndarray (m,)
            求值点
        alpha : float
            广义参数

        Returns
        -------
        H : ndarray (n_max+1, m)
            H[k, :] = H_k^{(α)}(x)
        """
        m = len(x)
        H = np.zeros((n_max + 1, m))
        H[0, :] = 1.0
        if n_max >= 1:
            H[1, :] = x

        for n in range(1, n_max):
            # 三项递推
            correction = 0.0
            if alpha != 0.0:
                correction = (alpha / 2.0) * (1.0 - (-1.0)**n) * H[n - 1, :]
            H[n + 1, :] = x * H[n, :] - n * H[n - 1, :] - correction
        return H

    @staticmethod
    def normalization_constant(n: int, alpha: float = 0.0) -> float:
        """
        计算 ||H_n^{(α)}||² 的解析值.

        γ_n(α) = n! · √π · 2^{-n} · Γ((α+n+1)/2) / Γ((α+1)/2)
        """
        num = factorial(n, exact=True) * np.sqrt(np.pi) * (2.0 ** (-n))
        num *= gamma_fn((alpha + n + 1) / 2.0)
        den = gamma_fn((alpha + 1) / 2.0)
        return float(num / den)


class GaussHermiteQuadrature:
    """
    Gauss-Hermite 求积规则.

    N 点规则:
        ∫_{-∞}^{∞} f(x) exp(-x²) dx ≈ Σ_{k=1}^N w_k f(x_k)

    其中 x_k 为 Hermite 多项式 H_N(x) 的零点.

    权重:
        w_k = 2^{N-1} N! √π / (N² [H_{N-1}(x_k)]²)

    精确性: 对 2N-1 次多项式精确.
    """

    def __init__(self, n_points: int):
        """
        Parameters
        ----------
        n_points : int
            求积点数 N
        """
        self.n_points = n_points
        self.nodes, self.weights = self._compute_nodes_weights(n_points)

    def _compute_nodes_weights(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过伴随矩阵的特征分解计算 Gauss-Hermite 节点和权重.

        伴随矩阵 (对称三对角):
            a_i = 0 (对角)
            b_i = √(i/2) (次对角)
        节点 = 特征值, 权重 = √π · (特征向量第一分量)²
        """
        if N <= 0:
            return np.array([]), np.array([])
        if N == 1:
            return np.array([0.0]), np.array([np.sqrt(np.pi)])

        # 构建伴随矩阵
        i = np.arange(1, N)
        b = np.sqrt(i / 2.0)
        J = np.diag(b, -1) + np.diag(b, 1)

        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(J)
        nodes = eigvals
        weights = np.sqrt(np.pi) * eigvecs[0, :]**2

        # 排序
        idx = np.argsort(nodes)
        return nodes[idx], weights[idx]

    def integrate(self, f_values: np.ndarray) -> float:
        """
        对函数值进行 Gauss-Hermite 积分.

        ∫ f(x) exp(-x²) dx ≈ Σ w_k f(x_k)

        Parameters
        ----------
        f_values : ndarray (N,)
            函数在节点处的值
        """
        if len(f_values) != self.n_points:
            raise ValueError(f"需要 {self.n_points} 个函数值, 得到 {len(f_values)}")
        return float(np.sum(self.weights * f_values))

    def integrate_gaussian(self, f_values: np.ndarray, mu: float = 0.0,
                           sigma: float = 1.0) -> float:
        """
        对标准 Gaussian 测度积分:
            E[f(X)] = ∫ f(x) φ(x) dx,  φ(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

        变换: x = μ + σ√2 · t,  则
            E[f(X)] ≈ (1/√π) Σ w_k f(μ + σ√2 t_k)
        """
        x_transformed = mu + sigma * np.sqrt(2.0) * self.nodes
        f_transformed = f_values
        return float(np.sum(self.weights * f_transformed) / np.sqrt(np.pi))

    def test_exactness(self, degree_max: int, alpha: float = 0.0) -> np.ndarray:
        """
        测试求积规则的多项式精确性.

        对 k = 0, 1, ..., degree_max, 比较:
            数值积分: Σ w_k x_k^k
            精确值:   ∫ x^k exp(-x²) dx = Γ((k+1)/2) / 2  (当 k 为偶数)
                      0                                      (当 k 为奇数)

        Returns
        -------
        errors : ndarray (degree_max+1,)
            各阶的绝对误差
        """
        errors = np.zeros(degree_max + 1)
        for k in range(degree_max + 1):
            # 精确值
            if k % 2 == 1:
                exact = 0.0
            else:
                exact = gamma_fn((k + 1) / 2.0) / 2.0
            # 数值值
            numerical = np.sum(self.weights * self.nodes**k)
            errors[k] = abs(numerical - exact)
        return errors


class StochasticVIGalerkin:
    """
    随机 VI 的 Hermite-Galerkin 投影.

    随机 VI: 求 x*(ω) ∈ K 使得
        ⟨F(x*(ω), ω), y - x*(ω)⟩ ≥ 0,  ∀ y ∈ K,  a.s.

    Hermite 展开:
        x*(ω) ≈ Σ_{k=0}^M c_k H_k(ξ),  ξ ~ N(0,1)

    Galerkin 条件:
        ⟨F(Σ c_k H_k(ξ)), H_j(ξ)⟩_{L²(P)} = 0,  j = 0, ..., M

    使用 Gauss-Hermite 求积计算内积.
    """

    def __init__(self, M: int, n_quadrature: int = 32):
        self.M = M
        self.quadrature = GaussHermiteQuadrature(n_quadrature)
        self.hermite = GeneralizedHermitePolynomials()

    def build_coefficient_system(
        self,
        F_eval: callable,
        xi_samples: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建 Galerkin 系统的系数矩阵和右端项.

        对确定性 F(x) = Mx + q, 展开为:
            Σ_k c_k · M · ⟨H_k H_j⟩ + q · ⟨H_j⟩ = 0
        由于 Hermite 正交, ⟨H_k H_j⟩ = δ_{kj} γ_k,
        系统退化为:
            c_j · M · γ_j + q · δ_{j0} · √π = 0
        """
        M_sys = np.zeros(((self.M + 1), (self.M + 1)))
        rhs = np.zeros(self.M + 1)

        for j in range(self.M + 1):
            gamma_j = self.hermite.normalization_constant(j)
            M_sys[j, j] = gamma_j
            if j == 0:
                rhs[j] = -np.sqrt(np.pi)

        return M_sys, rhs
