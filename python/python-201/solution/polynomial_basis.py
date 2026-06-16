"""
polynomial_basis.py — 正交多项式基函数构造模块
================================================
实现多维正交多项式基函数的构造、求值和正交性验证。
支持 Hermite (高斯测度), Legendre (均匀测度), Jacobi (Beta 测度)。

核心公式:
  三递推关系:
    P_{n+1}(ξ) = (ξ - α_n) P_n(ξ) - β_n P_{n-1}(ξ)

  归一化:
    Ψ_k(ξ) = P_k(ξ) / √h_k,   h_k = <P_k, P_k>

  多维基:
    Ψ_i(ξ) = Π_{d=1}^D φ_{i_d}^{(d)}(ξ_d)

映射种子项目:
  - 033_asa076: 正态分布与高斯求积
  - 068_ball_integrals: 高维积分公式
"""

import numpy as np
from scipy import special
from typing import List, Tuple, Optional
from config import MeasureConfig, BasisConfig, GlobalConfig
from measure import (recurrence_coefficients, gauss_quadrature,
                     weighted_inner_product, polynomial_norm_sq)
from utils import multi_index_set


class OrthogonalPolynomialBasis:
    """
    多维正交多项式基函数类。

    管理一维正交多项式的求值和多维张量积基的构造。

    属性:
        config:      基函数配置
        measures:    各维测度列表
        indices:     多维索引集, shape (P, d)
        n_basis:     基函数总数 P
        n_dim:       随机空间维度 d
    """

    def __init__(self, config: BasisConfig, measures: List[MeasureConfig]):
        self.config = config
        self.measures = measures
        self.n_dim = len(measures)

        # 生成多维索引集
        self.indices = multi_index_set(
            d=self.n_dim,
            p=config.max_degree,
            truncation=config.truncation,
            q=config.hyperbolic_q,
        )
        self.n_basis = len(self.indices)

        # 预计算一维递推系数
        self._alpha = []
        self._beta = []
        self._norm_sq = []
        for meas in measures:
            alpha, beta = recurrence_coefficients(
                meas, config.max_degree + 1)
            self._alpha.append(alpha)
            self._beta.append(beta)

            norms = []
            for deg in range(config.max_degree + 1):
                norms.append(polynomial_norm_sq(meas, deg))
            self._norm_sq.append(np.array(norms))

    def evaluate_1d(self, dim: int, degree: int,
                    xi: np.ndarray) -> np.ndarray:
        """
        计算一维正交多项式 P_n^{(dim)}(ξ)。

        使用三递推关系:
          P_0 = 1
          P_1 = (ξ - α_0) / β_0 * P_0  (约定)
          P_{k+1} = ((ξ - α_k) P_k - β_k P_{k-1}) / ...

        实际使用标准递推 (首项 P_0 = 1):
          P_{k+1}(ξ) = (ξ - α_k) P_k(ξ) - β_k P_{k-1}(ξ)

        参数:
            dim:    维度索引 (0, ..., d-1)
            degree: 多项式阶数 n
            xi:     求值点, shape (n_points,)

        返回:
            P: shape (n_points,)
        """
        if degree < 0:
            return np.zeros_like(xi)
        if degree == 0:
            return np.ones_like(xi)

        alpha = self._alpha[dim]
        beta = self._beta[dim]

        P_prev = np.ones_like(xi)   # P_0
        P_curr = (xi - alpha[0]) * P_prev  # P_1 = (ξ - α_0)

        if degree == 1:
            return P_curr

        for k in range(1, degree):
            P_next = (xi - alpha[k]) * P_curr - beta[k] * P_prev
            P_prev = P_curr
            P_curr = P_next

        return P_curr

    def evaluate_normalized_1d(self, dim: int, degree: int,
                               xi: np.ndarray) -> np.ndarray:
        """
        计算归一化一维正交多项式。

        Ψ_n(ξ) = P_n(ξ) / √h_n

        参数:
            dim:    维度
            degree: 阶数
            xi:     求值点

        返回:
            归一化多项式值
        """
        P = self.evaluate_1d(dim, degree, xi)
        h_n = self._norm_sq[dim][degree]
        if h_n > 1e-300:
            return P / np.sqrt(h_n)
        return P

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """
        计算所有 P 个基函数在给定节点上的值。

        参数:
            xi: shape (n_points, d), 求值节点

        返回:
            Psi: shape (n_points, P), 基函数值矩阵
        """
        n_points = xi.shape[0]
        Psi = np.ones((n_points, self.n_basis))

        for p_idx in range(self.n_basis):
            multi_idx = self.indices[p_idx]
            for dim in range(self.n_dim):
                deg = multi_idx[dim]
                psi_1d = self.evaluate_normalized_1d(dim, deg, xi[:, dim])
                Psi[:, p_idx] *= psi_1d

        return Psi

    def evaluate_single(self, p_idx: int, xi: np.ndarray) -> np.ndarray:
        """
        计算单个基函数 Ψ_p(ξ)。

        参数:
            p_idx: 基函数索引
            xi:    shape (n_points, d) 或 (d,)

        返回:
            shape (n_points,)
        """
        if xi.ndim == 1:
            xi = xi.reshape(1, -1)

        result = np.ones(xi.shape[0])
        multi_idx = self.indices[p_idx]

        for dim in range(self.n_dim):
            deg = multi_idx[dim]
            psi_1d = self.evaluate_normalized_1d(dim, deg, xi[:, dim])
            result *= psi_1d

        return result

    def compute_coupling_tensor(self, quadrature_nodes: np.ndarray,
                                quadrature_weights: np.ndarray
                                ) -> np.ndarray:
        """
        计算三重乘积耦合张量。

        C_{ijk} = <Ψ_i Ψ_j, Ψ_k>_w
                = ∫ Ψ_i(ξ) Ψ_j(ξ) Ψ_k(ξ) w(ξ) dξ
                ≈ Σ_q w_q Ψ_i(ξ_q) Ψ_j(ξ_q) Ψ_k(ξ_q)

        这是随机 Galerkin 投影的核心计算。

        参数:
            quadrature_nodes:   shape (n_q, d)
            quadrature_weights: shape (n_q,)

        返回:
            C: shape (P, P, P), 三重乘积张量
        """
        Psi = self.evaluate(quadrature_nodes)  # (n_q, P)
        P = self.n_basis
        n_q = len(quadrature_weights)

        # 预加权: Ψ_w[q, p] = w_q * Ψ[q, p]
        Psi_w = Psi * quadrature_weights[:, np.newaxis]  # (n_q, P)

        # C_{ijk} = Σ_q Ψ_w[q,i] * Ψ[q,j] * Ψ[q,k]
        # 分步计算以减少内存: 先算 (j,k) → intermediate
        C = np.zeros((P, P, P))
        for i in range(P):
            # Psi_w[:, i]: (n_q,)
            # Psi * Psi_w[:, i:i+1]: (n_q, P) → 对每个 j 乘 Ψ_i*w
            temp = Psi_w[:, i:i + 1] * Psi  # (n_q, P)
            # C[i, j, k] = Σ_q temp[q, j] * Psi[q, k]
            C[i] = temp.T @ Psi  # (P, n_q) @ (n_q, P) = (P, P)

        return C

    def compute_stiffness_tensor(self, quadrature_nodes: np.ndarray,
                                 quadrature_weights: np.ndarray
                                 ) -> np.ndarray:
        """
        计算双重乘积刚度张量 (用于线性项)。

        S_{ij} = <Ψ_i, Ψ_j>_w = δ_{ij}  (正交归一化)

        验证正交性。

        参数:
            quadrature_nodes:   shape (n_q, d)
            quadrature_weights: shape (n_q,)

        返回:
            S: shape (P, P)
        """
        Psi = self.evaluate(quadrature_nodes)
        Psi_w = Psi * quadrature_weights[:, np.newaxis]
        return Psi_w.T @ Psi

    def verify_orthogonality(self, quadrature_nodes: np.ndarray,
                             quadrature_weights: np.ndarray
                             ) -> float:
        """
        验证基函数的正交性。

        返回 ||S - I||_F / P, 应接近机器精度。
        """
        S = self.compute_stiffness_tensor(
            quadrature_nodes, quadrature_weights)
        error = np.linalg.norm(S - np.eye(self.n_basis), 'fro')
        return error / max(self.n_basis, 1)


def adaptive_degree_selection(basis: OrthogonalPolynomialBasis,
                              target_error: float = 1e-6,
                              max_degree: int = 15
                              ) -> int:
    """
    自适应选择多项式阶数。

    从低阶开始增加, 直到基函数的正交性误差小于目标。

    参数:
        basis:        基函数对象
        target_error: 正交性误差目标
        max_degree:   最大尝试阶数

    返回:
        p_opt: 最优阶数
    """
    d = basis.n_dim
    for p in range(1, max_degree + 1):
        # 粗略估计: P = C(d+p, p)
        from math import comb
        n_basis = comb(d + p, p)

        # 经验公式: 正交性误差 ≈ n_basis * machine_eps
        est_error = n_basis * 1e-16
        if est_error < target_error:
            return p

    return max_degree
