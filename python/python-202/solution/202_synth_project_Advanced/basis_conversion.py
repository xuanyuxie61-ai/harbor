"""
多项式混沌基转换模块 (Polynomial Chaos Basis Conversion)
==========================================================
实现不同多项式混沌 (PC) 基之间的转换。

多项式混沌展开:
  u(x,ω) ≈ Σ_{|α|≤p} c_α(x) Ψ_α(ξ)

其中:
  - Ψ_α(ξ) = Π_{d=1}^D P_{α_d}^{(d)}(ξ_d) 是多元正交多项式
  - P_n^{(d)} 是第 d 维度的正交多项式
  - α = (α_1,...,α_D) 是多指标, |α| = Σ α_d
  - c_α 是 PC 系数

基转换:
  当输入随机变量的分布变化时, 需要将 PC 展开从一个基转换到另一个基。
  例如: 从 Hermite (高斯输入) 转换到 Legendre (均匀输入)

  转换公式 (单变量):
    P_n^{(H)}(ξ) = Σ_{k=0}^{⌊n/2⌋} a_{n,k} P_{n-2k}^{(L)}(ξ)

  其中 Hermite-Legendre 转换系数:
    a_{n,k} = (-1)^k n! / (k! (n-2k)! 2^k) × (2(n-2k)+1) / 2^{n-2k}

多变量基转换通过张量积分解:
  Ψ_α^{(H)}(ξ) = Π_d He_{α_d}(ξ_d) → Σ_β T_{αβ} Π_d P_{β_d}(ξ_d)

  转换矩阵 T 是各维度转换矩阵的 Kronecker 积:
    T = ⊗_{d=1}^D T^{(d)}

参考文献:
  Xiu, D. & Karniadakis, G.E. (2002). The Wiener-Askey scheme for PC.
  Wan, X. & Karniadakis, G.E. (2006). Beyond Wiener-Askey: Stochastic PC.
"""

import numpy as np
from scipy import special
from typing import List, Tuple, Optional, Dict
import math


class PolynomialChaosBasis:
    """
    多项式混沌基: 管理和操作 PC 展开。

    支持:
    - 多指标集生成
    - 多元正交多项式求值
    - PC 系数投影 (通过求积)
    - 基转换 (Hermite ↔ Legendre ↔ Laguerre)
    """

    def __init__(
        self,
        dimension: int,
        max_order: int,
        basis_type: str = 'hermite'
    ):
        """
        参数:
            dimension: 随机空间维度 D
            max_order: 最大总阶数 p
            basis_type: 基类型 'hermite' | 'legendre' | 'laguerre'
        """
        self.D = dimension
        self.p = max_order
        self.basis_type = basis_type

        # 生成多指标集 {α: |α|₁ ≤ p}
        self.multi_indices = self._generate_multi_indices()
        self.n_basis = len(self.multi_indices)

    def _generate_multi_indices(self) -> List[Tuple[int, ...]]:
        """
        生成全阶多指标集:
          I_p = {α ∈ ℕ₀^D : |α|₁ = Σ α_d ≤ p}

        基数: C(p+D, D) = (p+D)! / (p! D!)
        """
        D = self.D
        p = self.p
        indices = []
        self._enumerate(D, p, [], indices)
        return indices

    def _enumerate(
        self, remaining: int, remaining_sum: int,
        current: List[int], result: List[Tuple[int, ...]]
    ):
        """递归枚举多指标集。"""
        if remaining == 0:
            result.append(tuple(current))
            return
        for v in range(remaining_sum + 1):
            current.append(v)
            self._enumerate(remaining - 1, remaining_sum - v, current, result)
            current.pop()

    def evaluate_basis(
        self, xi: np.ndarray
    ) -> np.ndarray:
        """
        在所有多指标处求值多元正交多项式:
          Ψ_α(ξ) = Π_{d=1}^D P_{α_d}(ξ_d)

        参数:
            xi: shape (N_points, D) 或 shape (D,)

        返回:
            Psi: shape (N_points, n_basis) 或 shape (n_basis,)
        """
        xi = np.atleast_2d(xi)
        N = xi.shape[0]

        # 单变量多项式求值
        from polynomial_utils import hermite_evaluate, legendre_evaluate, laguerre_evaluate

        eval_func = {
            'hermite': hermite_evaluate,
            'legendre': legendre_evaluate,
            'laguerre': laguerre_evaluate,
        }[self.basis_type]

        # 各维度, 各阶的求值
        poly_vals = {}  # (d, n) -> shape (N,)
        for d in range(self.D):
            vals = eval_func(xi[:, d], self.p)  # shape (p+1, N)
            for n in range(self.p + 1):
                poly_vals[(d, n)] = vals[n, :]

        # 多元多项式 = 各维度的乘积
        Psi = np.ones((N, self.n_basis))
        for j, alpha in enumerate(self.multi_indices):
            for d in range(self.D):
                Psi[:, j] *= poly_vals[(d, alpha[d])]

        return Psi.squeeze()

    def compute_pc_coefficients(
        self,
        solution_values: np.ndarray,
        quad_nodes: np.ndarray,
        quad_weights: np.ndarray
    ) -> np.ndarray:
        """
        通过数值投影计算 PC 系数:
          c_α = ∫ u(ξ) Ψ_α(ξ) dμ(ξ) / ∫ Ψ_α²(ξ) dμ(ξ)
              ≈ Σ_k w_k u(ξ_k) Ψ_α(ξ_k) / ||Ψ_α||²

        参数:
            solution_values: u(ξ_k), shape (N_quad,) 或 (N_quad, N_x)
            quad_nodes: 求积节点, shape (N_quad, D)
            quad_weights: 求积权重, shape (N_quad,)

        返回:
            coefficients: shape (n_basis,) 或 (n_basis, N_x)
        """
        # 求值基函数
        Psi = self.evaluate_basis(quad_nodes)  # (N_quad, n_basis)

        # 范数平方
        norms_sq = self._basis_norms_squared()  # (n_basis,)

        # 投影
        if solution_values.ndim == 1:
            # c_α = Σ_k w_k u(ξ_k) Ψ_α(ξ_k) / ||Ψ_α||²
            coeffs = np.zeros(self.n_basis)
            for j in range(self.n_basis):
                integrand = quad_weights * solution_values * Psi[:, j]
                coeffs[j] = np.sum(integrand) / norms_sq[j]
        else:
            N_x = solution_values.shape[1]
            coeffs = np.zeros((self.n_basis, N_x))
            for j in range(self.n_basis):
                integrand = quad_weights[:, np.newaxis] * solution_values * Psi[:, j:j + 1]
                coeffs[j] = np.sum(integrand, axis=0) / norms_sq[j]

        return coeffs

    def _basis_norms_squared(self) -> np.ndarray:
        """
        计算各基函数的范数平方:
          ||Ψ_α||² = Π_{d=1}^D ||P_{α_d}||²_d

        Hermite: ||He_n||² = n!
        Legendre: ||P_n||² = 2/(2n+1)
        Laguerre: ||L_n||² = 1
        """
        from polynomial_utils import hermite_norm_sq, legendre_norm_sq, laguerre_norm_sq

        norm_func = {
            'hermite': hermite_norm_sq,
            'legendre': legendre_norm_sq,
            'laguerre': laguerre_norm_sq,
        }[self.basis_type]

        norms = np.ones(self.n_basis)
        for j, alpha in enumerate(self.multi_indices):
            for d in range(self.D):
                norms[j] *= norm_func(alpha[d])
        return norms

    def evaluate_from_coefficients(
        self, coefficients: np.ndarray, xi: np.ndarray
    ) -> np.ndarray:
        """
        从 PC 系数重构解:
          u(ξ) ≈ Σ_α c_α Ψ_α(ξ)

        参数:
            coefficients: shape (n_basis,) 或 (n_basis, N_x)
            xi: shape (N_points, D)

        返回:
            u: shape (N_points,) 或 shape (N_points, N_x)
        """
        Psi = self.evaluate_basis(xi)  # (N_points, n_basis)
        if Psi.ndim == 1:
            Psi = Psi[:, np.newaxis]

        if coefficients.ndim == 1:
            return Psi @ coefficients
        else:
            return Psi @ coefficients


class BasisConverter:
    """
    多项式混沌基转换器: 在不同 PC 基之间转换系数。

    核心: 计算转换矩阵元素
      T_{mn} = <P_m^{(target)}, P_n^{(source)}> / ||P_m^{(target)}||²

    其中内积关于目标测度计算。

    对于 Hermite → Legendre (在 [-1,1] 上):
      T_{mn} = ∫_{-1}^{1} He_m(ξ) P_n(ξ) dξ / ||P_n||²

    这涉及不同族多项式之间的积分, 通常没有封闭形式,
    需要数值积分。
    """

    @staticmethod
    def hermite_to_legendre_1d(max_order: int, n_quad: int = 50) -> np.ndarray:
        """
        1D Hermite → Legendre 转换矩阵:
          T_{mn} = <He_m, P_n>_ρ / ||P_n||²

        其中 <f,g>_ρ = ∫ f(ξ) g(ξ) ρ_H(ξ) dξ
        ρ_H(ξ) = (2π)^{-1/2} exp(-ξ²/2)

        使用 Gauss-Hermite 求积:
          T_{mn} ≈ Σ_k w_k^H He_m(ξ_k^H) P_n(ξ_k^H) / ||P_n||²

        参数:
            max_order: 最大阶数
            n_quad: 求积点数

        返回:
            T: shape (max_order+1, max_order+1)
        """
        from polynomial_utils import gauss_hermite, hermite_evaluate, legendre_evaluate, legendre_norm_sq

        nodes, weights = gauss_hermite(n_quad)
        H = hermite_evaluate(nodes, max_order)  # (max_order+1, n_quad)
        P = legendre_evaluate(nodes, max_order)  # (max_order+1, n_quad)

        T = np.zeros((max_order + 1, max_order + 1))
        for m in range(max_order + 1):
            for n in range(max_order + 1):
                integrand = weights * H[m] * P[n]
                T[m, n] = np.sum(integrand) / legendre_norm_sq(n)

        return T

    @staticmethod
    def legendre_to_hermite_1d(max_order: int, n_quad: int = 50) -> np.ndarray:
        """
        1D Legendre → Hermite 转换矩阵:
          T_{mn} = <P_m, He_n>_ρ / ||He_n||²

        使用 Gauss-Legendre 求积 (在 [-1,1] 上):
          T_{mn} ≈ Σ_k w_k^L P_m(ξ_k^L) He_n(ξ_k^L) ρ_H(ξ_k^L) / ||He_n||²
        """
        from polynomial_utils import gauss_legendre, hermite_evaluate, legendre_evaluate
        from polynomial_utils import hermite_norm_sq, gaussian_pdf

        nodes, weights = gauss_legendre(n_quad)
        P = legendre_evaluate(nodes, max_order)
        H = hermite_evaluate(nodes, max_order)
        rho = gaussian_pdf(nodes)

        T = np.zeros((max_order + 1, max_order + 1))
        for m in range(max_order + 1):
            for n in range(max_order + 1):
                integrand = weights * P[m] * H[n] * rho
                T[m, n] = np.sum(integrand) / hermite_norm_sq(n)

        return T

    @staticmethod
    def convert_coefficients(
        coeffs_source: np.ndarray,
        conversion_matrix: np.ndarray,
        multi_indices_source: List[Tuple[int, ...]],
        multi_indices_target: List[Tuple[int, ...]],
        D: int
    ) -> np.ndarray:
        """
        多变量 PC 系数转换:
          c^target_β = Σ_α T_{α,β} c^source_α

        其中多变量转换矩阵 T 是各维度转换矩阵的 Kronecker 积:
          T_{αβ} = Π_{d=1}^D T^{(d)}_{α_d, β_d}

        参数:
            coeffs_source: 源系数, shape (n_source,)
            conversion_matrix: 1D 转换矩阵, shape (p+1, p+1)
            multi_indices_source: 源多指标集
            multi_indices_target: 目标多指标集
            D: 维度

        返回:
            coeffs_target: 目标系数
        """
        n_target = len(multi_indices_target)
        coeffs_target = np.zeros(n_target)

        for j, beta in enumerate(multi_indices_target):
            for i, alpha in enumerate(multi_indices_source):
                # T_{αβ} = Π_d T_{α_d, β_d}
                T_ab = 1.0
                for d in range(D):
                    T_ab *= conversion_matrix[alpha[d], beta[d]]

                if abs(T_ab) > 1e-15:
                    coeffs_target[j] += T_ab * coeffs_source[i]

        return coeffs_target
