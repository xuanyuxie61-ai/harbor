"""
variational_inequality.py
========================
定义变分不等式 (Variational Inequality, VI) 问题的核心数学结构。

数学背景
--------
给定闭凸集 K ⊂ R^n 和映射 F: K → R^n, 变分不等式问题 VI(K, F) 是:
    求 x* ∈ K, 使得 ⟨F(x*), y - x*⟩ ≥ 0,  ∀ y ∈ K.

等价互补问题 (NCP):
    求 x* ≥ 0, 使得 x* ≥ 0,  F(x*) ≥ 0,  ⟨x*, F(x*)⟩ = 0.

本模块实现:
    - VI 问题的参数化定义
    - 投影残差 ||x - Π_K(x - F(x))|| 的计算
    - 自然残差 min(x, F(x)) 的逐点计算
    - 误差界 (error bound) 与条件数估计

关键公式
--------
自然残差:   r_nat(x) = x - Π_K(x - α F(x))       (α > 0)
正规残差:   r_norm(x) = ||min(x, F(x))||_2
Gap 函数:   gap(x) = sup_{y∈K} ⟨F(x), x - y⟩

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple


class VariationalInequalityProblem:
    """
    参数化变分不等式 VI(K, F, α, β) 的数学定义。

    K = {x ∈ R^n : Ax ≤ b, x ≥ 0}  (多面体约束)
    F(x) = Mx + q + Φ(x)           (仿射 + 非线性算子)

    其中:
        M ∈ R^{n×n} 为 P0-矩阵 (保证解的存在性)
        q ∈ R^n 为常向量
        Φ: R^n → R^n 为 Lipschitz 连续非线性映射
    """

    def __init__(
        self,
        n: int,
        M: np.ndarray,
        q: np.ndarray,
        nonlinear: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        lipschitz_L: float = 1.0,
        strong_monotone_mu: float = 0.0,
    ):
        """
        Parameters
        ----------
        n : int
            问题维度
        M : ndarray (n, n)
            仿射部分的系数矩阵
        q : ndarray (n,)
            仿射部分的常数项
        nonlinear : callable, optional
            非线性映射 Φ(x)
        A_ub, b_ub : ndarray, optional
            线性不等式约束 Ax ≤ b
        lipschitz_L : float
            F 的 Lipschitz 常数上界
        strong_monotone_mu : float
            强单调性常数 (≥ 0 时为强单调 VI)
        """
        if M.shape != (n, n):
            raise ValueError(f"M 必须是 {n}x{n} 矩阵, 实际 {M.shape}")
        if q.shape != (n,):
            raise ValueError(f"q 必须是 ({n},) 向量")

        self.n = n
        self.M = M.copy()
        self.q = q.copy()
        self.nonlinear = nonlinear
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.lipschitz_L = lipschitz_L
        self.strong_monotone_mu = strong_monotone_mu

        # 预计算 M 的对称/反对称分解: M = (M+M^T)/2 + (M-M^T)/2
        self.M_sym = 0.5 * (M + M.T)
        self.M_skew = 0.5 * (M - M.T)

        # 检查 P-矩阵性质 (P0: 所有主子式 ≥ 0)
        self._check_p0_property()

    def _check_p0_property(self) -> None:
        """检查 M 是否为 P0-矩阵 (充分条件: 对称部分正半定)."""
        eigvals = np.linalg.eigvalsh(self.M_sym)
        self.min_eigenvalue_sym = float(np.min(eigvals))
        self.is_p0 = self.min_eigenvalue_sym >= -1e-10
        self.is_strongly_monotone = self.min_eigenvalue_sym > 1e-10

    def evaluate_F(self, x: np.ndarray) -> np.ndarray:
        """
        计算映射 F(x) = Mx + q + Φ(x).

        数学公式:
            F_i(x) = Σ_j M_{ij} x_j + q_i + Φ_i(x)

        Returns
        -------
        F_x : ndarray (n,)
        """
        F_x = self.M @ x + self.q
        if self.nonlinear is not None:
            F_x = F_x + self.nonlinear(x)
        return F_x

    def jacobian_F(self, x: np.ndarray) -> np.ndarray:
        """
        计算 ∇F(x) = M + ∇Φ(x).

        当 Φ 为恒等零映射时, ∇F = M (常数).
        """
        if self.nonlinear is None:
            return self.M.copy()
        # 数值差分近似 (前向差分, 步长 h = sqrt(eps) * max(1, |x|))
        h = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
        J = np.zeros((self.n, self.n))
        F0 = self.evaluate_F(x)
        for j in range(self.n):
            x_plus = x.copy()
            x_plus[j] += h[j]
            J[:, j] = (self.evaluate_F(x_plus) - F0) / h[j]
        return J

    def natural_residual(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        自然残差 r(x) = x - Π_K(x - α F(x)).

        对非负约束 K = R^n_+, 投影简化为:
            Π_{R+}(z) = max(0, z)

        故 r_i(x) = x_i - max(0, x_i - α F_i(x))
                   = min(x_i, x_i - (x_i - α F_i(x)))  [当 x_i - αF_i ≥ 0]
                   = min(x_i, α F_i(x))

        归一化: ||r(x)|| / (1 + ||x||) 作为停止准则.
        """
        z = x - alpha * self.evaluate_F(x)
        proj_z = np.maximum(0.0, z)
        return x - proj_z

    def min_residual(self, x: np.ndarray) -> np.ndarray:
        """
        Fischer-Burmeister / min 函数的逐点残差:
            r_i(x) = min(x_i, F_i(x))

        互补条件等价于: x ≥ 0, F(x) ≥ 0, min(x, F(x)) = 0.
        """
        F_x = self.evaluate_F(x)
        return np.minimum(x, F_x)

    def gap_function(self, x: np.ndarray) -> float:
        """
        对偶 gap 函数 (D-gap function):
            g_α(x) = ⟨F(x), x - y_α(x)⟩ - (α/2)||x - y_α(x)||^2
        其中 y_α(x) = Π_K(x - F(x)/α).

        当 α → ∞, g_α(x) → 原始 gap 函数 sup_{y∈K} ⟨F(x), x-y⟩.
        """
        alpha = 1.0
        y_alpha = np.maximum(0.0, x - self.evaluate_F(x) / alpha)
        diff = x - y_alpha
        gap = float(np.dot(self.evaluate_F(x), diff) - 0.5 * alpha * np.dot(diff, diff))
        return gap

    def condition_number_estimate(self) -> float:
        """
        条件数估计 κ = L / μ, 其中:
            L = Lipschitz 常数
            μ = 强单调性常数
        若 μ = 0, 返回 np.inf (退化 VI).
        """
        if self.strong_monotone_mu > 1e-14:
            return self.lipschitz_L / self.strong_monotone_mu
        return np.inf

    def __repr__(self) -> str:
        return (
            f"VariationalInequalityProblem(n={self.n}, "
            f"min_eig_sym={self.min_eigenvalue_sym:.4e}, "
            f"κ={self.condition_number_estimate():.2e})"
        )
