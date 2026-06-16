"""
jacobian_estimator.py
=====================
映射 F 的 Jacobian 矩阵的数值估计与自适应差分.

数学背景
--------
对 VI 的 Newton 型方法, 需要计算 ∇F(x). 当 F 由黑箱给出时,
使用有限差分近似:

前向差分:
    (∇F)_ij ≈ (F_i(x + h·e_j) - F_i(x)) / h
成本: n 次函数求值, 误差 O(h)

中心差分:
    (∇F)_ij ≈ (F_i(x + h·e_j) - F_i(x - h·e_j)) / (2h)
成本: 2n 次函数求值, 误差 O(h²)

复步差分 (Lynn 1998):
    (∇F)_ij ≈ Im(F_i(x + ih·e_j)) / h
成本: n 次复函数求值, 误差 O(h²) 且无消去误差!

最优步长选择:
    前向差分: h_opt = √(ε_mach) · max(1, |x_j|)
    中心差分: h_opt = ε_mach^{1/3} · max(1, |x_j|)
    复步差分: h_opt = 10^{-100} (几乎任意小)

在 VI 中的应用:
    1. 半光滑 Newton 的广义 Jacobian
    2. 伪 Newton 法的 Broyden 更新
    3. 条件数估计与预条件

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional


class JacobianEstimator:
    """
    Jacobian 矩阵的数值估计器, 支持多种差分格式.
    """

    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        n: int,
        method: str = 'central',
    ):
        """
        Parameters
        ----------
        F : callable
            映射 F: R^n → R^n
        n : int
            维度
        method : str
            差分方法: 'forward', 'central', 'complex'
        """
        if method not in ('forward', 'central', 'complex'):
            raise ValueError(f"未知差分方法: {method}")
        self.F = F
        self.n = n
        self.method = method
        self.n_evals = 0  # 函数求值计数器

    def estimate(self, x: np.ndarray) -> np.ndarray:
        """
        估计 Jacobian 矩阵 ∇F(x).

        Returns
        -------
        J : ndarray (n, n)
        """
        self.n_evals = 0
        if self.method == 'forward':
            return self._forward_diff(x)
        elif self.method == 'central':
            return self._central_diff(x)
        else:
            return self._complex_step(x)

    def _forward_diff(self, x: np.ndarray) -> np.ndarray:
        """
        前向差分:
            J_ij = (F_i(x + h·e_j) - F_i(x)) / h_j
        h_j = √ε · max(1, |x_j|)
        """
        eps = np.finfo(float).eps
        F0 = self.F(x)
        self.n_evals += 1
        J = np.zeros((self.n, self.n))
        for j in range(self.n):
            h = np.sqrt(eps) * max(1.0, abs(x[j]))
            x_plus = x.copy()
            x_plus[j] += h
            F_plus = self.F(x_plus)
            self.n_evals += 1
            J[:, j] = (F_plus - F0) / h
        return J

    def _central_diff(self, x: np.ndarray) -> np.ndarray:
        """
        中心差分:
            J_ij = (F_i(x + h·e_j) - F_i(x - h·e_j)) / (2h_j)
        h_j = ε^{1/3} · max(1, |x_j|)
        """
        eps = np.finfo(float).eps
        J = np.zeros((self.n, self.n))
        for j in range(self.n):
            h = eps**(1.0/3.0) * max(1.0, abs(x[j]))
            x_plus = x.copy()
            x_plus[j] += h
            x_minus = x.copy()
            x_minus[j] -= h
            F_plus = self.F(x_plus)
            F_minus = self.F(x_minus)
            self.n_evals += 2
            J[:, j] = (F_plus - F_minus) / (2.0 * h)
        return J

    def _complex_step(self, x: np.ndarray) -> np.ndarray:
        """
        复步差分:
            J_ij = Im(F_i(x + ih·e_j)) / h
        h = 10^{-100} (无消去误差)

        注意: 要求 F 支持复数输入.
        """
        h = 1e-100
        J = np.zeros((self.n, self.n))
        for j in range(self.n):
            x_complex = x.astype(complex).copy()
            x_complex[j] += 1j * h
            try:
                F_complex = self.F(x_complex)
                self.n_evals += 1
                J[:, j] = np.imag(F_complex) / h
            except (TypeError, ValueError):
                # 回退到中心差分
                eps = np.finfo(float).eps
                hh = eps**(1.0/3.0) * max(1.0, abs(x[j]))
                x_plus = x.copy()
                x_plus[j] += hh
                x_minus = x.copy()
                x_minus[j] -= hh
                F_plus = self.F(x_plus)
                F_minus = self.F(x_minus)
                self.n_evals += 2
                J[:, j] = (F_plus - F_minus) / (2.0 * hh)
        return J


class BroydenUpdate:
    """
    Broyden 拟 Newton 更新 (避免每次重新计算 Jacobian).

    初始化: J_0 = ∇F(x_0) (数值差分)
    更新:
        J_{k+1} = J_k + (ΔF_k - J_k Δx_k) · Δx_k^T / ||Δx_k||²

    其中 Δx_k = x_{k+1} - x_k, ΔF_k = F(x_{k+1}) - F(x_k).

    Sherman-Morrison 逆更新:
        J_{k+1}^{-1} = J_k^{-1} + (Δx_k - J_k^{-1} ΔF_k) · Δx_k^T · J_k^{-1} / (Δx_k^T · J_k^{-1} · ΔF_k)

    收敛性: 局部 q-超线性收敛 (在无奇点假设下).
    """

    def __init__(self, n: int):
        self.n = n
        self.J_inv: Optional[np.ndarray] = None
        self.x_prev: Optional[np.ndarray] = None
        self.F_prev: Optional[np.ndarray] = None

    def initialize(self, x: np.ndarray, F_x: np.ndarray, J_inv: np.ndarray) -> None:
        """用初始 Jacobian 逆初始化."""
        self.x_prev = x.copy()
        self.F_prev = F_x.copy()
        self.J_inv = J_inv.copy()

    def update(self, x_new: np.ndarray, F_new: np.ndarray) -> np.ndarray:
        """
        执行一次 Broyden 更新.

        Returns
        -------
        J_inv_new : ndarray (n, n)
            更新后的 Jacobian 逆
        """
        if self.J_inv is None or self.x_prev is None:
            raise RuntimeError("请先调用 initialize()")

        dx = x_new - self.x_prev
        dF = F_new - self.F_prev

        # Sherman-Morrison 更新
        v = self.J_inv @ dF
        denom = np.dot(dx, v)

        if abs(denom) < 1e-30:
            # 更新失败, 保持不变
            return self.J_inv

        numerator = np.outer(dx - v, dx @ self.J_inv)
        self.J_inv = self.J_inv + numerator / denom

        self.x_prev = x_new.copy()
        self.F_prev = F_new.copy()
        return self.J_inv

    def solve_direction(self, F_x: np.ndarray) -> np.ndarray:
        """计算 Newton 方向 d = -J^{-1} F(x)."""
        if self.J_inv is None:
            raise RuntimeError("请先初始化")
        return -self.J_inv @ F_x


class ConditionNumberEstimator:
    """
    Jacobian 条件数的快速估计 (无需完整 SVD).

    方法:
        1. 幂迭代估计 ||J||_∞ 和 ||J^{-1}||_∞
        2. 通过 LU 分解的 U 矩阵对角元素估计 ||J^{-1}||
    """

    @staticmethod
    def estimate_norm_inf_power(J: np.ndarray, n_iter: int = 20) -> float:
        """幂迭代估计 ||J||_∞."""
        n = J.shape[0]
        x = np.ones(n) / np.sqrt(n)
        for _ in range(n_iter):
            y = J @ x
            x = J.T @ y
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-30:
                return 0.0
            x = x / norm_x
        return float(np.linalg.norm(J @ x, ord=np.inf))

    @staticmethod
    def estimate_condition(J: np.ndarray) -> float:
        """
        基于 SVD 的精确条件数.
        对小矩阵 (n ≤ 200) 使用; 大矩阵应使用迭代法.
        """
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            if svals[-1] < 1e-30:
                return np.inf
            return float(svals[0] / svals[-1])
        except np.linalg.LinAlgError:
            return np.inf
