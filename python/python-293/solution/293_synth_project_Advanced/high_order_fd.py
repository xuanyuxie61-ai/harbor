# -*- coding: utf-8 -*-
"""
high_order_fd.py
=================
高阶有限差分算子与数值色散分析

融合种子项目:
  063_backward_euler — 隐式时间积分器 (Backward Euler)

物理背景
--------
空间等离子体中 Langmuir 波的数值模拟要求高阶差分格式以正确捕捉
波-粒子共振处的精细结构。低阶格式 (2阶) 存在严重的数值色散，
导致虚假的波反射和错误的阻尼率。

P 阶中心差分格式的截断误差:
  f'(x) = [Σ_{k=-P}^{P} c_k f(x + kh)] / h + O(h^{2P})

数值色散关系 (von Neumann 分析):
  k̃(k) = -i/h Σ c_k exp(ikh)
有效波数 k̃ 与真实波数 k 的偏差决定数值色散误差。

Fornberg 算法
-------------
给定模板点 {x_0, x_1, ..., x_N} 和导数阶数 M, 递推计算权系数:

δ_{n,j}^{(m)} = 递推公式 (Fornberg 1988, Math. Comp. 51, 699-706)
"""

import numpy as np
from typing import Tuple, Optional
from banded_matrix_ops import BandedMatrix


# ===== Fornberg 算法 =======================================================

def fd_weights(x: np.ndarray, x0: float, deriv_order: int) -> np.ndarray:
    """
    Fornberg (1988) 算法计算任意非均匀模板的有限差分权系数

    参数
    ----------
    x : array_like, shape (N+1,)
        模板点坐标 (可以非均匀).
    x0 : float
        求导点坐标.
    deriv_order : int
        导数阶数 (0=插值, 1=一阶导, 2=二阶导, ...).

    返回
    -------
    np.ndarray, shape (N+1, deriv_order+1)
        weights[i, m] = 第 i 个模板点对 f^(m)(x0) 的权系数.

    算法 (Fornberg 1988, Math. Comp. 51, 699-706)
    ----
    递推初始化:
      delta[0, 0] = 1,  其余 = 0
    对 n = 1, 2, ..., N:
      对 i = 0, ..., n-1: 更新旧点系数
      计算新点 n 的系数
    """
    x = np.asarray(x, dtype=np.float64)
    n_pts = len(x)
    if deriv_order < 0:
        raise ValueError("deriv_order must be >= 0")
    if n_pts == 0:
        raise ValueError("fd_weights: need at least one stencil point")

    M = deriv_order
    d = np.zeros((n_pts, M + 1), dtype=np.float64)
    d1 = np.zeros((n_pts, M + 1), dtype=np.float64)
    d[0, 0] = 1.0
    d1[0, 0] = 1.0

    c1 = 1.0
    for n in range(1, n_pts):
        c2 = 1.0
        for i in range(n):
            c3 = x[n] - x[i]
            if abs(c3) < 1.0e-300:
                raise ValueError(
                    f"fd_weights: duplicate stencil points")
            c4 = x[i] - x0
            d_old = d[i, :].copy()
            d1_old = d1[i, :].copy()

            d[i, 0] = c4 * d_old[0] / c3
            d1[i, 0] = c4 * d1_old[0] / c3
            for m in range(1, min(n, M) + 1):
                d[i, m] = (c4 * d_old[m] - m * d_old[m - 1]) / c3
                d1[i, m] = (c4 * d1_old[m] - m * d1_old[m - 1]
                            + m * d_old[m - 1]) / c3

            d[n, 0] = -c1 * d_old[0] / c3
            d1[n, 0] = c1 * (d_old[0] - (x[n] - x0) * d[n, 0]) / c3
            for m in range(1, min(n, M) + 1):
                d[n, m] = -c1 * d_old[m] / c3
                d1[n, m] = c1 * (m * d_old[m - 1]
                                  - (x[n] - x0) * d[n, m]) / c3

            c1 = c2
            c2 = c3

    return d1


# ===== 有限差分系数矩阵 ====================================================

class FDCoefficientMatrix:
    """
    高阶有限差分系数矩阵

    存储非均匀网格上 P 阶差分格式的权系数，支持任意模板宽度。

    对于 Vlasov 方程的速度空间绝热项:
      ∂(E f)/∂v ≈ Σ_k w_k f(v + kΔv) / Δv
    """

    def __init__(self, grid: np.ndarray, stencil_half_width: int = 2):
        """
        Parameters
        ----------
        grid : np.ndarray
            一维网格点坐标 (均匀或非均匀).
        stencil_half_width : int
            模板半宽度 (2 → 4阶, 3 → 6阶).
        """
        self.grid = np.asarray(grid, dtype=np.float64)
        self.n = len(self.grid)
        self.half_width = stencil_half_width
        self.stencil_width = 2 * stencil_half_width + 1

        # 计算一阶导数权系数矩阵
        self.w1 = self._build_weights(1)
        # 二阶导数权系数矩阵
        self.w2 = self._build_weights(2)

    def _build_weights(self, deriv: int) -> np.ndarray:
        """为每个网格点计算 derivate阶导数的差分权系数."""
        n = self.n
        hw = self.half_width
        sw = self.stencil_width
        W = np.zeros((n, sw), dtype=np.float64)

        for i in range(n):
            # 确定模板范围 (边界处使用非对称模板)
            j_start = max(0, i - hw)
            j_end = min(n, i + hw + 1)
            # 确保模板宽度一致
            if j_end - j_start < sw:
                if j_start == 0:
                    j_end = min(n, sw)
                elif j_end == n:
                    j_start = max(0, n - sw)

            stencil = self.grid[j_start:j_end]
            x0 = self.grid[i]
            w = fd_weights(stencil, x0, deriv)
            actual_width = j_end - j_start
            offset = i - j_start
            W[i, hw - offset:hw - offset + actual_width] = w[:, deriv]

        return W

    def derivative(self, f: np.ndarray, order: int = 1) -> np.ndarray:
        """
        计算 f 的 order 阶导数

        Parameters
        ----------
        f : np.ndarray, shape (n,)
            网格函数值.
        order : int
            导数阶数 (1 或 2).

        Returns
        -------
        np.ndarray
            导数近似值.
        """
        if order == 1:
            W = self.w1
        elif order == 2:
            W = self.w2
        else:
            raise ValueError(f"FDCoefficientMatrix.derivative: order must be 1 or 2")

        f = np.asarray(f, dtype=np.float64)
        n = len(f)
        if n != self.n:
            raise ValueError(f"derivative: f length {n} != grid size {self.n}")

        result = np.zeros(n, dtype=np.float64)
        hw = self.half_width
        for i in range(n):
            j_start = max(0, i - hw)
            j_end = min(n, i + hw + 1)
            if j_end - j_start < self.stencil_width:
                if j_start == 0:
                    j_end = min(n, self.stencil_width)
                elif j_end == n:
                    j_start = max(0, n - self.stencil_width)
            actual_width = j_end - j_start
            offset = i - j_start
            for k in range(actual_width):
                result[i] += W[i, hw - offset + k] * f[j_start + k]

        return result

    def differentiation_matrix(self, order: int = 1) -> np.ndarray:
        """
        构造稠密微分矩阵 D, 使得 D @ f ≈ f^(order)

        Returns
        -------
        np.ndarray, shape (n, n)
            微分矩阵.
        """
        W = self.w1 if order == 1 else self.w2
        n = self.n
        hw = self.half_width
        D = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            j_start = max(0, i - hw)
            j_end = min(n, i + hw + 1)
            if j_end - j_start < self.stencil_width:
                if j_start == 0:
                    j_end = min(n, self.stencil_width)
                elif j_end == n:
                    j_start = max(0, n - self.stencil_width)
            actual_width = j_end - j_start
            offset = i - j_start
            for k in range(actual_width):
                D[i, j_start + k] = W[i, hw - offset + k]

        return D


# ===== 4阶耗散算子 =========================================================

class FourthOrderDissipationOperator:
    """
    4阶人工耗散算子

    .. math::
        D_4 f_i = -\\epsilon_4 (f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2})

    用于压制 Vlasov 模拟中的 grid-scale (2Δx) 数值振荡，
    同时不影响物理尺度的波结构。

    ε₄ 的选取需在数值耗散与物理精度之间平衡:
      ε₄ ~ O(Δx⁴ / Δt)  可压制最高频振荡
      ε₄ << v_th Δt       不影响 Landau 阻尼物理
    """

    def __init__(self, n: int, epsilon: float = 0.01):
        self.n = n
        self.eps = epsilon

    def apply(self, f: np.ndarray) -> np.ndarray:
        """施加 4 阶耗散: f_new = f + eps * D4(f)."""
        f = np.asarray(f, dtype=np.float64)
        n = len(f)
        if n != self.n:
            raise ValueError(f"Dissipation: f length {n} != {self.n}")

        result = f.copy()
        for i in range(2, n - 2):
            d4 = f[i - 2] - 4.0 * f[i - 1] + 6.0 * f[i] - 4.0 * f[i + 1] + f[i + 2]
            result[i] -= self.eps * d4

        # 边界: 3阶非对称模板
        if n >= 5:
            d4_2 = f[0] - 4.0 * f[1] + 6.0 * f[2] - 4.0 * f[3] + f[4]
            result[2] -= self.eps * d4_2 * 0.5
            d4_nm2 = f[n - 5] - 4.0 * f[n - 4] + 6.0 * f[n - 3] - 4.0 * f[n - 2] + f[n - 1]
            result[n - 3] -= self.eps * d4_nm2 * 0.5

        return result


# ===== 谱分辨率分析 ========================================================

def spectral_resolution_analysis(
    stencil_order: int = 4, n_sample: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分析 P 阶差分格式的谱分辨率

    数值有效波数 k̃ 与真实波数 k 的关系:
      k̃(θ) = -i/h Σ_{j} c_j exp(ijθ),   θ = kh

    分辨率度量:  |k̃/k - 1| < ε 的最大 θ 范围

    Returns
    -------
    theta : np.ndarray
        归一化波数 θ = kh ∈ [0, π].
    error : np.ndarray
        色散误差 |k̃/k - 1|.
    """
    # 标准 P 阶中心差分模板
    hw = stencil_order // 2
    stencil_pts = np.arange(-hw, hw + 1, dtype=np.float64)

    # 计算权系数
    w = fd_weights(stencil_pts, 0.0, 1)[:, 1]

    # 分析波数范围
    theta = np.linspace(0, np.pi, n_sample)
    k_tilde = np.zeros_like(theta, dtype=np.complex128)

    for j_idx, j in enumerate(range(-hw, hw + 1)):
        k_tilde += w[j_idx] * np.exp(1j * j * theta)

    # k̃ / (i·k) 应接近 1 (对于小 θ)
    # 注意 k̃ ≈ ik̃_real, 所以 k̃ / (i·θ) = -i·k̃ / θ
    ratio = np.zeros_like(theta)
    for idx in range(n_sample):
        if abs(theta[idx]) > 1e-10:
            ratio[idx] = abs(k_tilde[idx] / (1j * theta[idx]))

    error = np.abs(ratio - 1.0)

    return theta, error


# ===== Backward Euler 隐式求解器 ===========================================

class BackwardEulerVlasov:
    """
    Backward Euler 隐式时间积分器

    用于 Vlasov 方程的隐式推进:
      f^{n+1} = f^n + dt · L[f^{n+1}]

    其中 L 为离散化的 Vlasov 算子。通过 Newton 迭代求解隐式方程:
      R(f^{n+1}) = f^{n+1} - f^n - dt · L[f^{n+1}] = 0

    优势: 无条件稳定 (对线性问题), 允许大时间步长。
    代价: 每步需求解大型线性/非线性系统。
    """

    def __init__(self, n: int, dt: float, tol: float = 1.0e-8,
                 max_iter: int = 20):
        self.n = n
        self.dt = dt
        self.tol = tol
        self.max_iter = max_iter

    def step(self, f: np.ndarray, rhs_func, jacobian_func=None) -> np.ndarray:
        """
        执行一步 Backward Euler

        Parameters
        ----------
        f : np.ndarray
            当前时刻解.
        rhs_func : callable
            右端函数 L(f).
        jacobian_func : callable, optional
            Jacobian 矩阵 J(f) = dL/df. 若为 None 则使用简化的 Picard 迭代.

        Returns
        -------
        np.ndarray
            新时刻解 f^{n+1}.
        """
        f = np.asarray(f, dtype=np.float64).copy()
        f_old = f.copy()

        # Newton/Picard 迭代
        for iteration in range(self.max_iter):
            Lf = rhs_func(f)
            # 残差: R = f_new - f_old - dt * L(f_new)
            residual = f - f_old - self.dt * Lf

            res_norm = np.max(np.abs(residual))
            if res_norm < self.tol:
                break

            if jacobian_func is not None:
                # Full Newton:  (I - dt·J) δf = -R
                J = jacobian_func(f)
                A = np.eye(self.n) - self.dt * J
                try:
                    delta_f = np.linalg.solve(A, -residual)
                except np.linalg.LinAlgError:
                    delta_f = -residual * 0.5
            else:
                # Simplified Picard:  f_new = f_old + dt * L(f_old)
                delta_f = -residual

            f = f + delta_f

            # 防止发散
            if np.max(np.abs(delta_f)) > 10.0 * np.max(np.abs(f_old)) + 1.0:
                f = f_old + 0.1 * (f - f_old)

        return f
