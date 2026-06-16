"""
lagrange_reconstructor.py
=========================
基于 Lagrange 插值的 VI 解路径重构.

数学背景
--------
在参数化 VI 中, 解 x*(t) 是参数 t 的函数.
给定若干离散参数值 t_0, t_1, ..., t_m 处的解 x*(t_k),
可以用 Lagrange 插值重构连续解路径.

Lagrange 插值多项式:
    P(t) = Σ_{k=0}^m x*(t_k) · L_k(t)
其中
    L_k(t) = Π_{j≠k} (t - t_j) / (t_k - t_j)

误差分析:
    |x*(t) - P(t)| ≤ |ω(t)| / (m+1)! · max |x*^{(m+1)}(ξ)|
其中 ω(t) = Π_{k=0}^m (t - t_k) 为节点多项式.

Runge 现象: 等距节点的高次插值可能发散.
解决方案: 使用 Chebyshev 节点 t_k = cos(kπ/m).

在 VI 中的应用:
    1. 参数化 VI 的解路径连续化
    2. 预测-校正 continuation 方法的预测步
    3. 灵敏度分析: dx*/dt 的近似

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable


class LagrangeInterpolator1D:
    """
    一维 Lagrange 插值器.

    给定 n 个数据点 (t_k, f_k), 构建 Lagrange 插值多项式 P(t),
    并支持在任意点求值和求导.
    """

    def __init__(self, t_data: np.ndarray, f_data: np.ndarray):
        """
        Parameters
        ----------
        t_data : ndarray (n,)
            插值节点
        f_data : ndarray (n,) 或 (n, d)
            节点处的函数值
        """
        if len(t_data) != len(f_data):
            raise ValueError("节点和函数值数量不匹配")
        self.t_data = t_data.copy()
        self.f_data = f_data.copy()
        self.n = len(t_data)
        # 检查节点是否互异
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(t_data[i] - t_data[j]) < 1e-14:
                    raise ValueError(f"节点 {i} 和 {j} 重合: t={t_data[i]}")

    def basis(self, t: np.ndarray) -> np.ndarray:
        """
        计算 Lagrange 基函数 L_k(t), k = 0, ..., n-1.

        L_k(t) = Π_{j≠k} (t - t_j) / (t_k - t_j)

        Parameters
        ----------
        t : ndarray (m,)
            求值点

        Returns
        -------
        L : ndarray (m, n)
            L[i, k] = L_k(t[i])
        """
        m = len(t)
        L = np.ones((m, self.n))
        for k in range(self.n):
            for j in range(self.n):
                if j != k:
                    L[:, k] *= (t - self.t_data[j]) / (self.t_data[k] - self.t_data[j])
        return L

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        计算插值多项式 P(t) = Σ_k f_k · L_k(t).

        Returns
        -------
        P : ndarray (m,) 或 (m, d)
        """
        L = self.basis(t)
        if self.f_data.ndim == 1:
            return L @ self.f_data
        else:
            return L @ self.f_data

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """
        计算插值多项式的导数 P'(t).

        使用解析导数公式:
            L_k'(t) = L_k(t) · Σ_{j≠k} 1/(t - t_j)
        (当 t 不是节点时)

        Returns
        -------
        dP : ndarray (m,) 或 (m, d)
        """
        m = len(t)
        dP = np.zeros_like(self.f_data) if self.f_data.ndim == 1 else \
             np.zeros((m, self.f_data.shape[1]))
        for k in range(self.n):
            # L_k(t)
            Lk = np.ones(m)
            for j in range(self.n):
                if j != k:
                    Lk *= (t - self.t_data[j]) / (self.t_data[k] - self.t_data[j])

            # L_k'(t) = L_k(t) · Σ_{j≠k} 1/(t - t_j)
            dLk = np.zeros(m)
            for i in range(m):
                s = 0.0
                for j in range(self.n):
                    if j != k:
                        denom = t[i] - self.t_data[j]
                        if abs(denom) > 1e-14:
                            s += 1.0 / denom
                        else:
                            # t[i] ≈ t_j, 使用极限
                            s += 1.0 / (self.t_data[k] - self.t_data[j])
                dLk[i] = Lk[i] * s

            if self.f_data.ndim == 1:
                dP += self.f_data[k] * dLk
            else:
                dP += self.f_data[k][np.newaxis, :] * dLk[:, np.newaxis]
        return dP


class ChebyshevNodes:
    """
    Chebyshev 节点的生成器 (避免 Runge 现象).

    第一类 Chebyshev 节点 (在 [-1, 1] 上):
        t_k = cos((2k+1)π / (2n)),  k = 0, ..., n-1

    映射到 [a, b]:
        t_k = (a+b)/2 + (b-a)/2 · cos((2k+1)π / (2n))

    节点多项式的极小性:
        max_{t∈[a,b]} |ω(t)| = (b-a)^n / 2^{2n-1}
    这保证了插值的数值稳定性.
    """

    @staticmethod
    def generate(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
        """生成 n 个 Chebyshev 节点."""
        k = np.arange(n)
        t_cheb = np.cos((2 * k + 1) * np.pi / (2 * n))
        return 0.5 * (a + b) + 0.5 * (b - a) * t_cheb

    @staticmethod
    def generate_clenshaw_curtis(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
        """
        Clenshaw-Curtis 节点 (含端点):
            t_k = cos(kπ/(n-1)),  k = 0, ..., n-1
        """
        k = np.arange(n)
        t_cc = np.cos(k * np.pi / (n - 1))
        return 0.5 * (a + b) + 0.5 * (b - a) * t_cc


class SolutionPathReconstructor:
    """
    VI 解路径的重构器.

    给定参数化 VI: 求 x*(t) ∈ K 使得
        ⟨F(x*(t), t), y - x*(t)⟩ ≥ 0,  ∀ y ∈ K

    在参数 t_0, ..., t_m 处求解得到 x*(t_k),
    然后使用 Lagrange 插值重构 x*(t) 的连续近似.
    """

    def __init__(self, use_chebyshev: bool = True):
        self.use_chebyshev = use_chebyshev
        self.solver_data: list = []
        self.interpolator: Optional[LagrangeInterpolator1D] = None

    def add_solution_point(self, t: float, x: np.ndarray) -> None:
        """添加一个解点 (t, x*(t))."""
        self.solver_data.append((float(t), x.copy()))

    def build_interpolator(self) -> None:
        """构建解路径的插值器."""
        if len(self.solver_data) < 2:
            raise ValueError("至少需要 2 个解点来构建插值")
        t_vals = np.array([s[0] for s in self.solver_data])
        x_vals = np.array([s[1] for s in self.solver_data])

        if self.use_chebyshev:
            # 确保节点按 Chebyshev 分布 (重排序)
            sort_idx = np.argsort(t_vals)
            t_vals = t_vals[sort_idx]
            x_vals = x_vals[sort_idx]

        self.interpolator = LagrangeInterpolator1D(t_vals, x_vals)

    def predict(self, t_new: float) -> np.ndarray:
        """
        预测新参数值处的解.

        用于 continuation 方法的预测步.
        """
        if self.interpolator is None:
            raise RuntimeError("请先调用 build_interpolator()")
        t_arr = np.array([t_new])
        return self.interpolator.evaluate(t_arr)[0]

    def sensitivity(self, t_new: float) -> np.ndarray:
        """
        计算解的灵敏度 dx*/dt.

        这对参数化 VI 的分歧分析至关重要.
        """
        if self.interpolator is None:
            raise RuntimeError("请先调用 build_interpolator()")
        t_arr = np.array([t_new])
        return self.interpolator.derivative(t_arr)[0]

    def interpolation_error_estimate(self, t: np.ndarray) -> np.ndarray:
        """
        估计插值误差 (基于节点多项式).

        |ω(t)|^{1/(n+1)} 作为误差的粗略度量.
        """
        if self.interpolator is None:
            return np.zeros_like(t)
        omega = np.ones_like(t, dtype=float)
        for t_k in self.interpolator.t_data:
            omega *= np.abs(t - t_k)
        return omega ** (1.0 / (self.interpolator.n + 1))
