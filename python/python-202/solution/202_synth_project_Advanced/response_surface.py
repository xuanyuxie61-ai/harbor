"""
响应面重构模块 (Stochastic Response Surface Reconstruction)
==============================================================
使用自然三次样条和多项式混沌从稀疏配置点重构随机响应面。

自然三次样条:
  S(x) 是分段三次多项式, 满足:
    1. S(x_i) = y_i  (插值条件)
    2. S ∈ C²[a,b]  (二阶连续可微)
    3. S''(a) = S''(b) = 0  (自然边界条件)

  在每个区间 [x_i, x_{i+1}]:
    S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)² + d_i(x-x_i)³

  其中系数通过三弯矩方程求解:
    h_i c_{i-1} + 2(h_i + h_{i+1}) c_i + h_{i+1} c_{i+1} = 3(a_{i+1}-a_i)/h_{i+1} - 3(a_i-a_{i-1})/h_i

  自然边界条件 ⟹ c_0 = c_n = 0

误差估计:
  ||f - S||_∞ ≤ (5/384) h⁴ ||f''''||_∞
  其中 h = max(x_{i+1} - x_i)

随机响应面重构:
  给定配置点 {ξ_k, u(ξ_k)}_{k=1}^{N}, 重构:
    û(ξ) = Σ_k c_k(ξ) u(ξ_k)  (插值)
  或
    û(ξ) = Σ_α c_α Ψ_α(ξ)  (PC 投影)

  然后从 û 计算统计矩:
    E[u] ≈ ∫ û(ξ) dμ(ξ)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import linalg as la


class NaturalCubicSpline:
    """
    自然三次样条插值器。

    自然边界条件: S''(a) = S''(b) = 0
    这最小化了 ∫ |S''(x)|² dx (最小曲率性质)

    应用: 在 1D 随机方向上重构响应面切片。
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.n = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'NaturalCubicSpline':
        """
        拟合自然三次样条。

        算法 (三弯矩法):
          设 h_i = x_{i+1} - x_i
          三弯矩方程:
            μ_i M_{i-1} + 2 M_i + λ_i M_{i+1} = d_i

          其中:
            μ_i = h_{i-1} / (h_{i-1} + h_i)
            λ_i = h_i / (h_{i-1} + h_i) = 1 - μ_i
            d_i = 6 / (h_{i-1} + h_i) [(y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}]

          自然边界: M_0 = M_n = 0

        参数:
            x: 插值节点, shape (n+1,), 严格递增
            y: 函数值, shape (n+1,)

        返回: self
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x = x
        self.y = y
        self.n = len(x) - 1
        n = self.n

        if n < 1:
            raise ValueError("Need at least 2 points for spline")

        if n == 1:
            # 线性插值
            self.a = y[:1].copy()
            self.b = np.array([(y[1] - y[0]) / (x[1] - x[0])])
            self.c = np.array([0.0])
            self.d = np.array([0.0])
            return self

        # 步长
        h = np.diff(x)

        # 构建三对角系统 (自然边界: c_0 = c_n = 0)
        # 内部点: i = 1, ..., n-1
        n_int = n - 1
        if n_int > 0:
            mu = np.zeros(n_int)
            lam = np.zeros(n_int)
            rhs = np.zeros(n_int)

            for i in range(n_int):
                ii = i + 1  # 原始索引
                mu[i] = h[ii - 1] / (h[ii - 1] + h[ii])
                lam[i] = h[ii] / (h[ii - 1] + h[ii])
                rhs[i] = 6.0 / (h[ii - 1] + h[ii]) * (
                    (y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]
                )

            # 求解三对角系统
            diag = 2.0 * np.ones(n_int)
            lower = mu[1:]
            upper = lam[:-1]

            A_banded = np.zeros((3, n_int))
            A_banded[0, 1:] = upper
            A_banded[1, :] = diag
            A_banded[2, :-1] = lower

            c_int = la.solve_banded((1, 1), A_banded, rhs)

            # 完整 c 向量 (含自然边界 c_0 = c_n = 0)
            self.c = np.zeros(n + 1)
            self.c[1:-1] = c_int
        else:
            self.c = np.zeros(n + 1)

        # 计算其他系数
        self.a = y[:-1].copy()
        self.b = np.zeros(n)
        self.d = np.zeros(n)

        for i in range(n):
            self.b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * self.c[i] + self.c[i + 1]) / 3.0
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])

        return self

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """
        在查询点求值样条。

        S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)² + d_i(x-x_i)³

        参数:
            xi: 查询点, shape (M,)

        返回:
            yi: shape (M,)
        """
        xi = np.asarray(xi, dtype=np.float64)
        result = np.zeros_like(xi)

        for k in range(len(xi)):
            x_val = xi[k]
            # 找到所在区间
            idx = np.searchsorted(self.x, x_val) - 1
            idx = max(0, min(idx, self.n - 1))

            dx = x_val - self.x[idx]
            result[k] = (self.a[idx] + self.b[idx] * dx
                         + self.c[idx] * dx ** 2 + self.d[idx] * dx ** 3)

        return result

    def evaluate_derivative(self, xi: np.ndarray) -> np.ndarray:
        """
        求值样条的一阶导数:
          S'_i(x) = b_i + 2 c_i (x-x_i) + 3 d_i (x-x_i)²
        """
        xi = np.asarray(xi, dtype=np.float64)
        result = np.zeros_like(xi)

        for k in range(len(xi)):
            x_val = xi[k]
            idx = np.searchsorted(self.x, x_val) - 1
            idx = max(0, min(idx, self.n - 1))

            dx = x_val - self.x[idx]
            result[k] = (self.b[idx] + 2.0 * self.c[idx] * dx
                         + 3.0 * self.d[idx] * dx ** 2)

        return result

    def curvature(self) -> float:
        """
        计算样条的总曲率:
          C = ∫ |S''(x)|² dx

        自然样条最小化此泛函 (最小曲率性质)。
        """
        total = 0.0
        for i in range(self.n):
            h = self.x[i + 1] - self.x[i]
            # ∫_0^h (2c_i + 6d_i t)² dt
            # = 4c_i² h + 24 c_i d_i h²/2 + 36 d_i² h³/3
            total += (4.0 * self.c[i] ** 2 * h
                      + 12.0 * self.c[i] * self.d[i] * h ** 2
                      + 12.0 * self.d[i] ** 2 * h ** 3)
        return total


class ResponseSurfaceReconstructor:
    """
    随机响应面重构器。

    从配置点数据重构完整的随机响应面:
      u(ξ) ≈ û(ξ)

    方法:
      1. 1D 切片: 自然三次样条
      2. 多维: 张量积样条或 PC 投影
    """

    def __init__(self):
        self.splines_1d = {}
        self.pc_coeffs = None
        self.pc_basis = None

    def reconstruct_1d(
        self, nodes: np.ndarray, values: np.ndarray
    ) -> NaturalCubicSpline:
        """
        1D 响应面重构 (自然三次样条)。

        参数:
            nodes: 配置点 (1D), shape (N,)
            values: 解值, shape (N,)

        返回:
            spline: 拟合的样条对象
        """
        # 排序
        sort_idx = np.argsort(nodes)
        sorted_nodes = nodes[sort_idx]
        sorted_values = values[sort_idx]

        # 去重
        unique_mask = np.concatenate([[True], np.diff(sorted_nodes) > 1e-12])
        sorted_nodes = sorted_nodes[unique_mask]
        sorted_values = sorted_values[unique_mask]

        spline = NaturalCubicSpline()
        spline.fit(sorted_nodes, sorted_values)
        return spline

    def reconstruct_from_pc(
        self,
        pc_coefficients: np.ndarray,
        pc_basis
    ) -> Callable:
        """
        从 PC 系数重构响应面。

        û(ξ) = Σ_α c_α Ψ_α(ξ)

        返回一个可调用的函数。
        """
        self.pc_coeffs = pc_coefficients
        self.pc_basis = pc_basis

        def response_surface(xi):
            return pc_basis.evaluate_from_coefficients(pc_coefficients, xi)

        return response_surface

    def compute_statistics_from_spline(
        self,
        spline: NaturalCubicSpline,
        n_quad: int = 100,
        measure: str = 'uniform',
        a: float = -1.0, b: float = 1.0
    ) -> dict:
        """
        从样条响应面计算统计矩。

        使用 Gauss 求积:
          E[u] ≈ Σ_k w_k û(ξ_k)
          Var[u] ≈ Σ_k w_k û(ξ_k)² - (E[u])²

        参数:
            spline: 拟合的样条
            n_quad: 求积点数
            measure: 测度类型
            a, b: 均匀测度的区间

        返回:
            dict: {'mean': ..., 'variance': ..., 'std': ...}
        """
        from polynomial_utils import gauss_legendre, gauss_hermite

        if measure == 'uniform':
            nodes, weights = gauss_legendre(n_quad)
            # 映射到 [a, b]
            nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
            weights *= 0.5 * (b - a)
        elif measure == 'gaussian':
            nodes, weights = gauss_hermite(n_quad)
        else:
            nodes, weights = gauss_legendre(n_quad)

        values = spline.evaluate(nodes)

        mean = np.sum(weights * values)
        variance = np.sum(weights * values ** 2) - mean ** 2
        variance = max(variance, 0.0)

        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(np.sqrt(variance)),
        }
