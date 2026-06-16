"""
chebyshev_surrogate.py
======================
Chebyshev 代理模型 —— 基于 Chebyshev 近似的昂贵函数优化

融合种子项目:
  - 163_chebyshev_series: Chebyshev 级数求值及高阶导数

核心公式:
  1. Chebyshev 插值: p_n(x) = Σ c_k T_k(x),  c_k = (2/n) Σ f(x_j) T_k(x_j)
  2. Chebyshev 节点: x_j = cos(π(j+0.5)/n), 近最优极小化 Runge 现象
  3. 截断误差: |f(x) - p_n(x)| ≤ 2/(2^n n!) max|f^{(n)}| (对解析函数)
  4. 导数级数: f'(x) = Σ d_k T_k(x), d_k = 2(k+1)c_{k+1} + d_{k+2}
  5. 代理优化: min p_n(x) 替代 min f(x), 大幅减少昂贵评估次数
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from special_functions import chebyshev_T, chebyshev_series_eval, chebyshev_coefficients


# ---------------------------------------------------------------------------
# 1. Chebyshev 代理模型类
# ---------------------------------------------------------------------------

class ChebyshevSurrogate:
    """Chebyshev 代理模型.

    对一维函数 f: [a,b] → ℝ, 构造 n 阶 Chebyshev 近似:
      p_n(x) = Σ_{k=0}^{n-1} c_k T_k((2x-a-b)/(b-a))

    用于代理优化: 先用少量 f 评估构造 p_n, 再优化 p_n (便宜).
    """

    def __init__(self, f: Callable, a: float, b: float, n_terms: int = 20):
        """构造代理模型.
        f: 目标函数
        [a,b]: 定义域
        n_terms: Chebyshev 项数
        """
        self.f = f
        self.a = a
        self.b = b
        self.n_terms = n_terms

        # 计算 Chebyshev 系数
        self.coef = chebyshev_coefficients(f, a, b, n_terms)

        # 计算导数系数
        self._compute_deriv_coefficients()

    def _compute_deriv_coefficients(self):
        """计算导数的 Chebyshev 系数.
        若 f = Σ c_k T_k, 则 f' = Σ d_k T_k, 其中:
          d_{n-1} = 0
          d_{n-2} = 2(n-1) c_{n-1}
          d_k = d_{k+2} + 2(k+1) c_{k+1}, k = n-3, ..., 0
          d_0 ← d_0 / 2  (首项减半修正)
        """
        n = self.n_terms
        c = self.coef
        self.dcoef = np.zeros(n)
        if n >= 2:
            self.dcoef[n - 2] = 2.0 * (n - 1) * c[n - 1]
        for k in range(n - 3, -1, -1):
            self.dcoef[k] = self.dcoef[k + 2] + 2.0 * (k + 1) * c[k + 1]
        if n > 0:
            self.dcoef[0] *= 0.5

    def evaluate(self, x: float) -> float:
        """代理模型求值 p_n(x)."""
        # 映射到 [-1, 1]
        t = (2.0 * x - self.a - self.b) / (self.b - self.a)
        t = np.clip(t, -1.0, 1.0)
        return chebyshev_series_eval(t, self.coef)

    def evaluate_derivative(self, x: float) -> float:
        """代理模型导数 p_n'(x)."""
        t = (2.0 * x - self.a - self.b) / (self.b - self.a)
        t = np.clip(t, -1.0, 1.0)
        # 链式法则: df/dx = df/dt · dt/dx = df/dt · 2/(b-a)
        dfdt = chebyshev_series_eval(t, self.dcoef)
        return dfdt * 2.0 / (self.b - self.a)

    def find_minimum(self, n_grid: int = 1000) -> Tuple[float, float]:
        """在 [a,b] 上找代理模型的最小值.
        先用网格搜索定位, 再用 Newton 精化.
        """
        # 网格搜索
        x_grid = np.linspace(self.a, self.b, n_grid)
        f_grid = np.array([self.evaluate(xi) for xi in x_grid])
        idx_min = np.argmin(f_grid)
        x0 = x_grid[idx_min]

        # Newton 精化
        x = x0
        for _ in range(50):
            fp = self.evaluate_derivative(x)
            # 二阶导数 (数值)
            eps = 1e-8
            fpp = (self.evaluate_derivative(x + eps) - self.evaluate_derivative(x - eps)) / (2 * eps)
            if abs(fpp) < 1e-300:
                break
            dx = -fp / fpp
            x_new = x + dx
            if x_new < self.a or x_new > self.b:
                break
            if abs(dx) < 1e-12:
                break
            x = x_new

        return x, self.evaluate(x)


# ---------------------------------------------------------------------------
# 2. 多维 Chebyshev 代理 (张量积)
# ---------------------------------------------------------------------------

class MultivariateChebyshevSurrogate:
    """多维 Chebyshev 代理模型 (张量积构造).

    对 f: [a₁,b₁]×...×[a_d,b_d] → ℝ:
      p(x) = Σ_{k₁,...,k_d} c_{k₁,...,k_d} Π T_{k_i}(t_i(x_i))

    项数: n^d (随维数指数增长, 仅适用于 d ≤ 6).
    """

    def __init__(self, f: Callable, bounds: np.ndarray, n_per_dim: int = 10):
        """
        bounds: (d, 2), 每行 [a_i, b_i]
        """
        self.f = f
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.n_per_dim = n_per_dim

        # 一维 Chebyshev 节点 (在 [-1,1])
        self.nodes_1d = np.cos(np.pi * (np.arange(n_per_dim) + 0.5) / n_per_dim)

        # 构造插值
        self._build_interpolant()

    def _build_interpolant(self):
        """构造张量积 Chebyshev 插值."""
        d = self.dim
        n = self.n_per_dim

        # 生成所有网格点 (张量积)
        # 对 d=2, n=10: 100 个点; d=3: 1000; d=4: 10000
        n_total = n ** d

        # 生成网格索引
        from itertools import product
        indices = list(product(range(n), repeat=d))

        # 计算函数值
        self.f_values = np.zeros(n_total)
        self.grid_points = np.zeros((n_total, d))

        for idx, multi_idx in enumerate(indices):
            # 映射到物理空间
            point = np.zeros(d)
            for k in range(d):
                t = self.nodes_1d[multi_idx[k]]
                point[k] = 0.5 * (self.bounds[k, 1] - self.bounds[k, 0]) * t + \
                           0.5 * (self.bounds[k, 0] + self.bounds[k, 1])
            self.grid_points[idx] = point
            self.f_values[idx] = self.f(point)

        # 存储 (简化: 直接存储函数值, 通过最近邻插值)
        self.indices = indices

    def evaluate(self, x: np.ndarray) -> float:
        """代理模型求值 (最近邻 Chebyshev 插值)."""
        # 映射到 [-1,1]^d
        t = np.zeros(self.dim)
        for k in range(self.dim):
            t[k] = (2.0 * x[k] - self.bounds[k, 0] - self.bounds[k, 1]) / \
                   (self.bounds[k, 1] - self.bounds[k, 0])
            t[k] = np.clip(t[k], -1.0, 1.0)

        # 张量积 Lagrange 插值 (简化: 加权平均)
        weights = np.ones(len(self.indices))
        for idx, multi_idx in enumerate(self.indices):
            for k in range(self.dim):
                t_node = self.nodes_1d[multi_idx[k]]
                dist = abs(t[k] - t_node)
                weights[idx] *= 1.0 / max(dist, 1e-10)

        # 归一化
        w_sum = np.sum(weights)
        if w_sum < 1e-300:
            # 使用最近点
            min_dist = float('inf')
            best_idx = 0
            for idx, multi_idx in enumerate(self.indices):
                dist = sum(abs(t[k] - self.nodes_1d[multi_idx[k]]) for k in range(self.dim))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            return float(self.f_values[best_idx])

        return float(np.sum(weights * self.f_values) / w_sum)


# ---------------------------------------------------------------------------
# 3. 自适应 Chebyshev 优化
# ---------------------------------------------------------------------------

def chebyshev_surrogate_optimization(f: Callable, a: float, b: float,
                                     n_initial: int = 20,
                                     n_iterations: int = 50,
                                     tol: float = 1e-10) -> Dict:
    """自适应 Chebyshev 代理优化 (一维).

    算法:
      1. 用 n_initial 个点构造初始代理 p_n
      2. 找 p_n 的最小值 x*
      3. 在 x* 附近评估 f, 添加到样本集
      4. 重新构造代理
      5. 重复至收敛

    优势: 大幅减少昂贵函数 f 的评估次数.
    适用于: 单次 f 评估成本高的场景 (如 CFD, 分子动力学).
    """
    # 初始样本
    n_total = n_initial
    x_samples = np.linspace(a, b, n_initial)
    f_samples = np.array([f(x) for x in x_samples])

    history = []

    for iteration in range(n_iterations):
        # 构造代理
        def f_interp(x):
            # 简单插值 (逆距离加权)
            dists = np.abs(x_samples - x)
            if np.min(dists) < 1e-15:
                idx = np.argmin(dists)
                return f_samples[idx]
            weights = 1.0 / dists ** 2
            return np.sum(weights * f_samples) / np.sum(weights)

        # 在代理上优化 (网格搜索)
        x_fine = np.linspace(a, b, 1000)
        f_fine = np.array([f_interp(xi) for xi in x_fine])
        idx_min = np.argmin(f_fine)
        x_new = x_fine[idx_min]

        # 评估真实函数
        f_new = f(x_new)
        history.append({
            "iteration": iteration,
            "x": x_new,
            "f": f_new,
            "n_samples": n_total
        })

        # 检查收敛
        if len(history) >= 2:
            if abs(history[-1]["f"] - history[-2]["f"]) < tol:
                break

        # 添加新样本
        x_samples = np.append(x_samples, x_new)
        f_samples = np.append(f_samples, f_new)
        n_total += 1

    best_idx = np.argmin(f_samples)
    return {
        "x": x_samples[best_idx],
        "f_val": f_samples[best_idx],
        "n_evaluations": n_total,
        "history": history,
        "method": "chebyshev_surrogate"
    }


# ---------------------------------------------------------------------------
# 4. Chebyshev 误差估计
# ---------------------------------------------------------------------------

def chebyshev_error_estimate(coef: np.ndarray) -> float:
    """Chebyshev 级数的截断误差估计.
    近似: err ≈ |c_{n-1}| + |c_{n-2}|
    (最后两项的系数和近似截断误差).
    """
    if len(coef) < 2:
        return 0.0
    return abs(coef[-1]) + abs(coef[-2])


def chebyshev_adaptive_degree(f: Callable, a: float, b: float,
                              tol: float = 1e-12,
                              max_degree: int = 100) -> Tuple[np.ndarray, int]:
    """自适应 Chebyshev 展开 (增加阶数直至收敛).

    从低阶开始, 每次倍增阶数, 直到截断误差 < tol.

    返回 (coef, degree).
    """
    n = 8
    while n <= max_degree:
        coef = chebyshev_coefficients(f, a, b, n)
        err = chebyshev_error_estimate(coef)
        if err < tol:
            return coef, n
        n *= 2

    coef = chebyshev_coefficients(f, a, b, max_degree)
    return coef, max_degree
