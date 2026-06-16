"""
hypercube_projection.py
=======================
超立方体投影度量与 VI 投影算子的几何分析.

数学背景
--------
在投影型 VI 算法中, 核心操作是到凸集 K 的 Euclid 投影:
    Π_K(x) = argmin_{y ∈ K} ||x - y||_2

对非负象限 K = R^n_+:
    Π_K(x)_i = max(0, x_i)

对盒约束 K = [l, u]:
    Π_K(x)_i = min(max(x_i, l_i), u_i)

超立方体距离的几何意义:
    在 Monte Carlo 分析中, 我们需要度量参数空间中两个点的距离.
    当参数属于超立方体 [0,1]^m 时, 距离的统计性质为:
        E[||X - Y||_2²] = m / 6
        Var[||X - Y||_2] ≈ m · (1/6 - 1/(4π))  (大 m 近似)

本模块实现:
    1. 多种凸集的精确投影
    2. 超立方体中随机点距离的 Monte Carlo 估计
    3. 投影残差的收敛性检验

关键公式
--------
Moreau 分解:
    x = Π_K(x) + Π_{K°}(x)
其中 K° 为 K 的极锥. 对 K = R^n_+:
    x = max(0, x) + min(0, x)

投影的非扩张性:
    ||Π_K(x) - Π_K(y)|| ≤ ||x - y||

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable


class ProjectionOperators:
    """
    凸集上的 Euclid 投影算子集合.

    所有投影满足:
        1. 幂等性: Π_K(Π_K(x)) = Π_K(x)
        2. 非扩张性: ||Π_K(x) - Π_K(y)|| ≤ ||x - y||
        3. 变分特征: ⟨x - Π_K(x), y - Π_K(x)⟩ ≤ 0, ∀ y ∈ K
    """

    @staticmethod
    def project_nonnegative(x: np.ndarray) -> np.ndarray:
        """
        到非负象限 R^n_+ 的投影:
            Π_{R+}(x)_i = max(0, x_i)
        """
        return np.maximum(0.0, x)

    @staticmethod
    def project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        到盒约束 [l, u] 的投影:
            Π_{[l,u]}(x)_i = min(max(x_i, l_i), u_i)
        """
        return np.minimum(np.maximum(x, lower), upper)

    @staticmethod
    def project_simplex(x: np.ndarray, s: float = 1.0) -> np.ndarray:
        """
        到单位单纯形 Δ_s = {x ≥ 0 : Σx_i = s} 的投影.

        算法 (Condat 2016, O(n log n)):
            1. 降序排序: x_{(1)} ≥ x_{(2)} ≥ ... ≥ x_{(n)}
            2. 找 ρ = max{j : x_{(j)} - (Σ_{k≤j} x_{(k)} - s)/j > 0}
            3. θ = (Σ_{k≤ρ} x_{(k)} - s) / ρ
            4. Π(x)_i = max(0, x_i - θ)
        """
        n = len(x)
        x_sorted = np.sort(x)[::-1]
        cumsum = np.cumsum(x_sorted)
        j = np.arange(1, n + 1)
        test = x_sorted - (cumsum - s) / j
        rho = np.max(np.where(test > 0)[0]) + 1
        theta = (np.sum(x_sorted[:rho]) - s) / rho
        return np.maximum(0.0, x - theta)

    @staticmethod
    def project_l2_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        到 L2 球 {x : ||x||_2 ≤ r} 的投影:
            Π(x) = x · min(1, r / ||x||_2)
        """
        norm_x = np.linalg.norm(x)
        if norm_x <= radius:
            return x.copy()
        return x * (radius / norm_x)

    @staticmethod
    def project_hyperplane(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
        """
        到超平面 {y : a^T y = b} 的投影:
            Π(x) = x - (a^T x - b) / ||a||^2 · a
        """
        a_norm_sq = np.dot(a, a)
        if a_norm_sq < 1e-30:
            return x.copy()
        return x - (np.dot(a, x) - b) / a_norm_sq * a

    @staticmethod
    def verify_projection_properties(
        x: np.ndarray,
        y: np.ndarray,
        proj_func: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[bool, bool, float]:
        """
        验证投影算子的数学性质.

        Returns
        -------
        idempotent : bool
            ||Π(Π(x)) - Π(x)|| < ε
        nonexpansive : bool
            ||Π(x) - Π(y)|| ≤ ||x - y|| + ε
        nonexpansive_ratio : float
            ||Π(x) - Π(y)|| / ||x - y||
        """
        Px = proj_func(x)
        PPx = proj_func(Px)
        idempotent = np.linalg.norm(PPx - Px) < 1e-10

        Py = proj_func(y)
        dist_xy = np.linalg.norm(x - y)
        dist_PxPy = np.linalg.norm(Px - Py)
        if dist_xy > 1e-14:
            ratio = dist_PxPy / dist_xy
            nonexpansive = ratio <= 1.0 + 1e-10
        else:
            ratio = 0.0
            nonexpansive = True

        return idempotent, nonexpansive, ratio


class HypercubeDistanceAnalyzer:
    """
    超立方体中随机点距离的统计分析.

    理论结果 (m 维单位超立方体 [0,1]^m):
        E[D²] = m/6
        E[D] ≈ √(m/6) · (1 - 1/(4m) + O(1/m²))  (大 m)
        Var[D] ≈ 1/6 - 1/(4π) · √(6/m)  + ... (近似)

    在 VI 中的应用:
        - Monte Carlo 参数的距离度量
        - 随机 VI 中场景集的分散度分析
    """

    @staticmethod
    def monte_carlo_estimate(
        dimension: int,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Monte Carlo 估计 E[D] 和 Var[D].

        Returns
        -------
        mu, var, std_error : float
        """
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, (dimension, n_samples))
        Y = rng.uniform(0.0, 1.0, (dimension, n_samples))
        distances = np.sqrt(np.sum((X - Y)**2, axis=0))

        mu = float(np.mean(distances))
        var = float(np.var(distances, ddof=1)) if n_samples > 1 else 0.0
        std_error = float(np.std(distances, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else 0.0
        return mu, var, std_error

    @staticmethod
    def theoretical_E_D_squared(dimension: int) -> float:
        """E[D²] = m/6 的精确值."""
        return dimension / 6.0

    @staticmethod
    def large_m_approx_E_D(dimension: int) -> float:
        """大 m 时 E[D] 的近似: √(m/6) · (1 - 1/(4m))."""
        m = dimension
        return np.sqrt(m / 6.0) * (1.0 - 1.0 / (4.0 * m))


class NormalConeProjection:
    """
    法锥投影: 用于 VI 的最优性条件分析.

    在解 x* 处, VI 的最优性条件为:
        -F(x*) ∈ N_K(x*)
    其中 N_K(x*) 为 K 在 x* 处的法锥.

    对 K = R^n_+:
        N_K(x)_i = {0}        当 x_i > 0
        N_K(x)_i = (-∞, 0]   当 x_i = 0

    投影到法锥等价于互补条件的验证.
    """

    @staticmethod
    def project_normal_cone_nonneg(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        将向量 v 投影到 R^n_+ 在 x 处的法锥.

        对每个分量:
            x_i > 0: N_i = {0}, Π(v_i) = 0
            x_i = 0: N_i = (-∞, 0], Π(v_i) = min(0, v_i)
        """
        result = np.zeros_like(v)
        active = x < 1e-12
        result[active] = np.minimum(0.0, v[active])
        return result

    @staticmethod
    def optimality_residual(x: np.ndarray, Fx: np.ndarray) -> float:
        """
        VI 最优性残差: dist(-F(x), N_K(x)).

        对非负约束:
            r_i = F_i(x)               当 x_i > 0 (应 = 0)
            r_i = min(0, F_i(x))       当 x_i = 0 (应 ≥ 0, 故 min=0)
        故 r = min(x, F(x)) (逐点).
        """
        return float(np.linalg.norm(np.minimum(x, Fx)))
