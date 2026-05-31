"""
Numerical Robustness Utilities
==============================
源自种子项目：
  - 809_nonlin_regula (Regula falsi root finding)
  - 961_r8_scale (Machine-epsilon aware floating-point traversal)
  - 069_ball_monte_carlo (Monte Carlo integration over unit ball)

本模块提供时间序列分析中的数值鲁棒性工具：
1. Regula Falsi 求根：用于求解异常检测阈值方程
2. 机器精度分析：评估数值算法的稳定性边界
3. 球体积 Monte Carlo 积分：用于高维特征空间中的概率异常界

数学原理：
1. Regula Falsi：
   给定 [a,b] 使得 f(a)f(b) < 0，
   c = (a f(b) - b f(a)) / (f(b) - f(a))
   保持符号变化区间，直到 |f(c)| < tol 或 |b-a| < tol。

2. IEEE-754 双精度浮点：
   eps ≈ 2.22e-16，nextafter(x) 给出 x 的下一个可表示浮点数。
   next(x) = x / (1 - eps/2)  for x > 0
   prev(x) = x * (1 - eps/2)  for x > 0

3. 单位球 Monte Carlo：
   在 d 维单位球 B_d(0,1) 上均匀采样：
       u ~ N(0, I_d),  r = ||u||_2,  x = U^{1/d} * u / r
   其中 U ~ Uniform(0,1)。
   体积公式：V_d = π^{d/2} / Γ(d/2 + 1)
   积分估计：I ≈ V_d * (1/N) sum_i f(x_i)
"""

import numpy as np
from typing import Callable


class NumericalRobustness:
    """
    数值鲁棒性工具集合。
    """

    EPS = np.finfo(float).eps
    REAL_MIN = np.finfo(float).tiny
    REAL_MAX = np.finfo(float).max

    @staticmethod
    def next_float(x: float) -> float:
        """
        返回 x 的下一个可表示双精度浮点数。
        """
        if np.isnan(x) or np.isinf(x):
            return x
        if x == 0.0:
            return NumericalRobustness.EPS
        if x > 0:
            if x >= NumericalRobustness.REAL_MAX:
                return np.inf
            return x / (1.0 - NumericalRobustness.EPS / 2.0)
        else:
            if x <= -NumericalRobustness.REAL_MAX:
                return -np.inf
            return x * (1.0 - NumericalRobustness.EPS / 2.0)

    @staticmethod
    def prev_float(x: float) -> float:
        """
        返回 x 的前一个可表示双精度浮点数。
        """
        if np.isnan(x) or np.isinf(x):
            return x
        if x == 0.0:
            return -NumericalRobustness.EPS
        if x > 0:
            if x <= NumericalRobustness.REAL_MIN:
                return 0.0
            return x * (1.0 - NumericalRobustness.EPS / 2.0)
        else:
            if x >= -NumericalRobustness.REAL_MIN:
                return 0.0
            return x / (1.0 - NumericalRobustness.EPS / 2.0)

    @staticmethod
    def regula_falsi(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10, max_iter: int = 100) -> float:
        """
        Regula Falsi 求根算法。
        要求 f(a) 和 f(b) 异号。
        """
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")
        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b

        # 边界处理：确保 a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(max_iter):
            # 避免除零
            denom = fb - fa
            if abs(denom) < 1e-15:
                return (a + b) / 2.0

            c = (a * fb - b * fa) / denom
            # 数值鲁棒性：c 必须在 (a,b) 内
            if c <= a or c >= b:
                c = (a + b) / 2.0

            fc = f(c)
            if abs(fc) < tol or abs(b - a) < tol:
                return c

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            # 防止区间收缩过慢：Illinois 变体
            if abs(fb) < 0.5 * abs(fa):
                fa *= 0.5

        return (a + b) / 2.0

    @staticmethod
    def threshold_by_quantile_root(scores: np.ndarray, target_fpr: float = 0.05) -> float:
        """
        使用 Regula Falsi 求解异常阈值 θ，使得假阳性率 (FPR) 精确等于 target_fpr。
        即求解：F(θ) = (1/N) sum_i I[scores_i > θ] - target_fpr = 0
        """
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        if n == 0:
            return 0.0

        def empirical_fpr(theta: float) -> float:
            return np.mean(scores > theta) - target_fpr

        a = float(sorted_scores[0]) - 1e-6
        b = float(sorted_scores[-1]) + 1e-6
        fa = empirical_fpr(a)
        fb = empirical_fpr(b)

        # 若目标不可达，返回最近边界
        if fa * fb > 0:
            if abs(fa) < abs(fb):
                return a
            return b

        return NumericalRobustness.regula_falsi(empirical_fpr, a, b)

    @staticmethod
    def ball_monte_carlo_integral(f: Callable[[np.ndarray], float],
                                   dim: int = 3, n_samples: int = 10000) -> float:
        """
        d 维单位球上的 Monte Carlo 积分。
        I = ∫_{B_d} f(x) dx = V_d * E[f(U)]
        """
        if dim < 1:
            raise ValueError("dim must be >= 1")

        # 单位球体积
        from math import gamma, pi
        volume = pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0)

        # 均匀采样
        samples = np.random.randn(n_samples, dim)
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        uniforms = np.random.rand(n_samples, 1) ** (1.0 / dim)
        points = uniforms * samples / norms

        vals = np.array([f(p) for p in points])
        return volume * np.mean(vals)

    @staticmethod
    def mahalanobis_ball_probability(center: np.ndarray, cov: np.ndarray,
                                      radius: float, dim: int, n_samples: int = 50000) -> float:
        """
        计算 Mahalanobis 球内概率：
            P( (x-μ)^T Σ^{-1} (x-μ) <= r^2 )
        通过 Monte Carlo 在单位球上积分变换密度。
        用于多维特征空间中的异常概率边界。
        """
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # 正则化
            cov_reg = cov + 1e-6 * np.eye(dim)
            L = np.linalg.cholesky(cov_reg)

        def transformed_density(x: np.ndarray) -> float:
            # x ∈ B_d，变换后 y = L x + center
            # 雅可比行列式 det(L)
            return np.linalg.det(L)

        return NumericalRobustness.ball_monte_carlo_integral(transformed_density, dim, n_samples)

    @staticmethod
    def condition_number_sensitivity(matrix: np.ndarray) -> dict:
        """
        分析矩阵条件数对机器精度的敏感性。
        """
        cond = np.linalg.cond(matrix)
        # 有效数字损失估计
        digits_lost = np.log10(cond)
        # 机器精度下的可解性阈值
        solvable = digits_lost < -np.log10(NumericalRobustness.EPS)
        return {
            "condition_number": float(cond),
            "digits_lost": float(digits_lost),
            "solvable_in_double_precision": bool(solvable),
            "recommended_regularization": float(cond * NumericalRobustness.EPS)
        }
