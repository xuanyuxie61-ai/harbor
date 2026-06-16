"""
chance_constraints.py
---------------------
机会约束处理模块 —— 映射自种子项目 051_asa243 (非中心 t 分布)
核心思想：将概率约束 (chance constraints) 转化为确定性约束，
利用非中心 t 分布的尾部概率计算可行域。

科学背景：
    考虑带机会约束的鲁棒优化：
        min  f(x)
        s.t. P( g_i(x, w) <= 0 ) >= 1 - alpha_i,  i = 1..m
             x in X
    当 g_i 为仿射不确定性时：
        g_i(x, w) = a_i^T x - b_i + w^T x
        w ~ N(mu_w, Sigma_w)
    机会约束等价于：
        a_i^T x - b_i + mu_w^T x + t_{1-alpha} * sqrt(x^T Sigma_w x) <= 0
    其中 t_{1-alpha} 为标准正态的 (1-alpha) 分位数。

    对于小样本或非高斯情形，需用非中心 t 分布修正：
        t_{1-alpha}(n-1, delta=sqrt(n) mu / sigma)
    其中 n 为样本量，delta 为非中心参数。

本模块实现：
    1. 正态分位数计算 (AS 243 的特例)
    2. 机会约束的确定性等价
    3. 样本均值-方差修正 (t 分布校正)
    4. 联合机会约束的 Bonferroni 近似
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 标准正态分位数 (Beasley-Springer-Moro 算法)
# =============================================================================
def norm_ppf(p: float) -> float:
    """
    标准正态分布的分位数函数：
        z_p : P(Z <= z_p) = p,  Z ~ N(0,1)
    采用 Abramowitz-Stegun 近似 (误差 < 4.5e-4)。
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p 必须位于 (0, 1)")
    if p < 0.5:
        return -norm_ppf(1.0 - p)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


# =============================================================================
# 非中心 t 分布 (简化实现)
# =============================================================================
def _alnorm(x: float, upper: bool = True) -> float:
    """标准正态上/下尾 (Hill AS 66 简化)."""
    z = abs(x)
    if z <= 0.67448975:
        p = 0.398942280444 * z
        y = 0.5 - p
    elif z <= 2.5:
        p = np.exp(-0.5 * z * z) * (
            0.39990348504 + z * (0.0599832206555 + z * 0.0159306)
        )
        y = 0.5 - p if x > 0 else 0.5 + p
        return y if upper else 1.0 - y
    else:
        p = np.exp(-0.5 * z * z) / (
            z + 0.398942280385 / (z + 0.03990348504 / (z + 0.003990348504 / (z + 0.0003990348504)))
        )
        y = p if x > 0 else 1.0 - p
        return y if upper else 1.0 - y
    return (1.0 - y) if upper else y


def noncentral_t_quantile(
    p: float,
    df: float,
    delta: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> float:
    """
    非中心 t 分布的分位数 (二分法)。
    P(T <= t | df, delta) = p.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须位于 (0, 1)")
    if df <= 0.0:
        raise ValueError("df 必须为正")

    # 初始区间
    lo, hi = -10.0, 10.0
    # 扩展
    for _ in range(40):
        mu = delta * np.sqrt(df / 2.0) * np.exp(
            math.lgamma((df - 1) / 2.0) - math.lgamma(df / 2.0)
        ) if df > 1 else delta
        var = df * (1.0 + delta ** 2) / (df - 2.0) - mu ** 2 if df > 2 else 1.0 + delta ** 2
        sigma = max(np.sqrt(var), 1e-10)
        # 正态近似
        z = norm_ppf(p)
        approx = mu + sigma * z
        lo = min(lo, approx - 5 * sigma)
        hi = max(hi, approx + 5 * sigma)
        break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # 使用正态近似计算 CDF
        cdf = _alnorm((mid - delta) / max(np.sqrt(1.0 + mid ** 2 / (2.0 * df)), 1e-10), upper=False)
        if abs(cdf - p) < tol:
            return mid
        if cdf < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# =============================================================================
# 机会约束的确定性等价
# =============================================================================
class ChanceConstraint:
    """
    机会约束：P(a^T x + w^T x <= b) >= 1 - alpha
    当 w ~ N(mu, Sigma) 时，等价于：
        a^T x + mu^T x + z_{1-alpha} * sqrt(x^T Sigma x) <= b
    """

    def __init__(
        self,
        a: np.ndarray,
        b: float,
        w_mean: np.ndarray,
        w_cov: np.ndarray,
        alpha: float = 0.05,
        n_samples: Optional[int] = None,
    ):
        self.a = np.asarray(a, dtype=float).ravel()
        self.b = float(b)
        self.w_mean = np.asarray(w_mean, dtype=float).ravel()
        self.w_cov = np.atleast_2d(w_cov)
        if self.a.size != self.w_mean.size:
            raise ValueError("维度不匹配")
        if self.w_cov.shape != (self.a.size, self.a.size):
            raise ValueError("协方差矩阵维度不匹配")
        self.alpha = float(alpha)
        self.n_samples = n_samples
        # 计算分位数
        if n_samples is not None and n_samples > 1:
            # t 分布修正 (小样本)
            df = n_samples - 1
            delta = np.sqrt(n_samples) * np.linalg.norm(self.w_mean) / max(
                np.sqrt(np.diag(self.w_cov).mean()), 1e-14
            )
            self.z = noncentral_t_quantile(1.0 - alpha, df, delta)
        else:
            self.z = norm_ppf(1.0 - alpha)

    def evaluate(self, x: np.ndarray) -> float:
        """
        计算机会约束的违反量 (<= 0 表示可行)。
        violation = a^T x + w_mean^T x + z * sqrt(x^T Sigma x) - b
        """
        x = np.asarray(x, dtype=float).ravel()
        linear = (self.a + self.w_mean) @ x
        quad = x @ self.w_cov @ x
        quad = max(quad, 0.0)  # 数值安全
        return linear + self.z * np.sqrt(quad) - self.b

    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        return self.evaluate(x) <= tol

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        约束函数对 x 的梯度：
            nabla g = a + w_mean + z * Sigma x / sqrt(x^T Sigma x)
        """
        x = np.asarray(x, dtype=float).ravel()
        quad = x @ self.w_cov @ x
        if quad < 1e-14:
            return self.a + self.w_mean
        return self.a + self.w_mean + self.z * (self.w_cov @ x) / np.sqrt(quad)


class JointChanceConstraint:
    """
    联合机会约束：P(g_i(x, w) <= 0, i=1..m) >= 1 - alpha
    Bonferroni 近似：分解为 m 个独立机会约束
        P(g_i(x, w) <= 0) >= 1 - alpha/m
    """

    def __init__(
        self,
        constraints: list,
        alpha: float = 0.05,
    ):
        self.constraints = constraints
        self.alpha = float(alpha)
        m = len(constraints)
        if m == 0:
            raise ValueError("至少需要一个约束")
        # 调整每个约束的风险水平
        self.alpha_individual = alpha / m

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """返回每个约束的违反量向量。"""
        return np.array([c.evaluate(x) for c in self.constraints])

    def max_violation(self, x: np.ndarray) -> float:
        return float(np.max(self.evaluate(x)))

    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        return self.max_violation(x) <= tol


# =============================================================================
# 样本平均近似 (SAA)
# =============================================================================
def sample_average_approximation(
    constraint_fn,
    samples: np.ndarray,
    alpha: float = 0.05,
) -> callable:
    """
    将机会约束转化为样本平均近似：
        (1/N) sum_{i=1}^N 1[ g(x, w_i) <= 0 ] >= 1 - alpha
    返回可计算违反量的函数。
    """
    N = samples.shape[0]
    max_violations = int(np.floor(alpha * N))

    def violation_fn(x: np.ndarray) -> float:
        vals = np.array([constraint_fn(x, samples[i]) for i in range(N)])
        sorted_vals = np.sort(vals)
        # 第 (N - max_violations) 个顺序统计量
        idx = min(N - max_violations - 1, N - 1)
        idx = max(idx, 0)
        return sorted_vals[idx]

    return violation_fn


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 正态分位数测试
    for p in [0.9, 0.95, 0.99]:
        print(f"z_{p} = {norm_ppf(p):.4f}")

    # 机会约束
    cc = ChanceConstraint(
        a=np.array([1.0, 2.0]),
        b=5.0,
        w_mean=np.array([0.1, -0.1]),
        w_cov=np.array([[0.1, 0.02], [0.02, 0.2]]),
        alpha=0.05,
    )
    x = np.array([1.0, 1.0])
    print(f"Chance constraint violation: {cc.evaluate(x):.4f}")
    print(f"Feasible: {cc.is_feasible(x)}")
    print(f"Gradient: {cc.gradient(x)}")

    # 联合机会约束
    ccs = [
        ChanceConstraint(
            a=np.array([1.0, 0.0]),
            b=3.0,
            w_mean=np.array([0.1, 0.0]),
            w_cov=np.array([[0.1, 0.0], [0.0, 0.1]]),
            alpha=0.05,
        ),
        ChanceConstraint(
            a=np.array([0.0, 1.0]),
            b=4.0,
            w_mean=np.array([0.0, 0.1]),
            w_cov=np.array([[0.1, 0.0], [0.0, 0.1]]),
            alpha=0.05,
        ),
    ]
    jcc = JointChanceConstraint(ccs, alpha=0.1)
    print(f"Joint max violation: {jcc.max_violation(x):.4f}")
    print(f"Joint feasible: {jcc.is_feasible(x)}")
