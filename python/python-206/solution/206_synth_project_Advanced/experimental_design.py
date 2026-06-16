"""
experimental_design.py  --  序贯实验设计 (优惠券收集统计)
===============================================================
来源种子项目:
    449_full_deck_simulation : 全副牌问题 (coupon collector)
                                的统计特性: 期望、方差、蒙特卡洛.
科学问题角色:
    在贝叶斯校准中, 我们需在多个观测通道上分配有限实验预算.
    优惠券收集问题给出: 为覆盖全部 k 个通道, 期望采样次数
        E[T] = k * H_k = k * sum_{i=1}^k 1/i
    方差 Var[T] = k^2 * sum_{i=1}^k 1/i^2 - k * H_k + ...
    (Wilf 2006, SIAM Rev.)
    我们用此统计设计 *自适应采样策略*:
        1. 每轮选 *未被充分采样* 的通道.
        2. 预算耗尽后, 用覆盖度作为后验质量的代理.
核心公式:
    信息增益: IG(channel j) = H(y_j) - H(y_j | theta)
    贪心选择: j* = argmax_j IG(j) / cost(j)
    覆盖度: C = |{j : n_j >= threshold}| / k
"""
from __future__ import annotations
import math
from typing import List, Dict, Tuple
from numerical_base import NUMERICS


# ======================================================================
# 1. 优惠券收集理论 (Wilf 2006)
# ======================================================================
def harmonic_number(k: int) -> float:
    """H_k = sum_{i=1}^k 1/i."""
    return sum(1.0 / i for i in range(1, k + 1))


def coupon_collector_expected(k: int) -> float:
    """E[T] = k * H_k."""
    return k * harmonic_number(k)


def coupon_collector_variance(k: int) -> float:
    """
    Var[T] = k^2 * sum_{i=1}^k 1/i^2 - E[T] * (1 + o(1)).
    精确公式见 Wilf (2006).
    """
    s2 = sum(1.0 / (i * i) for i in range(1, k + 1))
    return k * k * s2 - coupon_collector_expected(k)


def coupon_collector_monte_carlo(k: int, n_trials: int = 500,
                                  seed: int = 0) -> Dict[str, float]:
    """
    蒙特卡洛模拟 coupon collector.
    返回 {min, max, mean, var}.
    """
    rng = _LCG(seed)
    results = []
    for _ in range(n_trials):
        collected = [False] * k
        n_collected = 0
        tries = 0
        while n_collected < k:
            tries += 1
            j = int(rng.next() * k) % k
            if not collected[j]:
                collected[j] = True
                n_collected += 1
        results.append(tries)
    rmin = min(results)
    rmax = max(results)
    rmean = sum(results) / len(results)
    rvar = sum((r - rmean) ** 2 for r in results) / max(1, len(results) - 1)
    return {"min": rmin, "max": rmax, "mean": rmean, "var": rvar}


# ======================================================================
# 2. 自适应实验设计
# ======================================================================
class SequentialDesign:
    """
    自适应采样策略: 在 k 个通道上分配 N 次观测.
    每轮选 *信息增益最大* 的通道.
    """
    def __init__(self, n_channels: int, total_budget: int,
                 threshold: int = 3, seed: int = 0):
        self.n_channels = n_channels
        self.total_budget = total_budget
        self.threshold = threshold
        self.counts = [0] * n_channels
        self._rng = _LCG(seed)
        self.history: List[Dict[str, float]] = []

    def _information_gain(self, j: int) -> float:
        """
        简化信息增益: IG(j) = log(1 + 1 / (counts[j] + 1)).
        采样越少, 增益越大.
        """
        return math.log(1.0 + 1.0 / (self.counts[j] + 1))

    def select_channel(self) -> int:
        """贪心选择信息增益最大的通道."""
        gains = [self._information_gain(j) for j in range(self.n_channels)]
        # 加小随机扰动打破平局
        for j in range(self.n_channels):
            gains[j] += 0.01 * self._rng.next()
        return gains.index(max(gains))

    def run_design(self) -> List[int]:
        """
        运行完整设计, 返回每个通道的采样次数.
        """
        for _ in range(self.total_budget):
            j = self.select_channel()
            self.counts[j] += 1
            # 记录覆盖度
            covered = sum(1 for c in self.counts if c >= self.threshold)
            coverage = covered / self.n_channels
            self.history.append({
                "step": _, "channel": j, "coverage": coverage
            })
        return self.counts

    def coverage(self) -> float:
        """当前覆盖度."""
        covered = sum(1 for c in self.counts if c >= self.threshold)
        return covered / self.n_channels

    def summary(self) -> Dict[str, float]:
        return {
            "counts": self.counts,
            "coverage": self.coverage(),
            "expected_theory": coupon_collector_expected(self.n_channels),
            "variance_theory": coupon_collector_variance(self.n_channels),
        }


# ======================================================================
# 3. 后验精度与采样预算关系
# ======================================================================
def posterior_variance_vs_budget(base_var: float, n_samples: int,
                                  effective_dim: int) -> List[float]:
    """
    后验方差随采样预算衰减:
        Var[theta | y_N] approx base_var / (1 + N / effective_dim)
    返回 [Var_0, Var_1, ..., Var_N].
    """
    out = []
    for N in range(n_samples + 1):
        v = base_var / (1.0 + N / max(1, effective_dim))
        out.append(v)
    return out


# ======================================================================
# 辅助
# ======================================================================
class _LCG:
    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 1 << 32

    def next(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m


__all__ = ["SequentialDesign", "coupon_collector_expected",
           "coupon_collector_variance", "coupon_collector_monte_carlo",
           "posterior_variance_vs_budget", "harmonic_number"]
