# -*- coding: utf-8 -*-
"""
convergence_analysis.py
=======================

SAA 收敛性分析模块.

核心内容:
  1. 经验收敛速率估计 (对 N 个样本量的 F_N 误差)
  2. 置信区间宽度随 N 的变化
  3. 样本复杂度预算 (达到 epsilon 精度需要的 N)
  4. 运行平均的方差缩减效果

数学基础 (Shapiro, Dentcheva, Ruszczynski 2021):
  - LLD (Law of Large Numbers):  F_N(x) -> F(x) a.s.
  - CLT:  sqrt(N)(F_N(x*) - F(x*)) -> N(0, sigma^2(x*))
  - 均匀收敛:  sup_{x in X} |F_N(x) - F(x)| -> 0 a.s.
  - 优化解收敛:  x_N -> x* a.s.  (在适当条件下)
"""

import math
from typing import List, Tuple, Dict
from saa_objective import RunningStatistics
from scientific_formulas import (
    saa_statistical_error_bound,
    sgd_convergence_rate,
    variance_reduction_factor
)


class ConvergenceAnalyzer:
    """收敛分析器."""

    def __init__(self):
        self.sample_sizes: List[int] = []
        self.objective_values: List[float] = []
        self.std_values: List[float] = []
        self.ci_widths: List[float] = []

    def add_observation(self, N: int, stats: RunningStatistics) -> None:
        """添加一个 (样本量, 统计量) 观测."""
        self.sample_sizes.append(N)
        self.objective_values.append(stats.mean())
        self.std_values.append(stats.std())
        lo, hi = stats.confidence_interval(0.95)
        self.ci_widths.append(hi - lo)

    def empirical_rate(self) -> float:
        """估计经验收敛速率 (log-log 回归斜率).

        假设 CI 宽度 w(N) ~ C * N^{-alpha}, 则
            log w = log C - alpha log N
        alpha 的 OLS 估计 = -Cov(log N, log w) / Var(log N).

        理论值: alpha = 0.5 (标准 SAA).
        """
        if len(self.sample_sizes) < 2:
            return 0.5  # 默认
        log_n = [math.log(max(N, 1)) for N in self.sample_sizes]
        log_w = [math.log(max(w, 1e-15)) for w in self.ci_widths]
        n = len(log_n)
        x_mean = sum(log_n) / n
        y_mean = sum(log_w) / n
        num = sum((log_n[i] - x_mean) * (log_w[i] - y_mean)
                  for i in range(n))
        den = sum((log_n[i] - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-14:
            return 0.5
        slope = num / den
        return -slope  # 期望为正

    def sample_complexity_budget(self, target_eps: float,
                                 sigma_est: float,
                                 confidence: float = 0.95) -> int:
        """达到目标精度 epsilon 所需的样本量.

        N >= (z_{alpha/2} * sigma / epsilon)^2

        这是 SAA 理论的核心样本复杂度公式.
        """
        z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.96)
        N = math.ceil((z * sigma_est / max(target_eps, 1e-10)) ** 2)
        return max(N, 4)

    def summarize(self) -> Dict:
        """汇总收敛分析结果."""
        return {
            "sample_sizes": list(self.sample_sizes),
            "objective_values": list(self.objective_values),
            "std_values": list(self.std_values),
            "ci_widths": list(self.ci_widths),
            "empirical_rate": self.empirical_rate(),
            "theoretical_rate": 0.5,
        }


class SGATrajectoryAnalyzer:
    """SGD 轨迹分析器."""

    def __init__(self):
        self.iterates: List[List[float]] = []
        self.objectives: List[float] = []
        self.step_sizes: List[float] = []
        self.grad_norms: List[float] = []

    def add_step(self, x: List[float], f: float,
                 eta: float, grad: List[float]) -> None:
        self.iterates.append(list(x))
        self.objectives.append(f)
        self.step_sizes.append(eta)
        g_norm = math.sqrt(sum(g * g for g in grad))
        self.grad_norms.append(g_norm)

    def polyak_ruppert_average(self) -> List[float]:
        """Polyak-Ruppert 平均: x_bar_T = (1/T) sum_{t=1}^T x_t."""
        if not self.iterates:
            return []
        dim = len(self.iterates[0])
        T = len(self.iterates)
        avg = [0.0] * dim
        for x in self.iterates:
            for k in range(dim):
                avg[k] += x[k]
        return [a / T for a in avg]

    def tail_average(self, fraction: float = 0.3) -> List[float]:
        """尾部平均: 取最后 fraction 比例的迭代求平均.
        这通常比全序列平均更有效 (Burn-in 后).
        """
        if not self.iterates:
            return []
        T = len(self.iterates)
        start = max(0, T - int(T * fraction))
        dim = len(self.iterates[0])
        n_tail = T - start
        if n_tail <= 0:
            return list(self.iterates[-1])
        avg = [0.0] * dim
        for i in range(start, T):
            for k in range(dim):
                avg[k] += self.iterates[i][k]
        return [a / n_tail for a in avg]

    def estimate_convergence_rate(self) -> float:
        """估计 f(x_t) - f* 的收敛速率 (假设最后 10% 的均值近似 f*)."""
        if len(self.objectives) < 10:
            return 1.0
        f_star_est = sum(self.objectives[-max(1, len(self.objectives) // 10):]) \
                     / max(1, len(self.objectives) // 10)
        # 对前 80% 的点拟合 log(f - f*) vs log t 的斜率
        pts = []
        for t, f in enumerate(self.objectives[:int(0.8 * len(self.objectives))]):
            gap = f - f_star_est
            if gap > 1e-10 and t > 0:
                pts.append((math.log(t + 1), math.log(gap)))
        if len(pts) < 3:
            return 1.0
        n = len(pts)
        x_mean = sum(p[0] for p in pts) / n
        y_mean = sum(p[1] for p in pts) / n
        num = sum((p[0] - x_mean) * (p[1] - y_mean) for p in pts)
        den = sum((p[0] - x_mean) ** 2 for p in pts)
        if abs(den) < 1e-14:
            return 1.0
        return -num / den

    def summarize(self) -> Dict:
        return {
            "n_iter": len(self.iterates),
            "f_initial": self.objectives[0] if self.objectives else 0.0,
            "f_final": self.objectives[-1] if self.objectives else 0.0,
            "f_best": min(self.objectives) if self.objectives else 0.0,
            "f_mean": (sum(self.objectives) / len(self.objectives)
                       if self.objectives else 0.0),
            "grad_norm_initial": self.grad_norms[0] if self.grad_norms else 0.0,
            "grad_norm_final": self.grad_norms[-1] if self.grad_norms else 0.0,
            "polyak_ruppert_avg": self.polyak_ruppert_average(),
            "tail_average": self.tail_average(0.3),
            "empirical_rate": self.estimate_convergence_rate(),
        }


class VarianceReductionAnalyzer:
    """方差缩减效果分析.

    比较:
      1. 纯 MC (iid): Var(F_N) = sigma^2 / N
      2. 排列洗牌: 略优于 MC (无放回 vs 有放回)
      3. 椭球采样: O(1/N) 低差异, 常数更小
      4. 相关样本 (CRN): Var = sigma^2 * (1 + (B-1)rho) / (N*B)
    """

    @staticmethod
    def compute_vr_factors(batch_sizes: List[int],
                           correlations: List[float]) -> Dict:
        """对每个 (batch_size, correlation) 计算方差缩减因子."""
        results = {}
        for bs in batch_sizes:
            for corr in correlations:
                vr = variance_reduction_factor(bs, corr)
                results[(bs, corr)] = vr
        return results

    @staticmethod
    def effective_sample_size(N: int, batch_size: int,
                              corr: float) -> float:
        """有效样本量: N_eff = N / VR_factor.
        当 corr=0, N_eff = N.
        当 corr>0, N_eff < N (相关样本浪费信息).
        """
        vr = variance_reduction_factor(batch_size, corr)
        if vr < 1e-12:
            return float('inf')
        return N / (vr * batch_size)
