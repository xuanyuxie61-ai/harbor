"""
stability_analyzer.py — 数值稳定性、误差传播与动力学分析
=========================================================
融合来源:
  - 343_euler (Euler法截断误差分析)
  - 700_logistic_bifurcation (Logistic映射分岔与Lyapunov指数)
  - 1086_sir_ode (SIR模型稳定性与R0分析)

在外排序算法的数值实现中，稳定性分析确保：
  1. 采样和插值误差不会导致分区严重倾斜
  2. 迭代动力学（缓冲区管理、归并调度）不会进入混沌或发散区域
  3. 截断误差在可接受范围内
"""

import math
from typing import List, Tuple, Dict


class EulerErrorAnalysis:
    """
    显式Euler法的误差分析。

    对 ODE y' = f(t, y)，Euler 格式：
        y_{n+1} = y_n + h · f(t_n, y_n)

    局部截断误差（LTE）：
        τ_n = [y(t_{n+1}) - y(t_n)] / h - f(t_n, y(t_n))
            = (h/2) · y''(ξ_n) + O(h^2)

    全局误差（假设Lipschitz常数为L）：
        |e_n| ≤ (h · M_2 / 2L) · (e^{L·t_n} - 1)
    其中 M_2 = max|y''|。
    """

    def __init__(self, f, df_dt, df_dy, y0: float, t0: float, t_end: float):
        self.f = f
        self.df_dt = df_dt
        self.df_dy = df_dy
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end

    def local_truncation_error(self, h: float, t: float, y: float) -> float:
        """
        估计局部截断误差：
            τ ≈ (h/2) · |d²y/dt²|
        其中 d²y/dt² = ∂f/∂t + f · ∂f/∂y（全导数）。
        """
        d2y = self.df_dt(t, y) + self.f(t, y) * self.df_dy(t, y)
        return 0.5 * h * abs(d2y)

    def global_error_bound(self, h: float, L: float, M2: float) -> float:
        """
        全局误差上界估计。
        """
        if L < 1e-15:
            return 0.5 * h * M2 * self.t_end
        return (h * M2 / (2.0 * L)) * (math.exp(L * self.t_end) - 1.0)

    def recommend_step(self, target_error: float, L: float, M2: float) -> float:
        """
        根据目标全局误差推荐最大步长。
        """
        if L < 1e-15:
            return 2.0 * target_error / (M2 * self.t_end + 1e-15)
        denom = (M2 / (2.0 * L)) * (math.exp(L * self.t_end) - 1.0)
        if denom < 1e-15:
            return 1.0
        return target_error / denom


class LogisticBifurcationAnalyzer:
    """
    Logistic 映射分岔分析器。

    动力学：x_{n+1} = r · x_n · (1 - x_n)

    不动点分析：
        x* = 0 或 x* = 1 - 1/r
        稳定性要求 |f'(x*)| < 1，即 |r · (1 - 2x*)| < 1

    Lyapunov 指数（判断混沌）：
        λ = lim_{N→∞} (1/N) Σ_{n=0}^{N-1} ln|f'(x_n)|
        λ > 0 → 混沌（对初值敏感依赖）
        λ < 0 → 稳定/周期性
    """

    def __init__(self, r: float, x0: float = 0.5):
        self.r = r
        self.x0 = x0

    def fixed_points(self) -> List[float]:
        """
        计算不动点。
        """
        fps = [0.0]
        if self.r > 1.0:
            fps.append(1.0 - 1.0 / self.r)
        return fps

    def stability_of_fixed_point(self, x_star: float) -> float:
        """
        计算不动点处的导数值 f'(x*) = r · (1 - 2x*)。
        |f'(x*)| < 1 时稳定。
        """
        return self.r * (1.0 - 2.0 * x_star)

    def lyapunov_exponent(self, n_iter: int = 5000, n_transient: int = 1000) -> float:
        """
        数值估计Lyapunov指数。
        """
        x = self.x0
        # 瞬态丢弃
        for _ in range(n_transient):
            x = self.r * x * (1.0 - x)
            if x <= 0 or x >= 1:
                x = 0.5

        lam_sum = 0.0
        for _ in range(n_iter):
            x = self.r * x * (1.0 - x)
            if x <= 0 or x >= 1:
                x = 0.5
            deriv = abs(self.r * (1.0 - 2.0 * x))
            if deriv < 1e-15:
                deriv = 1e-15
            lam_sum += math.log(deriv)
        return lam_sum / n_iter

    def is_chaotic(self, threshold: float = 0.005) -> bool:
        """
        判断系统是否处于混沌区域。
        """
        return self.lyapunov_exponent() > threshold


class SIRStabilityAnalyzer:
    """
    SIR 模型的稳定性分析。

    无病平衡点 E0 = (N, 0, 0) 的稳定性由基本再生数决定：
        R0 = α / β

    若 R0 < 1：无病平衡点全局渐近稳定（数据传播自然消退）
    若 R0 > 1：存在地方病平衡点 E* = (S*, I*, R*)，其中
        S* = N / R0
        I* = (γ·N/β) · (R0 - 1)
    """

    def __init__(self, alpha: float, beta: float, gamma: float, N: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.N = N

    def reproduction_number(self) -> float:
        if self.beta < 1e-15:
            return float('inf')
        return self.alpha / self.beta

    def endemic_equilibrium(self) -> Tuple[float, float, float]:
        """
        计算地方病平衡点（仅当 R0 > 1 时存在物理意义）。

        对于 SIRS 模型（γ > 0），地方病平衡点满足：
            S* = βN / α = N / R0
            I* = γ(N - S*) / (β + γ) = Nγ(R0 - 1) / (R0(β + γ))
            R* = N - S* - I*
        """
        r0 = self.reproduction_number()
        if r0 <= 1.0:
            return (self.N, 0.0, 0.0)
        S_star = self.N / r0
        I_star = self.gamma * (self.N - S_star) / (self.beta + self.gamma)
        R_star = max(self.N - S_star - I_star, 0.0)
        return (S_star, I_star, R_star)

    def jacobian_eigenvalues(self, S: float, I: float) -> Tuple[complex, complex, complex]:
        """
        在平衡点 (S, I, R) 处计算Jacobian矩阵的特征值。

        Jacobian J =
            | -αI/N - γ    -αS/N + γ    γ       |
            |  αI/N         αS/N - β    0       |
            |  0            β          -γ       |
        """
        a11 = -self.alpha * I / self.N - self.gamma
        a12 = -self.alpha * S / self.N + self.gamma
        a13 = self.gamma
        a21 = self.alpha * I / self.N
        a22 = self.alpha * S / self.N - self.beta
        a23 = 0.0
        a31 = 0.0
        a32 = self.beta
        a33 = -self.gamma

        # 解析计算3x3矩阵特征值较复杂，此处计算迹和判别式
        trace = a11 + a22 + a33
        # 简化：返回近似特征值（实际应解三次方程）
        # 使用粗略估计
        det2 = a11 * a22 - a12 * a21
        disc = trace * trace - 4.0 * det2
        if disc >= 0:
            l1 = (-trace + math.sqrt(disc)) / 2.0
            l2 = (-trace - math.sqrt(disc)) / 2.0
        else:
            l1 = complex(-trace / 2.0, math.sqrt(-disc) / 2.0)
            l2 = complex(-trace / 2.0, -math.sqrt(-disc) / 2.0)
        l3 = a33  # 简化近似
        return (l1, l2, l3)


class SortStabilityMonitor:
    """
    外排序算法的运行时稳定性监控。

    监控指标：
        1. 分区倾斜度（熵值偏离度）
        2. 归并段长度波动系数
        3. 缓冲区溢出风险
    """

    def __init__(self, target_partitions: int, memory_size: int):
        self.P = target_partitions
        self.M = memory_size

    def partition_skew(self, partition_sizes: List[int]) -> float:
        """
        计算分区倾斜度：
            skew = max_size / mean_size - 1
        skew = 0 表示完全均衡。
        """
        if not partition_sizes:
            return 0.0
        mean_size = sum(partition_sizes) / len(partition_sizes)
        if mean_size < 1e-15:
            return 0.0
        max_size = max(partition_sizes)
        return max_size / mean_size - 1.0

    def run_length_cv(self, run_lengths: List[int]) -> float:
        """
        归并段长度的变异系数（CV）。
        """
        if len(run_lengths) < 2:
            return 0.0
        mean_len = sum(run_lengths) / len(run_lengths)
        if mean_len < 1e-15:
            return 0.0
        var = sum((l - mean_len) ** 2 for l in run_lengths) / len(run_lengths)
        std = math.sqrt(var)
        return std / mean_len

    def overflow_risk(self, current_buffer: int, predicted_peak: float) -> float:
        """
        缓冲区溢出风险概率估计（简化为逻辑斯蒂函数）：
            risk = 1 / (1 + exp( -k · (predicted_peak - M) / M ))
        """
        if self.M < 1:
            return 1.0
        ratio = (predicted_peak - self.M) / self.M
        k = 5.0
        return 1.0 / (1.0 + math.exp(-k * ratio))

    def diagnose(self, partition_sizes: List[int], run_lengths: List[int],
                 predicted_peak: float) -> Dict[str, float]:
        """
        综合诊断，返回各指标字典。
        """
        return {
            "partition_skew": self.partition_skew(partition_sizes),
            "run_length_cv": self.run_length_cv(run_lengths),
            "overflow_risk": self.overflow_risk(0, predicted_peak),
            "stability_score": max(0.0, 1.0 - self.partition_skew(partition_sizes)
                                          - 0.5 * self.run_length_cv(run_lengths)
                                          - self.overflow_risk(0, predicted_peak))
        }
