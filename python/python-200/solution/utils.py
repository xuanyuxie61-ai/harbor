"""
utils.py
========
通用工具函数：计时、直方图统计、文件解析、数据验证。

融合 wtime、histogram_display、calendar_nyt 等项目的核心功能，
转化为科学计算辅助工具。

核心数学公式
------------
直方图统计量：
    均值: μ = Σ x_i / N
    方差: σ² = Σ (x_i - μ)² / (N-1)
    偏度: γ₁ = (1/N) Σ [(x_i - μ)/σ]³
    峰度: γ₂ = (1/N) Σ [(x_i - μ)/σ]⁴ - 3

直方图熵（信息论）：
    H = -Σ p_i log₂(p_i)

性能计时（墙钟时间）：
    Δt = t₁ - t₀
    浮点运算速率 FLOPS = (操作数 × 频率) / Δt

 Julian Ephemeris Date（简化版）：
    JED = JD - 0.5
    用于模拟时间戳管理
"""

import time
import numpy as np
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# 计时工具（源自 wtime）
# ----------------------------------------------------------------------

class Timer:
    """高精度墙钟计时器。"""

    def __init__(self):
        self._start = None
        self._elapsed = 0.0
        self._running = False

    def start(self):
        if not self._running:
            self._start = time.perf_counter()
            self._running = True

    def stop(self):
        if self._running:
            self._elapsed += time.perf_counter() - self._start
            self._running = False
        return self._elapsed

    def reset(self):
        self._elapsed = 0.0
        self._start = None
        self._running = False

    @property
    def elapsed(self) -> float:
        if self._running:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def benchmark_function(func, *args, n_runs: int = 3, **kwargs) -> dict:
    """
    对函数进行多次运行并返回性能统计。
    
    返回:
        {'mean_time': ..., 'min_time': ..., 'max_time': ..., 'std_time': ...}
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        'mean_time': float(np.mean(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'std_time': float(np.std(times, ddof=1)),
        'result': result,
    }


# ----------------------------------------------------------------------
# 直方图与统计（源自 histogram_display）
# ----------------------------------------------------------------------

class HistogramStats:
    """一维数据直方图统计工具。"""

    def __init__(self, data: np.ndarray, n_bins: int = 20,
                 range_limits: Tuple[float, float] = None):
        self.data = np.asarray(data)
        self.n_bins = n_bins
        if range_limits is None:
            self.range = (float(np.min(data)), float(np.max(data)))
        else:
            self.range = range_limits

        self.counts, self.bin_edges = np.histogram(
            self.data, bins=n_bins, range=self.range
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]

    @property
    def probabilities(self) -> np.ndarray:
        """归一化概率密度。"""
        total = np.sum(self.counts)
        if total < 1:
            return np.zeros_like(self.counts, dtype=float)
        return self.counts.astype(float) / (total * self.bin_width)

    @property
    def entropy(self) -> float:
        """直方图香农熵（自然对数）。"""
        p = self.probabilities * self.bin_width
        p = p[p > 1e-30]
        return -np.sum(p * np.log(p))

    @property
    def mean(self) -> float:
        return float(np.mean(self.data))

    @property
    def variance(self) -> float:
        return float(np.var(self.data, ddof=1))

    @property
    def skewness(self) -> float:
        """偏度：衡量分布不对称性。"""
        mu = self.mean
        sigma = np.sqrt(self.variance)
        if sigma < 1e-30:
            return 0.0
        return float(np.mean(((self.data - mu) / sigma) ** 3))

    @property
    def kurtosis(self) -> float:
        """超额峰度：衡量分布尾部厚度。"""
        mu = self.mean
        sigma = np.sqrt(self.variance)
        if sigma < 1e-30:
            return 0.0
        return float(np.mean(((self.data - mu) / sigma) ** 4) - 3.0)

    def summary(self) -> dict:
        """返回完整统计摘要。"""
        return {
            'n_samples': len(self.data),
            'mean': self.mean,
            'variance': self.variance,
            'std': np.sqrt(self.variance),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'median': float(np.median(self.data)),
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'entropy': self.entropy,
            'n_bins': self.n_bins,
        }


def compute_cumulative_distribution(data: np.ndarray,
                                    n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算经验累积分布函数（ECDF）。
    
    F(x) = (1/N) Σ_i Θ(x - x_i)
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf


# ----------------------------------------------------------------------
# 数值鲁棒性工具
# ----------------------------------------------------------------------

def safe_divide(a: np.ndarray, b: np.ndarray,
                default: float = 0.0) -> np.ndarray:
    """安全除法：当除数接近零时返回默认值。"""
    result = np.full_like(a, default, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    return result


def clip_to_range(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """将数值裁剪到指定范围。"""
    return np.clip(x, low, high)


def relative_error(approx: float, exact: float) -> float:
    """计算相对误差（带边界保护）。"""
    if abs(exact) < 1e-30:
        return abs(approx)
    return abs((approx - exact) / exact)


def convergence_rate(errors: List[float]) -> List[float]:
    """
    从误差序列估计收敛速率。
    
    假设误差按 e_{n+1} = C · e_n^p 衰减，则
        p ≈ log(e_{n+1}/e_n) / log(e_n/e_{n-1})
    """
    rates = []
    for i in range(2, len(errors)):
        e0, e1, e2 = errors[i - 2], errors[i - 1], errors[i]
        if e0 < 1e-30 or e1 < 1e-30:
            rates.append(0.0)
        else:
            r = np.log(e2 / e1) / np.log(e1 / e0)
            rates.append(float(r))
    return rates


# ----------------------------------------------------------------------
# 时间戳管理（源自 calendar_nyt 的简化）
# ----------------------------------------------------------------------

def simulation_time_to_seconds(step: int, dt: float,
                                time_unit: str = "reduced") -> float:
    """
    将模拟步数转换为物理时间（简化版）。
    
    在自然单位制（reduced units）中：
        t* = t · √(ε/mσ²)
    
    对于氩（Ar）：
        ε/k_B ≈ 120 K, σ ≈ 3.4 Å, m ≈ 39.95 amu
        1 时间单位 ≈ 2.16 ps
    """
    t_reduced = step * dt
    if time_unit == "reduced":
        return t_reduced
    elif time_unit == "picoseconds":
        # 氩的转换因子
        return t_reduced * 2.16
    elif time_unit == "femtoseconds":
        return t_reduced * 2160.0
    else:
        return t_reduced


def format_time_interval(seconds: float) -> str:
    """将秒数格式化为人类可读字符串。"""
    if seconds < 1e-6:
        return f"{seconds*1e9:.3f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.3f} μs"
    elif seconds < 1.0:
        return f"{seconds*1e3:.3f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds/3600:.2f} h"


# ----------------------------------------------------------------------
# 矩阵工具
# ----------------------------------------------------------------------

def condition_number_estimate(A: np.ndarray) -> float:
    """估计矩阵条件数（使用 SVD）。"""
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[-1] < 1e-30:
        return 1e30
    return float(s[0] / s[-1])


def is_symmetric_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    """检查矩阵是否对称正定。"""
    if A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.all(eigvals > -tol)
    except np.linalg.LinAlgError:
        return False


def safe_cholesky(A: np.ndarray) -> Optional[np.ndarray]:
    """安全的 Cholesky 分解（失败时返回 None）。"""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
