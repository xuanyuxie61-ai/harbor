"""
utils.py
========
数值工具与鲁棒性保障模块。
融合来源：r8ss（skyline矩阵格式）、histogram_display（数据解析与统计框架）

本模块提供：
- 高精度浮点边界判定
- 安全数值运算（避免除零、溢出）
- 向量/矩阵统计直方图（非可视化，纯数值统计）
- Skyline 对称稀疏矩阵的压缩存储与矩阵-向量乘法
"""

import numpy as np
from typing import Tuple, List


EPS_MACHINE = np.finfo(float).eps
EPS_SQRT = np.sqrt(EPS_MACHINE)


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """安全除法，避免除零。"""
    if np.abs(b) < EPS_MACHINE:
        return fallback
    return a / b


def safe_sqrt(x: float) -> float:
    """安全开方，负值返回0。"""
    if x < 0.0:
        if x > -EPS_MACHINE:
            return 0.0
        raise ValueError(f"safe_sqrt: negative argument {x}")
    return np.sqrt(x)


def clip_spin_norm(s: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """
    将自旋向量裁剪到目标范数，保证数值稳定性。
    Heisenberg 自旋必须满足 |S| = S，此处默认 S=1。
    """
    norm = np.linalg.norm(s)
    if norm < EPS_MACHINE:
        # 退化情况：返回沿 z 轴的单位向量
        out = np.zeros_like(s)
        out[2] = target_norm
        return out
    return s * (target_norm / norm)


def rms_norm(v: np.ndarray) -> float:
    """均方根范数，融合自 laplacian_matrix 的 rms_norm 测试思想。"""
    n = v.size
    if n == 0:
        return 0.0
    return np.sqrt(np.sum(v ** 2) / n)


def skyline_mv(n: int, diag: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Skyline 对称稀疏矩阵与向量乘法。
    融合来源：r8ss（R8SS 存储格式）。

    对于对称矩阵 M，仅存储每列从第一个非零元到对角线的元素。
    diag[j] 给出第 j 列对角元在 a 中的索引（0-based）。

    参数
    ----
    n : int
        矩阵阶数。
    diag : np.ndarray, shape (n,)
        对角元在 a 中的索引。
    a : np.ndarray
        压缩存储的一维数组。
    x : np.ndarray, shape (n,)
        乘数向量。

    返回
    ----
    y : np.ndarray, shape (n,)
        结果向量 y = M * x。
    """
    y = np.zeros(n, dtype=float)
    for j in range(n):
        dj = diag[j]
        # 列 j 的高度（包含对角元）
        if j == 0:
            height = 1
        else:
            height = dj - diag[j - 1]
        i_start = j - height + 1
        for idx in range(height):
            i = i_start + idx
            val = a[dj - height + 1 + idx]
            y[i] += val * x[j]
            if i != j:
                y[j] += val * x[i]
    return y


def build_skyline_from_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    将三对角对称矩阵转换为 skyline 格式。
    对于三对角矩阵，skyline 每列高度为 2（首列除外），高度规则。
    """
    n = diag.size
    na = 2 * n - 1
    a = np.zeros(na, dtype=float)
    diag_idx = np.zeros(n, dtype=int)
    pos = 0
    for j in range(n):
        if j > 0:
            a[pos] = lower[j - 1]
            pos += 1
        a[pos] = diag[j]
        diag_idx[j] = pos
        pos += 1
    return na, diag_idx, a


def histogram_stats_1d(data: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    一维数值直方图统计（纯数值，无可视化）。
    融合来源：histogram_display（数据列统计思想）。

    返回
    ----
    counts : np.ndarray
        每个区间的计数。
    edges : np.ndarray
        区间边界。
    stats : dict
        包含 min, max, mean, variance, skewness, kurtosis。
    """
    if data.size == 0:
        counts = np.zeros(bins, dtype=int)
        edges = np.linspace(0.0, 1.0, bins + 1)
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        return counts, edges, stats

    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if dmax - dmin < EPS_MACHINE:
        dmax = dmin + 1.0

    edges = np.linspace(dmin, dmax, bins + 1)
    counts, _ = np.histogram(data, bins=edges)

    mean = float(np.mean(data))
    variance = float(np.var(data))
    std = safe_sqrt(variance)
    if std < EPS_MACHINE:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.mean(((data - mean) / std) ** 3))
        kurtosis = float(np.mean(((data - mean) / std) ** 4)) - 3.0

    stats = {
        "min": dmin,
        "max": dmax,
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    return counts, edges, stats


def triangle_area_histogram_2d(
    points: np.ndarray, n_sub: int = 5
) -> Tuple[np.ndarray, dict]:
    """
    在单位三角形 ((0,0),(1,0),(0,1)) 内对点集进行子三角形区域直方图统计。
    融合来源：triangle_histogram（子三角形划分与重心坐标映射）。

    参数
    ----
    points : np.ndarray, shape (N, 2)
        位于单位三角形内的二维点。
    n_sub : int
        每条边的细分数。

    返回
    ----
    histo : np.ndarray
        每个子三角形的计数。
    info : dict
        统计摘要。
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    # 排除单位三角形外的点
    valid = (
        (points[:, 0] >= -EPS_MACHINE)
        & (points[:, 1] >= -EPS_MACHINE)
        & (points[:, 0] + points[:, 1] <= 1.0 + EPS_MACHINE)
    )
    pts = points[valid]
    n_points = pts.shape[0]

    sub_num = n_sub * n_sub
    histo = np.zeros(sub_num + 1, dtype=int)

    for p in pts:
        x, y = p
        # 映射到子三角形索引（简化版：基于规则网格的重心坐标）
        i = int(np.floor(x * n_sub))
        j = int(np.floor(y * n_sub))
        k = int(np.floor((1.0 - x - y) * n_sub))
        i = max(0, min(n_sub - 1, i))
        j = max(0, min(n_sub - 1, j))
        k = max(0, min(n_sub - 1, k))

        # 使用线性化索引近似
        t = i + j * n_sub
        if t < 0 or t >= sub_num:
            t = sub_num
        histo[t] += 1

    histo_ave = np.sum(histo[:sub_num]) / max(sub_num, 1)
    histo_max = int(np.max(histo[:sub_num])) if sub_num > 0 else 0
    histo_min = int(np.min(histo[:sub_num])) if sub_num > 0 else 0
    histo_var = float(np.var(histo[:sub_num])) if sub_num > 0 else 0.0

    info = {
        "n_points": n_points,
        "n_sub": n_sub,
        "sub_num": sub_num,
        "min": histo_min,
        "max": histo_max,
        "average": histo_ave,
        "variance": histo_var,
        "out_of_range": int(histo[sub_num]),
    }
    return histo[:sub_num], info


def is_prime(n: int) -> bool:
    """素性检测，融合来源：prime_plot。"""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(np.sqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def primes_up_to(n_max: int) -> List[int]:
    """返回不超过 n_max 的所有素数列表。"""
    if n_max < 2:
        return []
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(np.sqrt(n_max)) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.nonzero(sieve)[0].tolist()
