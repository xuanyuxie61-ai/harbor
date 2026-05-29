"""
numeric_utils.py
数值工具与鲁棒性模块

融入种子项目:
  - 961_r8_scale: 浮点精度控制与机器精度边界
  - 1142_square_distance: 统计量计算与距离分布

功能:
  - 机器精度边界检测
  - 数值稳定性检查
  - 统计矩计算
  - 单位正方形内随机距离分布的解析公式
"""

import numpy as np
from typing import Tuple, Optional


def nextafter(x: float, direction: str = "up") -> float:
    """
    返回给定浮点数在指定方向上的相邻可表示浮点数。
    对应 r8_next / r8_previous 的功能。

    数学上，对于浮点数 x，其 ULP (Unit in the Last Place) 满足:
        nextafter(x) = x + ulp(x)

    参数:
        x: 输入浮点数
        direction: "up" 或 "down"

    返回:
        相邻浮点数
    """
    if direction == "up":
        return np.nextafter(x, np.inf)
    else:
        return np.nextafter(x, -np.inf)


def machine_epsilon() -> float:
    """
    返回双精度机器精度 epsilon。

    IEEE 754 双精度浮点数中:
        eps = 2^{-52} \\approx 2.22 \\times 10^{-16}
    """
    return np.finfo(float).eps


def safe_divide(a: np.ndarray, b: np.ndarray, tol: float = None) -> np.ndarray:
    """
    安全除法，避免除以零。

    参数:
        a: 被除数数组
        b: 除数数组
        tol: 最小容差，默认 machine_epsilon * 10

    返回:
        a / b，其中 |b| < tol 时返回符号处理后的 large 值
    """
    if tol is None:
        tol = machine_epsilon() * 10.0
    b_safe = np.where(np.abs(b) < tol, np.sign(b + tol) * tol, b)
    return a / b_safe


def condition_number_check(A: np.ndarray, threshold: float = 1e12) -> bool:
    """
    检查矩阵条件数是否超过阈值，确保数值稳定性。

    对于线性系统 A x = b，相对误差满足:
        \\frac{\|\\delta x\|}{\|x\|} \\le \\kappa(A) \\frac{\|\\delta b\|}{\|b\|}

    其中 \\kappa(A) = \|A\| \|A^{-1}\| 为条件数。
    """
    cond = np.linalg.cond(A)
    return cond < threshold


def square_distance_pdf(d: np.ndarray) -> np.ndarray:
    """
    计算单位正方形内两随机点距离的解析概率密度函数 (PDF)。
    来源于 1142_square_distance 的核心公式。

    对于距离 r \\in [0, \\sqrt{2}]，PDF 为分段函数:

    当 0 \\le r \\le 1 时:
        f(r) = 2r \\\left( r^2 - 4r + \\pi \\right)

    当 1 < r \\le \\sqrt{2} 时:
        f(r) = 2r \\\left[ 4\\sqrt{r^2 - 1} - (r^2 + 2 - \\pi) - 4\\arctan(\\sqrt{r^2 - 1}) \\right]

    参数:
        d: 距离数组

    返回:
        PDF 值数组
    """
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    mask1 = (d >= 0.0) & (d <= 1.0)
    mask2 = (d > 1.0) & (d <= np.sqrt(2.0))

    pdf[mask1] = 2.0 * d[mask1] * (d[mask1]**2 - 4.0 * d[mask1] + np.pi)

    sqrt_term = np.sqrt(d[mask2]**2 - 1.0)
    pdf[mask2] = 2.0 * d[mask2] * (
        4.0 * sqrt_term
        - (d[mask2]**2 + 2.0 - np.pi)
        - 4.0 * np.arctan(sqrt_term)
    )
    return pdf


def square_distance_cdf(r: float) -> float:
    """
    单位正方形内两随机点距离的累积分布函数 (CDF)。

    通过积分 PDF 得到，用于概率采样和统计分析。
    """
    if r <= 0.0:
        return 0.0
    if r >= np.sqrt(2.0):
        return 1.0

    # 使用数值积分计算 CDF
    n_points = 1000
    d_samples = np.linspace(0.0, r, n_points)
    pdf_vals = square_distance_pdf(d_samples)
    cdf = np.trapezoid(pdf_vals, d_samples)
    # 归一化（理论积分应为1，但数值积分有误差）
    d_full = np.linspace(0.0, np.sqrt(2.0), n_points)
    pdf_full = square_distance_pdf(d_full)
    total = np.trapezoid(pdf_full, d_full)
    return cdf / total if total > 0 else 0.0


def compute_moments(samples: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算样本的统计矩: 均值、方差、偏度、峰度。

    对于样本 {x_i}_{i=1}^N:
        \\mu = \\frac{1}{N} \\\sum_{i=1}^N x_i
        \\sigma^2 = \\frac{1}{N-1} \\\sum_{i=1}^N (x_i - \\mu)^2
        \\gamma_1 = \\frac{1}{N} \\\sum_{i=1}^N \\\left(\\frac{x_i - \\mu}{\\sigma}\\right)^3
        \\gamma_2 = \\frac{1}{N} \\\sum_{i=1}^N \\\left(\\frac{x_i - \\mu}{\\sigma}\\right)^4 - 3

    参数:
        samples: 样本数组

    返回:
        (mean, variance, skewness, kurtosis)
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    mu = np.mean(samples)
    if n < 2:
        return mu, 0.0, 0.0, 0.0

    var = np.var(samples, ddof=1)
    std = np.sqrt(var) if var > 0 else 1.0

    skew = np.mean(((samples - mu) / std) ** 3) if std > 0 else 0.0
    kurt = np.mean(((samples - mu) / std) ** 4) - 3.0 if std > 0 else 0.0

    return float(mu), float(var), float(skew), float(kurt)


def chebyshev_bound(mean: float, std: float, k: float) -> float:
    """
    Chebyshev 不等式给出的概率上界。

    对于任意随机变量 X，有:
        P(|X - \\mu| \\ge k\\sigma) \\le \\frac{1}{k^2}

    参数:
        mean: 均值 \\mu
        std: 标准差 \\sigma
        k: 偏离倍数

    返回:
        概率上界
    """
    if k <= 0:
        return 1.0
    return 1.0 / (k ** 2)


def gershgorin_discs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gershgorin 圆盘定理：估计矩阵特征值范围。

    对于矩阵 A = (a_{ij})，第 i 个 Gershgorin 圆盘定义为:
        D_i = { z \\in \\mathbb{C} : |z - a_{ii}| \\le \\\sum_{j \\ne i} |a_{ij}| }

    所有特征值都落在这些圆盘的并集中。

    参数:
        A: 方阵

    返回:
        (centers, radii): 圆盘中心和半径
    """
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii


def is_diagonally_dominant(A: np.ndarray, strict: bool = True) -> bool:
    """
    检查矩阵是否对角占优。

    严格对角占优: |a_{ii}| > \\\sum_{j \\ne i} |a_{ij}| 对所有 i 成立
    弱对角占优: |a_{ii}| \\ge \\\sum_{j \\ne i} |a_{ij}| 对所有 i 成立

    参数:
        A: 方阵
        strict: 是否要求严格占优

    返回:
        是否对角占优
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(np.abs(A[i, :])) - diag
        if strict:
            if diag <= off_diag:
                return False
        else:
            if diag < off_diag:
                return False
    return True


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    计算线性系统的相对残差:
        \\eta = \\frac{\|b - Ax\|}{\|b\|}

    参数:
        A: 系数矩阵
        x: 解向量
        b: 右端项

    返回:
        相对残差
    """
    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        return np.linalg.norm(b - A @ x)
    return np.linalg.norm(b - A @ x) / norm_b
