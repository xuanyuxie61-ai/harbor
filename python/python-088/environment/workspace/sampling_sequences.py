"""
sampling_sequences.py
低差异序列与准蒙特卡洛采样模块

融入种子项目:
  - 498_hammersley: Hammersley 低差异序列
  - 1142_square_distance: 单位正方形采样与统计

功能:
  - Hammersley 序列生成 (基于素数反演)
  - Halton 序列
  - 拉丁超立方采样 (LHS)
  - 准蒙特卡洛积分
"""

import numpy as np
from typing import Tuple


# 前 100 个素数，用于 Hammersley 和 Halton 序列
_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def _radical_inverse(n: int, base: int) -> float:
    """
    计算 n 在 base 进制下的 radical inverse (van der Corput 序列)。

    对于整数 n，将其表示为 base 进制:
        n = \\\sum_{k=0}^{\\\infty} a_k b^k

    则 radical inverse 为:
        \\phi_b(n) = \\\sum_{k=0}^{\\\infty} a_k b^{-(k+1)}

    参数:
        n: 非负整数索引
        base: 基 (素数)

    返回:
        [0, 1) 内的低差异数
    """
    n = int(abs(n))
    base = int(base)
    if base < 2:
        raise ValueError("Base must be >= 2")
    result = 0.0
    inv_base = 1.0 / base
    factor = inv_base
    while n > 0:
        result += (n % base) * factor
        n //= base
        factor *= inv_base
    return result


def hammersley_sequence(i1: int, i2: int, m: int, n_base: int = 1) -> np.ndarray:
    """
    计算 Hammersley 序列的第 i1 到 i2 项。

    Hammersley 序列在 M 维空间中的第 i 个点定义为:
        x_i = ( i/N, \\phi_{p_1}(i), \\phi_{p_2}(i), ..., \\phi_{p_{M-1}}(i) )

    其中 N 为采样总数，\\phi_{p_k} 为第 k 个素数基下的 radical inverse。

    与纯随机采样相比，Hammersley 序列的星偏差 (star discrepancy)
    满足 D_N^* = O(N^{-1} (\\log N)^{M-1})，远低于随机采样的 O(N^{-1/2})。

    参数:
        i1: 起始索引 (>= 0)
        i2: 结束索引
        m: 空间维度 (1 <= m <= 100)
        n_base: 第一分母的基数

    返回:
        形状为 (m, |i2-i1|+1) 的数组
    """
    if m < 1 or m > 100:
        raise ValueError("Dimension m must be in [1, 100]")
    if i1 < 0 or i2 < 0:
        raise ValueError("Indices must be non-negative")
    if n_base < 1:
        raise ValueError("n_base must be >= 1")

    step = 1 if i1 <= i2 else -1
    length = abs(i2 - i1) + 1
    r = np.zeros((m, length))

    k = 0
    for i in range(i1, i2 + step, step):
        r[0, k] = (i % (n_base + 1)) / n_base if n_base > 0 else 0.0
        for dim in range(1, m):
            r[dim, k] = _radical_inverse(i, int(_PRIMES[dim - 1]))
        k += 1

    return r


def halton_sequence(n_samples: int, m: int, skip: int = 0) -> np.ndarray:
    """
    生成 Halton 低差异序列。

    Halton 序列是 Hammersley 序列去掉第一个分量 (i/N) 的变体:
        x_i = ( \\phi_{p_1}(i), \\phi_{p_2}(i), ..., \\phi_{p_M}(i) )

    参数:
        n_samples: 样本数量
        m: 维度
        skip: 跳过的初始点数 (用于消除相关性)

    返回:
        形状为 (n_samples, m) 的数组
    """
    if m < 1 or m > 100:
        raise ValueError("Dimension m must be in [1, 100]")
    r = np.zeros((n_samples, m))
    for i in range(n_samples):
        idx = i + skip
        for dim in range(m):
            r[i, dim] = _radical_inverse(idx, int(_PRIMES[dim]))
    return r


def latin_hypercube_sampling(n_samples: int, m: int) -> np.ndarray:
    """
    拉丁超立方采样 (LHS)。

    将每个维度分成 n_samples 个等概率区间，在每个区间中随机采样一个点，
    然后通过随机置换确保一维投影的均匀性。

    参数:
        n_samples: 样本数
        m: 维度

    返回:
        形状为 (n_samples, m) 的数组
    """
    samples = np.zeros((n_samples, m))
    for dim in range(m):
        # 在每个区间内均匀随机采样
        perm = np.random.permutation(n_samples)
        samples[:, dim] = (perm + np.random.rand(n_samples)) / n_samples
    return samples


def quasi_monte_carlo_integral(
    f, dim: int, n_samples: int, domain: Tuple[float, float] = (0.0, 1.0),
    method: str = "hammersley"
) -> Tuple[float, float]:
    """
    使用准蒙特卡洛方法计算高维积分。

    对于积分:
        I = \\\\int_{[0,1]^d} f(x) dx

    QMC 估计为:
        \hat{I}_N = \\frac{1}{N} \\\sum_{i=1}^N f(x_i)

    其中 {x_i} 为低差异序列。误差界为 O(N^{-1} (\\log N)^{d-1})。

    参数:
        f: 被积函数，接受形状为 (n, dim) 的数组
        dim: 维度
        n_samples: 样本数
        domain: 积分区间 (a, b)
        method: "hammersley", "halton", "lhs", "random"

    返回:
        (估计值, 标准误差)
    """
    a, b = domain

    if method == "hammersley":
        points = hammersley_sequence(0, n_samples - 1, dim, n_base=n_samples)
        points = points.T
    elif method == "halton":
        points = halton_sequence(n_samples, dim, skip=100)
    elif method == "lhs":
        points = latin_hypercube_sampling(n_samples, dim)
    else:
        points = np.random.rand(n_samples, dim)

    # 缩放到积分域
    points = a + (b - a) * points

    vals = f(points)
    mean = np.mean(vals)
    std_err = np.std(vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0

    # 考虑区间体积
    volume = (b - a) ** dim
    return float(mean * volume), float(std_err * volume)


def transform_to_gaussian(uniform_samples: np.ndarray) -> np.ndarray:
    """
    使用 Box-Muller 变换将均匀采样转换为标准正态采样。

    对于 U_1, U_2 \\sim U(0,1) 且独立:
        Z_1 = \\sqrt{-2 \\ln U_1} \\cos(2\\pi U_2)
        Z_2 = \\sqrt{-2 \\ln U_1} \\sin(2\\pi U_2)

    则 Z_1, Z_2 \\sim N(0, 1) 且独立。

    参数:
        uniform_samples: 形状为 (n, m) 的均匀采样，m 必须为偶数

    返回:
        标准正态采样
    """
    n, m = uniform_samples.shape
    if m % 2 != 0:
        # 复制最后一列使其变为偶数
        uniform_samples = np.column_stack([uniform_samples, uniform_samples[:, -1:]])
        m += 1

    result = np.zeros((n, m))
    for i in range(0, m, 2):
        u1 = np.clip(uniform_samples[:, i], 1e-15, 1.0 - 1e-15)
        u2 = np.clip(uniform_samples[:, i + 1], 1e-15, 1.0 - 1e-15)
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        result[:, i] = r * np.cos(theta)
        result[:, i + 1] = r * np.sin(theta)

    return result[:, :m] if m % 2 == 0 else result[:, :-1]


def stratified_sampling(n_strata: int, dim: int) -> np.ndarray:
    """
    分层采样：将空间均匀划分为网格，在每个网格单元中随机采样。

    总样本数为 n_strata^dim，当维度高时不可行，适合低维问题。

    参数:
        n_strata: 每个维度的层数
        dim: 维度

    返回:
        采样点数组
    """
    total = n_strata ** dim
    if total > 1000000:
        raise ValueError("Too many strata for high dimensions")

    if dim == 1:
        samples = np.zeros((total, 1))
        for i in range(n_strata):
            samples[i, 0] = (i + np.random.rand()) / n_strata
        return samples

    # 多维网格
    grids = [np.linspace(0, 1, n_strata + 1) for _ in range(dim)]
    samples = []
    for idx in np.ndindex(*([n_strata] * dim)):
        point = np.zeros(dim)
        for d in range(dim):
            low = grids[d][idx[d]]
            high = grids[d][idx[d] + 1]
            point[d] = low + np.random.rand() * (high - low)
        samples.append(point)
    return np.array(samples)
