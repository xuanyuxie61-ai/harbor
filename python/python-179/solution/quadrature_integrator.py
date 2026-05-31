"""
quadrature_integrator.py
高维数值积分与误差估计模块
==========================
对应原项目 713_maple_area（Monte Carlo / Quasi-Monte Carlo 面积估计），
扩展至张量范数计算、高维参数积分及低秩张量积分的误差估计。

核心方法：
- 均匀网格积分（Riemann 和）
- 标准 Monte Carlo（伪随机）
- Quasi-Monte Carlo（Hammersley 低差异序列）
"""

import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK


# ---------------------------------------------------------------------------
# 低差异序列（Hammersley / van der Corput）
# ---------------------------------------------------------------------------

def van_der_corput_sequence(n: int, base: int = 2) -> np.ndarray:
    """
    一维 van der Corput 低差异序列：
        对整数 i，将其表示为 base 进制数 i = Σ a_k base^k，
        则 x_i = Σ a_k / base^{k+1}。

    星差异（star discrepancy）满足 D_n^* ≤ (log n) / (n * log base)。
    """
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        idx = i
        f = 1.0
        r = 0.0
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


def hammersley_sequence(n: int, d: int) -> np.ndarray:
    """
    d 维 Hammersley 序列：
        x_i^{(1)} = i / n
        x_i^{(k)} = van_der_corput(i, prime(k-1)),  k=2,...,d

    星差异界：D_n^* = O( (log n)^{d-1} / n )，远优于纯随机采样的 O(1/√n)。
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    # 简单质数表
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    points = np.zeros((n, d), dtype=float)
    points[:, 0] = np.arange(n, dtype=float) / max(n, 1)
    for k in range(1, d):
        base = primes[k - 1] if k - 1 < len(primes) else primes[-1]
        points[:, k] = van_der_corput_sequence(n, base)
    return points


# ---------------------------------------------------------------------------
# 积分器接口
# ---------------------------------------------------------------------------

def grid_integrate_1d(f, a: float, b: float, n: int) -> float:
    """
    一维均匀网格复合梯形积分：
        ∫_a^b f(x) dx ≈ h * [ 0.5 f(x0) + f(x1) + ... + f(x_{n-1}) + 0.5 f(xn) ]
    误差阶 O(h²) 对光滑函数。
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    val = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return float(val)


def monte_carlo_integrate(f, dim: int, n: int,
                          bounds: np.ndarray = None,
                          seed: int = None) -> Tuple[float, float]:
    """
    标准 Monte Carlo 积分：
        I ≈ V * (1/n) Σ f(x_i)
        σ² ≈ V² * Var(f) / n

    其中 V = ∏ (b_k - a_k) 为超矩形体积。

    返回 (估计值, 标准误差)。
    """
    if seed is not None:
        np.random.seed(seed)
    if bounds is None:
        bounds = np.tile([0.0, 1.0], (dim, 1))
    bounds = np.asarray(bounds, dtype=float)
    x = np.random.rand(n, dim)
    # 映射到实际区间
    for k in range(dim):
        x[:, k] = bounds[k, 0] + x[:, k] * (bounds[k, 1] - bounds[k, 0])
    y = np.array([f(xi) for xi in x], dtype=float)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    mean = np.mean(y)
    var = np.var(y, ddof=1) if n > 1 else 0.0
    estimate = volume * mean
    stderr = volume * np.sqrt(var / max(n, 1))
    return float(estimate), float(stderr)


def qmc_integrate(f, dim: int, n: int,
                  bounds: np.ndarray = None) -> float:
    """
    Quasi-Monte Carlo 积分（Hammersley 序列）：
        I ≈ V * (1/n) Σ f(x_i)

    误差界为 O( (log n)^{d-1} / n )，在适中维度（d≤10）下远优于 MC。
    """
    if bounds is None:
        bounds = np.tile([0.0, 1.0], (dim, 1))
    bounds = np.asarray(bounds, dtype=float)
    x = hammersley_sequence(n, dim)
    for k in range(dim):
        x[:, k] = bounds[k, 0] + x[:, k] * (bounds[k, 1] - bounds[k, 0])
    y = np.array([f(xi) for xi in x], dtype=float)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    return float(volume * np.mean(y))


# ---------------------------------------------------------------------------
# 张量范数的 Monte Carlo 估计
# ---------------------------------------------------------------------------

def estimate_tensor_frobenius_norm_mc(tensor: np.ndarray, n_samples: int = 10000,
                                      seed: int = None) -> float:
    """
    通过随机采样估计稠密张量的 Frobenius 范数：
        ‖A‖_F² = Σ_{i1,...,id} A_{i1...id}²

    对大规模张量（无法全部载入内存），从所有索引中均匀随机抽取 n_samples 个，
    估计总和。无偏估计：
        ‖A‖_F² ≈ (N / n_samples) * Σ_{sampled} A_i²
    其中 N = ∏ n_k 为总元素数。
    """
    tensor = np.asarray(tensor)
    shape = tensor.shape
    d = len(shape)
    N = int(np.prod(shape))
    if seed is not None:
        np.random.seed(seed)
    # 生成随机多维索引
    flat_idx = np.random.randint(0, N, size=n_samples)
    # 转换回多维并取值
    strides = [int(np.prod(shape[k+1:], dtype=np.int64)) for k in range(d)]
    samples = np.zeros(n_samples, dtype=float)
    for s in range(n_samples):
        idx = flat_idx[s]
        multi = []
        rem = idx
        for stride in strides:
            multi.append(rem // stride)
            rem = rem % stride
        samples[s] = tensor[tuple(multi)]
    norm_sq_est = (N / n_samples) * np.sum(samples * samples)
    return float(np.sqrt(norm_sq_est))


from typing import Tuple
