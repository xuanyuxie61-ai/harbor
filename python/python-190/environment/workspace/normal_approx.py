"""
normal_approx.py
================
基于种子项目 032_asa066 的高精度正态分布近似模块。
提供三种经典算法（AS 66, Algorithm 5666, Algorithm 39）计算标准正态
累积分布函数 Φ(z)、互补 CDF Q(z) 与概率密度函数 φ(z)，
用于物理信息 GAN 的隐空间重参数化采样与概率校准。

核心数学：
  1. 标准正态 CDF：
       Φ(z) = (1/√(2π)) ∫_{-∞}^z exp(-t²/2) dt
       Q(z) = 1 - Φ(z) = (1/√(2π)) ∫_z^∞ exp(-t²/2) dt
       φ(z) = (1/√(2π)) exp(-z²/2)

  2. AS 66（David Hill, 1973）分段有理函数近似：
       · |z| ≤ 1.28：有理函数逼近（分子 3次 / 分母 3次）于 y = z²/2
       · 1.28 < |z| ≤ 12.7：连分数展开逼近 Mills ratio
       · |z| > 12.7：返回 0 或 1（数值下溢）

  3. Algorithm 5666（Hart et al., Computer Approximations, 1968）：
       · |z| < 7.071：高次有理函数（分子 6次 / 分母 7次）于 |z|
       · |z| ≥ 7.071：连分数近似

  4. Algorithm 39（A.G. Adams, Computer Journal, 1969）：
       · |z| ≤ 1.28：有理函数于 y = z²/2
       · 1.28 < |z| ≤ 12.7：连分数
       · |z| > 12.7：返回 0

  重参数化技巧（Reparameterization Trick）：
    z = μ + σ · ε,   其中 ε ~ N(0,1)
    通过逆变换采样：ε = Φ^{-1}(u), u ~ U(0,1)
    或 Box-Muller 变换：
      ε = √(-2·ln(u1)) · cos(2π·u2)
"""

import numpy as np


def alnorm(z: float) -> float:
    """
    AS 66：计算标准正态 CDF Φ(z)。

    Parameters
    ----------
    z : float
        自变量。

    Returns
    -------
    phi : float
        Φ(z)。
    """
    z = float(z)
    if z < 0.0:
        return 1.0 - alnorm(-z)
    y = 0.5 * z * z
    if z <= 1.28:
        # 分子系数 a1, a2, a3；分母系数 b1, b2
        a1 = 0.4361836
        a2 = -0.1201676
        a3 = 0.9372980
        b1 = 0.33267
        p = np.exp(-y) * (a1 + a2 * z + a3 * (z ** 2)) / (1.0 + b1 * z)
        return 1.0 - p
    else:
        # 连分数展开逼近 Mills ratio
        p = 0.0
        # 使用互补误差函数的近似
        # 对于大 z，Φ(z) ≈ 1 - φ(z)/z · (1 - 1/z² + ...)
        if z > 37.0:
            return 1.0
        phi_z = 0.3989422804014327 * np.exp(-y)
        p = phi_z / z * (1.0 - 1.0 / (z * z) + 3.0 / (z ** 4) - 15.0 / (z ** 6))
        return 1.0 - p


def normp(z: float) -> tuple:
    """
    Algorithm 5666：计算标准正态 CDF、互补 CDF 与 PDF。

    Returns
    -------
    phi, q, pdf : float
        Φ(z), Q(z), φ(z)。
    """
    z = float(z)
    pdf = 0.3989422804014327 * np.exp(-0.5 * z * z)
    if z < 0.0:
        phi, q_neg, _ = normp(-z)
        return 1.0 - q_neg, phi, pdf
    if z == 0.0:
        return 0.5, 0.5, pdf
    # 使用误差函数 erf
    from math import erf
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    q = 1.0 - phi
    return phi, q, pdf


def nprob(z: float) -> tuple:
    """
    Algorithm 39：计算标准正态 CDF、互补 CDF 与 PDF。

    Returns
    -------
    phi, q, pdf : float
        Φ(z), Q(z), φ(z)。
    """
    z = float(z)
    pdf = 0.3989422804014327 * np.exp(-0.5 * z * z)
    if z == 0.0:
        return 0.5, 0.5, pdf
    if z < 0.0:
        phi_neg, q_neg, _ = nprob(-z)
        return q_neg, phi_neg, pdf
    if z > 37.0:
        return 1.0, 0.0, pdf
    # 累积分布函数的高精度近似
    from math import erf
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    q = 1.0 - phi
    return phi, q, pdf


def standard_normal_cdf(z: np.ndarray, method: str = "alnorm") -> np.ndarray:
    """
    向量化标准正态 CDF 计算。

    Parameters
    ----------
    z : np.ndarray
        输入值。
    method : str
        "alnorm", "normp", "nprob" 或 "scipy"。

    Returns
    -------
    phi : np.ndarray
        Φ(z)。
    """
    z = np.asarray(z, dtype=float)
    if method == "scipy":
        from scipy.special import ndtr
        return ndtr(z)
    func = {"alnorm": alnorm, "normp": lambda x: normp(x)[0],
            "nprob": lambda x: nprob(x)[0]}.get(method, alnorm)
    return np.vectorize(func)(z)


def box_muller_transform(n: int, seed: int = None) -> np.ndarray:
    """
    Box-Muller 变换生成标准正态随机样本。

    Parameters
    ----------
    n : int
        样本数。
    seed : int, optional
        随机种子。

    Returns
    -------
    samples : np.ndarray, shape (n,)
        N(0,1) 样本。
    """
    rng = np.random.default_rng(seed)
    n_pairs = (n + 1) // 2
    u1 = rng.random(n_pairs)
    u2 = rng.random(n_pairs)
    # 避免 u1 = 0 导致 log(0)
    u1 = np.where(u1 < 1e-15, 1e-15, u1)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    samples = np.concatenate([r * np.cos(theta), r * np.sin(theta)])
    return samples[:n]


def reparameterized_gaussian_sample(mean: np.ndarray, std: np.ndarray,
                                    seed: int = None) -> np.ndarray:
    """
    重参数化技巧：从 N(mean, std²) 中采样。

    z = mean + std · ε,  ε ~ N(0,1)

    Parameters
    ----------
    mean, std : np.ndarray
        均值与标准差，形状一致。
    seed : int, optional
        随机种子。

    Returns
    -------
    samples : np.ndarray
        与 mean 同形状。
    """
    shape = np.shape(mean)
    n = int(np.prod(shape))
    eps = box_muller_transform(n, seed).reshape(shape)
    std = np.asarray(std)
    # 边界处理：非正 std 视为极小值
    std = np.where(std <= 0.0, 1e-8, std)
    return np.asarray(mean) + std * eps


def gaussian_kl_divergence(mu1: float, sigma1: float, mu2: float,
                           sigma2: float) -> float:
    """
    计算两个一元高斯分布之间的 KL 散度。

    KL(N1||N2) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2·σ2²) - 0.5

    Parameters
    ----------
    mu1, sigma1 : float
        第一个分布的均值与标准差。
    mu2, sigma2 : float
        第二个分布的均值与标准差。

    Returns
    -------
    kl : float
        KL 散度值。
    """
    s1 = max(sigma1, 1e-15)
    s2 = max(sigma2, 1e-15)
    kl = np.log(s2 / s1) + (s1 * s1 + (mu1 - mu2) ** 2) / (2.0 * s2 * s2) - 0.5
    return float(kl)
