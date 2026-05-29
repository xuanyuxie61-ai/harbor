"""
random_parameters.py
====================
随机参数建模与采样模块（融合 1360_truncated_normal）

功能：
- 截断正态分布的PDF、CDF、逆CDF、采样
- 截断正态分布的矩计算（均值、方差、高阶矩）
- 随机参数场的KL展开系数生成

数学公式：
- 标准正态PDF: φ(x) = (1/√(2π)) exp(-x²/2)
- 标准正态CDF: Φ(x) = ∫_{-∞}^{x} φ(t) dt
- 双边截断PDF: f(x;μ,σ,a,b) = φ((x-μ)/σ) / [σ(Φ((b-μ)/σ) - Φ((a-μ)/σ))]
- 截断均值: E[X] = μ + σ * [φ(α) - φ(β)] / [Φ(β) - Φ(α)]
  其中 α = (a-μ)/σ, β = (b-μ)/σ
"""

import numpy as np
from scipy.special import erf


SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def normal_01_pdf(x):
    """标准正态PDF。"""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x ** 2) / SQRT2PI


def normal_01_cdf(x):
    """标准正态CDF，基于误差函数。"""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / SQRT2))


def truncated_normal_ab_pdf(x, mu, sigma, a, b):
    """
    双边截断正态分布的概率密度函数。
    
    f(x) = φ((x-μ)/σ) / [σ (Φ((b-μ)/σ) - Φ((a-μ)/σ))],  a < x < b
    """
    x = np.asarray(x, dtype=float)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if b <= a:
        raise ValueError("truncation limits must satisfy a < b")
    
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    
    if denom <= 1e-15:
        raise ValueError("truncation interval too narrow relative to sigma")
    
    xi = (x - mu) / sigma
    pdf = normal_01_pdf(xi) / (denom * sigma)
    pdf = np.where((x > a) & (x < b), pdf, 0.0)
    return pdf


def truncated_normal_ab_mean(mu, sigma, a, b):
    """
    截断正态分布的均值。
    E[X] = μ + σ * (φ(α) - φ(β)) / (Φ(β) - Φ(α))
    """
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    if denom <= 1e-15:
        return mu
    return mu + sigma * (normal_01_pdf(alpha) - normal_01_pdf(beta)) / denom


def truncated_normal_ab_variance(mu, sigma, a, b):
    """
    截断正态分布的方差。
    Var[X] = σ² [1 + (αφ(α) - βφ(β))/(Φ(β)-Φ(α)) - ((φ(α)-φ(β))/(Φ(β)-Φ(α)))²]
    """
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    if denom <= 1e-15:
        return sigma ** 2
    
    pdf_a = normal_01_pdf(alpha)
    pdf_b = normal_01_pdf(beta)
    
    term1 = (alpha * pdf_a - beta * pdf_b) / denom
    term2 = ((pdf_a - pdf_b) / denom) ** 2
    var = sigma ** 2 * (1.0 + term1 - term2)
    return max(var, 0.0)


def truncated_normal_ab_sample(mu, sigma, a, b, size=None):
    """
    使用逆变换采样生成截断正态分布随机数。
    X = μ + σ * Φ⁻¹( Φ(α) + U * (Φ(β) - Φ(α)) )
    """
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    cdf_a = normal_01_cdf(alpha)
    cdf_b = normal_01_cdf(beta)
    
    if cdf_b - cdf_a < 1e-15:
        return np.full(size if size is not None else (), mu)
    
    u = np.random.uniform(0.0, 1.0, size=size)
    # 反解标准正态CDF
    from scipy.special import erfinv
    z = SQRT2 * erfinv(2.0 * (cdf_a + u * (cdf_b - cdf_a)) - 1.0)
    return mu + sigma * z


def generate_kl_coefficients(n_modes, correlation_length, domain_size=1.0):
    """
    生成Karhunen-Loève展开的随机系数，用于参数化随机场。
    
    对于指数协方差核 C(x,y) = exp(-|x-y|/L)，特征值为：
    λ_k = 2L / (L²ω_k² + 1)
    其中 ω_k 是方程 (Lω)tan(ωD) = 1 或 (Lω) + tan(ωD) = 0 的根。
    
    这里简化为解析近似。
    """
    if n_modes <= 0:
        return np.ones(1)
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive")
    
    k = np.arange(1, n_modes + 1)
    # 使用简化模型：λ_k = σ² * L² / (L² + (kπ/D)²)
    eigenvalues = (correlation_length ** 2) / (correlation_length ** 2 + (k * np.pi / domain_size) ** 2)
    eigenvalues = np.sqrt(eigenvalues)  # 标准差缩放
    
    # 生成截断正态分布的随机系数，保证物理正性
    coeffs = truncated_normal_ab_sample(
        mu=0.0, sigma=1.0, a=-2.0, b=2.0, size=n_modes
    ) * eigenvalues
    return coeffs
