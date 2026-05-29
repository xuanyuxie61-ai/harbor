"""
molecular_weight_distribution.py
==================================
分子量分布 (MWD) 建模与矩分析

基于种子项目 1360_truncated_normal 与 886_polygon_integrals 融合重构。

科学背景：
---------
自由基聚合产物的分子量分布(MWD)通常可用对数正态分布、
Flory-Schulz 分布或广义Gamma分布描述。然而，在实际工业
反应器中，由于混合不均匀、局部过热点和链转移反应，
MWD 往往呈现多峰或截断特征。

本模块实现以下核心功能：
1. 基于矩方法的 MWD 参数重构
2. 截断对数正态分布用于描述链长分布
3. 多边形域上的矩积分（模拟非理想反应器几何效应）
4. 多分散指数(PDI)与累积分布函数(CDF)计算

核心公式：
----------
Flory-Schulz 最可几分布（理想均相聚合）：

    P(n) = (1-p) p^{n-1}

其中 p 为链增长概率：

    p = R_p / (R_p + R_t + R_tr) = k_p [M] / (k_p [M] + k_t [P^\bullet] + k_tr [S])

数均聚合度：

    \bar{X}_n = 1 / (1-p) = R_p / (R_t + R_tr)

重均聚合度：

    \bar{X}_w = (1+p) / (1-p)

多分散指数：

    PDI = \bar{X}_w / \bar{X}_n = 1 + p

截断对数正态分布（描述非理想分布）：

    f(n; \mu, \sigma, a, b) =
        \frac{1}{\sigma n \sqrt{2\pi}}
        \frac{ \exp( -\frac{(\ln n - \mu)^2}{2\sigma^2} ) }
             { \Phi(\frac{\ln b - \mu}{\sigma}) - \Phi(\frac{\ln a - \mu}{\sigma}) }

其中 Φ 为标准正态累积分布函数，[a,b] 为截断区间。

广义矩定义（多边形域上）：

    \nu_{pq} = \iint_{\Omega} x^p y^q \rho(x,y) \, dx dy

对于反应器截面 Ω，此矩可用于估计局部浓度梯度对 MWD 的影响。
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.special import erf, gammaln, factorial


def flory_schulz_distribution(n: np.ndarray, p: float) -> np.ndarray:
    """
    Flory-Schulz 链长分布：
        P(n) = (1-p) * p^{n-1}

    参数：
        n : 聚合度数组 (>=1)
        p : 链增长概率 (0 < p < 1)

    返回：
        概率质量函数值
    """
    n = np.asarray(n)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    n = np.maximum(n, 1.0)
    pmf = (1.0 - p) * (p ** (n - 1.0))
    # 归一化修正（数值稳定性）
    pmf = np.where(n < 1.0, 0.0, pmf)
    return pmf


def flory_schulz_moments(p: float, max_moment: int = 4) -> np.ndarray:
    """
    Flory-Schulz 分布的解析矩：
        μ_k = Σ n^k P(n) = (1-p) * Li_{-k}(p)

    对于低阶矩有闭式：
        μ_0 = 1
        μ_1 = 1/(1-p)
        μ_2 = (1+p)/(1-p)^2
        μ_3 = (1+4p+p^2)/(1-p)^3

    返回 k=0,...,max_moment 的矩值。
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    moments = np.zeros(max_moment + 1)
    moments[0] = 1.0
    if max_moment >= 1:
        moments[1] = 1.0 / (1.0 - p)
    if max_moment >= 2:
        moments[2] = (1.0 + p) / (1.0 - p) ** 2
    if max_moment >= 3:
        moments[3] = (1.0 + 4.0 * p + p ** 2) / (1.0 - p) ** 3
    if max_moment >= 4:
        moments[4] = (1.0 + 11.0 * p + 11.0 * p ** 2 + p ** 3) / (1.0 - p) ** 4
    return moments


def normal_01_cdf(x: float) -> float:
    """
    标准正态累积分布函数：
        Φ(x) = 0.5 * [1 + erf(x / sqrt(2))]
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def normal_01_cdf_inv(p: float) -> float:
    """
    标准正态 CDF 的反函数（近似实现，基于 truncated_normal_ab_sample.m 思想）
    使用 scipy.special.erfinv 直接计算：
        x = sqrt(2) * erfinv(2p - 1)
    """
    from scipy.special import erfinv
    p = np.clip(p, 1.0e-10, 1.0 - 1.0e-10)
    return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def truncated_normal_pdf(n: np.ndarray,
                         mu: float,
                         sigma: float,
                         a: float,
                         b: float) -> np.ndarray:
    """
    截断正态分布 PDF（用于描述对数变换后的链长分布）

    f(n) = φ((n-μ)/σ) / [σ (Φ((b-μ)/σ) - Φ((a-μ)/σ))]

    其中 φ 为标准正态 PDF。
    """
    n = np.asarray(n, dtype=float)
    sigma = max(sigma, 1.0e-12)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    denom = sigma * (normal_01_cdf(beta) - normal_01_cdf(alpha))
    denom = max(denom, 1.0e-15)

    z = (n - mu) / sigma
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2) / denom
    pdf = np.where((n >= a) & (n <= b), pdf, 0.0)
    return pdf


def truncated_normal_sample(mu: float,
                            sigma: float,
                            a: float,
                            b: float,
                            size: int = 1,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    截断正态分布采样（逆变换法）
    基于 truncated_normal_ab_sample.m 的算法：

        α = (a-μ)/σ,   β = (b-μ)/σ
        U ~ Uniform(0,1)
        Z = Φ^{-1}( Φ(α) + U*(Φ(β)-Φ(α)) )
        X = μ + σ Z
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    sigma = max(sigma, 1.0e-12)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_cdf = normal_01_cdf(alpha)
    beta_cdf = normal_01_cdf(beta)

    u = rng.random(size=size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    xi = normal_01_cdf_inv(xi_cdf)
    x = mu + sigma * xi
    x = np.clip(x, a, b)
    return x


def lognormal_mwd_pdf(n: np.ndarray,
                      mu_log: float,
                      sigma_log: float,
                      a: float = 1.0,
                      b: float = 1.0e6) -> np.ndarray:
    """
    截断对数正态分子量分布：

    设 X = ln(M_n)，则 X ~ N(μ, σ²) 在 [ln a, ln b] 上截断。

    PDF: f(M) = 1/(M σ sqrt(2π)) * exp( -(ln M - μ)²/(2σ²) ) / Z

    其中 Z = Φ((ln b - μ)/σ) - Φ((ln a - μ)/σ)
    """
    n = np.asarray(n, dtype=float)
    n = np.maximum(n, 1.0e-3)
    sigma_log = max(sigma_log, 1.0e-12)

    alpha = (np.log(a) - mu_log) / sigma_log
    beta = (np.log(b) - mu_log) / sigma_log
    Z = normal_01_cdf(beta) - normal_01_cdf(alpha)
    Z = max(Z, 1.0e-15)

    z = (np.log(n) - mu_log) / sigma_log
    pdf = (1.0 / (n * sigma_log * np.sqrt(2.0 * np.pi))
           * np.exp(-0.5 * z ** 2) / Z)
    pdf = np.where((n >= a) & (n <= b), pdf, 0.0)
    return pdf


def polygon_moment(n_vertices: int,
                   x: np.ndarray,
                   y: np.ndarray,
                   p: int,
                   q: int) -> float:
    """
    计算多边形域上的未归一化矩 ν_{pq}
    基于 moment_polygon.m 的算法：

    ν_{pq} = ∬_Ω x^p y^q dx dy

    计算公式（Steger 公式）：
        ν_{pq} = 1/((p+q+2)(p+q+1)C(p+q,p)) *
                 Σ_{i=1}^{n} (x_{i-1} y_i - x_i y_{i-1}) *
                 Σ_{k=0}^{p} Σ_{l=0}^{q}
                     C(k+l,l) C(p+q-k-l, q-l) x_i^k x_{i-1}^{p-k} y_i^l y_{i-1}^{q-l}

    参数：
        n_vertices : 顶点数
        x, y       : 顶点坐标数组（逆时针排列）
        p, q       : 矩的阶数
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != n_vertices or y.size != n_vertices:
        raise ValueError("x and y must have length n_vertices")

    nu_pq = 0.0
    xj = x[-1]
    yj = y[-1]

    for i in range(n_vertices):
        xi = x[i]
        yi = y[i]
        s_pq = 0.0
        for k in range(p + 1):
            for l in range(q + 1):
                # 组合数 C(n,k)
                ck = factorial(k + l) / (factorial(k) * factorial(l))
                ckl = factorial(p + q - k - l) / (factorial(q - l) * factorial(p - k))
                s_pq += ck * ckl * (xi ** k) * (xj ** (p - k)) * (yi ** l) * (yj ** (q - l))
        nu_pq += (xj * yi - xi * yj) * s_pq
        xj = xi
        yj = yi

    denom = (p + q + 2) * (p + q + 1) * factorial(p + q) / (factorial(p) * factorial(q))
    nu_pq = nu_pq / denom
    return float(nu_pq)


def mwd_from_moments(moments: np.ndarray,
                     n_grid: np.ndarray,
                     method: str = 'maxent') -> np.ndarray:
    """
    由前 N 阶矩重构分子量分布。

    方法：
      'maxent'   : 最大熵原理近似（指数型分布族）
      'gamma'    : Gamma 分布拟合
      'lognormal': 对数正态分布拟合

    最大熵分布形式：
        P(n) = exp( -Σ_{k=0}^{N} λ_k n^k )

    其中 Lagrange 乘子 λ_k 由矩约束确定。
    此处采用简化的一阶近似（两个矩拟合Gamma分布）。
    """
    n_grid = np.asarray(n_grid, dtype=float)
    n_grid = np.maximum(n_grid, 1.0e-3)

    if method == 'gamma':
        # Gamma 分布: 形状 α = μ_1^2 / (μ_2 - μ_1^2), 尺度 β = (μ_2 - μ_1^2) / μ_1
        mu1 = moments[1] if len(moments) > 1 else 100.0
        mu2 = moments[2] if len(moments) > 2 else mu1 ** 2 * 2.0
        var = max(mu2 - mu1 ** 2, 1.0e-6)
        alpha = mu1 ** 2 / var
        beta = var / mu1
        from scipy.stats import gamma
        pdf = gamma.pdf(n_grid, alpha, scale=beta)
    elif method == 'lognormal':
        mu1 = moments[1] if len(moments) > 1 else 100.0
        mu2 = moments[2] if len(moments) > 2 else mu1 ** 2 * 2.0
        var = max(mu2 - mu1 ** 2, 1.0e-6)
        sigma2 = np.log(var / mu1 ** 2 + 1.0)
        mu_log = np.log(mu1) - 0.5 * sigma2
        sigma_log = np.sqrt(sigma2)
        pdf = lognormal_mwd_pdf(n_grid, mu_log, sigma_log)
    else:
        # 简化 maxent：使用两个矩的指数修正
        mu1 = moments[1] if len(moments) > 1 else 100.0
        lam = 1.0 / mu1
        pdf = lam * np.exp(-lam * n_grid)

    pdf = np.maximum(pdf, 0.0)
    # 归一化
    integral = np.trapezoid(pdf, n_grid)
    if integral > 1.0e-15:
        pdf /= integral
    return pdf


def compute_pdi_from_moments(moments: np.ndarray) -> Tuple[float, float, float]:
    """
    由矩计算数均/重均聚合度与PDI

    DP_n = μ_1 / μ_0
    DP_w = μ_2 / μ_1
    PDI  = DP_w / DP_n = μ_0 μ_2 / μ_1^2
    """
    mu0 = moments[0] if len(moments) > 0 else 1.0
    mu1 = moments[1] if len(moments) > 1 else 1.0
    mu2 = moments[2] if len(moments) > 2 else 1.0
    # TODO(Hole 2): 由前 3 阶矩计算数均聚合度 DP_n、重均聚合度 DP_w 与多分散指数 PDI
    # 核心公式：
    #   DP_n = μ₁ / μ₀
    #   DP_w = μ₂ / μ₁
    #   PDI  = DP_w / DP_n = μ₀·μ₂ / μ₁²
    # 注意数值稳定性：需防止 μ₀ 或 μ₁ 接近零导致除零错误。
    raise NotImplementedError("Hole 2: 请实现由矩计算 DP_n、DP_w、PDI 的公式")


def local_mwd_broadening(moments_local: np.ndarray,
                         velocity_gradient: float,
                         diffusion_coeff: float,
                         reaction_rate: float) -> np.ndarray:
    """
    考虑局部流场（速度梯度）和扩散对 MWD 展宽的非理想修正。

    物理模型：
      在拉伸流场中，聚合物链受到拉伸导致链断裂概率增加，
      从而使 MWD 向低分子量方向偏移。

    修正矩方程（一阶扰动）：
      μ_k^{eff} = μ_k * (1 + χ * γ̇ / (D + k_p [M]))

    其中 χ 为无量纲流场敏感系数。
    """
    chi = 0.05  # 经验敏感系数
    denominator = max(diffusion_coeff + reaction_rate, 1.0e-12)
    correction = 1.0 + chi * velocity_gradient / denominator
    correction = min(correction, 2.0)  # 上限保护
    return moments_local * correction
