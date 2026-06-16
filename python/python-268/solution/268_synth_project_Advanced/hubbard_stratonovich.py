"""
hubbard_stratonovich.py - Hubbard-Stratonovich 变换与辅助场采样
================================================================

科学背景 (Scientific Background):
    DQMC 的核心技巧是 Hubbard-Stratonovich (HS) 变换:
    将四费米子相互作用项 e^{-Δτ U n↑ n↓} 分解为辅助玻色场 σ 的二次型:

        e^{-Δτ U (n↑ - 1/2)(n↓ - 1/2)} =
            (1/2) Σ_{σ=±1} e^{α σ (n↑ - n↓)} e^{λ σ (n↑ + n↓ - 1)}

    其中 cosh(α) = e^{Δτ U / 2},  或等价地  α = arccosh(e^{Δτ U / 2}).
    对于连续 HS 场 (用于更精确的分解):
        σ ~ 截断高斯分布,  σ ∈ [-σ_max, σ_max]

    融合种子项目:
      - 1360_truncated_normal: 截断正态分布采样
      - 301_disk01_monte_carlo: 圆盘蒙特卡洛积分

核心公式 (Key Formulas):
    HS 变换的离散形式:
        e^{-(Δτ U/2)(n↑-n↓)²} = (1/2) Σ_{s=±1} e^{√(Δτ U) s (n↑-n↓)}

    辅助场的作用量 (action):
        S_HS = (1/2) Σ_{i,τ} σ_i(τ)² - Σ_{i,τ} σ_i(τ) √(U/Δτ) (n_{i↑} - n_{i↓})

    配分函数重写:
        Z = ∫ D[σ̄, σ] D[σ_HS] e^{-S[σ̄, σ, σ_HS]}
        其中 S = S_kinetic + S_HS + S_interaction
"""

import numpy as np
from typing import Tuple, Optional


# ==========================================================================
#  截断正态分布 (融合 1360_truncated_normal)
# ==========================================================================

def normal_01_pdf(x: np.ndarray) -> np.ndarray:
    """
    标准正态分布的概率密度函数.

    公式:
        φ(x) = (1/√(2π)) exp(-x²/2)

    物理应用:
        HS 辅助场在连续形式下服从高斯分布.
        截断正态分布用于限制辅助场的振幅, 防止数值溢出.
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def normal_01_cdf(x: np.ndarray) -> np.ndarray:
    """
    标准正态分布的累积分布函数.

    Φ(x) = (1/2) [1 + erf(x/√2)]

    使用高精度近似 (Abramowitz & Stegun).
    """
    return 0.5 * (1.0 + _erf_approx(x / np.sqrt(2.0)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """
    误差函数的高精度近似 (Abramowitz & Stegun 7.1.26).

    |ε(x)| ≤ 1.5 × 10^{-7}

    公式:
        erf(x) ≈ 1 - (a1 t + a2 t² + a3 t³) exp(-x²)
        t = 1 / (1 + 0.47047 x)
    """
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.47047 * x)
    poly = t * (0.3480242 + t * (-0.0958798 + t * 0.7478556))
    result = 1.0 - poly * np.exp(-x * x)
    return sign * result


def truncated_normal_sample(a: float, b: float, size: int,
                            mean: float = 0.0, std: float = 1.0,
                            rng: Optional[np.random.Generator] = None
                            ) -> np.ndarray:
    """
    从截断正态分布 TN(mean, std², a, b) 中采样.

    融合种子项目 1360_truncated_normal 的采样方法.

    截断正态分布 PDF:
        f(x) = φ((x-μ)/σ) / (σ [Φ((b-μ)/σ) - Φ((a-μ)/σ)])
        对 x ∈ [a, b],  否则 f(x) = 0

    采样算法 (逆 CDF 法):
        1. 计算 α = (a - μ) / σ,  β = (b - μ) / σ
        2. 计算 U ~ Uniform(Φ(α), Φ(β))
        3. 返回 X = μ + σ Φ^{-1}(U)

    在 DQMC 中:
        辅助场 σ_i(τ) ~ TN(0, 1, -σ_max, σ_max)
        截断防止 exp(α σ) 溢出 (当 ασ > 500 时浮点溢出).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    alpha = (a - mean) / std
    beta = (b - mean) / std
    phi_alpha = normal_01_cdf(np.array([alpha]))[0]
    phi_beta = normal_01_cdf(np.array([beta]))[0]

    # 逆 CDF 采样
    u = rng.uniform(phi_alpha, phi_beta, size=size)
    samples = mean + std * _normal_cdf_inv(u)
    return np.clip(samples, a, b)


def _normal_cdf_inv(p: np.ndarray) -> np.ndarray:
    """
    标准正态分布的逆 CDF (分位函数).

    使用 Beasley-Springer-Moro 算法:
        对 0.5 ≤ p < 1:
            t = √(-2 ln(1-p))
            x = t - (c0 + c1 t + c2 t²) / (1 + d1 t + d2 t² + d3 t³)
        对 p < 0.5:
            x = -x(1-p)
    """
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    sign = np.where(p >= 0.5, 1.0, -1.0)
    p_adj = np.where(p >= 0.5, 1.0 - p, p)

    t = np.sqrt(-2.0 * np.log(p_adj))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)
    return sign * x


def truncated_normal_pdf(x: np.ndarray, a: float, b: float,
                         mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    截断正态分布的 PDF.
    """
    mask = (x >= a) & (x <= b)
    result = np.zeros_like(x, dtype=float)
    z = (x[mask] - mean) / std
    norm_const = normal_01_cdf(np.array([(b - mean) / std]))[0] - \
                 normal_01_cdf(np.array([(a - mean) / std]))[0]
    result[mask] = normal_01_pdf(z) / (std * norm_const)
    return result


def truncated_normal_moment(n: int, a: float, b: float,
                            mean: float = 0.0, std: float = 1.0) -> float:
    """
    截断正态分布的 n 阶矩 E[X^n].

    E[X^n] = ∫_a^b x^n f(x) dx

    递归公式 (对标准截断正态):
        E[Z^n] = (n-1) E[Z^{n-2}] + (α^{n-1} φ(α) - β^{n-1} φ(β)) / (Φ(β) - Φ(α))
    """
    alpha = (a - mean) / std
    beta = (b - mean) / std
    phi_alpha = normal_01_pdf(np.array([alpha]))[0]
    phi_beta = normal_01_pdf(np.array([beta]))[0]
    Z_alpha = normal_01_cdf(np.array([alpha]))[0]
    Z_beta = normal_01_cdf(np.array([beta]))[0]
    denom = Z_beta - Z_alpha

    if denom < 1e-15:
        return mean ** n

    moments_standard = np.zeros(n + 1)
    moments_standard[0] = 1.0
    if n >= 1:
        moments_standard[1] = (phi_alpha - phi_beta) / denom
    for k in range(2, n + 1):
        moments_standard[k] = (
            (k - 1) * moments_standard[k - 2]
            + (alpha ** (k - 1) * phi_alpha - beta ** (k - 1) * phi_beta) / denom
        )

    # 转换为非标准: E[X^n] = Σ C(n,k) μ^{n-k} σ^k E[Z^k]
    result = 0.0
    for k in range(n + 1):
        binom = _binomial(n, k)
        result += binom * (mean ** (n - k)) * (std ** k) * moments_standard[k]
    return result


def _binomial(n: int, k: int) -> float:
    """二项式系数 C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


# ==========================================================================
#  HS 变换参数
# ==========================================================================

def compute_hs_alpha(delta_tau: float, U: float) -> float:
    """
    计算 HS 变换的耦合常数 α.

    离散 HS 变换:
        cosh(α) = exp(Δτ U / 2)
        α = arccosh(exp(Δτ U / 2))

    当 Δτ U << 1 时:
        α ≈ √(Δτ U) + O((Δτ U)^{3/2})

    当 Δτ U >> 1 时:
        α ≈ Δτ U / 2 - ln(2) + O(e^{-Δτ U})

    数值稳定性: 对大 Δτ U, 直接用渐近展开避免 arccosh 溢出.
    """
    arg = np.exp(delta_tau * U / 2.0)
    if arg > 1e10:
        # 渐近展开: arccosh(x) ≈ ln(2x) for x >> 1
        return np.log(2.0 * arg)
    return np.arccosh(arg)


def compute_hs_coupling(delta_tau: float, U: float) -> float:
    """
    返回 HS 辅助场与费米子的耦合强度 λ.

    对离散 HS 变换 (Ising 辅助场):
        λ = α = arccosh(exp(Δτ U / 2))

    对连续 HS 变换 (高斯辅助场):
        λ = √(Δτ U)
    """
    if abs(U) < 1e-15:
        return 0.0
    return compute_hs_alpha(delta_tau, U)


# ==========================================================================
#  圆盘蒙特卡洛积分 (融合 301_disk01_monte_carlo)
# ==========================================================================

def disk01_sample(n_points: int, radius: float = 1.0,
                  rng: Optional[np.random.Generator] = None
                  ) -> np.ndarray:
    """
    在单位圆盘中均匀采样.

    融合种子项目 301_disk01_monte_carlo 的圆盘采样方法.

    拒绝法:
        在 [-R, R]² 中均匀采样, 拒绝 r > R 的点.
    效率: π/4 ≈ 78.5%.

    或直接法:
        θ ~ Uniform(0, 2π)
        r = R √U,  U ~ Uniform(0, 1)
        x = r cos θ,  y = r sin θ
    """
    if rng is None:
        rng = np.random.default_rng(42)

    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = radius * np.sqrt(rng.uniform(0, 1, n_points))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def disk01_area_monte_carlo(n_points: int, radius: float = 1.0,
                            rng: Optional[np.random.Generator] = None
                            ) -> float:
    """
    蒙特卡洛估计圆盘面积.

    A = π R² (精确值)
    A_MC = (2R)² * (N_inside / N_total)  (拒绝法)

    或直接法: A_MC = (1/N) Σ_i π R² = π R² (无方差, 因为已知面积元)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pts = rng.uniform(-radius, radius, (n_points * 2, 2))
    r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    n_inside = np.sum(r2 <= radius * radius)
    return (2 * radius) ** 2 * n_inside / (n_points * 2)


def disk01_monomial_integral(p: int, q: int, radius: float = 1.0) -> float:
    """
    圆盘上单项式 x^p y^q 的精确积分.

    ∫∫_{x²+y²≤R²} x^p y^q dx dy

    若 p 或 q 为奇数, 积分 = 0 (对称性).
    若 p, q 均为偶数:
        = 2 R^{p+q+2} B((p+1)/2, (q+1)/2) / (p+q+2)

    其中 B 为 Beta 函数.

    物理应用:
        布里渊区积分中, 对称性分析决定哪些项对自能有贡献.
    """
    if p % 2 == 1 or q % 2 == 1:
        return 0.0
    from math import gamma
    numerator = 2.0 * (radius ** (p + q + 2)) * np.pi
    denominator = (p + q + 2) * _beta_func((p + 1) / 2.0, (q + 1) / 2.0)
    # 使用极坐标: ∫ r^{p+q+1} dr ∫ cos^p(θ) sin^q(θ) dθ
    radial = radius ** (p + q + 2) / (p + q + 2)
    angular = _beta_func((p + 1) / 2.0, (q + 1) / 2.0)
    return radial * angular


def _beta_func(a: float, b: float) -> float:
    """Beta 函数 B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
    from math import gamma
    return gamma(a) * gamma(b) / gamma(a + b)


def monte_carlo_bz_integral(func, n_points: int,
                            radius: float = 1.0,
                            rng: Optional[np.random.Generator] = None
                            ) -> Tuple[float, float]:
    """
    蒙特卡洛计算布里渊区圆盘上的积分.

    I = ∫∫_{disk} f(k) d²k ≈ (πR²/N) Σ_i f(k_i)

    返回:
        (estimate, standard_error)

    标准误差:
        SE = √(Var/N) = R²π √(Var[f]/N)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pts = disk01_sample(n_points, radius, rng)
    f_vals = np.array([func(pts[i]) for i in range(n_points)])
    area = np.pi * radius * radius
    estimate = area * np.mean(f_vals)
    std_err = area * np.std(f_vals, ddof=1) / np.sqrt(n_points)
    return estimate, std_err
