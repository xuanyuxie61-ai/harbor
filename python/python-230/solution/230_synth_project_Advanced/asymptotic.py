"""
asymptotic.py
=============

渐近公式 —— CLs 计算的解析近似。

种子项目 1225 (VSC HEOM) 与 1220 (Unruh 热力学) 启发:
将开放量子系统的谱分解思想映射到检验统计量的渐近分布计算。

渐近理论 (Cowan, Cranmer, Gross, Vitells 2012, JHEP 07 (2012) 033):

1. 检验统计量 q_μ 的渐近分布:

    在 H_0: μ = μ' 下 (μ' 为真实值):
        q_μ ~ (1/2) δ(0) + (1/2) χ²_1     若 μ ≥ μ'
        q_μ ~ 0                            若 μ < μ' (被截断)

    更精确 (非渐imov):
        f(q_μ | μ') = (1/(2√(2π q_μ))) · [exp(-(√q_μ + (μ-μ')/σ)²/2)
                                            + exp(-(√q_μ - (μ-μ')/σ)²/2)]

2. Asimov 数据集 (期望数据, n_i = ν_i(μ_A, θ_A)):
    - 用于预估实验灵敏度 (中位期望 limit)
    - Asimov μ'=0 数据集: n_i = ν_i(0, θ̂_0)
    - Asimov μ'=1 数据集: n_i = ν_i(1, θ̂_1)

3. 渐近 p 值:
    p_{s+b}(μ) = 1 - Φ(√q_μ - (μ - μ')/σ)
               ≈ 1 - Φ(√q_μ)              [当 μ' = μ]

    p_b(μ) = 1 - Φ(√q_μ + μ/σ)
           ≈ Φ(-√q_μ - μ/σ)

    其中 σ ≈ μ / √q_{μ,A} (从 Asimov 数据集估计)

4. CLs:
    CLs(μ) = p_{s+b}(μ) / (1 - p_b(μ))
           = [1 - Φ(√q_μ)] / Φ(√q_μ + μ/σ)

5. 上限 μ_up: CLs(μ_up) = α (通常 α = 0.05 对 95% CL)

   渐近求解:
   √q_{μ_up} + μ_up/σ = Φ^{-1}(1 - α · Φ(√q_{μ_up}))

   简化 (当 q_μ 不太小时):
   μ_up ≈ σ · [z_α + √(z_α² + 4 q_{μ,A}) / 2]
   其中 z_α = Φ^{-1}(1 - α) ≈ 1.645 (95% CL)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional, Dict

from physical_model import BinModel


# ---------------------------------------------------------------------------
# 标准正态分布函数
# ---------------------------------------------------------------------------
def Phi(x: float) -> float:
    """标准正态 CDF: Φ(x) = (1/√(2π)) ∫_{-∞}^{x} exp(-t²/2) dt"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def Phi_inv(p: float) -> float:
    """标准正态逆 CDF (分位数函数): Φ^{-1}(p)"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p 必须在 (0, 1) 内: p={p}")
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    """
    逆误差函数 (Abramowitz & Stegun 有理近似):
        erfinv(x) ≈ sign(x) · √(√((2/(πa) + ln(1-x²)/2)² - ln(1-x²)/a) - (2/(πa) + ln(1-x²)/2))
    其中 a = 0.147
    """
    if abs(x) >= 1.0:
        return math.copysign(float("inf"), x)
    a = 0.147
    ln1mx2 = math.log(1.0 - x * x)
    t1 = 2.0 / (math.pi * a) + ln1mx2 / 2.0
    t2 = math.sqrt(t1 * t1 - ln1mx2 / a)
    return math.copysign(math.sqrt(t2 - t1), x)


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """高斯概率密度: (1/(σ√(2π))) · exp(-(x-μ)²/(2σ²))"""
    if sigma <= 0.0:
        raise ValueError(f"sigma 必须为正: sigma={sigma}")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# Asimov 数据集
# ---------------------------------------------------------------------------
def asimov_data_background(
    model: BinModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asimov 背景数据集: n_i = ν_i(μ=0, θ̂_0)

    用于预估背景假设下的实验灵敏度。
    θ̂_0 = 0 (在 Asimov 近似下, nuisance 参数在其约束最大值)。

    Returns
    -------
    (n_asimov, theta_hat_0) : (ndarray, ndarray)
    """
    theta_0 = np.zeros(model.n_nuis)
    n_asimov = model.expected_rate(0.0, theta_0)
    return n_asimov, theta_0


def asimov_data_signal(
    model: BinModel, mu: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asimov 信号+背景数据集: n_i = ν_i(μ, θ̂_μ)
    """
    theta_mu = np.zeros(model.n_nuis)
    n_asimov = model.expected_rate(mu, theta_mu)
    return n_asimov, theta_mu


# ---------------------------------------------------------------------------
# Asimov q_μ,A (预估检验统计量)
# ---------------------------------------------------------------------------
def asimov_q_mu(
    model: BinModel, mu: float, mu_true: float = 0.0
) -> float:
    """
    Asimov 检验统计量 q_{μ,A}:

    在 μ' = mu_true 的 Asimov 数据集上计算的 q_μ。

    q_{μ,A} = 2 Σ_i [ν_i(μ, θ̂_μ) - ν_i(μ', θ̂_{μ'})
                       - n_i · ln(ν_i(μ, θ̂_μ) / ν_i(μ', θ̂_{μ'}))]

    在 Asimov 近似下 (n_i = ν_i(μ', 0)):

    q_{μ,A} ≈ μ² / σ²
    其中 σ² = μ² / q_{μ,A}

    精确公式 (Cowan et al. 2012, eq. 56):
    q_{μ,A} = Σ_i (μ s_i)² / ν_i(0, 0) + O(μ³)
    """
    if mu <= 0.0:
        return 0.0

    n_asimov, _ = asimov_data_background(model)
    # ν at μ
    theta_mu = np.zeros(model.n_nuis)
    nu_mu = model.expected_rate(mu, theta_mu)
    # ν at 0
    nu_0 = model.expected_rate(0.0, np.zeros(model.n_nuis))

    # 精确 Asimov q_μ
    q_A = 0.0
    for i in range(model.n_bins):
        if n_asimov[i] < 1e-12:
            continue
        q_A += 2.0 * (
            nu_mu[i] - nu_0[i] - n_asimov[i] * math.log(max(nu_mu[i], 1e-300) / max(nu_0[i], 1e-300))
        )
    return max(q_A, 0.0)


def estimate_sigma(
    model: BinModel, mu: float = 1.0
) -> float:
    """
    估计 σ = μ / √(q_{μ,A})

    用于渐近 p 值计算中的位移参数。
    """
    q_A = asimov_q_mu(model, mu)
    if q_A < 1e-12:
        return float("inf")
    return mu / math.sqrt(q_A)


# ---------------------------------------------------------------------------
# 渐近 p 值与 CLs
# ---------------------------------------------------------------------------
def asymptotic_pvalues(
    q_mu: float, mu: float, sigma: float, mu_prime: float = 0.0
) -> Tuple[float, float]:
    """
    渐近 p 值 (Cowan et al. 2012, eq. 59):

    p_{s+b} = 1 - Φ(√q_μ - (μ' - μ)/σ)
    p_b     = Φ(√q_μ + (μ - μ')/σ)  [背景假设 p 值]

    通常 μ' = μ (计算 observed p_{s+b}):
        p_{s+b} = 1 - Φ(√q_μ)

    对 CLs: μ' = 0 (背景假设):
        p_{s+b}(μ | μ'=0) = 1 - Φ(√q_μ + μ/σ)
        p_b(μ | μ'=0) = 1 - Φ(√q_μ)

    Returns
    -------
    (p_sb, p_b) : (float, float)
    """
    sqrt_q = math.sqrt(max(q_mu, 0.0))

    if sigma < 1e-10 or sigma == float("inf"):
        # σ 极大 → p_b → 1, CLs → p_sb
        p_sb = 1.0 - Phi(sqrt_q)
        p_b = 1.0 - 1e-15
        return max(p_sb, 1e-300), max(p_b, 1e-300)

    # p_{s+b}: 在 μ' = 0 (背景) 假设下
    p_sb = 1.0 - Phi(sqrt_q + mu / sigma)
    # p_b: 背景 fluctuation p 值
    p_b_raw = 1.0 - Phi(sqrt_q)

    # 安全截断
    p_sb = max(min(p_sb, 1.0), 1e-300)
    p_b_raw = max(min(p_b_raw, 1.0), 1e-300)

    return p_sb, p_b_raw


def asymptotic_cls(
    q_mu: float, mu: float, sigma: float
) -> Tuple[float, float, float]:
    """
    渐近 CLs:

        CLs = p_{s+b} / (1 - p_b)
            = [1 - Φ(√q_μ + μ/σ)] / Φ(√q_μ)

    Returns
    -------
    (CLs, p_sb, p_b) : (float, float, float)
    """
    p_sb, p_b = asymptotic_pvalues(q_mu, mu, sigma)
    denominator = 1.0 - p_b
    if denominator < 1e-300:
        # 1 - p_b → 0: CLs → 1 (无排除能力)
        return 1.0, p_sb, p_b
    cls_val = p_sb / denominator
    return max(min(cls_val, 1.0), 0.0), p_sb, p_b


# ---------------------------------------------------------------------------
# 渐近上限求解
# ---------------------------------------------------------------------------
def asymptotic_upper_limit(
    model: BinModel,
    alpha: float = 0.05,
    mu_scan_max: float = 20.0,
    n_scan: int = 200,
) -> Tuple[float, Dict]:
    """
    求解 CLs(μ_up) = α 的 μ_up。

    使用渐近公式 + Brent 搜索。

    Parameters
    ----------
    model : BinModel
    alpha : float
        显著性水平 (0.05 → 95% CL)
    mu_scan_max : float
        搜索上限
    n_scan : int
        粗扫描点数

    Returns
    -------
    (mu_up, info) : (float, dict)
    """
    from scipy.optimize import brentq

    sigma = estimate_sigma(model, mu=1.0)
    q_A_1 = asimov_q_mu(model, mu=1.0)

    info = {
        "sigma": sigma,
        "q_A_mu1": q_A_1,
        "alpha": alpha,
    }

    if sigma == float("inf") or sigma > 1e6:
        info["status"] = "sigma_divergent"
        return float("inf"), info

    def cls_at_mu(mu_val):
        # Asimov q_μ 近似
        q_A_mu = asimov_q_mu(model, mu_val)
        cls_val, _, _ = asymptotic_cls(q_A_mu, mu_val, sigma)
        return cls_val - alpha

    # 粗扫描找 bracket
    mu_grid = np.linspace(0.01, mu_scan_max, n_scan)
    cls_vals = []
    for mu in mu_grid:
        q_A = asimov_q_mu(model, mu)
        cls, _, _ = asymptotic_cls(q_A, mu, sigma)
        cls_vals.append(cls)
    cls_vals = np.array(cls_vals)

    info["mu_grid"] = mu_grid
    info["cls_grid"] = cls_vals

    # 找 CLs = alpha 交叉点
    crossings = np.where((cls_vals[:-1] > alpha) & (cls_vals[1:] < alpha))[0]
    if len(crossings) == 0:
        if cls_vals[-1] > alpha:
            info["status"] = "limit_beyond_scan"
            return mu_scan_max, info
        else:
            info["status"] = "limit_below_min"
            return 0.0, info

    # Brent 精化
    idx = crossings[0]
    mu_lo, mu_hi = mu_grid[idx], mu_grid[idx + 1]
    try:
        mu_up = brentq(cls_at_mu, mu_lo, mu_hi, xtol=1e-6, maxiter=100)
        info["status"] = "converged"
    except ValueError:
        mu_up = 0.5 * (mu_lo + mu_hi)
        info["status"] = "brent_failed"

    return float(mu_up), info


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from physical_model import make_default_binmodel

    mdl = make_default_binmodel()

    # 基本函数
    print(f"Φ(0) = {Phi(0.0):.6f}")
    print(f"Φ(1.645) = {Phi(1.645):.6f}")
    print(f"Φ^{-1}(0.975) = {Phi_inv(0.975):.6f}")

    # Asimov
    q_A = asimov_q_mu(mdl, mu=1.0)
    sigma = estimate_sigma(mdl, mu=1.0)
    print(f"\nAsimov q_{{μ=1,A}} = {q_A:.6f}")
    print(f"σ = {sigma:.6f}")

    # CLs
    cls_val, p_sb, p_b = asymptotic_cls(q_A, 1.0, sigma)
    print(f"\nCLs(μ=1) = {cls_val:.6f}")
    print(f"p_{{s+b}} = {p_sb:.6e}")
    print(f"p_b = {p_b:.6e}")

    # 上限
    mu_up, info = asymptotic_upper_limit(mdl, alpha=0.05)
    print(f"\n95% CL 上限: μ_up = {mu_up:.4f}")
    print(f"状态: {info['status']}")
