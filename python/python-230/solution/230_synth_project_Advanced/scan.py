"""
scan.py
=======

Profile Likelihood μ 扫描与上限确定。

种子项目 1297 (FormalCellular) 启发: 在 μ 的离散网格上做
CDF 式扫描, 结合 seed 1435 (Brent) 的精确求根定位上限。

扫描策略:
1. 粗扫描: μ ∈ [0, μ_max] 等距网格, 计算 profiled NLL 和 CLs
2. 精扫描: 在 CLs ≈ α 附近用 Brent 精化
3. 输出: CLs(μ) 曲线, 观测上限, 期望上限 ± 1σ, 2σ band

Profile Likelihood 扫描:
    对每个 μ_i:
        θ̂_{μ_i} = argmin_θ NLL(μ_i, θ)
        q_{μ_i} = 2 · [NLL(μ_i, θ̂_{μ_i}) - NLL(μ̂, θ̂)]
        ΔNLL_i = NLL(μ_i, θ̂_{μ_i}) - NLL(μ̂, θ̂)

期望限 (Asimov):
    对 b-only Asimov 数据, 计算 median 和 ±1σ, ±2σ band:
    μ_up^{exp} = 渐近上限(Asimov b-only)
    band_{1σ} = [μ_up^{exp} - σ_exp, μ_up^{exp} + σ_exp]
    band_{2σ} = [μ_up^{exp} - 2σ_exp, μ_up^{exp} + 2σ_exp]
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from physical_model import BinModel, generate_observed_data
from likelihood import (
    profile_nll,
    global_best_fit,
    profile_likelihood_ratio,
)
from asymptotic import (
    asymptotic_cls,
    asymptotic_upper_limit,
    estimate_sigma,
    asimov_q_mu,
    asimov_data_background,
    Phi,
    Phi_inv,
)


# ---------------------------------------------------------------------------
# Profile Likelihood 扫描
# ---------------------------------------------------------------------------
def scan_profile_likelihood(
    model: BinModel,
    n_obs: np.ndarray,
    mu_values: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    在 μ 网格上扫描 profile likelihood。

    Returns
    -------
    (mu_values, delta_nll, q_mu, info) : (ndarray, ndarray, ndarray, dict)

    delta_nll[i] = NLL(μ_i, θ̂_{μ_i}) - NLL(μ̂, θ̂)
    q_mu[i] = 2 · delta_nll[i]  (one-sided: 若 μ̂ > μ_i 则 q=0)
    """
    mu_values = np.asarray(mu_values, dtype=np.float64)
    n_scan = len(mu_values)
    delta_nll = np.zeros(n_scan)
    q_mu = np.zeros(n_scan)
    theta_hats = np.zeros((n_scan, model.n_nuis))

    # 全局最优
    mu_hat, nll_min, theta_hat_global = global_best_fit(
        model, n_obs, theta_init=theta_init,
    )
    info = {
        "mu_hat": mu_hat,
        "nll_min": nll_min,
        "theta_hat_global": theta_hat_global,
    }

    theta_prev = theta_init
    for i, mu in enumerate(mu_values):
        nll_prof, theta_prof = profile_nll(
            model, n_obs, mu, theta_init=theta_prev,
        )
        dnll = nll_prof - nll_min
        dnll = max(dnll, 0.0)
        delta_nll[i] = dnll
        q_mu[i] = 2.0 * dnll if mu_hat <= mu else 0.0
        theta_hats[i] = theta_prof
        theta_prev = theta_prof  # 热启动

    info["theta_hats"] = theta_hats
    return mu_values, delta_nll, q_mu, info


# ---------------------------------------------------------------------------
# CLs 扫描 (渐近)
# ---------------------------------------------------------------------------
def scan_cls_asymptotic(
    model: BinModel,
    n_obs: np.ndarray,
    mu_values: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    渐近 CLs 扫描。

    Returns
    -------
    (mu_values, cls_values, p_sb_values, p_b_values, info)
    """
    mu_values, delta_nll, q_mu, scan_info = scan_profile_likelihood(
        model, n_obs, mu_values, theta_init,
    )

    sigma = estimate_sigma(model, mu=1.0)
    n_scan = len(mu_values)
    cls_values = np.zeros(n_scan)
    p_sb_values = np.zeros(n_scan)
    p_b_values = np.zeros(n_scan)

    for i, mu in enumerate(mu_values):
        if mu <= 0.0:
            cls_values[i] = 1.0
            continue
        cls_val, p_sb, p_b = asymptotic_cls(q_mu[i], mu, sigma)
        cls_values[i] = cls_val
        p_sb_values[i] = p_sb
        p_b_values[i] = p_b

    info = {
        **scan_info,
        "sigma": sigma,
        "q_mu": q_mu,
        "delta_nll": delta_nll,
    }
    return mu_values, cls_values, p_sb_values, p_b_values, info


# ---------------------------------------------------------------------------
# 期望限与 Band
# ---------------------------------------------------------------------------
def compute_expected_band(
    model: BinModel,
    mu_values: np.ndarray,
    n_sigma_bands: List[int] = [1, 2],
) -> Dict[str, np.ndarray]:
    """
    计算期望限 band (Asimov b-only)。

    在背景假设下, CLs(μ) 的期望分布:
        median CLs(μ) = CLs_asymptotic(q_{μ,A}(μ'=0), μ, σ)

    ± Nσ band:
        CLs_{±Nσ}(μ) = Φ(Φ^{-1}(CLs_median) ± Nσ · σ_CLs)

    简化: 使用 Asimov 估计 μ_up^{exp}, 然后用 σ_μ 构建 band。

    Returns
    -------
    dict
        keys: "mu_up_median", "mu_up_pm1sigma", "mu_up_pm2sigma"
    """
    sigma = estimate_sigma(model, mu=1.0)
    band_info = {"sigma": sigma}

    # Asimov b-only 数据
    n_asimov, _ = asimov_data_background(model)

    # 期望上限 (median)
    mu_up_exp, exp_info = asymptotic_upper_limit(model, alpha=0.05)
    band_info["mu_up_median"] = mu_up_exp
    band_info["exp_status"] = exp_info.get("status", "unknown")

    # σ_μ (上限的不确定度)
    # 渐近: σ_μ ≈ σ (从 q_μ,A 估计)
    sigma_mu = sigma
    for ns in n_sigma_bands:
        band_info[f"mu_up_plus_{ns}sigma"] = mu_up_exp + ns * sigma_mu
        band_info[f"mu_up_minus_{ns}sigma"] = max(0.0, mu_up_exp - ns * sigma_mu)

    return band_info


# ---------------------------------------------------------------------------
# 上限精确定位 (Brent)
# ---------------------------------------------------------------------------
def find_cls_crossing(
    model: BinModel,
    n_obs: np.ndarray,
    alpha: float = 0.05,
    mu_lo: float = 0.01,
    mu_hi: float = 20.0,
    tol: float = 1e-4,
) -> Tuple[float, Dict]:
    """
    用 Brent 方法精确求解 CLs(μ_up) = α。

    Returns
    -------
    (mu_up, info) : (float, dict)
    """
    from scipy.optimize import brentq

    sigma = estimate_sigma(model, mu=1.0)

    # 全局拟合
    mu_hat, nll_min, _ = global_best_fit(model, n_obs)

    def cls_residual(mu_val):
        if mu_val <= 0.0:
            return 1.0 - alpha
        nll_prof, _ = profile_nll(model, n_obs, mu_val)
        dnll = max(nll_prof - nll_min, 0.0)
        q_mu = 2.0 * dnll if mu_hat <= mu_val else 0.0
        cls_val, _, _ = asymptotic_cls(q_mu, mu_val, sigma)
        return cls_val - alpha

    # 粗扫描找 bracket
    mu_grid = np.linspace(mu_lo, mu_hi, 100)
    residuals = []
    for mu in mu_grid:
        try:
            r = cls_residual(mu)
        except Exception:
            r = float("nan")
        residuals.append(r)
    residuals = np.array(residuals)

    # 找符号变化
    sign_changes = np.where(
        (residuals[:-1] * residuals[1:] < 0) & np.isfinite(residuals[:-1]) & np.isfinite(residuals[1:])
    )[0]

    info = {"mu_grid": mu_grid, "residuals": residuals}

    if len(sign_changes) == 0:
        if np.all(residuals > 0):
            info["status"] = "cls_always_above_alpha"
            return mu_hi, info
        else:
            info["status"] = "cls_always_below_alpha"
            return mu_lo, info

    # Brent 精化
    idx = sign_changes[0]
    try:
        mu_up = brentq(cls_residual, mu_grid[idx], mu_grid[idx + 1],
                       xtol=tol, maxiter=200)
        info["status"] = "converged"
    except ValueError:
        mu_up = 0.5 * (mu_grid[idx] + mu_grid[idx + 1])
        info["status"] = "brent_failed"

    return float(mu_up), info


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from physical_model import make_default_binmodel, generate_observed_data

    mdl = make_default_binmodel()
    data = generate_observed_data(mdl, mu_true=0.0, seed=230)

    mu_grid = np.linspace(0.0, 5.0, 25)
    mu_vals, delta_nll, q_mu, info = scan_profile_likelihood(mdl, data, mu_grid)

    print(f"μ̂ = {info['mu_hat']:.4f}")
    print(f"Profile NLL 扫描 (部分):")
    for i in range(0, len(mu_vals), 5):
        print(f"  μ={mu_vals[i]:.2f}: ΔNLL={delta_nll[i]:.4f}, q_μ={q_mu[i]:.4f}")

    # CLs 扫描
    mu_cls, cls, p_sb, p_b, cls_info = scan_cls_asymptotic(mdl, data, mu_grid)
    print(f"\nCLs 扫描 (部分):")
    for i in range(0, len(mu_cls), 5):
        print(f"  μ={mu_cls[i]:.2f}: CLs={cls[i]:.6f}")

    # 上限
    mu_up, up_info = find_cls_crossing(mdl, data, alpha=0.05)
    print(f"\n95% CL 上限: μ_up = {mu_up:.4f} ({up_info['status']})")

    # 期望 band
    band = compute_expected_band(mdl, mu_grid)
    print(f"期望上限: {band['mu_up_median']:.4f}")
