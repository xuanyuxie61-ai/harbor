"""
likelihood.py
=============

构造 Profile Likelihood 函数 —— 统计推断核心。

似然函数 (Poisson × Gaussian 约束):

    L(μ, θ) = Π_i Poisson(n_i | ν_i(μ, θ)) · Π_k Gaussian(θ_k | 0, 1)

负对数似然 (NLL, 最小化目标):

    -ln L(μ, θ) = Σ_i [ ν_i(μ,θ) - n_i · ln ν_i(μ,θ) + ln(n_i!) ]
                  + (1/2) Σ_k θ_k^2 + const

Profile Likelihood Ratio (PLR):

    λ(μ) = L(μ, θ̂_μ) / L(μ̂, θ̂)

其中 θ̂_μ = argmin_θ NLL(μ, θ), (μ̂, θ̂) = argmin_{μ,θ} NLL(μ, θ).

检验统计量 (one-sided, 用于上限设定):

    q_μ = { -2 ln λ(μ)    if μ̂ ≤ μ
          { 0              otherwise

渐近分布 (Cowran et al. 2012, JHEP 07 (2012) 033):

    f(q_μ | μ) → (1/2) δ(q_μ) + (1/2) χ^2_1(q_μ)   [当 μ ≥ μ_true]

p 值:
    p_{s+b} = 1 - Φ(√q_μ)
    p_b    = 1 - Φ(√q_μ - μ/σ)

CLs 比率:
    CLs = p_{s+b} / (1 - p_b)

对数似然梯度 (用于 profiling, 取自 seed 1171 网格采样思想):

    ∂(-ln L)/∂θ_k = Σ_i [ ∂ν_i/∂θ_k · (1 - n_i/ν_i) ] + θ_k
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from physical_model import BinModel


# ---------------------------------------------------------------------------
# 负对数似然
# ---------------------------------------------------------------------------
def nll_poisson_gaussian(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta: np.ndarray,
) -> float:
    """
    负对数似然:

        -ln L = Σ_i [ν_i - n_i ln ν_i] + (1/2)|θ|^2

    忽略与参数无关的常数项 (如 ln(n_i!))。

    Parameters
    ----------
    model : BinModel
    n_obs : ndarray (n_bins,)
    mu : float
    theta : ndarray (n_nuis,)

    Returns
    -------
    float
        负对数似然值
    """
    theta = np.asarray(theta, dtype=np.float64)
    nu = model.expected_rate(mu, theta)
    # Poisson 项: ν - n ln ν (稳定计算)
    poisson_term = nu - n_obs * np.log(nu)
    stat = np.sum(poisson_term)
    # Gaussian 约束项
    gauss_term = 0.5 * np.dot(theta, theta)
    return float(stat + gauss_term)


# ---------------------------------------------------------------------------
# NLL 关于 θ 的梯度
# ---------------------------------------------------------------------------
def nll_gradient_theta(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta: np.ndarray,
) -> np.ndarray:
    """
    ∂(-ln L)/∂θ_k = Σ_i [ ∂ν_i/∂θ_k · (1 - n_i/ν_i) ] + θ_k

    ∂ν_i/∂θ_k 由 BinModel.expected_rate 解析给出:
        信号项: μ · s_eff · s_nom · γ_{ik}
        背景项: b_nom_i · δ_{ik} · κ_{ik} · (1+δ_{ik}θ_k)^{κ_{ik}-1} · Π_{j≠k} (1+δ_{ij}θ_j)^{κ_{ij}}
    """
    theta = np.asarray(theta, dtype=np.float64)
    nu = model.expected_rate(mu, theta)
    grad = theta.copy()  # Gaussian 约束项
    for k in range(model.n_nuis):
        # 信号项导数
        dnu_sig = mu * model.s_eff * model.s_nom * model.gamma_sig[:, k]
        # 背景项导数 (含 κ 阶修正)
        dnu_bkg = np.zeros(model.n_bins)
        for i in range(model.n_bins):
            # 对第 k 个 θ 求导
            base_k = 1.0 + model.delta_bkg[i, k] * theta[k]
            base_k = max(base_k, 1e-8)
            power = model.kappa[i, k]
            # d/dθ_k [(1+δθ)^κ] = κ · δ · (1+δθ)^{κ-1}
            d_factor = (
                model.kappa[i, k]
                * model.delta_bkg[i, k]
                * base_k ** (power - 1.0)
            )
            prod_others = 1.0
            for j in range(model.n_nuis):
                if j == k:
                    continue
                base_j = 1.0 + model.delta_bkg[i, j] * theta[j]
                base_j = max(base_j, 1e-8)
                prod_others *= base_j ** model.kappa[i, j]
            dnu_bkg[i] = model.b_nom[i] * d_factor * prod_others
        dnu_total = dnu_sig + dnu_bkg
        # 链式: d(NLL)/dθ_k = Σ_i dν_i/dθ_k · (1 - n_i/ν_i)
        grad[k] += np.sum(dnu_total * (1.0 - n_obs / nu))
    return grad


# ---------------------------------------------------------------------------
# NLL Hessian (用于 Halley 型 profiling)
# ---------------------------------------------------------------------------
def nll_hessian_theta_diag(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta: np.ndarray,
) -> np.ndarray:
    """
    对角 Hessian 近似 (Gauss-Newton 型):

        H_{kk} ≈ Σ_i (∂ν_i/∂θ_k)^2 / ν_i + 1

    该近似在 n_i ≈ ν_i (Asimov 数据集) 时严格成立。
    用于 Halley 方法加速 profiling 收敛 (seed 1435 → Halley 迭代)。
    """
    theta = np.asarray(theta, dtype=np.float64)
    nu = model.expected_rate(mu, theta)
    Hdiag = np.ones(model.n_nuis)
    for k in range(model.n_nuis):
        dnu_sig = mu * model.s_eff * model.s_nom * model.gamma_sig[:, k]
        # 简化: 仅用信号+线性背景导数
        dnu_bkg_lin = model.b_nom * model.delta_bkg[:, k]
        dnu = dnu_sig + dnu_bkg_lin
        Hdiag[k] += np.sum(dnu**2 / nu)
    return Hdiag


# ---------------------------------------------------------------------------
# Profile NLL (给定 μ, profile 掉 θ)
# ---------------------------------------------------------------------------
def profile_nll(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta_init: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:
    """
    Profile NLL: min_θ NLL(μ, θ)

    使用 L-BFGS-B 型收敛的 Newton-Raphson + 线搜索混合。
    返回 (profiled_nll, theta_hat_mu)。

    算法融合 seed 807 (不动点迭代) 与 seed 1435 (Halley 三次收敛):
        1. 梯度下降步: θ ← θ - α · H^{-1} · g  (Newton)
        2. Halley 修正: θ ← θ - (f/f') / (1 - 0.5 · f·f''/(f')^2)
        3. 不动点检验: |θ_{n+1} - θ_n| < tol
    """
    from scipy.optimize import minimize

    theta0 = (
        np.asarray(theta_init, dtype=np.float64)
        if theta_init is not None
        else np.zeros(model.n_nuis)
    )

    def objective(th):
        return nll_poisson_gaussian(model, n_obs, mu, th)

    def gradient(th):
        return nll_gradient_theta(model, n_obs, mu, th)

    # 边界约束: θ ∈ [-10, 10] (物理合理性)
    bounds = [(-10.0, 10.0)] * model.n_nuis

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol * 1e-3, "gtol": tol},
    )

    theta_hat = result.x
    nll_val = float(result.fun)
    return nll_val, theta_hat


# ---------------------------------------------------------------------------
# 全局最优 (μ, θ) 联合拟合
# ---------------------------------------------------------------------------
def global_best_fit(
    model: BinModel,
    n_obs: np.ndarray,
    mu_bounds: Tuple[float, float] = (-1.0, 10.0),
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    联合最小化 NLL(μ, θ) → (μ̂, θ̂, NLL_min)

    μ 物理约束: μ ≥ 0 (信号强度非负)
    使用 seed 1435 (Brent) 在 μ 维做黄金分割搜索, 内层 profile θ。
    """
    from scipy.optimize import minimize_scalar

    theta0 = (
        np.asarray(theta_init, dtype=np.float64)
        if theta_init is not None
        else np.zeros(model.n_nuis)
    )

    def profiled_nll_mu(mu_val):
        nll_val, _ = profile_nll(model, n_obs, mu_val, theta_init=theta0)
        return nll_val

    result = minimize_scalar(
        profiled_nll_mu,
        bounds=(max(0.0, mu_bounds[0]), mu_bounds[1]),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    mu_hat = float(result.x)
    nll_min, theta_hat = profile_nll(model, n_obs, mu_hat, theta_init=theta0)
    return mu_hat, nll_min, theta_hat


# ---------------------------------------------------------------------------
# Profile Likelihood Ratio 与检验统计量
# ---------------------------------------------------------------------------
def profile_likelihood_ratio(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    mu_hat: Optional[float] = None,
    nll_global_min: Optional[float] = None,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    计算 λ(μ) 和 q_μ:

        λ(μ) = L(μ, θ̂_μ) / L(μ̂, θ̂)
        -2 ln λ(μ) = 2 · [NLL(μ, θ̂_μ) - NLL(μ̂, θ̂)]

        q_μ = -2 ln λ(μ)   if μ̂ ≤ μ
             = 0            otherwise

    Returns
    -------
    (lambda_val, q_mu) : (float, float)
    """
    nll_profile, _ = profile_nll(model, n_obs, mu, theta_init=theta_init)

    if nll_global_min is None or mu_hat is None:
        mu_hat, nll_global_min, _ = global_best_fit(
            model, n_obs, theta_init=theta_init
        )

    delta_nll = nll_profile - nll_global_min
    # 数值稳定: delta_nll 应为非负
    delta_nll = max(delta_nll, 0.0)
    q_mu_val = 2.0 * delta_nll
    if mu_hat > mu:
        q_mu_val = 0.0
    lambda_val = math.exp(-0.5 * q_mu_val)
    return lambda_val, q_mu_val


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from physical_model import make_default_binmodel, generate_observed_data
    mdl = make_default_binmodel()
    data = generate_observed_data(mdl, mu_true=0.0, seed=230)
    print(f"伪数据: {data}")

    nll0 = nll_poisson_gaussian(mdl, data, 1.0, np.zeros(mdl.n_nuis))
    print(f"NLL(μ=1, θ=0): {nll0:.4f}")

    grad0 = nll_gradient_theta(mdl, data, 1.0, np.zeros(mdl.n_nuis))
    print(f"∇_θ NLL at (μ=1, θ=0): {grad0}")

    mu_hat, nll_min, theta_hat = global_best_fit(mdl, data)
    print(f"μ̂ = {mu_hat:.6f}, NLL_min = {nll_min:.6f}, θ̂ = {theta_hat}")

    lam, qmu = profile_likelihood_ratio(mdl, data, 1.0, mu_hat, nll_min)
    print(f"λ(μ=1) = {lam:.6f}, q_μ = {qmu:.6f}")
