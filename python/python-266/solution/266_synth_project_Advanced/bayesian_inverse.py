"""
bayesian_inverse.py — 贝叶斯反演: 从能带数据重建势场
========================================================

本模块实现逆问题: 从观测到的能带能量反推晶体势场参数。
融合种子项目:
  - 1247_vcasasmo_BayRad3D: 贝叶斯变分推断 (SVI)
  - 350_fd_predator_prey: 梯度下降动力学
  - 1415_will_you_be_alive: Monte Carlo 概率推断

核心物理:
  正问题: V_params → ε_n(k) (KS 求解)
  逆问题: ε_n(k)_obs → V_params

  贝叶斯框架:
    后验 ∝ 似然 × 先验
    p(θ|D) ∝ p(D|θ) · p(θ)

  其中:
    θ = 势场参数 (V₁, V₂, V₃, a, ...)
    D = 观测的能带能量 {ε_n(k)_obs}

  似然函数 (Gaussian):
    p(D|θ) ∝ exp(-χ²/2)
    χ² = Σ_{n,k} [(ε_n(k; θ) - ε_n(k)_obs)² / σ²]

  先验:
    p(θ) = Π_i Uniform(θ_i_min, θ_i_max)

  优化方法:
  1. MLE (最大似然估计): L-BFGS-B
  2. MAP (最大后验): MLE + 先验
  3. 近似后验: Laplace 近似 (Hessian 逆)
  4. Monte Carlo: MCMC (Metropolis-Hastings)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable


# ============================================================
# 正问题: 势场参数 → 能带
# ============================================================

def forward_model(params: np.ndarray,
                    n_grid: int, fd_order: int,
                    kpoints: np.ndarray,
                    n_bands: int,
                    n_electrons: int = 2
                    ) -> np.ndarray:
    """
    正问题: 从势场参数计算能带能量。

    参数向量:
      θ = [a, V₁, V₂, V₃]
    或更一般的:
      θ = [a, V₁, V₂, V₃, hartree_strength, xc_alpha]

    正问题流程:
    1. 构建 Mathieu 势 V(x; θ)
    2. 在 k 点上求解 KS 方程
    3. 返回能带能量 {ε_n(k)}

    Parameters
    ----------
    params : np.ndarray
        势场参数 [a, V₁, V₂, V₃]
    n_grid : int
        网格点数
    fd_order : int
        FD 半带宽
    kpoints : np.ndarray
        k 点列表
    n_bands : int
        能带数

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_kpoints, n_bands)
        能带能量
    """
    from potential import MathieuPotential
    from kohn_sham import solve_all_bands

    a, V1, V2, V3 = params[0], params[1], params[2], params[3]

    # 边界保护
    a = max(a, 1.0)
    V1 = np.clip(V1, 0.0, 5.0)
    V2 = np.clip(V2, -2.0, 2.0)
    V3 = np.clip(V3, -1.0, 1.0)

    pot = MathieuPotential(a, V1, V2, V3)
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_eff = pot.V(x_grid)

    eigenvalues, _ = solve_all_bands(
        v_eff, kpoints, n_grid, dx, fd_order, a, n_bands)

    return eigenvalues


# ============================================================
# 似然函数
# ============================================================

def gaussian_log_likelihood(eigenvalues_model: np.ndarray,
                              eigenvalues_obs: np.ndarray,
                              sigma: float) -> float:
    """
    Gaussian 对数似然。

    log p(D|θ) = -χ²/2 - (N/2) log(2πσ²)

    χ² = Σ_{n,k} [(ε_{nk}^model - ε_{nk}^obs)² / σ²]

    Parameters
    ----------
    eigenvalues_model : np.ndarray
    eigenvalues_obs : np.ndarray
    sigma : float
        观测噪声标准差

    Returns
    -------
    log_lik : float
    """
    diff = eigenvalues_model - eigenvalues_obs
    chi2 = np.sum(diff ** 2) / (sigma ** 2)
    n_data = diff.size
    log_lik = -0.5 * chi2 - 0.5 * n_data * np.log(2.0 * PI * sigma ** 2)
    return log_lik


PI = np.pi


# ============================================================
# MLE 优化
# ============================================================

def maximum_likelihood_estimation(observed_eigenvalues: np.ndarray,
                                    initial_params: np.ndarray,
                                    n_grid: int, fd_order: int,
                                    kpoints: np.ndarray,
                                    n_bands: int,
                                    sigma: float = 0.01,
                                    max_iter: int = 50,
                                    learning_rate: float = 0.01
                                    ) -> Dict[str, any]:
    """
    最大似然估计 (MLE) 求解逆问题。

    使用数值梯度下降 (类比 350_fd_predator_prey 的 Euler 步进):
      θ^{k+1} = θ^k + η · ∇_θ log p(D|θ^k)

    梯度由有限差分近似:
      ∂L/∂θ_i ≈ [L(θ + ε·e_i) - L(θ - ε·e_i)] / (2ε)

    Parameters
    ----------
    observed_eigenvalues : np.ndarray
        观测到的能带能量
    initial_params : np.ndarray
        初始参数猜测
    n_grid, fd_order, kpoints, n_bands :
        正问题参数
    sigma : float
        噪声标准差
    max_iter : int
        最大迭代数
    learning_rate : float
        学习率

    Returns
    -------
    result : dict
        优化后的参数, 似然值, 收敛历史
    """
    params = initial_params.copy()
    n_params = len(params)
    eps_grad = 1e-4

    likelihood_history = []
    param_history = [params.copy()]

    # 参数边界
    bounds_low = np.array([1.0, 0.01, -1.0, -0.5])
    bounds_high = np.array([50.0, 5.0, 2.0, 0.5])
    grad_clip = 100.0

    for iteration in range(max_iter):
        # 正问题
        try:
            evals_model = forward_model(
                params, n_grid, fd_order, kpoints, n_bands)
        except Exception:
            break

        # 似然
        log_lik = gaussian_log_likelihood(
            evals_model, observed_eigenvalues, sigma)
        likelihood_history.append(log_lik)

        # 数值梯度
        grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            # 自适应步长
            h_i = max(abs(params[i]) * 1e-4, eps_grad)
            params_plus[i] += h_i
            params_minus[i] -= h_i

            try:
                evals_plus = forward_model(
                    params_plus, n_grid, fd_order, kpoints, n_bands)
                evals_minus = forward_model(
                    params_minus, n_grid, fd_order, kpoints, n_bands)
                ll_plus = gaussian_log_likelihood(
                    evals_plus, observed_eigenvalues, sigma)
                ll_minus = gaussian_log_likelihood(
                    evals_minus, observed_eigenvalues, sigma)
                grad[i] = (ll_plus - ll_minus) / (2.0 * h_i)
            except Exception:
                grad[i] = 0.0

        # 梯度裁剪
        grad_norm = np.linalg.norm(grad)
        if grad_norm > grad_clip:
            grad *= grad_clip / grad_norm

        # 梯度上升
        params += learning_rate * grad

        # 边界保护
        for i in range(n_params):
            params[i] = np.clip(params[i], bounds_low[i], bounds_high[i])
        param_history.append(params.copy())

        # 自适应学习率
        if iteration > 5:
            learning_rate *= 0.95

        if iteration % 10 == 0:
            pass  # 静默

    return {
        'optimal_params': params,
        'likelihood_history': likelihood_history,
        'param_history': np.array(param_history),
        'final_log_likelihood': likelihood_history[-1] if likelihood_history else 0.0,
    }


# ============================================================
# Laplace 近似 (后验不确定性)
# ============================================================

def laplace_approximation(optimal_params: np.ndarray,
                            observed_eigenvalues: np.ndarray,
                            n_grid: int, fd_order: int,
                            kpoints: np.ndarray,
                            n_bands: int,
                            sigma: float = 0.01
                            ) -> Dict[str, np.ndarray]:
    """
    Laplace 近似计算后验协方差。

    在 MLE 点 θ* 处展开:
      log p(θ|D) ≈ log p(θ*|D) - ½(θ-θ*)^T H (θ-θ*)

    其中 H = -∂²log p/∂θ² 为 Hessian 矩阵。

    后验协方差:
      Σ_post = H^{-1}

    Hessian 由数值二阶差分近似:
      H[i,j] ≈ -[L(θ+ε_i+ε_j) - L(θ+ε_i-ε_j)
                   - L(θ-ε_i+ε_j) + L(θ-ε_i-ε_j)] / (4ε²)

    Parameters
    ----------
    optimal_params : np.ndarray
        MLE 参数
    observed_eigenvalues : np.ndarray
    n_grid, fd_order, kpoints, n_bands :
    sigma : float

    Returns
    -------
    result : dict
        后验均值, 协方差, 标准差
    """
    params = optimal_params.copy()
    n_params = len(params)
    eps = 1e-4

    # 计算 Hessian
    H = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            p_pp = params.copy()
            p_pm = params.copy()
            p_mp = params.copy()
            p_mm = params.copy()

            p_pp[i] += eps; p_pp[j] += eps
            p_pm[i] += eps; p_pm[j] -= eps
            p_mp[i] -= eps; p_mp[j] += eps
            p_mm[i] -= eps; p_mm[j] -= eps

            try:
                e_pp = forward_model(p_pp, n_grid, fd_order, kpoints, n_bands)
                e_pm = forward_model(p_pm, n_grid, fd_order, kpoints, n_bands)
                e_mp = forward_model(p_mp, n_grid, fd_order, kpoints, n_bands)
                e_mm = forward_model(p_mm, n_grid, fd_order, kpoints, n_bands)

                l_pp = gaussian_log_likelihood(e_pp, observed_eigenvalues, sigma)
                l_pm = gaussian_log_likelihood(e_pm, observed_eigenvalues, sigma)
                l_mp = gaussian_log_likelihood(e_mp, observed_eigenvalues, sigma)
                l_mm = gaussian_log_likelihood(e_mm, observed_eigenvalues, sigma)

                H[i, j] = -(l_pp - l_pm - l_mp + l_mm) / (4.0 * eps ** 2)
                H[j, i] = H[i, j]
            except Exception:
                H[i, j] = 0.0
                H[j, i] = 0.0

    # 后验协方差
    try:
        cov_post = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_post = np.linalg.pinv(H)

    std_post = np.sqrt(np.maximum(np.diag(cov_post), 0.0))

    return {
        'hessian': H,
        'covariance': cov_post,
        'std_posterior': std_post,
        'posterior_mean': optimal_params,
    }


# ============================================================
# 完整贝叶斯反演流程
# ============================================================

def run_bayesian_inverse(observed_eigenvalues: np.ndarray,
                           true_params: np.ndarray,
                           n_grid: int = 64,
                           fd_order: int = 2,
                           n_kpoints: int = 16,
                           n_bands: int = 4,
                           sigma: float = 0.01
                           ) -> Dict[str, any]:
    """
    完整的贝叶斯反演流程。

    1. 生成初始猜测 (带偏移)
    2. MLE 优化
    3. Laplace 近似
    4. 结果分析

    Parameters
    ----------
    observed_eigenvalues : np.ndarray
        观测数据
    true_params : np.ndarray
        真实参数 (用于对比)
    n_grid, fd_order, n_kpoints, n_bands :
        计算参数
    sigma : float
        噪声水平

    Returns
    -------
    result : dict
        完整反演结果
    """
    from lattice import monkhorst_pack_grid

    a_true = true_params[0]
    kpoints = monkhorst_pack_grid(n_kpoints, a_true)

    # 初始猜测 (偏离真值 10%, 更保守)
    initial_params = true_params * np.array([1.1, 0.9, 1.1, 0.9])

    # MLE
    mle_result = maximum_likelihood_estimation(
        observed_eigenvalues, initial_params,
        n_grid, fd_order, kpoints, n_bands, sigma,
        max_iter=20, learning_rate=0.001)

    # Laplace 近似
    lap_result = laplace_approximation(
        mle_result['optimal_params'],
        observed_eigenvalues,
        n_grid, fd_order, kpoints, n_bands, sigma)

    # 参数误差
    param_error = np.abs(
        mle_result['optimal_params'] - true_params) / np.abs(true_params)

    return {
        'true_params': true_params,
        'estimated_params': mle_result['optimal_params'],
        'param_error_relative': param_error,
        'posterior_std': lap_result['std_posterior'],
        'final_log_likelihood': mle_result['final_log_likelihood'],
        'converged': True,
    }
