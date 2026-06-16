#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stochastic_robustness.py — 随机鲁棒性分析与不确定量化

对应种子项目:
  - 839_ornstein_uhlenbeck: OU 过程建模材料性能随机波动
  - 1109_marekgluza_Fidelity_witnesses_example: 保真度 witnesses 用于量化前沿质量

核心数学公式:
  鲁棒优化 (worst-case):
      min_x max_{ξ ∈ U} F(x, ξ)
  其中 U 为不确定集.

  期望值模型:
      min_x E_ξ [F(x, ξ)] + λ · Var_ξ [F(x, ξ)]

  Polynomial Chaos Expansion (PCE):
      F(x, ξ) ≈ Σ_{|α|≤p} c_α(x) Ψ_α(ξ)
  其中 Ψ_α 为正交多项式基.

  保真度 witness (来自 1109):
      F_w(ρ) = |⟨ψ_target | ψ(ρ)⟩|²
  映射到多目标: 衡量设计 ρ 与理想解的 "距离".

  广义 Polynomial Chaos (gPC):
      对 OU 过程的稳态分布 (Gauss):
      Ψ_k(ξ) = He_k(ξ / σ) / √(k!)  (Hermite 多项式)
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from adaptive_rk_integrator import ornstein_uhlenbeck_em, ou_analytical_moments


# ---------------------------------------------------------------------------
# 不确定量化 — Monte Carlo + gPC
# ---------------------------------------------------------------------------
class UncertaintyQuantifier:
    """
    不确定量化引擎.

    对设计变量 ρ 施加随机扰动:
        ρ̃ = ρ + σ_ρ · ξ(t)
    其中 ξ(t) 为 OU 过程, 模拟制造误差的时空相关性.

    鲁棒目标:
        F_robust = μ_F + β · σ_F
    其中 μ_F 为目标均值, σ_F 为目标标准差, β 为风险偏好参数.
    """

    def __init__(self, sigma_ou: float = 0.05,
                 theta_ou: float = 2.0,
                 beta_risk: float = 2.0,
                 n_mc: int = 100,
                 seed: int = 42):
        self.sigma_ou = sigma_ou      # OU 过程扩散系数
        self.theta_ou = theta_ou      # OU 过程回复速率
        self.beta_risk = beta_risk    # 风险偏好
        self.n_mc = n_mc              # MC 样本数
        self.seed = seed

    def generate_perturbations(self, rho: np.ndarray,
                               n_realizations: int = 50
                               ) -> np.ndarray:
        """
        生成材料密度的随机扰动场.

        对每个设计变量 ρ_i, 施加独立的 OU 过程扰动:
            δρ_i(t) = σ_ou · ξ_i(t)
            ξ_i ~ OU(θ, 0, 1)

        最终取稳态值: ξ ~ N(0, 1/(2θ))

        Returns
        -------
        rho_perturbed : shape (n_realizations, len(rho))
        """
        rng = np.random.default_rng(self.seed)
        n = len(rho)
        # OU 稳态: ξ ~ N(0, σ²/(2θ))
        var_stationary = self.sigma_ou ** 2 / (2 * self.theta_ou)
        std_stationary = np.sqrt(max(var_stationary, 1e-15))

        rho_perturbed = np.zeros((n_realizations, n))
        for r in range(n_realizations):
            xi = rng.normal(0, std_stationary, n)
            rho_tilde = rho + xi
            rho_tilde = np.clip(rho_tilde, 0.0, 1.0)
            rho_perturbed[r] = rho_tilde

        return rho_perturbed

    def robust_objective(self, f_samples: np.ndarray) -> Tuple[float, float]:
        """
        计算鲁棒目标值.

        F_robust = μ_F + β · σ_F

        Parameters
        ----------
        f_samples : shape (n_realizations,) — 各随机实现的目标值

        Returns
        -------
        (f_robust, f_std)
        """
        mu = np.mean(f_samples)
        sigma = np.std(f_samples, ddof=1) if len(f_samples) > 1 else 0.0
        f_robust = mu + self.beta_risk * sigma
        return float(f_robust), float(sigma)


# ---------------------------------------------------------------------------
# Hermite 多项式基 (gPC)
# ---------------------------------------------------------------------------
def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    概率学家 Hermite 多项式 He_n(x).

    递推关系:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x · He_n(x) - n · He_{n-1}(x)

    正交性:
        E[He_m(ξ) He_n(ξ)] = n! · δ_{mn},  ξ ~ N(0,1)
    """
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    He_prev2 = np.ones_like(x)
    He_prev1 = x.copy()
    for k in range(1, n):
        He_curr = x * He_prev1 - k * He_prev2
        He_prev2 = He_prev1
        He_prev1 = He_curr
    return He_prev1


def gpc_expansion(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    gPC 展开:
        F(ξ) ≈ Σ_{k=0}^{p} c_k He_k(ξ)

    Parameters
    ----------
    coeffs : shape (p+1,) — gPC 系数
    xi     : shape (N,)   — 标准正态样本

    Returns
    -------
    F_approx : shape (N,)
    """
    p = len(coeffs) - 1
    result = np.zeros_like(xi, dtype=np.float64)
    for k in range(p + 1):
        result += coeffs[k] * hermite_polynomial(k, xi)
    return result


def gpc_project(func_samples: np.ndarray, xi: np.ndarray,
                max_order: int = 4) -> np.ndarray:
    """
    通过投影估计 gPC 系数.

    c_k = (1/k!) · E[F(ξ) · He_k(ξ)]
        ≈ (1/k!) · (1/N) Σ_{i=1}^N F(ξ_i) He_k(ξ_i)

    Parameters
    ----------
    func_samples : shape (N,)
    xi           : shape (N,)
    max_order    : int

    Returns
    -------
    coeffs : shape (max_order+1,)
    """
    from math import factorial
    N = len(xi)
    coeffs = np.zeros(max_order + 1)
    for k in range(max_order + 1):
        He_k = hermite_polynomial(k, xi)
        coeffs[k] = np.mean(func_samples * He_k) / factorial(k)
    return coeffs


# ---------------------------------------------------------------------------
# 保真度 Witness (来自 1109 的映射)
# ---------------------------------------------------------------------------
def fidelity_witness(f_achieved: np.ndarray,
                     f_ideal: np.ndarray,
                     f_nadir: np.ndarray) -> float:
    """
    保真度 witness — 衡量近似解与理想解的 "接近度".

    类比量子保真度:
        F_w = |⟨ψ_ideal | ψ_achieved⟩|²

    映射到多目标:
        F_w = Π_{i=1}^{M} (1 - (f_i - f_i^*)/(f_i^nadir - f_i^*))

    其中 f_i^* 为第 i 个目标的理想值, f_i^nadir 为 nadir 值.
    F_w ∈ [0, 1], 越大表示越接近理想解.

    Parameters
    ----------
    f_achieved : shape (M,) —  achieved 目标值
    f_ideal    : shape (M,) — 各目标理想值 (utopia point)
    f_nadir    : shape (M,) — nadir 值

    Returns
    -------
    float ∈ [0, 1]
    """
    M = len(f_achieved)
    fw = 1.0
    for i in range(M):
        denom = f_nadir[i] - f_ideal[i]
        if abs(denom) < 1e-14:
            continue
        ratio = (f_achieved[i] - f_ideal[i]) / denom
        ratio = max(0.0, min(1.0, ratio))
        fw *= (1.0 - ratio)
    return float(fw)


def fidelity_matrix(pf: np.ndarray,
                    f_ideal: np.ndarray,
                    f_nadir: np.ndarray) -> np.ndarray:
    """
    计算 Pareto 前沿上每个解的保真度 witness.

    Returns: shape (K,) — 每个解的保真度值
    """
    K = pf.shape[0]
    fw = np.zeros(K)
    for i in range(K):
        fw[i] = fidelity_witness(pf[i], f_ideal, f_nadir)
    return fw


# ---------------------------------------------------------------------------
# OU 过程验证
# ---------------------------------------------------------------------------
def verify_ou_statistics(theta: float, mu: float, sigma: float,
                         x0: float, tmax: float,
                         n_steps: int = 10000,
                         n_paths: int = 200
                         ) -> dict:
    """
    验证 OU 过程的数值统计与解析统计的一致性.

    解析:
        E[x(t)] = μ + (x_0 - μ) exp(-θ t)
        Var[x(t)] = σ²/(2θ) (1 - exp(-2θ t))

    Returns
    -------
    dict with 't', 'mean_numerical', 'mean_analytical',
              'var_numerical', 'var_analytical'
    """
    all_paths = np.zeros((n_paths, n_steps + 1))
    t_arr = None
    for p in range(n_paths):
        t_arr, x_arr = ornstein_uhlenbeck_em(
            theta, mu, sigma, x0, tmax, n_steps, seed=p
        )
        all_paths[p] = x_arr

    mean_num = np.mean(all_paths, axis=0)
    var_num = np.var(all_paths, axis=0)

    mean_ana = np.array([ou_analytical_moments(theta, mu, sigma, x0, t)[0]
                         for t in t_arr])
    var_ana = np.array([ou_analytical_moments(theta, mu, sigma, x0, t)[1]
                        for t in t_arr])

    return {
        't': t_arr,
        'mean_numerical': mean_num,
        'mean_analytical': mean_ana,
        'var_numerical': var_num,
        'var_analytical': var_ana
    }
