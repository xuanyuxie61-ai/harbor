"""
entropy_norm.py — 熵产生率与 L2 范数诊断
========================================

种子项目映射:
  - 813_norm_l2: L2 范数的数值计算

物理基础:
  Boltzmann H-定理: 对于碰撞算子 C[f],
    dH/dt ≤ 0,  其中 H = ∫ f ln f d³v
  等号成立当且仅当 f 是 Maxwellian.

  熵产生率:
    σ = -dH/dt = -∫ C[f] (1 + ln f) d³v
               = ∫ D(v) (∂f/∂v)² / f · 4π v² dv  ≥ 0

  L2 范数:
    ||f - f_M||_2 = [∫ (f - f_M)² d³v]^{1/2}
  衡量分布函数与平衡态的距离.
"""

import math
import numpy as np

from physical_constants import PI, FOUR_PI, maxwellian_1d


# ===========================================================================
#  §1  L2 范数  (源自 norm_l2)
# ===========================================================================
def l2_norm_discrete(x, f, weight=None):
    """离散 L2 范数: ||f||_2 = [∫ f² d³v]^{1/2} ≈ [Σ f_i² w_i]^{1/2}.

    映射自 norm_l2:
      原项目: value = sqrt(∫_a^b f(x)² dx)  (连续积分)
      本项目: value = sqrt(4π ∫ f(v)² v² dv)  (速度空间 L2 范数)

    Parameters
    ----------
    x : ndarray  速度网格
    f : ndarray  分布函数
    weight : ndarray or None  额外权重 (如球壳体积元 4π v²)
    """
    if weight is None:
        weight = FOUR_PI * x**2

    integrand = f**2 * weight
    value = np.trapz(integrand, x)
    return math.sqrt(max(value, 0.0))


def l2_norm_perturbation(x, f, f_maxwellian):
    """扰动 L2 范数: ||δf||_2 = ||f - f_M||_2.

    衡量当前分布与 Maxwellian 的距离.
    """
    delta_f = f - f_maxwellian
    return l2_norm_discrete(x, delta_f)


def l2_norm_relative(x, f, f_reference):
    """相对 L2 误差: ||f - f_ref||_2 / ||f_ref||_2."""
    num = l2_norm_discrete(x, f - f_reference)
    den = l2_norm_discrete(x, f_reference)
    if den < 1e-30:
        return float('inf')
    return num / den


# ===========================================================================
#  §2  Boltzmann H-泛函与熵
# ===========================================================================
def boltzmann_H(x, f, eps=1e-30):
    """Boltzmann H-泛函: H[f] = ∫ f ln(f) d³v = 4π ∫ f(v) ln(f(v)) v² dv.

    H-定理: dH/dt ≤ 0.
    平衡态: H = H_min (对给定 n, E).

    eps 用于防止 f=0 时 ln(f) 发散.
    """
    f_safe = np.maximum(f, eps)
    integrand = f_safe * np.log(f_safe) * FOUR_PI * x**2
    return np.trapz(integrand, x)


def entropy_S(x, f, eps=1e-30):
    """Boltzmann 熵: S = -H = -∫ f ln f d³v."""
    return -boltzmann_H(x, f, eps)


def entropy_production_rate(x, f, C_f, eps=1e-30):
    """熵产生率: σ = -∫ C[f] (1 + ln f) d³v.

    由 H-定理, σ ≥ 0.

    Parameters
    ----------
    x : ndarray  速度网格
    f : ndarray  分布函数
    C_f : ndarray  碰撞算子 C[f]
    """
    f_safe = np.maximum(f, eps)
    integrand = -C_f * (1.0 + np.log(f_safe)) * FOUR_PI * x**2
    sigma = np.trapz(integrand, x)
    return max(sigma, 0.0)  # 理论上 ≥ 0, 但数值误差可能导致负值


def entropy_maxwellian(x, T_eff):
    """Maxwellian 的解析熵:

    S_M = (3/2) n (1 + ln(π T)) + n ln(n/π^{3/2})

    对于归一化 n=1:
    S_M = (3/2)(1 + ln(π T))
    """
    n = 1.0  # 归一化
    return 1.5 * n * (1.0 + math.log(PI * max(T_eff, 1e-30)))


# ===========================================================================
#  §3  相对熵 (KL 散度)
# ===========================================================================
def kl_divergence(x, f, f_target, eps=1e-30):
    """KL 散度: D_KL(f || f_target) = ∫ f ln(f/f_target) d³v.

    物理含义: 衡量 f 与目标分布 f_target 的 "信息距离".
    D_KL ≥ 0, 等号成立当且仅当 f = f_target a.e.
    """
    f_safe = np.maximum(f, eps)
    f_target_safe = np.maximum(f_target, eps)
    integrand = f_safe * np.log(f_safe / f_target_safe) * FOUR_PI * x**2
    return np.trapz(integrand, x)


# ===========================================================================
#  §4  收敛度量
# ===========================================================================
def convergence_metrics(x, f, f_initial, f_maxwellian, dt, step):
    """综合收敛度量.

    Returns
    -------
    metrics : dict  包含各种诊断量
    """
    # L2 范数
    l2_f = l2_norm_discrete(x, f)
    l2_delta = l2_norm_perturbation(x, f, f_maxwellian)
    l2_rel = l2_norm_relative(x, f, f_maxwellian)

    # 熵
    H = boltzmann_H(x, f)
    S = entropy_S(x, f)
    S_M = entropy_maxwellian(x, 1.0)

    # KL 散度
    kl = kl_divergence(x, f, f_maxwellian)

    # 粒子数守恒检查
    n_current = np.trapz(FOUR_PI * x**2 * f, x)
    n_initial = np.trapz(FOUR_PI * x**2 * f_initial, x)
    n_error = abs(n_current - n_initial) / max(abs(n_initial), 1e-30)

    # 能量守恒检查
    E_current = np.trapz(FOUR_PI * x**4 * f, x)
    E_initial = np.trapz(FOUR_PI * x**4 * f_initial, x)
    E_error = abs(E_current - E_initial) / max(abs(E_initial), 1e-30)

    return {
        "step": step,
        "time": step * dt,
        "l2_norm_f": l2_f,
        "l2_norm_delta_f": l2_delta,
        "l2_relative_error": l2_rel,
        "H_functional": H,
        "entropy_S": S,
        "entropy_maxwellian": S_M,
        "kl_divergence": kl,
        "particle_conservation_error": n_error,
        "energy_conservation_error": E_error,
    }
