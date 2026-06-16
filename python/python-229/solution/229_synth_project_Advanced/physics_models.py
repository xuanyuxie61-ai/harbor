"""
physics_models.py
=================

真实能谱模型 (用于 unfold 测试)
-------------------------------
提供 4 类高能物理典型能谱:

1. 幂律谱 (宇宙线 / QCD 喷注):
       dN/dE = A · (E / E_0)^{-γ}
   γ ≈ 2.7 (宇宙线 knee 以下), γ ≈ 4.7 (knee 以上)

2. 热谱 (Maxwell-Boltzmann, 夸克胶子等离子体):
       dN/dE = A · E^2 · exp(-E / T)
   T 为温度 (QGP: ~150-300 MeV)

3. 同步辐射谱 (弯曲磁场中的相对论电子):
       dI/dE = (√3 e^3 B / (m_e c^2)) · (E / E_c) ∫_{E/E_c}^∞ K_{5/3}(x) dx
   近似为:  dI/dE ∝ (E / E_c)^{1/3} exp(-E / E_c)  for E << E_c
                     ∝ (E / E_c)^{1/2} exp(-E / E_c)  for E >> E_c

4. 极坐标 ODE 轨迹谱 (来自 880_polar_ode 的解析函数):
       f(E) = r(E) = 1 - sin(E) cos(3E)   (作为非平凡测试函数)

本模块还提供:
  - 高斯泊松噪声生成 (模拟观测数据)
  - 响应矩阵应用 O = R T
"""

from __future__ import annotations
from typing import List, Callable, Tuple
import math
import random
import special_functions as sf


# ===========================================================================
#            能谱生成器
# ===========================================================================

def power_law_spectrum(
    E: float,
    A: float = 1.0,
    E0: float = 1.0,
    gamma: float = 2.7,
) -> float:
    """
    幂律谱: dN/dE = A (E/E0)^{-γ}.

    截断: E < 0.1 GeV → 0 (红外截断).
    """
    if E <= 0.1:
        return 0.0
    return A * (E / E0) ** (-gamma)


def thermal_spectrum(
    E: float,
    A: float = 1.0,
    T: float = 0.2,
) -> float:
    """
    热谱 (MB): dN/dE = A E^2 exp(-E/T).

    T 典型值: QGP 温度 0.15 - 0.3 GeV.
    """
    if E <= 0.0:
        return 0.0
    if E / T > 700:
        return 0.0
    return A * E * E * math.exp(-E / T)


def synchrotron_spectrum(
    E: float,
    A: float = 1.0,
    E_c: float = 5.0,
) -> float:
    """
    同步辐射谱近似:
        dI/dE ∝ (E/E_c)^{1/3} exp(-E/E_c)   for E < E_c
              ∝ (E/E_c)^{1/2} exp(-E/E_c)   for E ≥ E_c

    精确形式需要 ∫_{E/E_c}^∞ K_{5/3}(x) dx, 此处用渐近拼接.
    """
    if E <= 0.0:
        return 0.0
    x = E / E_c
    if x < 1.0:
        power = 1.0 / 3.0
    else:
        power = 0.5
    # 使用 K_{1/3} 近似 ∫ K_{5/3}
    k_val = sf.r8_besk0(x) + sf.r8_besk0(x + 0.5)  # 近似
    return A * (x ** power) * math.exp(-x) * (0.5 + 0.5 * k_val / (sf.r8_besk0(1.0) + 1e-300))


def polar_ode_spectrum(
    E: float,
    A: float = 10.0,
) -> float:
    """
    极坐标 ODE 轨迹作为测试谱:
        f(E) = A · (1 - sin(E) cos(3E))^2
    (平方保证非负)

    用于验证 unfolding 在非单调 / 振荡谱上的表现.
    """
    r = 1.0 - math.sin(E) * math.cos(3.0 * E)
    return A * r * r


def combined_spectrum(
    E: float,
    weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.3, 0.2),
) -> float:
    """
    组合谱: 所有模型的加权和.

    用于复杂测试.
    """
    return (
        weights[0] * power_law_spectrum(E)
        + weights[1] * thermal_spectrum(E, T=0.3)
        + weights[2] * synchrotron_spectrum(E, E_c=10.0)
        + weights[3] * polar_ode_spectrum(E, A=5.0)
    )


# ===========================================================================
#          观测数据生成
# ===========================================================================

def apply_response(
    R_dense: List[List[float]],
    T: List[float],
) -> List[float]:
    """
    应用响应矩阵: O = R T.

    R: N_rec × N_true,  T: N_true → O: N_rec.
    """
    N_rec = len(R_dense)
    N_true = len(T)
    O = [sum(R_dense[i][j] * T[j] for j in range(N_true)) for i in range(N_rec)]
    return O


def add_poisson_noise(
    O: List[float],
    seed: int = 42,
    scale: float = 1.0,
) -> List[float]:
    """
    对观测计数添加泊松噪声.

    对大期望值 λ: Poisson(λ) ≈ N(λ, √λ).
    对小 λ: 直接采样 (Knuth 算法).
    """
    random.seed(seed)
    O_noisy = []
    for o in O:
        lam = max(0.0, o * scale)
        if lam < 1e-10:
            O_noisy.append(0.0)
        elif lam < 25:
            # Knuth 算法
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= random.random()
                if p <= L:
                    break
            O_noisy.append(float(k - 1))
        else:
            # 高斯近似
            noise = random.gauss(0.0, math.sqrt(lam))
            O_noisy.append(max(0.0, lam + noise))
    return O_noisy


def add_gaussian_noise(
    O: List[float],
    sigma_frac: float = 0.05,
    seed: int = 42,
) -> List[float]:
    """
    高斯噪声: O_noisy = O · (1 + σ · N(0,1)).

    σ 为相对不确定度 (典型 5%).
    """
    random.seed(seed)
    return [max(0.0, o * (1.0 + sigma_frac * random.gauss(0.0, 1.0))) for o in O]


# ===========================================================================
#          真谱在 bin 上的积分
# ===========================================================================

def integrate_spectrum_in_bins(
    spectrum: Callable,
    E_edges: List[float],
    quadrature_order: int = 7,
) -> List[float]:
    """
    计算谱在每个 bin 上的积分:
        T_j = ∫_{E_j}^{E_{j+1}} f(E) dE

    使用 Gauss-Legendre 求积.
    """
    import phase_space_grid as psg
    nodes, weights = psg.gauss_legendre_cos_theta(quadrature_order)

    T = []
    for j in range(len(E_edges) - 1):
        Ej_lo, Ej_hi = E_edges[j], E_edges[j + 1]
        Dj = Ej_hi - Ej_lo
        s = 0.0
        for k in range(quadrature_order):
            E = Ej_lo + 0.5 * Dj * (nodes[k] + 1.0)
            s += 0.5 * Dj * weights[k] * spectrum(E)
        T.append(max(0.0, s))
    return T


# ===========================================================================
#         χ² 拟合优度
# ===========================================================================

def chi_squared(
    O_obs: List[float],
    O_exp: List[float],
    sigma: List[float] = None,
) -> float:
    """
    χ² = Σ_i (O_obs_i - O_exp_i)^2 / σ_i^2.

    若 σ 未给出, 假设 σ_i = √(O_obs_i) (泊松).
    """
    if len(O_obs) != len(O_exp):
        raise ValueError("长度不匹配")
    chi2 = 0.0
    for i in range(len(O_obs)):
        sig = sigma[i] if sigma is not None else math.sqrt(max(O_obs[i], 1.0))
        if sig < 1e-300:
            sig = 1.0
        chi2 += ((O_obs[i] - O_exp[i]) / sig) ** 2
    return chi2


__all__ = [
    "power_law_spectrum",
    "thermal_spectrum",
    "synchrotron_spectrum",
    "polar_ode_spectrum",
    "combined_spectrum",
    "apply_response",
    "add_poisson_noise",
    "add_gaussian_noise",
    "integrate_spectrum_in_bins",
    "chi_squared",
]
