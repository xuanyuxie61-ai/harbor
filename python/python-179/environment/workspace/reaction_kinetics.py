"""
reaction_kinetics.py
反应动力学模块
==============
对应原项目 1018_reaction_twoway_ode（双向线性化学反应）与 1386_vanderpol_ode（范德波尔振子），
将低维 ODE 动力学扩展为空间分布反应项，用于参数化反应-扩散方程的源项建模。
"""

import numpy as np
from system_utils import clip_to_range, check_finite


# ---------------------------------------------------------------------------
# 双向线性反应（原 reaction_twoway_ode 扩展）
# ---------------------------------------------------------------------------

def twoway_reaction_term(u: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """
    双向一级反应速率项 R(u) = -k1*u + k2*(1-u)。

    化学背景
    --------
    设 u 为组分 W1 的归一化浓度，W2 = 1-u 为其互补组分。
    反应体系
        W1 --(k1)--> W2
        W2 --(k2)--> W1
    总质量守恒：W1 + W2 = 1。

    动力学方程
        du/dt = -k1*u + k2*(1-u) = -(k1+k2)*u + k2
    稳态解 u* = k2 / (k1 + k2)。
    """
    k1 = float(k1)
    k2 = float(k2)
    k1 = max(k1, 1e-12)
    k2 = max(k2, 1e-12)
    R = -k1 * u + k2 * (1.0 - u)
    return R


def twoway_exact_solution(t: np.ndarray, u0: float, k1: float, k2: float) -> np.ndarray:
    """
    精确解（用于验证时间离散化）：
        u(t) = u* + (u0 - u*) * exp(-(k1+k2)*t)
    其中 u* = k2/(k1+k2)。
    """
    t = np.asarray(t, dtype=float)
    k_sum = k1 + k2
    u_star = k2 / k_sum
    return u_star + (u0 - u_star) * np.exp(-k_sum * t)


# ---------------------------------------------------------------------------
# 范德波尔型非线性反应（原 vanderpol_ode 扩展）
# ---------------------------------------------------------------------------

def vanderpol_reaction_term(u: np.ndarray, mu: float) -> np.ndarray:
    """
    范德波尔型非线性阻尼反应项：
        R(u) = μ * (1 - u²) * u

    物理背景
    --------
    对应范德波尔振子的速度方程 v' = μ(1-u²)v - u 中的非线性阻尼部分。
    当 |u|<1 时，R>0 提供负阻尼（能量注入）；当 |u|>1 时，R<0 提供正阻尼（能量耗散）。
    该非线性特性导致稳定的极限环振荡。

    参数 μ>0 控制非线性强度；μ>>1 时系统呈现强刚性。
    """
    mu = float(mu)
    mu = max(mu, 1e-6)
    u = clip_to_range(np.asarray(u, dtype=float), -1e3, 1e3)
    R = mu * (1.0 - u * u) * u
    return R


# ---------------------------------------------------------------------------
# 混合反应-扩散源项（参数化）
# ---------------------------------------------------------------------------

def parametric_reaction_source(u: np.ndarray,
                                k1: float, k2: float,
                                mu: float,
                                mix_ratio: float = 0.5) -> np.ndarray:
    """
    参数化混合反应源项：
        R(u; k1,k2,μ,α) = α * R_two-way(u;k1,k2) + (1-α) * R_vdp(u;μ)

    参数
    ----
    mix_ratio (α) ∈ [0,1]：线性化学动力学与范德波尔非线性振荡的混合权重。

    科学意义
    --------
    该模型描述同时存在化学反应（如酶催化）与电化学振荡（如 Belousov-Zhabotinsky
    反应中类范德波尔动力学）的复杂系统。参数空间 (k1,k2,μ,α) 构成四维张量索引，
    后续通过张量分解实现参数化模型降阶。
    """
    mix_ratio = clip_to_range(float(mix_ratio), 0.0, 1.0)
    R_lin = twoway_reaction_term(u, k1, k2)
    R_vdp = vanderpol_reaction_term(u, mu)
    R = mix_ratio * R_lin + (1.0 - mix_ratio) * R_vdp
    check_finite(R, "reaction_source")
    return R


# ---------------------------------------------------------------------------
# 反应项 Jacobian（用于隐式时间步）
# ---------------------------------------------------------------------------

def reaction_jacobian_diagonal(u: np.ndarray,
                                k1: float, k2: float,
                                mu: float,
                                mix_ratio: float = 0.5) -> np.ndarray:
    """
    反应项对 u 的逐点 Jacobian（对角矩阵，因反应项为局部）：
        J_ii = dR/du (u_i)

    推导
    ----
    dR_two-way/du = -(k1 + k2)
    dR_vdp/du     = μ * (1 - 3u²)
    dR_mix/du     = α * dR_two-way/du + (1-α) * dR_vdp/du
    """
    u = np.asarray(u, dtype=float)
    mix_ratio = clip_to_range(float(mix_ratio), 0.0, 1.0)
    dR_lin = -(k1 + k2)
    dR_vdp = mu * (1.0 - 3.0 * u * u)
    J = mix_ratio * dR_lin + (1.0 - mix_ratio) * dR_vdp
    return J
