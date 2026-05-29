"""
reaction_kinetics.py
Modified Selkov-type kinetic terms for infection and population dynamics.

Adapted from:
  - 472_glycolysis_ode: Selkov model of glycolysis with autocatalytic terms

Role in synthesis:
  Provides nonlinear reaction rate laws with autocatalytic infection dynamics,
  adapted from biochemical oscillator kinetics to ecological epidemiology.
"""

import numpy as np


def selkov_infection_rate(
    S,
    I,
    a: float = 0.08,
    b: float = 0.6,
    c: float = 1.0
):
    """
    Modified Selkov-type infection rate with autocatalytic enhancement.

    Original Selkov: du/dt = -u + a*v + u^2*v
    Adapted: infection force = c * (a * I + S * I^2) / (1 + S^2)

    The autocatalytic I^2 term represents infection amplification through
    social/behavioral feedback (e.g., increased contact rates during outbreaks).

    Parameters
    ----------
    S : float or ndarray
        Susceptible density.
    I : float or ndarray
        Infected density.
    a : float
        Baseline transmission coefficient.
    b : float
        Saturation parameter (not used directly but kept for parameter consistency).
    c : float
        Scaling factor.

    Returns
    -------
    rate : float or ndarray
        Force of infection contribution.
    """
    S = np.clip(np.asarray(S, dtype=float), 0.0, 1e4)
    I = np.clip(np.asarray(I, dtype=float), 0.0, 1e4)
    # Clip S*I^2 to prevent overflow
    SI2 = np.clip(S * I ** 2, 0.0, 1e8)
    numerator = a * I + SI2
    denominator = 1.0 + S ** 2
    return np.clip(c * numerator / denominator, 0.0, 1e6)


def selkov_growth_rate(
    N: float,
    K: float,
    a: float = 0.5,
    b: float = 0.1
) -> float:
    """
    Modified logistic growth with Allee effect, inspired by Selkov kinetics.

    dN/dt = r * N * (1 - N/K) * (N/K - a) / (1 - b*N/K)

    The Allee effect (N/K - a) introduces a critical population threshold below
    which the population cannot sustain itself.
    """
    if K <= 0:
        return 0.0
    n = N / K
    if n < 0:
        return 0.0
    # Allee effect with smooth denominator
    allee = max(n - a, -1.0)
    denom = max(1.0 - b * n, 0.1)
    return n * (1.0 - n) * allee / denom


def compute_reaction_terms(
    state: np.ndarray,
    K_field: np.ndarray,
    r_field: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Compute reaction terms for the coupled eco-epidemiological system.

    State vector at each grid point: [S1, I1, R1, S2, I2, R2]

    Parameters
    ----------
    state : ndarray, shape (6, nx, ny)
    K_field : ndarray, shape (nx, ny)
        Carrying capacity.
    r_field : ndarray, shape (nx, ny)
        Growth rate.
    params : dict
        Model parameters.

    Returns
    -------
    reaction : ndarray, shape (6, nx, ny)
    """
    S1, I1, R1, S2, I2, R2 = state
    N1 = S1 + I1 + R1
    N2 = S2 + I2 + R2

    D_s1 = params.get('D_s1', 0.01)
    D_s2 = params.get('D_s2', 0.01)
    beta11 = params.get('beta11', 0.3)
    beta12 = params.get('beta12', 0.1)
    beta21 = params.get('beta21', 0.1)
    beta22 = params.get('beta22', 0.3)
    gamma1 = params.get('gamma1', 0.1)
    gamma2 = params.get('gamma2', 0.1)
    mu1 = params.get('mu1', 0.02)
    mu2 = params.get('mu2', 0.02)
    alpha12 = params.get('alpha12', 0.5)
    alpha21 = params.get('alpha21', 0.5)

    # === HOLE 1 START ===
    # 修复要求：实现竞争调制逻辑斯蒂增长、跨物种传播力、Selkov自催化增强、恢复/移除动力学的完整反应项计算。
    # 科学知识要点：
    #   1. 竞争调制逻辑斯蒂增长：growth = r_field * S * (1 - (N_self + alpha_cross * N_other) / K_field)
    #   2. 跨物种传播力（Force of Infection）：foi = beta11*S1*I1 + beta12*S1*I2（物种1）
    #   3. Selkov型自催化增强：当 I > threshold 时调用 selkov_infection_rate(S, I, a=0.05, c=0.05)
    #   4. 恢复动力学：dR = gamma * I
    # 需进行合理的数值裁剪（np.clip）防止溢出，并返回 shape 为 (6, nx, ny) 的 ndarray，
    # 顺序严格为 [dS1, dI1, dR1, dS2, dI2, dR2]
    raise NotImplementedError("HOLE 1: 请实现 compute_reaction_terms 的核心反应项计算")
    # === HOLE 1 END ===


def equilibrium_analysis(params: dict) -> dict:
    """
    Compute analytical equilibrium points for the spatially homogeneous system.

    For the simplified model without cross-species terms:
    S* = (gamma + mu) / beta
    I* = r * S* * (1 - S*/K) / (gamma + mu)

    Returns
    -------
    equilibria : dict
    """
    beta = params.get('beta11', 0.3)
    gamma = params.get('gamma1', 0.1)
    mu = params.get('mu1', 0.02)
    r = params.get('r_mean', 1.0)
    K = params.get('K_mean', 100.0)

    if beta > 0:
        S_star = (gamma + mu) / beta
        I_star = max(0.0, r * S_star * (1.0 - S_star / K) / (gamma + mu))
        R_star = gamma * I_star / max(mu, 1e-10)
    else:
        S_star, I_star, R_star = K, 0.0, 0.0

    # Basic reproduction number: R0 = beta * K / (gamma + mu)
    R0 = beta * K / (gamma + mu)

    return {
        'S_star': S_star,
        'I_star': I_star,
        'R_star': R_star,
        'R0': R0,
        'disease_free_stable': R0 < 1.0,
    }
