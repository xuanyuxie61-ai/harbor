"""
贝叶斯热力学第二律诊断 (from 1205_abartolo-tb_Bayesian-Second-Law).

原项目：对谐振子系统验证贝叶斯形式的第二定律，
  ⟨ΔS_tot⟩ ≥ 0,  其中 S_tot = ΔS_sys + ΔS_bath.

超新星中：激波将引力束缚能转化为热能 + 动能 + 辐射,
  总熵增：
    ΔS = ΔS_gas + ΔS_rad + ΔS_ν
  气体熵 (单位质量):
    s_gas = (k_B / (μ m_p)) ln(P / ρ^γ) + const
  辐射熵:
    s_rad = (4/3) a T^3 / ρ
  中微子熵:
    s_ν ≈ (7/8) (11/4) (4/3) a T_ν^3 / ρ  (费米子修正)

贝叶斯更新 (from BSL):
  给定观测中微子通量 Φ_ν(t) 及其误差 σ(t),
  对熵产率 Σ(t) 的后验估计：
    P(Σ | Φ_ν) ∝ P(Φ_ν | Σ) P(Σ)
  先验 P(Σ) = Gamma(α_0, β_0),
  似然 P(Φ_ν | Σ) = N(Σ · K, σ^2).
  后验 P(Σ | Φ_ν) = Gamma(α_0 + N/2, β_0 + (SSR + 1/(2K^2))/2)

热力学一致性检验：
  若 ⟨ΔS⟩ ± 2 σ(ΔS) > 0, 则数据与第二律一致.
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


def specific_entropy_gas(P: np.ndarray, rho: np.ndarray,
                          gamma: float = 5.0 / 3.0,
                          mu: float = 0.61) -> np.ndarray:
    """单位质量气体熵 s_gas (erg/K/g).

  s = k_B / (μ m_p) [ ln(P / ρ^γ) - ln(P_0 / ρ_0^γ) ] + s_0
  取 s_0 = 0 为参考.
  """
    rho_safe = np.maximum(rho, C.TINY_RHO)
    P_safe = np.maximum(P, C.TINY_P)
    ln_term = np.log(P_safe / rho_safe ** gamma)
    return C.K_BOLTZMANN / (mu * C.M_PROTON) * ln_term


def specific_entropy_rad(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """单位质量辐射熵.

  s_rad = (4/3) a T^3 / ρ
  """
    return (4.0 / 3.0) * C.A_RADIATION * T ** 3 / np.maximum(rho, C.TINY_RHO)


def specific_entropy_neutrino(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """单位质量中微子熵 (费米子修正).

  s_ν = (7/8) (11/4) (4/3) a T_ν^3 / ρ
  对简并中微子，有额外因子.
  """
    T_nu = T * 0.7  # 中微子温度略低于物质
    return (7.0 / 8.0) * (11.0 / 4.0) * (4.0 / 3.0) * C.A_RADIATION \
           * T_nu ** 3 / np.maximum(rho, C.TINY_RHO)


def total_entropy(P: np.ndarray, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
    """单位质量总熵 s = s_gas + s_rad + s_ν."""
    return (specific_entropy_gas(P, rho)
            + specific_entropy_rad(T, rho)
            + specific_entropy_neutrino(T, rho))


class BayesianEntropyProduction:
    """贝叶斯推断熵产率 Σ(t).

  给定观测序列 Φ_ν(t), 假设 Φ_ν(t) = K Σ(t) + ε,
  先验 Σ ~ Gamma(α_0, β_0), 后验仍为 Gamma.
  """

    def __init__(self, alpha_0: float = 2.0, beta_0: float = 1.0,
                 K: float = 1.0e50, sigma_obs: float = 1.0e51):
        self.alpha = float(alpha_0)
        self.beta = float(beta_0)
        self.K = float(K)
        self.sigma_obs = float(sigma_obs)

    def update(self, Phi_obs: np.ndarray):
        """基于观测 Φ_ν 更新后验.

        Gamma(α, β) 的共轭更新：
          α_new = α + N/2
          β_new = β + (1/2) Σ (Φ_i/K)^2 / (2 σ^2)
        """
        N = len(Phi_obs)
        SSR = float(np.sum((Phi_obs / self.K) ** 2))
        self.alpha = self.alpha + N / 2.0
        self.beta = self.beta + SSR / (2.0 * self.sigma_obs ** 2)

    def posterior_mean(self) -> float:
        """后验均值 E[Σ] = α / β."""
        return self.alpha / max(self.beta, 1.0e-30)

    def posterior_variance(self) -> float:
        """后验方差 Var[Σ] = α / β^2."""
        return self.alpha / max(self.beta ** 2, 1.0e-60)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """后验可信区间 (正态近似, 对大 α).

        Σ ≈ N(α/β, α/β^2) 当 α 较大时.
        """
        mean = self.posterior_mean()
        std = math.sqrt(max(self.posterior_variance(), 0.0))
        z = 1.96 if level > 0.9 else 1.645
        return (mean - z * std, mean + z * std)

    def test_second_law(self) -> dict:
        """检验 ⟨ΔS⟩ > 0 (第二律)."""
        mean = self.posterior_mean()
        std = math.sqrt(max(self.posterior_variance(), 0.0))
        z = mean / max(std, 1.0e-30)
        # 单侧 p-value (检验 mean > 0)
        from math import erf
        p_value = 0.5 * (1.0 - erf(z / math.sqrt(2.0)))
        return {'mean': mean, 'std': std, 'z_score': z,
                'p_value': p_value,
                'consistent_with_2nd_law': mean > 2.0 * std}


def entropy_integral(s: np.ndarray, dm: np.ndarray) -> float:
    """总熵 S = ∫ s dm (对整个星体积分)."""
    return float(np.sum(s * dm))
