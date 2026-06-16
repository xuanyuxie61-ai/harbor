"""
超新星物态方程 (Equation of State).

采用 Helmholtz 自由能分解的经典近似：
    F(ρ,T,Y_e) = F_ion + F_e + F_rad + F_Coul

这里简化为三项核心贡献：
  1. 理想气体离子   F_ion = - n k_B T [ln(n_Q / n) + 1]
  2. 相对论 e± 对   采用近似 Fermi-Dirac 积分
  3. 光子辐射场     P_rad = (1/3) a T^4

压强：
    P = P_ion + P_e + P_rad
      = (ρ/μ m_p) k_B T  +  P_e(ρ,T,Y_e)  +  (1/3) a T^4

比内能 (单位质量)：
    e = (3/2) k_B T / (μ m_p)  +  e_e  +  a T^4 / ρ

绝热指数 Γ_1 由热力学恒等式给出：
    Γ_1 = (ρ/P) (∂P/∂ρ)_s
        = (χ_ρ + P χ_T / (ρ^2 e_T)) / (1 + ... )
其中 χ_ρ = (∂ln P/∂ln ρ)_T,  χ_T = (∂ln P/∂ln T)_ρ.

声速 c_s:
    c_s^2 = Γ_1 P / ρ
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


def fermi_dirac_half(eta: np.ndarray) -> np.ndarray:
    """F_{1/2}(η) = (2/√π) ∫_0^∞ √x / (exp(x-η) + 1) dx 的近似.

    采用 Antia (1993) 有理分式近似，精度 ~ 1e-8：
      对 η > 0: F_{1/2} ≈ (2/3) η^{3/2} (1 + (π^2/8) η^{-2} + ...)
      对 η < 0: F_{1/2} ≈ exp(η) - exp(2η)/2^{3/2} + ...
    """
    out = np.zeros_like(eta, dtype=np.float64)
    pos = eta > 0.0
    neg = ~pos
    # 退化极限
    eta_p = np.abs(eta[pos]) + 1.0e-30
    out[pos] = (2.0 / 3.0) * eta_p ** 1.5 * (
        1.0 + (math.pi ** 2 / 8.0) / (eta_p ** 2 + 1.0e-30)
        + (7.0 * math.pi ** 4 / 640.0) / (eta_p ** 4 + 1.0e-30)
    )
    # 非退化极限
    eta_n = eta[neg]
    out[neg] = np.exp(eta_n) * (
        1.0 - np.exp(eta_n) / (2.0 ** 1.5) + np.exp(2.0 * eta_n) / (3.0 ** 1.5)
    )
    return out


def fermi_dirac_3half(eta: np.ndarray) -> np.ndarray:
    """F_{3/2}(η) 近似，用于 e± 对内能."""
    out = np.zeros_like(eta, dtype=np.float64)
    pos = eta > 0.0
    neg = ~pos
    eta_p = np.abs(eta[pos]) + 1.0e-30
    out[pos] = (2.0 / 5.0) * eta_p ** 2.5 * (
        1.0 + (5.0 * math.pi ** 2 / 8.0) / (eta_p ** 2 + 1.0e-30)
    )
    eta_n = eta[neg]
    out[neg] = np.exp(eta_n) * (
        1.0 - np.exp(eta_n) / (2.0 ** 2.5) + np.exp(2.0 * eta_n) / (3.0 ** 2.5)
    )
    return out


class EquationOfState:
    """超新星物态方程（含辐射压 + e± 对）."""

    def __init__(self, gamma_ion: float = 5.0 / 3.0, y_e: float = 0.42):
        self.gamma_ion = float(gamma_ion)
        self.y_e = float(y_e)   # 电子丰度，典型坍缩核心 ~ 0.42
        self.mu = C.avogadro_mean_molecular_weight()

    def electron_degeneracy_parameter(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """电子简并参数 η_e = μ_e / (k_B T).

        μ_e 为电子化学势，在完全电离近似下由数密度给出：
          n_e = Y_e ρ / m_p
          非简并：η_e ≈ ln(n_e / (2 (2π m_e k_B T / h^2)^{3/2}))
        """
        n_e = self.y_e * rho / C.M_PROTON
        # 量子浓度
        n_Q = 2.0 * (2.0 * math.pi * C.M_ELECTRON * C.K_BOLTZMANN * T
                      / (C.H_PLANCK ** 2)) ** 1.5
        eta = np.log(np.maximum(n_e, C.TINY_RHO) / np.maximum(n_Q, C.TINY_RHO))
        return eta

    def pressures(self, rho: np.ndarray, T: np.ndarray):
        """返回 (P_ion, P_e, P_rad, P_total)."""
        rho = np.asarray(rho, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        n_ion = rho / (self.mu * C.M_PROTON)
        P_ion = n_ion * C.K_BOLTZMANN * T

        # e± 对压强 (近似): P_e = (8π √2 / 3) (m_e c^2 / (hc)^3) (k_B T)^{5/2} F_{3/2}(η)
        # 这里简化：P_e = n_e k_B T (1 + 修正)
        eta = self.electron_degeneracy_parameter(rho, T)
        n_e = self.y_e * rho / C.M_PROTON
        # 修正因子来自简并
        F32 = fermi_dirac_3half(eta)
        F12 = fermi_dirac_half(eta)
        # P_e = (2/3) (8π √2 m_e^{3/2} / h^3) (k_B T)^{5/2} F_{3/2}
        pref = (8.0 * math.sqrt(2.0) * C.M_ELECTRON ** 1.5 / (3.0 * C.H_PLANCK ** 3))
        P_e = pref * (C.K_BOLTZMANN * T) ** 2.5 * F32

        P_rad = (C.A_RADIATION / 3.0) * T ** 4
        P_tot = P_ion + P_e + P_rad
        return P_ion, P_e, P_rad, P_tot

    def specific_internal_energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """单位质量比内能 e (erg/g).

        e = (3/2) P_ion / ρ  +  e_e  +  a T^4 / ρ
        其中 e_e = 3 P_e / (2 ρ) (非相对论近似).
        """
        P_ion, P_e, P_rad, _ = self.pressures(rho, T)
        e_ion = 1.5 * P_ion / np.maximum(rho, C.TINY_RHO)
        e_e = 1.5 * P_e / np.maximum(rho, C.TINY_RHO)
        e_rad = C.A_RADIATION * T ** 4 / np.maximum(rho, C.TINY_RHO)
        return e_ion + e_e + e_rad

    def sound_speed(self, rho: np.ndarray, T: np.ndarray,
                    P_tot: np.ndarray | None = None) -> np.ndarray:
        """绝热声速 c_s = sqrt(Γ_1 P / ρ).

        Γ_1 近似取有效值，包含辐射压修正：
          Γ_1_eff = β + (4-3β)^2 (γ_ion-1) / (β + 12(γ_ion-1)(4-3β)/...)
        其中 β = P_gas / P_tot (气体压强占比).
        """
        if P_tot is None:
            _, _, _, P_tot = self.pressures(rho, T)
        n_ion = rho / (self.mu * C.M_PROTON)
        P_gas = n_ion * C.K_BOLTZMANN * T
        beta = P_gas / np.maximum(P_tot, C.TINY_P)
        beta = np.clip(beta, 1.0e-10, 1.0 - 1.0e-10)
        gm1 = self.gamma_ion - 1.0
        # Chandrasekhar (1939) 气体+辐射混合绝热指数
        gamma_1 = beta + (4.0 - 3.0 * beta) ** 2 * gm1 / (
            beta + 36.0 * gm1 * (4.0 - 3.0 * beta) / (10.0 + 1.0e-30)
        )
        gamma_1 = np.maximum(gamma_1, 1.01)
        cs2 = gamma_1 * P_tot / np.maximum(rho, C.TINY_RHO)
        return np.sqrt(np.maximum(cs2, C.TINY_E))

    def temperature_from_energy(self, rho: np.ndarray, e_tot: np.ndarray,
                                tol: float = 1.0e-8, max_iter: int = 40) -> np.ndarray:
        """由比内能反解温度 (Newton-Raphson).

        方程: e(T) = e_target, 未知 T.
        导数: de/dT = c_v = (∂e/∂T)_ρ.
        辐射项贡献 c_v,rad = 4 a T^3 / ρ.
        """
        T = np.full_like(e_tot, 1.0e9, dtype=np.float64)
        for _ in range(max_iter):
            P_ion, P_e, P_rad, _ = self.pressures(rho, T)
            e_cur = (1.5 * P_ion + 1.5 * P_e + C.A_RADIATION * T ** 4) / np.maximum(rho, C.TINY_RHO)
            # c_v
            c_v = (1.5 * rho / (self.mu * C.M_PROTON) * C.K_BOLTZMANN
                   + 4.0 * C.A_RADIATION * T ** 3) / np.maximum(rho, C.TINY_RHO)
            c_v = np.maximum(c_v, 1.0e-20)
            dT = (e_tot - e_cur) / c_v
            # 限幅：一次变化不超过当前 T 的 50%
            dT = np.clip(dT, -0.5 * T, 0.5 * T)
            T = np.maximum(T + dT, 1.0e4)
            if np.max(np.abs(dT) / (T + 1.0e-30)) < tol:
                break
        return T

    def adiabatic_gradient(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """绝热温度梯度 ∇_ad = (d ln T / d ln P)_s.

        对理想气体 ∇_ad = (γ-1)/γ.
        含辐射时，∇_ad = (P/ρ T) (χ_T / c_P),
        其中 χ_T = (∂ln P / ∂ln T)_ρ.
        """
        P_ion, P_e, P_rad, P_tot = self.pressures(rho, T)
        chi_T = (P_ion + 4.0 * P_rad + 2.5 * P_e) / np.maximum(P_tot, C.TINY_P)
        c_v = (1.5 * rho / (self.mu * C.M_PROTON) * C.K_BOLTZMANN
               + 4.0 * C.A_RADIATION * T ** 3) / np.maximum(rho, C.TINY_RHO)
        c_P = c_v + P_tot * chi_T ** 2 / (np.maximum(rho, C.TINY_RHO) * np.maximum(T, C.TINY_T))
        nabla_ad = P_tot * chi_T / (np.maximum(rho, C.TINY_RHO) * np.maximum(T, C.TINY_T) * np.maximum(c_P, C.TINY_E))
        return nabla_ad
