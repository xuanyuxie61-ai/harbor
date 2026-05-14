# -*- coding: utf-8 -*-
"""
stellar_structure.py
基于 142_cavity_flow_movie (数据稀疏化) 与 791_ncm (数值算法) 合成
恒星结构方程求解：流体静力学平衡、能量传输、物态方程。
"""

import numpy as np
from typing import Tuple, Optional
from numerical_utils import safe_divide, brent_root, tridiag_solve, solve_linear


class StellarStructure:
    """
    恒星一维结构方程求解器（拉格朗日质量坐标）。
    
    基本方程组：
      (1) dr/dm = 1 / (4πr²ρ)
      (2) dP/dm = -Gm / (4πr⁴)
      (3) dL/dm = ε_nuc - ε_ν - c_P dT/dt + (δ/ρ) dP/dt  (完整能量方程)
      (4) dT/dm = ∇ T / P · dP/dm
    
    温度梯度：
      辐射梯度: ∇_rad = (3 κ L P) / (16π a c G m T⁴)
      绝热梯度: ∇_ad = P δ / (T ρ c_P)
      若 ∇_rad > ∇_ad : 对流不稳定，取 ∇ = ∇_ad
      若 ∇_rad ≤ ∇_ad : 辐射稳定，取 ∇ = ∇_rad
    
    物态方程（理想气体+辐射压）：
      P = P_gas + P_rad = (R/μ) ρT + (a/3) T⁴
      其中 a = 7.5657e-15 erg cm^-3 K^-4 (辐射密度常数)
    """

    # 物理常数 CGS
    G = 6.67430e-8
    A_RAD = 7.5657e-15
    C_LIGHT = 2.99792458e10
    R_GAS = 8.314462618e7  # 通用气体常数 [erg K^-1 mol^-1]
    SIGMA_SB = 5.670374419e-5
    KAPPA_ES = 0.2 * (1.0 + 0.74)  # 电子散射不透明度 [cm^2 g^-1] (Thomson)

    def __init__(self, M_total: float, R_init: float, composition: Optional[np.ndarray] = None):
        self.M_total = M_total
        self.R_init = R_init
        if composition is None:
            self.composition = np.array([0.7, 0.0, 0.28, 0.01, 0.005, 0.003, 0.001, 0.001])
        else:
            self.composition = np.array(composition, dtype=np.float64)
            self.composition /= np.sum(self.composition)

    def mean_molecular_weight(self, X: np.ndarray) -> float:
        """
        计算平均分子量 μ。
        完全电离假设：
          1/μ = Σ X_i / A_i * (1 + Z_i)
        对 H: 1/μ_H = 2X
        对 He: 1/μ_He = 3Y/4
        对金属: 1/μ_Z ≈ Z/2
        """
        X_h1 = X[0]
        Y_he = X[2]
        Z_metal = np.sum(X[3:])
        if X_h1 + Y_he + Z_metal <= 0:
            return 0.6
        mu_inv = 2.0 * X_h1 + 3.0 * Y_he / 4.0 + Z_metal / 2.0
        mu_inv = max(mu_inv, 0.5)
        return 1.0 / mu_inv

    def equation_of_state(self, rho: float, T: float, X: np.ndarray) -> Tuple[float, float, float, float]:
        """
        物态方程：返回 P, P_gas, P_rad, 绝热指数 Γ1。
        
        P = P_gas + P_rad = R/μ ρ T + a/3 T^4
        
        导数：
          dP/dρ = R T / μ
          dP/dT = R ρ / μ + 4 a T^3 / 3
          Γ1 = (d ln P / d ln ρ)_ad = β + (4-3β)^2 (γ-1) / (β + 12(γ-1)(1-β))
          其中 β = P_gas / P, γ = 5/3 (单原子理想气体)
        """
        mu = self.mean_molecular_weight(X)
        P_gas = self.R_GAS / mu * rho * T
        P_rad = self.A_RAD / 3.0 * T ** 4
        P = P_gas + P_rad

        if P <= 0:
            P = 1e-5
            P_gas = 1e-5
            P_rad = 1e-10

        beta = P_gas / P
        gamma_gas = 5.0 / 3.0
        # 混合绝热指数
        Gamma1 = beta + (4.0 - 3.0 * beta) ** 2 * (gamma_gas - 1.0) / (
                beta + 12.0 * (gamma_gas - 1.0) * (1.0 - beta))
        return P, P_gas, P_rad, Gamma1

    def opacity(self, rho: float, T: float, X: np.ndarray) -> float:
        """
        不透明度近似公式（Kramers + 电子散射）。
        κ ≈ κ_es + 4.4e24 * (Z + 0.001) * ρ / T^3.5  [cm^2/g]
        这是 Kramers 不透明度定律的简化形式。
        实际应使用 OPAL 或 OP 表。
        """
        Z_metal = np.sum(X[3:])
        kappa_es = self.KAPPA_ES
        # Kramers 束缚-自由不透明度
        kappa_kramers = 4.4e24 * (Z_metal + 0.001) * rho * T ** (-3.5)
        kappa = kappa_es + kappa_kramers
        # 物理限制
        kappa = np.clip(kappa, 1e-4, 1e6)
        return kappa

    def temperature_gradients(self, m: float, r: float, P: float, T: float,
                              L: float, rho: float, X: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        计算温度梯度与对流稳定性。
        
        ∇_rad = (3 κ L P) / (16 π a c G m T⁴)
        ∇_ad = P β δ / (T ρ c_P) ≈ (γ-1)/γ * β / (4 - 3β)
        
        返回: (∇_rad, ∇_ad, ∇_actual, is_convective)
        """
        if m <= 0 or r <= 0 or T <= 0:
            return 0.0, 0.4, 0.4, False

        kappa = self.opacity(rho, T, X)
        # 辐射梯度
        nabla_rad = (3.0 * kappa * L * P) / (16.0 * np.pi * self.A_RAD * self.C_LIGHT
                                               * self.G * m * T ** 4)
        nabla_rad = max(0.0, min(nabla_rad, 10.0))

        # 绝热梯度（简化）
        mu = self.mean_molecular_weight(X)
        P_gas = self.R_GAS / mu * rho * T
        beta = P_gas / P if P > 0 else 1.0
        gamma = 5.0 / 3.0
        nabla_ad = (gamma - 1.0) / gamma * beta / (4.0 - 3.0 * beta)
        nabla_ad = max(0.1, min(nabla_ad, 0.5))

        if nabla_rad > nabla_ad:
            return nabla_rad, nabla_ad, nabla_ad, True
        else:
            return nabla_rad, nabla_ad, nabla_rad, False

    def hydrostatic_structure(self, mass: np.ndarray, rho: np.ndarray,
                              P_c: float) -> np.ndarray:
        """
        从中心到表面积分流体静力学平衡方程，得到半径分布。
        dr/dm = 1 / (4π r² ρ)
        使用中心边界条件 r(0) = 0, P(0) = P_c。
        """
        n = len(mass)
        r = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        P[0] = P_c

        for i in range(n - 1):
            dm = mass[i + 1] - mass[i]
            if dm <= 0:
                continue
            r_avg = max(r[i], 1e-3)
            rho_avg = 0.5 * (rho[i] + rho[i + 1])
            # dr = dm / (4π r² ρ)
            dr = dm / (4.0 * np.pi * r_avg ** 2 * rho_avg)
            r[i + 1] = r[i] + dr

            # dP = - G m dm / (4π r⁴)
            m_avg = 0.5 * (mass[i] + mass[i + 1])
            dP = -self.G * m_avg * dm / (4.0 * np.pi * r_avg ** 4)
            P[i + 1] = P[i] + dP
            if P[i + 1] <= 0:
                P[i + 1] = 1e-5

        return r, P

    def solve_luminosity_profile(self, mass: np.ndarray, epsilon: np.ndarray,
                                 L_surface: float) -> np.ndarray:
        """
        从内向外积分能量方程：dL/dm = ε。
        边界条件: L(0) = 0, L(M) = L_surface (光度)。
        返回光度分布 L(m)。
        """
        n = len(mass)
        L = np.zeros(n, dtype=np.float64)
        for i in range(n - 1):
            dm = mass[i + 1] - mass[i]
            eps_avg = 0.5 * (epsilon[i] + epsilon[i + 1])
            L[i + 1] = L[i] + eps_avg * dm
        # 归一化到表面光度
        if L[-1] > 0 and L_surface > 0:
            L *= L_surface / L[-1]
        return L

    def eddington_luminosity(self, M: float) -> float:
        """
        爱丁顿光度：
          L_Edd = 4π G M c / κ_es
        """
        return 4.0 * np.pi * self.G * M * self.C_LIGHT / self.KAPPA_ES

    def schwarzschild_radius(self, M: float) -> float:
        """史瓦西半径：r_s = 2 G M / c²"""
        return 2.0 * self.G * M / self.C_LIGHT ** 2

    def sound_speed(self, rho: float, T: float, X: np.ndarray) -> float:
        """
        声速：c_s = sqrt(Γ1 P / ρ)
        """
        P, _, _, Gamma1 = self.equation_of_state(rho, T, X)
        if rho <= 0:
            return 1e5
        return np.sqrt(Gamma1 * P / rho)

    def dynamical_timescale(self, M: float, R: float, rho_avg: float) -> float:
        """
        动力学时标：τ_dyn ≈ sqrt(R³ / GM) ≈ 1 / sqrt(Gρ)
        """
        if rho_avg <= 0:
            return 1e10
        return 1.0 / np.sqrt(self.G * rho_avg)

    def kelvin_helmholtz_timescale(self, M: float, R: float, L: float) -> float:
        """
        开尔文-亥姆霍兹时标：
          τ_KH ≈ G M² / (R L)
        """
        if L <= 0 or R <= 0:
            return 1e20
        return self.G * M ** 2 / (R * L)

    def nuclear_timescale(self, M: float, L: float, f_nuc: float = 0.007) -> float:
        """
        核时标：
          τ_nuc ≈ f_nuc M c² / L
          f_nuc ~ 0.007 (氢→氦的质量亏损)
        """
        if L <= 0:
            return 1e20
        return f_nuc * M * self.C_LIGHT ** 2 / L

    def thin_data(self, arrays: Tuple[np.ndarray, ...], thin_factor: int = 4) -> Tuple[np.ndarray, ...]:
        """
        数据稀疏化（基于 142_cavity_flow_movie 的 thin_index 思想）。
        对演化数据按 thin_factor 降采样，减少存储量。
        """
        if thin_factor <= 1:
            return arrays
        idx = np.arange(0, len(arrays[0]), thin_factor)
        return tuple(np.asarray(a)[idx] for a in arrays)
