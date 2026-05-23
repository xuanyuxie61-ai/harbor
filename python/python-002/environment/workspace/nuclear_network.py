# -*- coding: utf-8 -*-
"""
nuclear_network.py
基于 1086_sir_ode 与 915_prime_plot 合成
恒星核燃烧网络：求解核素丰度演化的刚性ODE系统。
pp链、CNO循环、3α过程耦合，含中微子损失与能量产生。
同时利用素性检验（915_prime_plot）进行核素质量数稳定性校验。
"""

import numpy as np
from typing import Tuple, Optional
from reaction_rates import NuclearReactionRates


class NuclearNetwork:
    """
    恒星核反应网络。
    追踪的核素：H1, He3, He4, C12, N14, O16, Ne20, Mg24
    反应通道由 NuclearReactionRates 提供。
    
    演化方程（基于SIR ODE思想的质量作用定律）：
      dY_i/dt = Σ_j ν_ij λ_j ∏_k Y_k^{|ν_kj|}
    其中 Y_i 是核素 i 的摩尔丰度（每核子），λ_j 是反应 j 的速率。
    
    能量方程：
      ε = Σ_j Q_j * rate_j / N_A
    中微子损失：
      ε_ν = f_ν * ε_pp (对pp链约2%)
    """

    # 核素名称与质量数
    SPECIES = ['H1', 'He3', 'He4', 'C12', 'N14', 'O16', 'Ne20', 'Mg24']
    MASS_NUMBERS = np.array([1, 3, 4, 12, 14, 16, 20, 24], dtype=np.float64)

    def __init__(self, rates_calculator: Optional[NuclearReactionRates] = None):
        self.rates = rates_calculator if rates_calculator is not None else NuclearReactionRates()
        self.n_species = len(self.SPECIES)
        # 化学计量矩阵 ν[i,j]：物种 i 在反应 j 中的净产生数
        self._build_stoichiometry()

    def _build_stoichiometry(self):
        """
        构建反应网络的化学计量矩阵。
        反应顺序与 reaction_rates.py 中一致（取前15个+3α）。
        """
        n_react = 16
        self.nu = np.zeros((self.n_species, n_react), dtype=np.float64)
        idx = {s: i for i, s in enumerate(self.SPECIES)}

        # 1: p+p -> d + e+ + ν (简化为 H1 消耗，后续 d 快速达到平衡)
        self.nu[idx['H1'], 0] = -2.0
        # 2: d+p -> He3 + γ (d 忽略不计)
        self.nu[idx['H1'], 1] = -1.0
        self.nu[idx['He3'], 1] = +1.0
        # 3: He3+He3 -> He4 + 2p
        self.nu[idx['He3'], 2] = -2.0
        self.nu[idx['He4'], 2] = +1.0
        self.nu[idx['H1'], 2] = +2.0
        # 4: He3+He4 -> Be7 + γ
        self.nu[idx['He3'], 3] = -1.0
        self.nu[idx['He4'], 3] = -1.0
        # 5: Be7+e -> Li7 + ν (Li7 快速反应，不追踪)
        # 6: Li7+p -> 2He4
        self.nu[idx['H1'], 5] = -1.0
        self.nu[idx['He4'], 5] = +2.0
        # 7: Be7+p -> B8 + γ
        self.nu[idx['H1'], 6] = -1.0
        # 8: B8 -> Be8* + e+ + ν
        # 9: Be8* -> 2He4
        self.nu[idx['He4'], 8] = +2.0
        # 10: C12+p -> N13 + γ
        self.nu[idx['C12'], 9] = -1.0
        self.nu[idx['H1'], 9] = -1.0
        # 11: N13 -> C13 + e+ + ν (C13 快速达到平衡，简化为 C12 再生)
        self.nu[idx['C12'], 10] = +1.0
        # 12: C13+p -> N14 + γ
        self.nu[idx['H1'], 11] = -1.0
        self.nu[idx['N14'], 11] = +1.0
        # 13: N14+p -> O15 + γ
        self.nu[idx['N14'], 12] = -1.0
        self.nu[idx['H1'], 12] = -1.0
        # 14: O15 -> N15 + e+ + ν
        self.nu[idx['H1'], 13] = +1.0  # 电子正电子湮灭产生热能
        # 15: N15+p -> C12 + He4
        self.nu[idx['H1'], 14] = -1.0
        self.nu[idx['C12'], 14] = +1.0
        self.nu[idx['He4'], 14] = +1.0
        # 16: 3He4 -> C12
        self.nu[idx['He4'], 15] = -3.0
        self.nu[idx['C12'], 15] = +1.0

    @staticmethod
    def is_prime(n: int) -> bool:
        """
        素性检验（基于 915_prime_plot 的试除法）。
        用于校验核素质量数的稳定性：幻数多为质数（如2,7,19,23等）。
        """
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        limit = int(np.sqrt(n)) + 1
        for d in range(3, limit, 2):
            if n % d == 0:
                return False
        return True

    def magic_number_stability(self, A: int) -> float:
        """
        根据质量数 A 的素性与幻数接近度评估核素稳定性。
        幻数: 2, 8, 20, 28, 50, 82, 126
        """
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        # 到最近幻数的距离
        dist = min(abs(A - m) for m in magic_numbers)
        stability = np.exp(-dist / 10.0)
        if self.is_prime(A):
            stability *= 1.1
        return min(stability, 1.0)

    def abundances_to_mass_fractions(self, Y: np.ndarray) -> np.ndarray:
        """
        将摩尔丰度 Y_i [mol/g] 转换为质量分数 X_i。
        X_i = A_i * Y_i / Σ_j A_j * Y_j
        且 Σ X_i = 1。
        """
        Y = np.asarray(Y, dtype=np.float64)
        mass_per_nucleon = self.MASS_NUMBERS * Y
        total = np.sum(mass_per_nucleon)
        if total <= 0:
            return np.ones_like(Y) / len(Y)
        X = mass_per_nucleon / total
        return np.clip(X, 1e-15, 1.0)

    def compute_derivatives(self, t: float, Y: np.ndarray,
                            T: float, rho: float) -> np.ndarray:
        """
        计算 dY/dt。
        当前假设温度 T 和密度 ρ 在壳层内为常数（局部近似）。
        """
        Y = np.asarray(Y, dtype=np.float64)
        Y = np.maximum(Y, 1e-30)
        X = self.abundances_to_mass_fractions(Y)
        X_h1 = X[0]
        X_he4 = X[2]
        Z_cno = X[3] + X[4] + X[5]
        Z_metal = np.sum(X[3:])
        Y_val = X[2]  # 氦质量分数

        rates_pp = self.rates.pp_chain_rates(T, rho, X_h1, Y_val, Z_metal)
        rates_cno = self.rates.cno_cycle_rates(T, rho, X_h1, Y_val, Z_metal)
        rate_3a = self.rates.triple_alpha_rate(T, rho, Y_val)

        # TODO: Hole 2 - 实现反应率合并与 dY/dt 计算
        # 需将 rates_pp, rates_cno, rate_3a 正确合并到 all_rates 数组
        # 注意反应顺序与 _build_stoichiometry 中定义的 16 个反应对应
        # 需根据反应阶数 order 正确处理密度 ρ 和阿伏伽德罗常数 NA_eff 的缩放
        # 最终返回 dYdt = Σ_j ν_ij * R_j / A_i
        raise NotImplementedError("Hole 2: 待实现反应率合并与核素丰度演化导数计算")

    def solve_network_rk4(self, Y0: np.ndarray, T: float, rho: float,
                          dt: float, n_steps: int = 1) -> np.ndarray:
        """
        用经典 RK4 求解核网络一个时间步。
        对刚性系统可能步长需很小；实际应用应使用隐式方法。
        """
        Y = np.asarray(Y0, dtype=np.float64).copy()
        for _ in range(n_steps):
            k1 = self.compute_derivatives(0.0, Y, T, rho)
            k2 = self.compute_derivatives(0.0, Y + 0.5 * dt * k1, T, rho)
            k3 = self.compute_derivatives(0.0, Y + 0.5 * dt * k2, T, rho)
            k4 = self.compute_derivatives(0.0, Y + dt * k3, T, rho)
            Y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y = np.maximum(Y, 1e-30)
        return Y

    def solve_network_euler(self, Y0: np.ndarray, T: float, rho: float,
                            dt: float) -> np.ndarray:
        """显式 Euler（用于快速粗略估计）。"""
        Y = np.asarray(Y0, dtype=np.float64)
        dY = self.compute_derivatives(0.0, Y, T, rho)
        Y_new = Y + dt * dY
        return np.maximum(Y_new, 1e-30)

    def energy_generation_rate(self, Y: np.ndarray, T: float, rho: float) -> float:
        """
        基于当前丰度计算局部核能源产生率 [erg g^-1 s^-1]。
        """
        X = self.abundances_to_mass_fractions(Y)
        X_h1 = X[0]
        Y_he = X[2]
        Z_cno = X[3] + X[4] + X[5]
        Z_metal = np.sum(X[3:])
        return self.rates.energy_generation(T, rho, X_h1, Y_he, Z_cno, Z_metal)
