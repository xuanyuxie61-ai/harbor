# -*- coding: utf-8 -*-
"""
electron_beam_source.py
=======================
相对论快电子束随机源采样.

物理模型:
---------
1) 快电子能量服从修正 Maxwellian (热) 分布:
   f(E) ∝ (E + m_e c^2) · sqrt(E^2 + 2 E m_e c^2) · exp(-E / T_hot)

2) 简化模型: 使用对数正态分布 + Box-Muller 采样:
   E_i = E_0 · exp(σ · N(0,1))
   其中 σ 由能散 FWHM 确定:  σ = FWHM / (2 sqrt(2 ln 2))

3) 角分布: 锥形分布, 半角 θ_div
   μ = cos(θ) 均匀分布于 [cos(θ_div), 1]

4) 空间分布: 高斯束斑
   r = σ_r · sqrt(-2 ln(R_1))    (Box-Muller 径向)
   φ = 2π · R_2

核心来源 (种子项目映射):
- 816_normal  : Box-Muller 正态采样
- 910_prime   : 素数序列用于准随机种子扰动
"""

import math
import random

from plasma_parameters import (
    ELECTRON_MASS, ELECTRON_CHARGE, SPEED_OF_LIGHT,
    EV_TO_JOULE, MEV_TO_JOULE,
    relativistic_gamma, relativistic_momentum,
)


# ============================================================
# Box-Muller 正态采样 (来自 816_normal)
# ============================================================
def box_muller_single(rng=None):
    """
    Box-Muller 方法生成单个标准正态样本 N(0,1).

    公式:
        Z = sqrt(-2 ln U_1) · cos(2π U_2)
    其中 U_1, U_2 ~ Uniform(0,1) 独立.

    参数:
        rng: 可选随机数生成器 (random.Random 实例)
    返回:
        float: N(0,1) 样本
    """
    _rng = rng if rng is not None else random
    u1 = _rng.random()
    u2 = _rng.random()
    # 防止 log(0)
    u1 = max(u1, 1.0e-300)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def box_muller_pair(rng=None):
    """
    Box-Muller 方法生成一对独立标准正态样本.

    公式:
        Z_1 = sqrt(-2 ln U_1) · cos(2π U_2)
        Z_2 = sqrt(-2 ln U_1) · sin(2π U_2)
    """
    _rng = rng if rng is not None else random
    u1 = max(_rng.random(), 1.0e-300)
    u2 = _rng.random()
    r = math.sqrt(-2.0 * math.log(u1))
    z1 = r * math.cos(2.0 * math.pi * u2)
    z2 = r * math.sin(2.0 * math.pi * u2)
    return z1, z2


def normal_vector(n, mu=0.0, sigma=1.0, seed=None):
    """
    生成 N 维正态随机向量 (来自 816_normal/r8vec_normal_01).

    参数:
        n    : 样本数
        mu   : 均值
        sigma: 标准差
        seed : 随机种子
    返回:
        list[float]: N 个 N(mu, sigma^2) 样本
    """
    rng = random.Random(seed)
    result = []
    # 偶数对采样
    i = 0
    while i + 1 < n:
        z1, z2 = box_muller_pair(rng)
        result.append(mu + sigma * z1)
        result.append(mu + sigma * z2)
        i += 2
    if i < n:
        z1 = box_muller_single(rng)
        result.append(mu + sigma * z1)
    return result


# ============================================================
# 素数序列用于准随机扰动 (来自 910_prime)
# ============================================================
def prime_sieve(n):
    """
    Eratosthenes 筛法 (来自 910_prime/prime_sieve).
    返回 [2, n] 内所有素数.
    """
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    i = 2
    while i * i <= n:
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
        i += 1
    return [k for k in range(2, n + 1) if is_prime[k]]


def prime_count(n):
    """
    朴素计数 (来自 910_prime/prime).
    返回 ≤ n 的素数个数 π(n).
    """
    if n < 2:
        return 0
    total = 0
    for i in range(2, n + 1):
        p = 1
        j = 2
        while j * j <= i:
            if i % j == 0:
                p = 0
                break
            j += 1
        total += p
    return total


# ============================================================
# 相对论快电子束采样
# ============================================================
class FastElectronBeam:
    """
    快电子束 Monte Carlo 采样器.

    每个宏粒子具有:
        (x, y, vx, vy, vz, E_kin [eV], weight)
    采样模型:
        - 能量: 对数正态 (Box-Muller) 中心 E_0, 展宽 σ_E
        - 角度: 锥角 θ ~ cos^-1(Uniform(cos θ_div, 1))
        - 位置: 高斯束斑 (Box-Muller 径向)
        - 权重: 1/N_total, 归一化到总能量
    """

    def __init__(self, n_particles, E0_ev, sigma_frac, r_beam_m,
                 theta_div_rad, total_energy_j, seed=42):
        self.n_particles = n_particles
        self.E0_ev = E0_ev
        # 由 FWHM 换算标准差: σ = FWHM / (2√(2 ln 2))
        self.sigma_E = sigma_frac * E0_ev / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        self.r_beam = r_beam_m
        self.theta_div = theta_div_rad
        self.total_energy_j = total_energy_j
        self.seed = seed

        # 素数扰动种子 (来自 910_prime)
        self.primes = prime_sieve(max(100, n_particles))

        self.particles = []
        self._sample()

    def _sample(self):
        """执行 Monte Carlo 采样."""
        rng = random.Random(self.seed)
        self.particles = []

        # 素数序列扰动每个粒子 (独特方法论)
        for i in range(self.n_particles):
            # 使用第 i 个素数作为子种子扰动
            prime_seed = self.primes[i % len(self.primes)]
            sub_rng = random.Random(self.seed + prime_seed + i * 7919)

            # (1) 能量: 对数正态采样
            z_E = box_muller_single(sub_rng)
            energy_ev = self.E0_ev * math.exp(self.sigma_E / self.E0_ev * z_E)
            energy_ev = max(energy_ev, 1.0e3)  # 下限 1 keV

            # (2) 极角: 锥形分布
            mu_min = math.cos(self.theta_div)
            mu = mu_min + (1.0 - mu_min) * sub_rng.random()
            theta = math.acos(mu)
            # 方位角
            phi = 2.0 * math.pi * sub_rng.random()

            # (3) 相对论运动学
            gamma = relativistic_gamma(energy_ev)
            p_mag = ELECTRON_MASS * SPEED_OF_LIGHT * math.sqrt(gamma ** 2 - 1.0)
            # 速度方向 (z 为主传播方向)
            vx = (p_mag / (gamma * ELECTRON_MASS)) * math.sin(theta) * math.cos(phi)
            vy = (p_mag / (gamma * ELECTRON_MASS)) * math.sin(theta) * math.sin(phi)
            vz = (p_mag / (gamma * ELECTRON_MASS)) * math.cos(theta)

            # (4) 位置: 高斯束斑 (Box-Muller 径向)
            z_r1, z_r2 = box_muller_pair(sub_rng)
            r = self.r_beam * math.sqrt(abs(z_r1))
            phi_pos = 2.0 * math.pi * z_r2
            x = r * math.cos(phi_pos)
            y = r * math.sin(phi_pos)

            # (5) 权重
            weight = self.total_energy_j / self.n_particles

            self.particles.append({
                "id": i,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "energy_ev": energy_ev,
                "gamma": gamma,
                "weight_j": weight,
            })

    def get_energies(self):
        """返回所有电子能量 [eV]."""
        return [p["energy_ev"] for p in self.particles]

    def get_mean_energy(self):
        """计算平均能量 [eV]."""
        if not self.particles:
            return 0.0
        return sum(p["energy_ev"] for p in self.particles) / len(self.particles)

    def get_energy_spread(self):
        """计算能量展宽 (标准差) [eV]."""
        if len(self.particles) < 2:
            return 0.0
        mean = self.get_mean_energy()
        var = sum((p["energy_ev"] - mean) ** 2 for p in self.particles)
        return math.sqrt(var / (len(self.particles) - 1))

    def get_total_weight(self):
        """总沉积能量 [J]."""
        return sum(p["weight_j"] for p in self.particles)

    def summary(self):
        """返回束流统计摘要."""
        e_mean = self.get_mean_energy()
        e_spread = self.get_energy_spread()
        return {
            "n_particles": self.n_particles,
            "E_mean_eV": e_mean,
            "E_spread_eV": e_spread,
            "E_spread_frac": e_spread / max(e_mean, 1.0),
            "total_energy_J": self.get_total_weight(),
            "r_beam_m": self.r_beam,
            "theta_div_rad": self.theta_div,
            "prime_seed_count": len(self.primes),
        }
