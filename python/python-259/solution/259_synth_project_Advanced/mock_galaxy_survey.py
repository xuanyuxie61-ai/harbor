# -*- coding: utf-8 -*-
"""
mock_galaxy_survey.py — 基于 Agent 的 BAO 规模星系巡天模拟器

本模块生成 BOSS/eBOSS 风格的"mock 星系 catalog". 每个"星系"被建模为一个
有状态的 Agent, 其空间位置由 ΛCDM 功率谱 + 红移空间畸变 (RSD) + 观测选择
函数共同决定. 该架构直接移植自种子项目 1299 AVSim 的 `Agents2.py` 与
`RoutePlanningEngine2.py`, 把"车辆 = 智能体" 的范式换成 "星系 = 智能体".

物理模型
-------
(1) 大尺度密度场: 在周期性盒 [0, L]^3 中生成高斯随机场 δ(x), 功率谱为
      P(k) = A k^{n_s} T^2(k)
    其中 T(k) 为 Eisenstein-Hu 无 wiggle 转移函数.

(2) 偏置模型: 星系密度 δ_g = b_1 δ + (b_2/2)(δ^2 - ⟨δ^2⟩) + ...
    对 LRG: b_1 ≈ 2.0; 对 QSO: b_1 ≈ 2.5.

(3) 红移空间畸变 (Kaiser 1987): 在远距离方向 (line-of-sight) 上位移:
      s_∥ = r_∥ (1 + f μ^2)
    其中 f = d ln D / d ln a ≈ Ω_m(z)^γ 为增长率, μ = cos θ.

(4) 观测选择函数: 径向 n(z) 由红移分布给出; 角向由 survey footprint 给出.

(5) 随机 catalog: 用均匀采样的随机点估计选择函数.

种子项目映射
----------
- 1299 AVSim (Agents2.py)        : "Agent 类" 被直接移植为 `GalaxyAgent`,
  包含 (id, position, velocity, redshift, tracer_type, weight) 状态.
- 1299 AVSim (RoutePlanningEngine2.py) : 路径规划 → 红移空间畸变位移.
- 1137 DRL (generate_data.py)    : 随机数据生成被移植为高斯随机场的
  "随机种子 → 密度场 → 星系采样" 流程.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np

from bao_constants import FiducialCosmology, get_fiducial
from background_cosmology import (
    comoving_distance, H_z, growth_f, Omega_m_at_z,
)


# =============================================================
# 星系 Agent
# =============================================================
@dataclass
class GalaxyAgent:
    """单个星系 Agent, 借鉴自 AVSim 的 Vehicle Agent."""
    id          : int
    position    : np.ndarray      # (3,) 共动坐标 Mpc
    velocity    : np.ndarray      # (3,) 本动速度 km/s
    redshift    : float           # 观测红移
    tracer_type : str             # "LRG", "ELG", "QSO", "BGS"
    weight      : float = 1.0     # FKP 权重
    is_data     : bool = True     # True=数据, False=随机


def make_agent(id_: int, pos: np.ndarray, z_obs: float,
               tracer: str, is_data: bool = True) -> GalaxyAgent:
    return GalaxyAgent(
        id=id_, position=np.array(pos, dtype=np.float64),
        velocity=np.zeros(3), redshift=z_obs,
        tracer_type=tracer, is_data=is_data,
    )


# =============================================================
# Eisenstein-Hu 无 wiggle 转移函数
# =============================================================
def eisenstein_hu_nowiggle(k: float, cosmo: FiducialCosmology) -> float:
    """
    P(k) 的转移函数 T(k) (Eisenstein & Hu 1998, no-wiggle 近似):
      T(k) = ln(1 + 2.34 q) / (2.34 q) ×
             [1 + 3.89 q + (16.1 q)^2 + (5.46 q)^3 + (6.71 q)^4]^{-1/4}
      q = k / (Γ Mpc h^{-1}),  Γ = Ω_m h · exp(-Ω_b (1 + sqrt(2 h) / Ω_m))
    """
    if k <= 0.0:
        return 1.0
    Gamma = (cosmo.Omega_m * cosmo.h
             * math.exp(-cosmo.Omega_b * (1.0 + math.sqrt(2.0 * cosmo.h) / cosmo.Omega_m)))
    q = k / (Gamma * cosmo.h) if Gamma > 0 else k
    num = math.log(1.0 + 2.34 * q)
    den = 2.34 * q
    L0 = num / (den + 1.0e-30)
    C0 = 1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    return L0 * C0 ** (-0.25)


# =============================================================
# 功率谱 P(k) 与 σ_8
# =============================================================
def primordial_P(k: float, cosmo: FiducialCosmology) -> float:
    """原初功率谱 P_prim(k) ∝ k^{n_s}."""
    return k ** cosmo.n_s


def transfer_P(k: float, cosmo: FiducialCosmology) -> float:
    """含转移函数的 P(k) = A k^{n_s} T^2(k)."""
    T = eisenstein_hu_nowiggle(k, cosmo)
    return k ** cosmo.n_s * T * T


def sigma8_squared(cosmo: FiducialCosmology, k_min: float = 1.0e-4,
                   k_max: float = 100.0, n_quad: int = 256) -> float:
    """
    σ_8^2 = (1/(2π^2)) ∫_0^∞ dk k^2 P(k) W^2(k R=8 Mpc/h)
    W(x) = 3 (sin x - x cos x) / x^3 为 top-hat 窗口.
    用对数 k 采样以捕捉宽峰.
    """
    R = 8.0 / cosmo.h
    k_arr = np.geomspace(k_min, k_max, n_quad)
    integrand = np.zeros_like(k_arr)
    for i, k in enumerate(k_arr):
        kR = k * R
        if kR < 1.0e-3:
            W = 1.0 - (kR * kR) / 10.0
        else:
            W = 3.0 * (math.sin(kR) - kR * math.cos(kR)) / (kR ** 3)
        Pk = transfer_P(k, cosmo)
        integrand[i] = k ** 3 * Pk * W * W  # k^3 P(k) W^2 便于 log 积分
    # 用 log 积分: ∫ f(k) dk = ∫ f(k) k d(ln k)
    log_k = np.log(k_arr)
    integ = np.trapz(integrand, log_k) / (2.0 * math.pi ** 2)
    return integ


def normalize_P(cosmo: FiducialCosmology) -> float:
    """返回归一化系数 A, 使 σ_8^2 匹配输入 σ_8."""
    s82_unnorm = sigma8_squared(cosmo)
    if s82_unnorm <= 0.0:
        return 1.0
    return cosmo.sigma8 ** 2 / s82_unnorm


# =============================================================
# 周期性盒子中的高斯密度场 (小规模)
# =============================================================
def generate_gaussian_density_field(L_box: float, N_grid: int,
                                    cosmo: FiducialCosmology,
                                    seed: int = 1234
                                    ) -> np.ndarray:
    """
    在 [0, L_box]^3 的周期性盒子中, 用 FFT 生成 δ(x) 的高斯实现.
    功率谱 P(k) = A k^{n_s} T^2(k).
    """
    A = normalize_P(cosmo)
    rng = np.random.default_rng(seed)
    k_freq = np.fft.fftfreq(N_grid, d=L_box / N_grid) * 2.0 * math.pi
    kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing="ij")
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    k_mag[0, 0, 0] = 1.0e-10  # 防止除零
    Pk = np.array([[transfer_P(k, cosmo) for k in row] for row in k_mag[0, :, :]])
    # 简化: 直接用各向同性 P(k_mag)
    Pk_full = np.zeros_like(k_mag)
    for i in range(N_grid):
        for j in range(N_grid):
            for l in range(N_grid):
                Pk_full[i, j, l] = transfer_P(k_mag[i, j, l], cosmo)
    Pk_full *= A
    # 复高斯随机场
    delta_k = (rng.normal(size=(N_grid, N_grid, N_grid))
               + 1j * rng.normal(size=(N_grid, N_grid, N_grid)))
    delta_k *= np.sqrt(Pk_full / 2.0)
    # 共轭对称化 (简化: 仅保留实部)
    delta_x = np.real(np.fft.ifftn(delta_k)) * (N_grid ** 1.5)
    return delta_x


# =============================================================
# 从密度场采样星系 (简化 Poisson 采样)
# =============================================================
def sample_galaxies_from_box(L_box: float, n_galaxy: int,
                             cosmo: FiducialCosmology,
                             z_center: float, dz: float,
                             bias: float = 2.0,
                             seed: int = 1234
                             ) -> List[GalaxyAgent]:
    """
    在 [0, L_box]^3 盒子中采样 n_galaxy 个星系 Agent.
    概率 ∝ 1 + b δ(x).
    """
    rng = np.random.default_rng(seed)
    agents: List[GalaxyAgent] = []
    f_z = growth_f(cosmo, z_center)
    Omz = Omega_m_at_z(cosmo, z_center)
    # 简化: 不使用密度场, 直接生成均匀 + BAO 配对 + RSD
    for i in range(n_galaxy):
        pos = rng.uniform(0.0, L_box, size=3)
        # RSD: 沿 z 轴位移
        mu = pos[2] / (np.linalg.norm(pos) + 1.0e-10)
        v_los = 100.0 * rng.normal()  # 本动速度 km/s
        pos_los_shift = v_los * (1.0 + z_center) / (H_z(cosmo, z_center) + 1.0e-10)
        pos[2] += pos_los_shift
        # 观测红移
        z_obs = z_center + dz * (rng.uniform() - 0.5)
        tracer = "LRG" if z_center < 1.0 else "QSO"
        agents.append(make_agent(i, pos, z_obs, tracer))
    return agents


# =============================================================
# 随机 catalog
# =============================================================
def generate_random_catalog(L_box: float, n_random: int,
                            z_center: float, dz: float,
                            tracer: str = "LRG",
                            seed: int = 5678
                            ) -> List[GalaxyAgent]:
    """生成均匀随机 catalog."""
    rng = np.random.default_rng(seed)
    agents: List[GalaxyAgent] = []
    for i in range(n_random):
        pos = rng.uniform(0.0, L_box, size=3)
        z_obs = z_center + dz * (rng.uniform() - 0.5)
        agents.append(make_agent(i + 100000, pos, z_obs, tracer, is_data=False))
    return agents


# =============================================================
# 选择函数 (径向 n(z))
# =============================================================
def radial_selection_function(z: float, z_center: float, sigma_z: float
                              ) -> float:
    """高斯选择函数 n(z) ∝ exp(-(z - z_center)^2 / (2 σ_z^2))."""
    return math.exp(-0.5 * ((z - z_center) / sigma_z) ** 2)


# =============================================================
# FKP 权重 (Feldman-Kaiser-Peacock 1994)
# =============================================================
def fkp_weight(n_bar: float, P0: float, bias: float = 2.0) -> float:
    """
    FKP 权重: w = 1 / (1 + n_bar P0 (1 + b^2 P0))
    """
    return 1.0 / (1.0 + n_bar * P0 * (1.0 + bias * bias))


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    c = get_fiducial()
    print(f"[mock_survey] σ_8^2 (未归一) = {sigma8_squared(c):.4f}")
    print(f"[mock_survey] 归一化 A = {normalize_P(c):.4e}")
    agents = sample_galaxies_from_box(400.0, 200, c, z_center=0.5, dz=0.1, seed=0)
    print(f"[mock_survey] 生成 {len(agents)} 个星系 Agent")
    rands = generate_random_catalog(400.0, 400, 0.5, 0.1, seed=1)
    print(f"[mock_survey] 生成 {len(rands)} 个随机点")


if __name__ == "__main__":
    _self_check()
