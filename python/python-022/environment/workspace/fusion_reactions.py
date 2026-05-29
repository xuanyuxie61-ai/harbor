"""
fusion_reactions.py
===================
DT聚变反应与中子输运模拟模块。

融合原项目 803_niederreiter2（Niederreiter 低差异序列准蒙特卡洛采样）、
134_california_migration（马尔可夫链转移矩阵）、与
543_histogramize（直方图分箱统计）的核心思想，
模拟 ICF 内爆中 DT 聚变反应率的计算、alpha 粒子能量沉积及中子输运。

物理模型：
1. DT 反应率密度:
     R = n_D * n_T * <sigma*v>(T_i)
   其中 <sigma*v> 采用 Bosch-Hale 参数化。

2. alpha 粒子能量沉积（局域）：
     S_alpha = R * E_alpha * f_stop
   f_stop 为停止份额（基于 Bragg 曲线近似）。

3. 中子输运（蒙特卡洛）:
   - 初始位置：聚变反应点
   - 初始方向：各向同性
   - 平均自由程: lambda_n = 1 / (n_DT * sigma_scattering)
   - 采用 niederreiter2 低差异序列优化采样效率

4. 能量统计：histogramize 对中子能谱进行分箱分析。
"""

import numpy as np
from typing import Tuple, List
from icf_parameters import PC, FP, NP, TP
from utils import clamp_array


# ========================================================================
# Niederreiter 准随机序列（基于原项目 803_niederreiter2）
# ========================================================================

class NiederreiterSequence:
    """Niederreiter 基-2 低差异序列生成器（简化实现）。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        # 方向数（预计算 primitive polynomial 的系数）
        self._init_direction_numbers()

    def _init_direction_numbers(self):
        """初始化 Sobol/Niederreiter 方向数（简化基-2 Gray code 版本）。"""
        max_bits = 31
        self.directions = np.zeros((self.dim, max_bits), dtype=np.uint32)

        for d in range(self.dim):
            # 使用简单方向数: v[j] = 2^(max_bits - j - 1) / (2*j + 1)
            for j in range(max_bits):
                self.directions[d, j] = np.uint32(1) << (max_bits - j - 1)

    def next(self) -> np.ndarray:
        """生成下一个准随机向量，分量在 [0, 1]。"""
        self.count += 1
        max_bits = 31
        recip = 2.0**(-max_bits)

        # Gray code 表示
        gray = self.count ^ (self.count >> 1)
        quasi = np.zeros(self.dim)

        for d in range(self.dim):
            x = np.uint32(0)
            g = gray
            bit = 0
            while g > 0 and bit < max_bits:
                if g & 1:
                    x ^= self.directions[d, bit]
                g >>= 1
                bit += 1
            quasi[d] = float(x) * recip

        return np.clip(quasi, 0.0, 1.0)


# ========================================================================
# 聚变反应率
# ========================================================================

def dt_reactivity(T_i_kev: float) -> float:
    """
    DT 聚变反应率 <sigma*v> [m^3/s]。
    使用 Bosch-Hale 参数化。
    """
    return FP.reactivity_dt(T_i_kev)


def compute_fusion_rate_density(n_d: np.ndarray, n_t: np.ndarray,
                                T_i: np.ndarray) -> np.ndarray:
    """
    计算每个单元的聚变反应率密度 [reactions/(m^3*s)]。

    参数
    ----
    n_d, n_t : np.ndarray
        氘、氚数密度 [m^-3]
    T_i : np.ndarray
        离子温度 [K]

    返回
    ----
    rate : np.ndarray
        反应率密度
    """
    n_cells = len(n_d)
    rate = np.zeros(n_cells)
    for i in range(n_cells):
        T_kev = T_i[i] * PC.BOLTZMANN / (1.0e3 * PC.ELEMENTARY_CHARGE)
        T_kev = max(T_kev, 0.1)
        sv = dt_reactivity(T_kev)
        rate[i] = n_d[i] * n_t[i] * sv
    return rate


def alpha_deposition_local(rate_density: np.ndarray, cell_volume: np.ndarray) -> np.ndarray:
    """
    alpha 粒子局域能量沉积 [W/m^3]。
    假设 alpha 粒子在产生位置附近完全沉积（高 rho*r 极限）。
    """
    return rate_density * FP.Q_ALPHA


# ========================================================================
# 马尔可夫链能量弛豫（基于原项目 134_california_migration）
# ========================================================================

def build_energy_relaxation_matrix(dt: float, tau_eq: float) -> np.ndarray:
    """
    构建离子-电子能量弛豫的离散时间马尔可夫转移矩阵。

    两态模型:
        状态 0: 离子热能
        状态 1: 电子热能

    转移概率（Spitzer 弛豫）:
        P(0->1) = dt / tau_eq   (离子向电子传热)
        P(1->0) = dt / tau_eq   (电子向离子传热)

    返回 2x2 转移矩阵 A，满足 pop_new = A @ pop_old。
    """
    p = min(dt / max(tau_eq, 1.0e-30), 0.5)
    A = np.array([
        [1.0 - p, p],
        [p, 1.0 - p]
    ])
    return A


def spitzer_equilibration_time(n_e: float, T_e: float, Z_eff: float,
                               A_ion: float) -> float:
    """
    Spitzer 离子-电子能量 equilibration 时间 [s]。

    tau_eq = (3 * m_i * m_e) / (8 * sqrt(2*pi) * n_e * Z^2 * e^4 * lnLambda)
              * (k_B*T_e/m_e + k_B*T_i/m_i)^(3/2)
    简化 [s]:
        tau_eq ≈ 3.0e-18 * A_ion * T_e^1.5 / (Z_eff^2 * n_e * lnLambda)
    """
    if n_e <= 0.0 or T_e <= 0.0 or Z_eff <= 0.0:
        return 1.0e30
    ln_lambda = max(23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0)), 2.0)
    tau = 3.0e-18 * A_ion * T_e**1.5 / (Z_eff**2 * n_e * ln_lambda)
    return max(tau, 1.0e-30)


def apply_energy_relaxation(E_ion: float, E_e: float, dt: float,
                            n_e: float, T_e: float, Z_eff: float,
                            A_ion: float) -> Tuple[float, float]:
    """
    对单个单元应用离子-电子能量弛豫。

    返回新的离子热能与电子热能密度 [J/m^3]。
    """
    tau_eq = spitzer_equilibration_time(n_e, T_e, Z_eff, A_ion)
    A = build_energy_relaxation_matrix(dt, tau_eq)
    pop = np.array([E_ion, E_e])
    pop_new = A @ pop
    return float(pop_new[0]), float(pop_new[1])


# ========================================================================
# 中子蒙特卡洛输运（基于 Niederreiter 低差异序列）
# ========================================================================

class NeutronMC:
    """中子蒙特卡洛输运模拟器。"""

    def __init__(self, n_samples: int = NP.MC_NEUTRON_SAMPLES):
        self.n_samples = n_samples
        self.sequence = NiederreiterSequence(dim=6)
        self.energies = []
        self.escaped = []

    def sample_isotropic_direction(self) -> np.ndarray:
        """各向同性方向采样（使用低差异序列）。"""
        q = self.sequence.next()
        mu = 2.0 * q[0] - 1.0  # cos(theta)
        phi = 2.0 * np.pi * q[1]
        sin_theta = np.sqrt(max(1.0 - mu**2, 0.0))
        return np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            mu
        ])

    def neutron_mean_free_path(self, rho: float, A_avg: float = 2.5) -> float:
        """
        DT 等离子体中 14.1 MeV 中子的平均自由程。
        近似散射截面 sigma ≈ 3.5 barn（DT 混合）。
        """
        n = rho * PC.AVOGADRO / (A_avg * 1.0e-3)
        sigma = 3.5e-28  # m^2
        return 1.0 / max(n * sigma, 1.0e-30)

    def transport_batch(self, r_cells: np.ndarray, rho_cells: np.ndarray,
                        source_positions: np.ndarray, source_weights: np.ndarray) -> dict:
        """
        批量中子输运模拟。

        参数
        ----
        r_cells : np.ndarray
            单元中心半径
        rho_cells : np.ndarray
            单元密度
        source_positions : np.ndarray
            源位置半径数组
        source_weights : np.ndarray
            各位置源的权重

        返回
        ----
        统计结果字典
        """
        n_cells = len(r_cells)
        cell_flux = np.zeros(n_cells)
        escaped_energy = 0.0
        deposited_energy = 0.0

        n_sources = len(source_positions)
        samples_per_source = self.n_samples // max(n_sources, 1)

        for src_idx in range(n_sources):
            r_src = source_positions[src_idx]
            weight = source_weights[src_idx]

            for _ in range(samples_per_source):
                # 初始位置与方向
                pos = np.array([r_src, 0.0, 0.0])
                direction = self.sample_isotropic_direction()
                energy = FP.Q_NEUTRON

                # 随机步长（指数分布）
                q = self.sequence.next()
                mfp = self.neutron_mean_free_path(
                    np.interp(r_src, r_cells, rho_cells))
                step = -mfp * np.log(max(q[2], 1.0e-30))

                new_pos = pos + step * direction
                r_new = np.sqrt(np.sum(new_pos**2))

                # 判断边界
                if r_new > r_cells[-1] + 1.0e-6:
                    escaped_energy += energy * weight / samples_per_source
                    self.escaped.append(energy)
                else:
                    # 沉积到最近单元
                    cell_idx = np.searchsorted(r_cells, r_new)
                    cell_idx = min(max(cell_idx, 0), n_cells - 1)
                    cell_flux[cell_idx] += weight / samples_per_source
                    deposited_energy += energy * weight / samples_per_source
                    self.energies.append(energy)

        return {
            "cell_flux": cell_flux,
            "escaped_energy": escaped_energy,
            "deposited_energy": deposited_energy,
            "total_samples": samples_per_source * n_sources,
        }


# ========================================================================
# 直方图统计（基于原项目 543_histogramize）
# ========================================================================

def histogramize_spectrum(energies: List[float], n_bins: int = 20,
                          e_min: float = 13.0e6 * PC.ELEMENTARY_CHARGE,
                          e_max: float = 15.0e6 * PC.ELEMENTARY_CHARGE) -> Tuple[np.ndarray, np.ndarray]:
    """
    对中子能量谱进行直方图分箱统计。

    参数
    ----
    energies : List[float]
        中子能量列表 [J]
    n_bins : int
        分箱数
    e_min, e_max : float
        能量范围 [J]

    返回
    ----
    bin_centers, bin_counts : np.ndarray
        箱中心与计数
    """
    if not energies:
        return np.zeros(n_bins), np.zeros(n_bins)

    bin_edges = np.linspace(e_min, e_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(n_bins)

    for e in energies:
        idx = int(n_bins * (e - e_min) / (e_max - e_min))
        if 0 <= idx < n_bins:
            counts[idx] += 1
        elif idx == n_bins and e < bin_edges[-1] + 1.0e-15 * (e_max - e_min):
            counts[-1] += 1

    return bin_centers, counts
