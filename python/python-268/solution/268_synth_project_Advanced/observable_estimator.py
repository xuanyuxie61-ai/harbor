"""
observable_estimator.py - 物理可观测量估计器
==============================================

科学背景 (Scientific Background):
    DQMC 可以测量 Hubbard 模型的各种物理可观测量:
        1. 动能:     E_kin = ⟨T⟩ = Σ_{ij,σ} t_{ij} ⟨c†_i c_j⟩
        2. 势能:     E_pot = U Σ_i ⟨n_{i↑} n_{i↓}⟩
        3. 双占据:   D = (1/Ns) Σ_i ⟨n_{i↑} n_{i↓}⟩
        4. 自旋结构因子: S(q) = (1/Ns) Σ_{ij} e^{iq(r_i-r_j)} ⟨S_i · S_j⟩
        5. 配对关联: P(q) = (1/Ns) Σ_{ij} e^{iq(r_i-r_j)} ⟨Δ†_i Δ_j⟩
        6. 压缩率:   κ = dn/dμ = β/Ns (⟨N²⟩ - ⟨N⟩²)
        7. 比热:     C = dE/dT = β²/Ns (⟨E²⟩ - ⟨E⟩²)

    本模块实现以上各量的估计器.

融合种子项目:
    - 660_legendre_fast_rule: 高精度积分用于 k 空间求和
    - 301_disk01_monte_carlo: 蒙特卡洛积分用于布里渊区

核心公式 (Key Formulas):
    自旋结构因子:
        S(q) = (1/Ns) Σ_{ij} e^{iq·(r_i-r_j)} ⟨S_z(i) S_z(j)⟩
        S_z(i) = (n_{i↑} - n_{i↓}) / 2

    在 DQMC 中 (对给定 HS 构型):
        ⟨c†_{i↑} c_{j↑}⟩ = G_{ij}
        ⟨c†_{i↓} c_{j↓}⟩ = G̃_{ij}  (↓ 的格林函数, 与 ↑ 相同对自旋对称)
        ⟨n_{i↑} n_{j↓}⟩ = ⟨n_{i↑}⟩⟨n_{j↓}⟩ - G_{ij} G̃_{ji}

    配对关联:
        Δ_i = c_{i↓} c_{i↑}  (s-波)
        P(q) = (1/Ns) Σ_{ij} e^{iq·(r_i-r_j)} ⟨Δ†_i Δ_j⟩
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ==========================================================================
#  能量估计
# ==========================================================================

def estimate_kinetic_energy(G: np.ndarray, T_mat: np.ndarray) -> float:
    """
    动能估计:
        E_kin = 2 × Tr[T × G]  (自旋因子 2)

    对自旋对称 Hubbard, G↑ = G↓ = G.
    """
    return 2.0 * np.trace(T_mat @ G).real


def estimate_potential_energy(G: np.ndarray, U: float) -> float:
    """
    势能估计:
        E_pot = U Σ_i ⟨n_{i↑} n_{i↓}⟩

    对等时格林函数 ( Wick 定理):
        ⟨n_{i↑} n_{i↓}⟩ = ⟨n_{i↑}⟩ ⟨n_{i↓}⟩ - |G_{ii}|² + G_{ii}(1-G_{ii})

    简化 (对角近似):
        ⟨n_{i↑} n_{i↓}⟩ ≈ (1 - G_{ii})²
    """
    Ns = G.shape[0]
    n_sites = np.array([1.0 - G[i, i].real for i in range(Ns)])
    return U * np.sum(n_sites ** 2)


def estimate_total_energy(G: np.ndarray, T_mat: np.ndarray, U: float) -> Dict[str, float]:
    """总能量及其分解."""
    E_kin = estimate_kinetic_energy(G, T_mat)
    E_pot = estimate_potential_energy(G, U)
    return {
        'E_kinetic': E_kin,
        'E_potential': E_pot,
        'E_total': E_kin + E_pot,
    }


# ==========================================================================
#  双占据与压缩率
# ==========================================================================

def estimate_double_occupancy(G: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    双占据 D = (1/Ns) Σ_i ⟨n_{i↑} n_{i↓}⟩.

    返回:
        D_mean: 平均双占据
        D_sites: (Ns,) 各格点的双占据
    """
    Ns = G.shape[0]
    D_sites = np.array([
        (1.0 - G[i, i].real) ** 2 for i in range(Ns)
    ])
    return float(np.mean(D_sites)), D_sites


def estimate_compressor_from_fluctuations(n_series: np.ndarray,
                                          beta: float, Ns: int) -> float:
    """
    压缩率由粒子数涨落计算:
        κ = (β / Ns) × (⟨N²⟩ - ⟨N⟩²)

    需要多次测量的 N 序列.
    """
    mean_N = np.mean(n_series)
    var_N = np.var(n_series)
    return beta * var_N / max(Ns, 1)


# ==========================================================================
#  自旋结构因子
# ==========================================================================

def spin_structure_factor(G: np.ndarray, q_vectors: np.ndarray,
                          positions: np.ndarray) -> np.ndarray:
    """
    自旋结构因子 S(q).

    S(q) = (1/Ns) Σ_{ij} e^{iq·(r_i-r_j)} ⟨S_z(i) S_z(j)⟩

    在 DQMC 中 (Wick 定理):
        ⟨S_z(i) S_z(j)⟩ = (1/4)[⟨n_i n_j⟩ - 2⟨n_{i↑} n_{j↓}⟩ + ...]

    简化 (仅对角贡献):
        ⟨S_z(i) S_z(j)⟩ ≈ (1/4) δ_{ij} ⟨n_i⟩ - (1/4) G_{ij} G_{ji}

    参数:
        G:         (Ns, Ns) 等时格林函数
        q_vectors: (Nq, 2) 动量转移向量
        positions: (Ns, 2) 格点位置

    返回:
        S_q: (Nq,) 各 q 点的结构因子
    """
    Ns = G.shape[0]
    Nq = q_vectors.shape[0]
    S_q = np.zeros(Nq)

    # 计算自旋-自旋关联函数 C_{ij} = ⟨S_z(i) S_z(j)⟩
    n_sites = np.array([1.0 - G[i, i].real for i in range(Ns)])

    for q_idx in range(Nq):
        q = q_vectors[q_idx]
        S_val = 0.0 + 0.0j
        for i in range(Ns):
            for j in range(Ns):
                dr = positions[i] - positions[j]
                phase = np.exp(1j * np.dot(q, dr))
                # Wick 分解
                C_ij = 0.25 * (
                    n_sites[i] * n_sites[j]
                    - G[i, j] * G[j, i]
                    + (1 if i == j else 0) * n_sites[i] * 0.5
                )
                S_val += phase * C_ij
        S_q[q_idx] = S_val.real / Ns

    return S_q


# ==========================================================================
#  配对关联函数
# ==========================================================================

def pairing_correlation(G: np.ndarray, q_vectors: np.ndarray,
                        positions: np.ndarray) -> np.ndarray:
    """
    s-波配对关联函数 P(q).

    Δ_i = c_{i↓} c_{i↑}  (s-波对算符)
    P(q) = (1/Ns) Σ_{ij} e^{iq·(r_i-r_j)} ⟨Δ†_i Δ_j⟩

    在 DQMC 中:
        ⟨Δ†_i Δ_j⟩ = ⟨c†_{i↑} c†_{i↓} c_{j↓} c_{j↑}⟩
                     = -⟨c†_{i↑} c_{j↑}⟩ ⟨c†_{i↓} c_{j↓}⟩ + ...
                     = -G_{ij} G̃_{ij}  (对自旋对称 G̃ = G)

    返回:
        P_q: (Nq,) 配对关联
    """
    Ns = G.shape[0]
    Nq = q_vectors.shape[0]
    P_q = np.zeros(Nq)

    for q_idx in range(Nq):
        q = q_vectors[q_idx]
        P_val = 0.0 + 0.0j
        for i in range(Ns):
            for j in range(Ns):
                dr = positions[i] - positions[j]
                phase = np.exp(1j * np.dot(q, dr))
                # s-波配对: ⟨Δ†_i Δ_j⟩ = |G_{ij}|²
                pair_corr = np.abs(G[i, j]) ** 2
                P_val += phase * pair_corr
        P_q[q_idx] = P_val.real / Ns

    return P_q


# ==========================================================================
#  高对称路径上的动量分布
# ==========================================================================

def high_symmetry_path_kpoints(a: float = 1.0, n_points_per_segment: int = 20
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造第一布里渊区高对称路径上的 k 点.

    路径: Γ → M → K → Γ

    返回:
        k_points: (Nk, 2) k 空间坐标
        distances: (Nk,) 沿路径的累积距离
    """
    b1 = np.array([2 * np.pi / a, -2 * np.pi / (a * np.sqrt(3))])
    b2 = np.array([0, 4 * np.pi / (a * np.sqrt(3))])

    Gamma = np.array([0.0, 0.0])
    M = (b1 + b2) / 2.0
    K = (b1 + b2) / 3.0

    segments = [
        (Gamma, M),
        (M, K),
        (K, Gamma),
    ]

    k_points = []
    distances = []
    cum_dist = 0.0

    for start, end in segments:
        for i in range(n_points_per_segment):
            t = i / n_points_per_segment
            k = start + t * (end - start)
            k_points.append(k)
            if i > 0 or len(distances) > 0:
                prev_k = k_points[-2] if len(k_points) >= 2 else start
                cum_dist += np.linalg.norm(k - prev_k)
            distances.append(cum_dist)

    return np.array(k_points), np.array(distances)


# ==========================================================================
#  态密度 (DOS) 与费米面
# ==========================================================================

def compute_dos_by_kernel_polishing(energies: np.ndarray,
                                    omega: np.ndarray,
                                    eta: float = 0.1) -> np.ndarray:
    """
    核平滑态密度.

    N(ω) = (1/N_k) Σ_k (η/π) / [(ω - ε_k)² + η²]

    即 Lorentzian 展宽的态密度.

    参数:
        energies: 本征值集合
        omega: 频率网格
        eta: 展宽参数
    """
    dos = np.zeros_like(omega)
    for eps in energies:
        dos += (eta / np.pi) / ((omega - eps) ** 2 + eta ** 2)
    dos /= len(energies)
    return dos


def find_fermi_surface(energies_k: np.ndarray, positions_k: np.ndarray,
                       mu: float, tolerance: float = 0.1
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    识别费米面: ε(k) ≈ μ 的 k 点集合.

    返回:
        k_fs: 费米面上的 k 点
        energies_fs: 对应的能量
    """
    mask = np.abs(energies_k - mu) < tolerance
    return positions_k[mask], energies_k[mask]
