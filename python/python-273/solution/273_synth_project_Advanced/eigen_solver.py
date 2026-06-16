"""
eigen_solver.py — 动力学矩阵本征值求解与声子频率提取
====================================================

融合种子项目:
  - 995_r8sm: 特征值问题的秩1修正更新
  - 1250_fjarri-attic_qsim_letter_2011: BEC 量子模拟中的 Bogoliubov 准粒子谱

物理背景:
  声子色散关系: omega_n(k) 通过对动力学矩阵 D(k) 对角化得到。
  D(k) * e_n(k) = omega_n^2(k) * e_n(k)

  对于含 n 个原子的原胞, 有 3n 个声子支:
    - 3 个声学支 (omega -> 0 当 k -> 0)
    - 3n-3 个光学支

  Bogoliubov 变换 (融合 qsim_letter):
    对含时晶格, 准粒子算符 alpha_k = u_k * a_k + v_k * a_{-k}^dag
    色散关系: E_k = sqrt(eps_k^2 + 2*n_0*U*eps_k)

  Hellmann-Feynman 定理用于群速度:
    v_g = d(omega)/dk = <e| dD/dk |e> / (2*omega)
"""

import numpy as np
from typing import Tuple, List, Dict


def solve_phonon_eigenproblem(
    D_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解声子本征问题 D(q)*e = omega^2 * e。

    由于 D(q) 是厄米矩阵, 使用 eigh 保证实数本征值。

    返回:
        omega_sq: (3N,) omega^2 (可能含负值 -> 虚频, 结构不稳定)
        eigvecs: (3N, 3N) 偏振向量 (列向量)
    """
    # 确保厄米性
    D_hermitian = 0.5 * (D_q + D_q.conj().T)
    omega_sq, eigvecs = np.linalg.eigh(D_hermitian)

    # 排序 (从小到大)
    idx = np.argsort(omega_sq)
    omega_sq = omega_sq[idx]
    eigvecs = eigvecs[:, idx]

    return omega_sq, eigvecs


def extract_phonon_frequencies(
    omega_sq: np.ndarray,
    handle_imaginary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 omega^2 提取声子频率 omega。

    omega = sqrt(omega^2)  若 omega^2 > 0 (稳定模式)
    omega = -sqrt(-omega^2) 若 omega^2 < 0 (虚频, 不稳定)

    虚频表明晶格动力学不稳定 (相变前兆)。

    返回:
        omega: (3N,) 声子频率 (THz 或 meV 取决于力常数单位)
        is_stable: (3N,) 布尔数组, True=稳定
    """
    is_stable = omega_sq >= 0
    omega = np.zeros_like(omega_sq)
    omega[is_stable] = np.sqrt(np.maximum(omega_sq[is_stable], 0.0))
    if handle_imaginary:
        unstable = ~is_stable
        omega[unstable] = -np.sqrt(np.maximum(-omega_sq[unstable], 0.0))
    else:
        omega[~is_stable] = 0.0
    return omega, is_stable


def compute_group_velocity_hellmann_feynman(
    D_q: np.ndarray,
    eigvecs: np.ndarray,
    omega: np.ndarray,
    dD_dq: np.ndarray,
    n_dim: int = 3,
) -> np.ndarray:
    """
    Hellmann-Feynman 定理计算声子群速度。

    v_g^{n,alpha} = (1 / (2*omega_n)) * <e_n| dD/dq_alpha |e_n>

    其中 dD/dq_alpha 是动力学矩阵对波矢分量的导数。

    对简单最近邻模型:
      dD/dq_a = sum_l i*R_l^a * D(l) * exp(i*q*R_l)

    参数:
        D_q: (3N, 3N) D(q)
        eigvecs: (3N, 3N) 偏振向量
        omega: (3N,) 声子频率
        dD_dq: (3, 3N, 3N) dD/dq_alpha (复数)
        n_dim: 维度

    返回:
        v_group: (3N, 3) 群速度向量
    """
    n_modes = len(omega)
    v_group = np.zeros((n_modes, n_dim))

    for n in range(n_modes):
        en = eigvecs[:, n]
        if abs(omega[n]) < 1e-12:
            v_group[n] = 0.0
            continue
        for alpha in range(n_dim):
            # <e_n| dD/dq_alpha |e_n>
            matrix_element = en.conj() @ dD_dq[alpha] @ en
            v_group[n, alpha] = np.real(matrix_element) / (2.0 * omega[n])

    return v_group


def compute_dD_dq(
    positions: np.ndarray,
    D_real_blocks: Dict[int, np.ndarray],
    q_vector: np.ndarray,
    box_length: float,
    n_atoms: int,
    n_dim: int = 3,
) -> np.ndarray:
    """
    计算 dD/dq (动力学矩阵对波矢的导数)。

    dD_{ab}/dq_c = sum_l i*R_l^c * D_{ab}(l) * exp(i*q*R_l)
    """
    n_dof = n_atoms * n_dim
    dD_dq = np.zeros((n_dim, n_dof, n_dof), dtype=complex)

    ref = positions[0]
    for j in range(n_atoms):
        diff = positions[j] - ref
        diff -= box_length * np.round(diff / box_length)
        phase = np.exp(1j * np.dot(q_vector, diff))
        block_key = j
        if block_key in D_real_blocks:
            phi_block = D_real_blocks[block_key]
            for a in range(n_dim):
                for b in range(n_dim):
                    for c in range(n_dim):
                        dD_dq[c, a, j * n_dim + b] += (
                            1j * diff[c] * phi_block[a, b] * phase
                        )
    # 厄米化
    for c in range(n_dim):
        dD_dq[c] = 0.5 * (dD_dq[c] + dD_dq[c].conj().T)
    return dD_dq


def compute_bogoliubov_spectrum(
    epsilon_k: np.ndarray,
    interaction_strength: float,
    condensate_density: float,
) -> np.ndarray:
    """
    Bogoliubov 准粒子色散 (融合 qsim_letter)。

    E_k = sqrt(eps_k * (eps_k + 2*n_0*U))
    其中 eps_k = hbar^2*k^2/(2m), U = 4*pi*hbar^2*a/m

    声子极限 (k->0): E_k ≈ hbar*c_s*k, 声速 c_s = sqrt(n_0*U/m)
    自由粒子极限 (k->inf): E_k ≈ eps_k + n_0*U

    参数:
        epsilon_k: (Nk,) 自由粒子色散
        interaction_strength: n_0*U (相互作用能)
        condensate_density: n_0 (凝聚体密度)

    返回:
        E_k: (Nk,) Bogoliubov 色散
    """
    gap = 2.0 * interaction_strength * condensate_density
    E_sq = epsilon_k * (epsilon_k + gap)
    E_sq = np.maximum(E_sq, 0.0)  # 数值保护
    return np.sqrt(E_sq)


def separate_acoustic_optical(
    omega: np.ndarray,
    n_atoms_per_cell: int,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将声子支分为声学支和光学支。

    对含 n_atoms_per_cell 个原子的原胞:
      - 3 个声学支: omega -> 0 当 q -> Gamma
      - 3*(n_atoms_per_cell - 1) 个光学支

    返回:
        acoustic: (3,) 声学支频率
        optical: (3*(n-1),) 光学支频率
    """
    n_acoustic = 3
    acoustic = omega[:n_acoustic]
    optical = omega[n_acoustic:]
    return acoustic, optical


def compute_debye_temperature(
    acoustic_velocities: np.ndarray,
    volume_per_atom: float,
    n_atoms: int = 1,
) -> float:
    """
    德拜温度 Theta_D 计算。

    Theta_D = (hbar/k_B) * (6*pi^2*n/V)^{1/3} * v_D
    其中 v_D = (1/3 * (1/v_L^3 + 2/v_T^3))^{-1/3} 为德拜速度,
    v_L 为纵波速度, v_T 为横波速度。

    参数:
        acoustic_velocities: (3,) [v_L, v_T1, v_T2] (m/s)
        volume_per_atom: 每个原子体积 (m^3)
        n_atoms: 原胞原子数

    返回:
        theta_D: 德拜温度 (K)
    """
    hbar = 1.0546e-34  # J*s
    k_B = 1.3806e-23   # J/K
    vels = np.abs(acoustic_velocities)
    vels = np.maximum(vels, 1.0)  # 防止零速度

    # 德拜速度 (平均)
    inv_v3 = np.sum(1.0 / vels ** 3)
    v_debye = (inv_v3 / 3.0) ** (-1.0 / 3.0)

    # 数密度
    n_density = n_atoms / volume_per_atom

    # 德拜频率
    omega_D = v_debye * (6.0 * np.pi ** 2 * n_density) ** (1.0 / 3.0)

    # 德拜温度
    theta_D = hbar * omega_D / k_B
    return theta_D
