"""
dynamical_matrix.py — 动力学矩阵构建与声学求和规则
==================================================

融合种子项目:
  - 995_r8sm: 动力学矩阵的秩1修正 (缺陷/杂质)
  - 368_fd2d_poisson: 差分模板 -> 力常数组装
  - 746_md_parfor: 对势的力与能量 -> 力常数矩阵元素

物理背景:
  晶格动力学矩阵 D(q) 是力常数矩阵的 Fourier 变换:
    D_{ab}(q) = (1/sqrt(m_a * m_b)) * sum_l Phi_{ab}(l) * exp(i*q*R_l)

  声子色散关系由本征方程确定:
    D(q) * e_{lambda}(q) = omega_{lambda}^2(q) * e_{lambda}(q)

  其中 lambda = (branch, polarization), omega 为声子频率,
  e 为偏振向量。

声学求和规则 (ASR):
  sum_l Phi_{ab}(l) = 0  (对每个 a, b)
  确保在 q=0 处有 3 个零频率声学模式。
"""

import numpy as np
from typing import Tuple, Dict, List
from lattice_geometry import compute_neighbor_shells, generate_bravais_lattice
from interatomic_potential import InteratomicPotential, compute_force_constant_matrix


def build_dynamical_matrix(
    positions: np.ndarray,
    atom_types: np.ndarray,
    masses: np.ndarray,
    box_length: float,
    potential: InteratomicPotential,
    n_shells: int = 3,
    n_dim: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建实空间力常数矩阵 (动力学矩阵的前身)。

    步骤:
      1. 识别近邻壳层 (shell 1..n_shells)
      2. 对每对近邻计算力常数矩阵 Phi_{ab}
      3. 组装 N*n_dim × N*n_dim 的力常数矩阵
      4. 除以质量因子 -> 动力学矩阵

    参数:
        positions: (N, 3) 原子坐标
        atom_types: (N,) 原子类型索引
        masses: (N,) 原子质量
        box_length: 周期性盒子边长
        potential: 原子间势函数
        n_shells: 近邻壳层数
        n_dim: 空间维度

    返回:
        D_real: (3N, 3N) 实空间动力学矩阵
        shells: 近邻壳层信息列表
    """
    n_atoms = len(positions)
    n_dof = n_atoms * n_dim

    shells = compute_neighbor_shells(positions, box_length, n_shells)

    # 初始化力常数矩阵
    Phi = np.zeros((n_dof, n_dof))

    for shell in shells:
        bond_vecs = shell['bond_vectors']
        bond_dists = np.linalg.norm(bond_vecs, axis=1)

        if len(bond_vecs) == 0:
            continue

        # 计算该壳层的力常数矩阵
        phi_shell = compute_force_constant_matrix(
            bond_vecs, bond_dists, potential, n_dim
        )

        # 分配到对应的自由度
        pair_indices = shell['pair_indices']
        for j, atom_j in enumerate(pair_indices):
            for a in range(n_dim):
                for b in range(n_dim):
                    # 非对角块: Phi_{0a, jb} = -phi_shell_{ab}
                    Phi[a, atom_j * n_dim + b] -= phi_shell[a, b]
                    # 对角块: 声学求和规则 -> Phi_{0a, 0b} = -sum_{j!=0} Phi_{0a, jb}
                    Phi[a, b] += phi_shell[a, b]

    # 复制到其他原子 (利用平移对称性)
    for i in range(1, n_atoms):
        for j in range(n_atoms):
            # 简化: 假设所有原子等价 (单元素晶格)
            src_i, src_j = 0, (j - i) % n_atoms
            for a in range(n_dim):
                for b in range(n_dim):
                    Phi[i * n_dim + a, j * n_dim + b] = Phi[src_i * n_dim + a, src_j * n_dim + b]

    # 质量归一化 -> 动力学矩阵
    D_real = np.zeros((n_dof, n_dof))
    for i in range(n_atoms):
        for j in range(n_atoms):
            m_factor = 1.0 / np.sqrt(masses[i] * masses[j])
            D_real[i * n_dim:(i + 1) * n_dim,
                   j * n_dim:(j + 1) * n_dim] = (
                Phi[i * n_dim:(i + 1) * n_dim,
                    j * n_dim:(j + 1) * n_dim] * m_factor
            )

    return D_real, shells


def apply_acoustic_sum_rule(D: np.ndarray, n_atoms: int, n_dim: int = 3) -> np.ndarray:
    """
    强制施加声学求和规则 (ASR)。

    确保 sum_j D_{ij} = 0 对所有 i。
    通过对角块修正: D_{ii} = -sum_{j!=i} D_{ij}

    物理: 保证平移不变性 -> q=0 处 3 个零频率声学模式。
    """
    n_dof = n_atoms * n_dim
    D_fixed = D.copy()
    for i in range(n_atoms):
        for a in range(n_dim):
            row = i * n_dim + a
            # 计算该行的非对角元素之和
            off_diag_sum = 0.0
            for j in range(n_atoms):
                if j != i:
                    for b in range(n_dim):
                        col = j * n_dim + b
                        off_diag_sum += D_fixed[row, col]
            # 修正对角元素
            diag_col = i * n_dim + a
            D_fixed[row, diag_col] = -off_diag_sum
    return D_fixed


def fourier_transform_dynamical_matrix(
    D_real: np.ndarray,
    positions: np.ndarray,
    q_vector: np.ndarray,
    box_length: float,
    n_dim: int = 3,
) -> np.ndarray:
    """
    对实空间动力学矩阵做 Fourier 变换得到 D(q)。

    D_{ab}(q) = sum_l D_{ab}(l) * exp(i * q * R_l)

    其中 R_l 是格矢, l 标记原胞。

    参数:
        D_real: (3N, 3N) 实空间动力学矩阵
        positions: (N, 3) 原子坐标
        q_vector: (3,) 波矢
        box_length: 盒子边长
        n_dim: 维度

    返回:
        D_q: (3N, 3N) 复数动力学矩阵 D(q)
    """
    n_atoms = len(positions)
    n_dof = n_atoms * n_dim
    D_q = np.zeros((n_dof, n_dof), dtype=complex)

    ref_pos = positions[0]
    for j in range(n_atoms):
        # 最小镜像约定
        diff = positions[j] - ref_pos
        diff -= box_length * np.round(diff / box_length)
        phase = np.exp(1j * np.dot(q_vector, diff))
        for a in range(n_dim):
            for b in range(n_dim):
                D_q[a, j * n_dim + b] = D_real[a, j * n_dim + b] * phase

    # 厄米化 (确保本征值为实数)
    D_q = 0.5 * (D_q + D_q.conj().T)
    return D_q


def build_dynamical_matrix_from_force_constants(
    force_constants: Dict[Tuple[int, int], np.ndarray],
    masses: np.ndarray,
    n_atoms: int,
    n_dim: int = 3,
) -> np.ndarray:
    """
    从力常数字典构建动力学矩阵。

    force_constants[(i,j)] = Phi_{ij} (n_dim x n_dim)

    融合 fd2d_poisson 的稀疏矩阵组装模式。
    """
    n_dof = n_atoms * n_dim
    Phi = np.zeros((n_dof, n_dof))

    for (i, j), phi_ij in force_constants.items():
        for a in range(n_dim):
            for b in range(n_dim):
                Phi[i * n_dim + a, j * n_dim + b] = phi_ij[a, b]

    # 质量归一化
    D = np.zeros((n_dof, n_dof))
    for i in range(n_atoms):
        for j in range(n_atoms):
            m_factor = 1.0 / np.sqrt(masses[i] * masses[j])
            for a in range(n_dim):
                for b in range(n_dim):
                    D[i * n_dim + a, j * n_dim + b] = (
                        Phi[i * n_dim + a, j * n_dim + b] * m_factor
                    )

    return D
