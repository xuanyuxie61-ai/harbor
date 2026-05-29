"""
hubbard_hamiltonian.py

Hubbard 模型哈密顿量的精确对角化。

单带Hubbard模型:
    H = -t Σ_{<ij>,σ} (c†_{iσ} c_{jσ} + h.c.)
        + U Σ_i n_{i↑} n_{i↓}
        - μ Σ_{i,σ} n_{iσ}

其中 t 为跃迁积分，U 为在位库仑排斥，μ 为化学势。
本模块使用全对角化获取小团簇的精确基态与热力学量，
作为更大规模近似方法（如DQMC）的基准。
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional


def fermion_hilbert_dimension(nsites: int) -> int:
    """N 格点、自旋↑↓各一个费米子模式的Hilbert空间维度 = 4^N。"""
    if nsites < 0 or nsites > 6:
        # 限制为6个格点，维度 4^6 = 4096，可全对角化
        raise ValueError("nsites 必须满足 0 <= nsites <= 6")
    return 4 ** nsites


def _occupation_state(nsites: int, state_idx: int, sigma: int) -> int:
    """
    返回给定Hilbert态中，自旋σ(0=↑, 1=↓)在第state_idx格点上的占据数(0或1)。
    每个格点用2比特编码: bit(2*i) = ↑, bit(2*i+1) = ↓
    """
    if not (0 <= state_idx < nsites):
        raise IndexError("state_idx 越界")
    bit_pos = 2 * state_idx + sigma
    return (state_idx >> bit_pos) & 1


def _apply_hopping(state: int, i: int, j: int, sigma: int, nsites: int) -> Tuple[int, float]:
    """
    对态 |state> 作用 c†_{iσ} c_{jσ}，返回 (新态, 符号)。
    若 j 位未被占据或 i 位已被占据，返回 (-1, 0.0)。
    """
    bit_j = 2 * j + sigma
    bit_i = 2 * i + sigma
    # 检查 c_{jσ} 能否作用 (j 必须被占据)
    if not ((state >> bit_j) & 1):
        return -1, 0.0
    # 检查 c†_{iσ} 能否作用 (i 必须为空)
    if (state >> bit_i) & 1:
        return -1, 0.0
    # 消去 j，创建 i
    new_state = state ^ (1 << bit_j)
    new_state = new_state ^ (1 << bit_i)
    # Jordan-Wigner 符号: 计算 j 与 i 之间被占据的费米子数
    low, high = sorted((bit_i, bit_j))
    sign = (-1) ** bin((state >> low) & ((1 << (high - low)) - 1)).count("1")
    return new_state, float(sign)


def build_hubbard_hamiltonian(nsites: int, neighbors: list, t: float, U: float, mu: float = 0.0) -> np.ndarray:
    """
    构建 Hubbard 哈密顿矩阵 (稠密矩阵)。
    
    参数:
        nsites: 格点数
        neighbors: neighbors[i] = [j1, j2, ...] 为近邻索引列表
        t: 跃迁积分
        U: 在位库仑排斥
        mu: 化学势
    
    返回:
        H: 形状 (4^nsites, 4^nsites) 的复Hermite矩阵
    """
    if nsites < 0 or nsites > 6:
        raise ValueError("nsites 必须在 [0, 6] 范围内")
    dim = 4 ** nsites
    H = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):
        # 在位相互作用 + 化学势项 (对角)
        n_up_total = 0
        n_dn_total = 0
        for i in range(nsites):
            n_up = (state >> (2 * i)) & 1
            n_dn = (state >> (2 * i + 1)) & 1
            n_up_total += n_up
            n_dn_total += n_dn
            H[state, state] += U * n_up * n_dn
        H[state, state] -= mu * (n_up_total + n_dn_total)
        # 跃迁项 (非对角)
        for i in range(nsites):
            for j in neighbors[i]:
                if j <= i:
                    continue  # 只算一次，利用Hermite对称
                for sigma in [0, 1]:
                    new_state, sign = _apply_hopping(state, i, j, sigma, nsites)
                    if sign != 0.0:
                        H[state, new_state] -= t * sign
    # 确保厄米
    H = 0.5 * (H + H.T.conj())
    return H


def exact_diagonalization(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对哈密顿量进行全对角化，返回 (本征值, 本征矢)。"""
    if H.shape[0] != H.shape[1]:
        raise ValueError("H 必须是方阵")
    evals, evecs = eigh(H)
    return evals, evecs


def thermal_average(evals: np.ndarray, evecs: np.ndarray, operator: np.ndarray, beta: float) -> float:
    """
    计算热力学期望值 <O> = Tr[O exp(-βH)] / Z。
    
    参数:
        evals: 本征能级
        evecs: 本征矢矩阵 (列矢)
        operator: 观测量算符
        beta: 逆温度 β = 1/(k_B T)
    
    返回:
        热力学期望值
    """
    if beta < 0:
        raise ValueError("beta 必须 >= 0")
    if len(evals) == 0:
        return 0.0
    # 减去基态能以防止溢出
    E0 = np.min(evals)
    weights = np.exp(-beta * (evals - E0))
    Z = np.sum(weights)
    if Z == 0:
        return 0.0
    # O_{nn} = <n|O|n>
    O_diag = np.einsum('ni,ij,nj->n', evecs, operator, evecs)
    return np.sum(weights * O_diag) / Z


def double_occupancy_operator(nsites: int) -> np.ndarray:
    """构造双占据数算符 D = Σ_i n_{i↑} n_{i↓}。"""
    dim = 4 ** nsites
    D = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):
        d = 0
        for i in range(nsites):
            n_up = (state >> (2 * i)) & 1
            n_dn = (state >> (2 * i + 1)) & 1
            d += n_up * n_dn
        D[state, state] = float(d)
    return D


def density_operator(nsites: int, sigma: int) -> np.ndarray:
    """构造总粒子数算符 N_σ = Σ_i n_{iσ}。"""
    dim = 4 ** nsites
    Nop = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):
        n = 0
        for i in range(nsites):
            n += (state >> (2 * i + sigma)) & 1
        Nop[state, state] = float(n)
    return Nop


def compute_ground_state_properties(nsites: int, neighbors: list, t: float, U: float, mu: float = 0.0) -> dict:
    """
    计算基态性质，返回字典。
    """
    H = build_hubbard_hamiltonian(nsites, neighbors, t, U, mu)
    evals, evecs = exact_diagonalization(H)
    gs = evecs[:, 0]
    E0 = evals[0]
    D = double_occupancy_operator(nsites)
    d_occ = float(np.vdot(gs, D @ gs).real)
    Nup = density_operator(nsites, 0)
    Ndn = density_operator(nsites, 1)
    n_up = float(np.vdot(gs, Nup @ gs).real)
    n_dn = float(np.vdot(gs, Ndn @ gs).real)
    return {
        "E0": E0,
        "double_occupancy": d_occ,
        "n_up": n_up,
        "n_dn": n_dn,
        "n_total": n_up + n_dn,
        "energy_gap": evals[1] - evals[0] if len(evals) > 1 else 0.0,
    }


if __name__ == "__main__":
    # 2-site Hubbard model 测试
    nsites = 2
    neighbors = [[1], [0]]
    props = compute_ground_state_properties(nsites, neighbors, t=1.0, U=4.0)
    print(props)
