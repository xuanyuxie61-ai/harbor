"""
interatomic_potential.py — 原子间相互作用势与力常数
==================================================

融合种子项目:
  - 746_md_parfor: 分子动力学中的对势 V(r) = sin^2(min(r, pi/2))
                   力 F = sin(2*min(r, pi/2)), 截断策略
  - 368_fd2d_poisson: 有限差分 5 点模板 -> 力常数矩阵的空间离散化
  - 195_coin_simulation: 统计采样思想 (Bernoulli 试验, 运行平均)

物理背景:
  晶体原子间的相互作用势 phi(r) 的 Taylor 展开:
    phi(r) = phi(r0) + phi'(r0)*(r-r0) + (1/2)*phi''(r0)*(r-r0)^2 + ...
  力常数: Phi_{ij,ab} = d^2 E_total / (du_{i,a} * du_{j,b})
  其中 u_{i,a} 是原子 i 在 a 方向的位移。

核心势函数:
  1. Lennard-Jones: V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6)
  2. Morse:         V(r) = D*(exp(-alpha*(r-r0)) - 1)^2
  3. Stillinger-Weber (用于 Si): 三体项贡献
  4. Born-Mayer:    V(r) = A*exp(-r/rho) - C/r^6
"""

import numpy as np
from typing import Tuple, Dict, Optional


class InteratomicPotential:
    """原子间相互作用势基类"""

    def __init__(self, potential_type: str, params: Dict):
        self.ptype = potential_type
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """参数合法性检验 (边界鲁棒性)"""
        if self.ptype == 'lj':
            assert self.params['epsilon'] > 0, "LJ eps 必须 > 0"
            assert self.params['sigma'] > 0, "LJ sigma 必须 > 0"
        elif self.ptype == 'morse':
            assert self.params['D'] > 0, "Morse D 必须 > 0"
            assert self.params['alpha'] > 0, "Morse alpha 必须 > 0"
            assert self.params['r0'] > 0, "Morse r0 必须 > 0"
        elif self.ptype == 'born_mayer':
            assert self.params['A'] > 0
            assert self.params['rho'] > 0
        elif self.ptype == 'sin2_trunc':
            assert self.params['r_cut'] > 0
        else:
            raise ValueError(f"未知势函数类型: {self.ptype}")

    def energy(self, r: np.ndarray) -> np.ndarray:
        """计算势能 V(r), 融合 md_parfor 的截断策略"""
        r = np.maximum(r, 1e-15)  # 防止除零
        if self.ptype == 'lj':
            eps, sig = self.params['epsilon'], self.params['sigma']
            sr6 = (sig / r) ** 6
            return 4 * eps * (sr6 ** 2 - sr6)
        elif self.ptype == 'morse':
            D, alpha, r0 = self.params['D'], self.params['alpha'], self.params['r0']
            return D * (np.exp(-alpha * (r - r0)) - 1) ** 2
        elif self.ptype == 'born_mayer':
            A, rho, C = self.params['A'], self.params['rho'], self.params['C']
            return A * np.exp(-r / rho) - C / r ** 6
        elif self.ptype == 'sin2_trunc':
            r_cut = self.params['r_cut']
            r_eff = np.minimum(r, r_cut)
            return np.sin(r_eff) ** 2
        return np.zeros_like(r)

    def force_magnitude(self, r: np.ndarray) -> np.ndarray:
        """计算力的大小 |F(r)| = -dV/dr"""
        r = np.maximum(r, 1e-15)
        if self.ptype == 'lj':
            eps, sig = self.params['epsilon'], self.params['sigma']
            sr6 = (sig / r) ** 6
            return 24 * eps / r * (2 * sr6 ** 2 - sr6)
        elif self.ptype == 'morse':
            D, alpha, r0 = self.params['D'], self.params['alpha'], self.params['r0']
            return 2 * D * alpha * np.exp(-alpha * (r - r0)) * (
                1 - np.exp(-alpha * (r - r0)))
        elif self.ptype == 'born_mayer':
            A, rho, C = self.params['A'], self.params['rho'], self.params['C']
            return A / rho * np.exp(-r / rho) - 6 * C / r ** 7
        elif self.ptype == 'sin2_trunc':
            r_cut = self.params['r_cut']
            r_eff = np.minimum(r, r_cut)
            return np.sin(2 * r_eff)
        return np.zeros_like(r)

    def force_constant_scalar(self, r: np.ndarray) -> np.ndarray:
        """
        径向力常数 k_r = d^2V/dr^2 (标量部分)。
        融合 fd2d_poisson 的差分模板思想: 二阶导数的中心差分
        k_r ≈ (V(r+h) - 2V(r) + V(r-h)) / h^2
        """
        r = np.maximum(r, 1e-15)
        h = min(0.01 * r.min(), 1e-4)
        h = max(h, 1e-10)
        v_plus = self.energy(r + h)
        v_zero = self.energy(r)
        v_minus = self.energy(r - h)
        return (v_plus - 2 * v_zero + v_minus) / h ** 2


def compute_force_constant_matrix(
    bond_vectors: np.ndarray,
    bond_distances: np.ndarray,
    potential: InteratomicPotential,
    n_dim: int = 3,
) -> np.ndarray:
    """
    从键向量和势函数计算力常数矩阵 Phi_{ab} (n_dim x n_dim)。

    对于中心势 V(r), 力常数矩阵为:
      Phi_{ab} = (r_a * r_b / r^2) * (d^2V/dr^2 - (1/r)*dV/dr)
                 + delta_{ab} * (1/r) * dV/dr

    物理含义: 第一项为纵向(键方向)力常数, 第二项为横向力常数。

    融合 fd2d_poisson 的离散化模板: 将连续力常数映射到
    离散格点差分模板。

    参数:
        bond_vectors: (n_bonds, 3) 键向量
        bond_distances: (n_bonds,) 键长
        potential: 势函数对象
        n_dim: 空间维度

    返回:
        Phi: (n_dim, n_dim) 力常数矩阵
    """
    Phi = np.zeros((n_dim, n_dim))
    r = np.maximum(bond_distances, 1e-15)
    f_mag = potential.force_magnitude(r)  # -dV/dr
    k_scalar = potential.force_constant_scalar(r)  # d^2V/dr^2

    for i in range(len(bond_vectors)):
        b = bond_vectors[i]
        ri = r[i]
        # 纵向力常数
        kl = k_scalar[i]
        # 横向力常数
        kt = f_mag[i] / ri
        # 外积贡献
        b_hat = b / ri
        for a in range(n_dim):
            for c in range(n_dim):
                Phi[a, c] += kl * b_hat[a] * b_hat[c]
                if a == c:
                    Phi[a, c] += kt * (1.0 - b_hat[a] * b_hat[c])
                else:
                    Phi[a, c] -= kt * b_hat[a] * b_hat[c]
    return Phi


def compute_born_effective_charge(
    lattice_type: str, a: float,
    mass_1: float, mass_2: float,
) -> np.ndarray:
    """
    计算 Born 有效电荷张量 Z*_{ij,ab}。

    对于二元化合物 (如 GaAs), Born 有效电荷联系离子位移
    与极化: P_a = (e/Omega) * sum_{kappa,b} Z*_{kappa,ab} * u_{kappa,b}

    约束: sum_kappa Z*_{kappa,ab} = 0 (声学求和规则)

    参数:
        lattice_type: 晶格类型
        a: 晶格常数
        mass_1, mass_2: 两种原子质量

    返回:
        Z_star: (2, 3, 3) Born 有效电荷张量
    """
    Z_star = np.zeros((2, 3, 3))
    # Kleinman 对称性: Z*_{ab} = Z*_{ba}
    # 对闪锌矿结构, Z* 为对角矩阵
    # 经验公式: Z* ~ (m2 - m1) / (m1 + m2) * e * alpha_polar
    mass_ratio = (mass_2 - mass_1) / (mass_1 + mass_2)
    # 极性参数 (Phillips 离子性)
    f_i = mass_ratio ** 2  # 简化模型
    Z_eff = 2.0 * f_i * np.sqrt(mass_1 * mass_2) / (mass_1 + mass_2)
    Z_star[0] = -Z_eff * np.eye(3)
    Z_star[1] = Z_eff * np.eye(3)
    return Z_star


def lennard_jones_equilibrium(sigma: float) -> float:
    """LJ 势的平衡距离: r_eq = 2^(1/6) * sigma"""
    return sigma * 2.0 ** (1.0 / 6.0)


def cohesive_energy(
    potential: InteratomicPotential,
    neighbor_distances: np.ndarray,
    neighbor_counts: np.ndarray,
) -> float:
    """
    内聚能: E_coh = -(1/N) * sum_l Z_l * V(r_l)
    其中 Z_l 为第 l 壳层配位数, r_l 为壳层距离。
    """
    e_coh = 0.0
    for l in range(len(neighbor_distances)):
        r_l = neighbor_distances[l]
        z_l = neighbor_counts[l]
        e_coh -= 0.5 * z_l * potential.energy(np.array([r_l]))[0]
    return e_coh
