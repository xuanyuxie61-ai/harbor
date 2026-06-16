"""
jet_clustering.py — 喷注聚类算法模块

融合种子项目:
  - 793_nearest_neighbor : 最近邻搜索 (KNN 聚类核心)
  - 192_closest_point_brute : 暴力最近点搜索
  - 849_partition_brute : 集合划分 (用于子结构分析)

核心算法:
  1. anti-kT 算法 (LHC 标准喷注定义):
       d_ij = min(p_{Ti}^{-2}, p_{Tj}^{-2}) · ΔR_{ij}² / R²
       d_iB = p_{Ti}^{-2}
     每次合并 d 最小的粒子对, 或声明 d_iB 最小的为喷注.

  2. Cambridge/Aachen (C/A) 算法:
       d_ij = ΔR_{ij}² / R²
     仅基于几何距离, 不含 pT 权重.

  3. kT 算法:
       d_ij = min(p_{Ti}^{2}, p_{Tj}^{2}) · ΔR_{ij}² / R²
       d_iB = p_{Ti}^{2}

  距离度量的统一形式:
    d_ij = min(f(p_{Ti}), f(p_{Tj})) · ΔR_{ij}² / R²
    d_iB = f(p_{Ti})
    其中 f(pT) = pT^p, p=1(kT), p=0(C/A), p=-2(anti-kT)
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from jet_fourvector import FourVector, partition_momentum_brute


# ─────────────────────────────────────────────────────────────────────────────
# 距离矩阵计算 (源自 793_nearest_neighbor + 192_closest_point_brute)
# ─────────────────────────────────────────────────────────────────────────────

def delta_R_sq(pi: FourVector, pj: FourVector) -> float:
    """ΔR² = (Δη)² + (Δφ)²."""
    return pi.delta_R_sq(pj)


def compute_distance_matrix(particles: List[FourVector],
                            R: float,
                            power: int = -2) -> Tuple[np.ndarray, np.ndarray]:
    """计算粒子间的聚类距离矩阵.

    d_ij = min(p_{Ti}^p, p_{Tj}^p) · ΔR_{ij}² / R²
    d_iB = p_{Ti}^p

    Args:
        particles: 粒子列表
        R: 喷注半径参数
        power: 距离度量幂次 p
            p = -2 → anti-kT
            p =  0 → Cambridge/Aachen
            p =  1 → kT

    Returns:
        (d_pair, d_beam): d_pair[i,j] 为粒子对距离,
                          d_beam[i] 为粒子-光束距离
    """
    n = len(particles)
    d_pair = np.full((n, n), np.inf)
    d_beam = np.zeros(n)

    pT_list = [max(p.pT, 1e-10) for p in particles]

    for i in range(n):
        # 光束距离: d_iB = p_{Ti}^p
        d_beam[i] = pT_list[i] ** power

        for j in range(i + 1, n):
            # 粒子对距离
            dr2 = delta_R_sq(particles[i], particles[j])
            f_min = min(pT_list[i] ** power, pT_list[j] ** power)
            d_pair[i, j] = f_min * dr2 / (R ** 2)
            d_pair[j, i] = d_pair[i, j]

    return d_pair, d_beam


def find_nearest_pair(d_pair: np.ndarray,
                      d_beam: np.ndarray) -> Tuple[str, int, int, float]:
    """在所有粒子对距离和光束距离中找到最小值.

    返回:
        (action, i, j, d_min):
        - 'merge' + i, j → 合并粒子 i, j
        - 'beam' + i, -1 → 粒子 i 成为最终喷注

    源自 793_nearest_neighbor 的核心搜索逻辑.
    """
    n = len(d_beam)
    min_d_beam = np.min(d_beam)
    min_d_pair = np.min(d_pair) if n > 1 else np.inf

    if min_d_beam <= min_d_pair:
        idx = int(np.argmin(d_beam))
        return ('beam', idx, -1, min_d_beam)
    else:
        idx = np.unravel_index(np.argmin(d_pair), d_pair.shape)
        return ('merge', int(idx[0]), int(idx[1]), min_d_pair)


def brute_closest_pair(particles: List[FourVector]) -> Tuple[int, int, float]:
    """暴力搜索最近粒子对 (源自 192_closest_point_brute).

    对每个粒子 i, 找到使 ΔR 最小的 j ≠ i.
    时间复杂度 O(n²), 对小规模 (< 1000) 可行.
    """
    n = len(particles)
    if n < 2:
        return (-1, -1, float('inf'))

    best_i, best_j = 0, 1
    best_dr = particles[0].delta_R(particles[1])

    for i in range(n):
        for j in range(i + 1, n):
            dr = particles[i].delta_R(particles[j])
            if dr < best_dr:
                best_dr = dr
                best_i, best_j = i, j

    return (best_i, best_j, best_dr)


# ─────────────────────────────────────────────────────────────────────────────
# E 方案重组 (四动量加法)
# ─────────────────────────────────────────────────────────────────────────────

def recombine_E_scheme(pi: FourVector, pj: FourVector) -> FourVector:
    """E-scheme 四动量重组: p_new = p_i + p_j.

    这是最常用的喷注重组方案, 保持四动量守恒.
    """
    return pi + pj


def recombine_WTA(pi: FourVector, pj: FourVector) -> FourVector:
    """Winner-Takes-All 重组: 方向取 pT 较大者的方向,
    能量为两者之和.

    用于减少非微扰效应对喷注轴的影响.
    """
    if pi.pT >= pj.pT:
        winner, loser = pi, pj
    else:
        winner, loser = pj, pi

    E_new = winner.E + loser.E
    p_mag = winner.p3_mag
    if p_mag < 1e-15:
        return FourVector(E_new, 0, 0, 0)

    # 方向保持 winner 的方向
    scale = E_new / p_mag if p_mag > 1e-15 else 1.0
    return FourVector(E_new,
                      winner.px * scale,
                      winner.py * scale,
                      winner.pz * scale)


# ─────────────────────────────────────────────────────────────────────────────
# 通用迭代聚类算法
# ─────────────────────────────────────────────────────────────────────────────

class JetClusterResult:
    """聚类结果容器."""

    def __init__(self):
        self.jets: List[FourVector] = []
        self.history: List[dict] = []  # 聚类历史 (用于子结构分析)
        self.n_steps: int = 0

    def __repr__(self):
        return (f"JetClusterResult(n_jets={len(self.jets)}, "
                f"n_steps={self.n_steps})")


def sequential_recombination(particles: List[FourVector],
                             R: float = 0.4,
                             power: int = -2,
                             pT_min: float = 5.0,
                             recomb_scheme: str = 'E') -> JetClusterResult:
    """顺序重组喷注聚类算法的统一实现.

    算法流程:
      1. 计算所有 d_ij 和 d_iB
      2. 找到最小的 d = min(min d_ij, min d_iB)
      3. 如果 d = d_ij: 合并粒子 i, j → 新粒子
      4. 如果 d = d_iB: 粒子 i 成为最终喷注
      5. 重复直到所有粒子处理完毕

    Args:
        particles: 输入粒子列表
        R: 喷注半径
        power: 距离度量幂次 (-2=anti-kT, 0=C/A, 1=kT)
        pT_min: 最小 pT 阈值 (GeV)
        recomb_scheme: 重组方案 ('E' 或 'WTA')

    Returns:
        JetClusterResult
    """
    result = JetClusterResult()
    active = list(particles)  # 活跃粒子列表
    recomb = recombine_E_scheme if recomb_scheme == 'E' else recombine_WTA

    step = 0
    while len(active) > 0:
        step += 1

        if len(active) == 1:
            # 最后一个粒子直接成为喷注
            if active[0].pT >= pT_min:
                result.jets.append(active[0])
                result.history.append({
                    'step': step, 'action': 'beam',
                    'particles': [active[0]],
                    'pT': active[0].pT
                })
            active.clear()
            break

        # 计算距离矩阵
        d_pair, d_beam = compute_distance_matrix(active, R, power)

        # 找到最近对
        action, i, j, d_min = find_nearest_pair(d_pair, d_beam)

        if action == 'beam':
            # 粒子 i 成为喷注
            jet = active.pop(i)
            if jet.pT >= pT_min:
                result.jets.append(jet)
            result.history.append({
                'step': step, 'action': 'beam',
                'pT': jet.pT, 'eta': jet.eta, 'phi': jet.phi
            })
        else:
            # 合并粒子 i 和 j (确保 i < j)
            if i > j:
                i, j = j, i
            pj_removed = active.pop(j)
            pi_removed = active.pop(i)
            new_particle = recomb(pi_removed, pj_removed)
            active.append(new_particle)

            result.history.append({
                'step': step, 'action': 'merge',
                'i': i, 'j': j,
                'pT_new': new_particle.pT,
                'dr': math.sqrt(delta_R_sq(pi_removed, pj_removed))
            })

    result.n_steps = step

    # 按 pT 降序排列喷注
    result.jets.sort(key=lambda j: -j.pT)

    return result


def anti_kt(particles: List[FourVector], R: float = 0.4,
            pT_min: float = 5.0) -> JetClusterResult:
    """Anti-kT 算法 (p = -2). LHC 标准喷注定义."""
    return sequential_recombination(particles, R=R, power=-2,
                                    pT_min=pT_min)


def cambridge_aachen(particles: List[FourVector], R: float = 0.4,
                     pT_min: float = 5.0) -> JetClusterResult:
    """Cambridge/Aachen 算法 (p = 0). 纯几何距离."""
    return sequential_recombination(particles, R=R, power=0,
                                    pT_min=pT_min)


def kt_algorithm(particles: List[FourVector], R: float = 0.4,
                 pT_min: float = 5.0) -> JetClusterResult:
    """kT 算法 (p = 1)."""
    return sequential_recombination(particles, R=R, power=1,
                                    pT_min=pT_min)


# ─────────────────────────────────────────────────────────────────────────────
# 聚类历史的二叉树表示 (用于子结构分析)
# ─────────────────────────────────────────────────────────────────────────────

class ClusterTree:
    """聚类历史的二叉树表示.

    每个节点代表一次合并, 叶子节点是原始粒子.
    用于后续计算:
    - 剪枝 (pruning / trimming / pruning)
    - 质量下降 (mass drop)
    - N-subjettiness
    """

    def __init__(self, jet: FourVector,
                 left: Optional['ClusterTree'] = None,
                 right: Optional['ClusterTree'] = None,
                 step: int = 0):
        self.jet = jet
        self.left = left
        self.right = right
        self.step = step
        self.is_leaf = (left is None and right is None)

    @property
    def mass(self) -> float:
        return self.jet.invariant_mass()

    @property
    def pT(self) -> float:
        return self.jet.pT

    @property
    def z(self) -> float:
        """对称性变量 z = min(pT1, pT2) / (pT1 + pT2).

        用于 Mass Drop Condition 和 pruning.
        """
        if self.is_leaf or self.left is None or self.right is None:
            return 0.5
        pT1 = self.left.pT
        pT2 = self.right.pT
        total = pT1 + pT2
        if total < 1e-15:
            return 0.5
        return min(pT1, pT2) / total

    @property
    def delta_R(self) -> float:
        """两个子喷注之间的距离."""
        if self.is_leaf or self.left is None or self.right is None:
            return 0.0
        return self.left.jet.delta_R(self.right.jet)

    def count_leaves(self) -> int:
        """计算叶子节点数 (原始 constituents 数)."""
        if self.is_leaf:
            return 1
        n_left = self.left.count_leaves() if self.left else 0
        n_right = self.right.count_leaves() if self.right else 0
        return n_left + n_right
