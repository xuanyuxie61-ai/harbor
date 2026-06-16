"""
jet_substructure.py — 喷注子结构观测量模块

融合种子项目:
  - 1233_eileenrmartin_dissertation-reproducibility-passive-DAS :
    互相关函数 (cross-correlation) 的计算框架
  - 1131_cchrisgong_aip_rockstar : merger tree 分析,
    层次聚类历史的递归遍历

核心观测量:
  1. N-subjettiness τ_N:
       τ_N = (1/d₀) Σ_i p_{Ti} min(ΔR_{i,1}, ..., ΔR_{i,N})^β
     其中 d₀ = Σ_i p_{Ti} R^β, β 通常为 1 或 2.
     比值 τ_{N,N-1} = τ_N / τ_{N-1} 用于区分 N-prong 结构.

  2. 能量关联函数 (ECF):
       e2(α) = Σ_{i<j} p_{Ti} p_{Tj} (ΔR_{ij})^α / (Σ_i p_{Ti})²
       e3(α) = Σ_{i<j<k} p_{Ti} p_{Tj} p_{Tk} (ΔR_{ij} ΔR_{jk} ΔR_{ik})^α
                / (Σ_i p_{Ti})³
     D₂ = e3 · e1³ / e2³  (用于区分 quark/gluon 和 W/Z/H 喷注)

  3. 能量密度分布 ρ(r):
       ρ(r) = (1/(2πr Δr)) Σ_{i: r<ΔR_i<R} p_{Ti}
     用于分析喷注内部能量流分布.

  4. 互相关函数 C(Δη, Δφ):
       C(Δη, Δφ) = ⟨ε(η,φ) · ε(η+Δη, φ+Δφ)⟩
     源自 1233 的 DAS 交叉相关分析, 用于提取喷注内部角度关联.
"""

import numpy as np
import math
from typing import List, Optional
from jet_fourvector import FourVector


# ─────────────────────────────────────────────────────────────────────────────
# N-subjettiness (核心喷注子结构观测量)
# ─────────────────────────────────────────────────────────────────────────────

def n_subjettiness(constituents: List[FourVector],
                   axes: List[FourVector],
                   beta: float = 1.0,
                   R: float = 0.4) -> float:
    """计算 N-subjettiness τ_N.

    τ_N = (1/d₀) Σ_i p_{Ti} · min_k(ΔR_{i,k})^β

    其中 d₀ = Σ_i p_{Ti} · R^β 为归一化因子,
    k 遍历 N 个喷注轴.

    β 的物理意义:
      β → 0: 对软辐射敏感
      β = 1: 线性权重, 对硬宽角度辐射敏感
      β = 2: 对宽角度辐射更敏感, 减少非微扰影响

    τ_N ∈ [0, 1]:
      τ_N ≈ 0 → 高度对准 N 个方向 (N-prong 结构)
      τ_N ≈ 1 → 各向同性辐射

    判别量: τ_{N,N-1} = τ_N / τ_{N-1}
      小值 → N-prong 结构
      大值 → (N-1)-prong 或各向同性
    """
    if len(constituents) == 0 or len(axes) == 0:
        return 0.0

    N_axes = len(axes)

    # 归一化因子 d₀
    sum_pT = sum(c.pT for c in constituents)
    d0 = sum_pT * (R ** beta)
    if d0 < 1e-30:
        return 0.0

    # 分子
    numerator = 0.0
    for c in constituents:
        # 找到距 c 最近的轴
        min_dr = float('inf')
        for axis in axes:
            dr = c.delta_R(axis)
            if dr < min_dr:
                min_dr = dr
        numerator += c.pT * (min_dr ** beta)

    return numerator / d0


def n_subjettiness_ratio(constituents: List[FourVector],
                         axes_1: List[FourVector],
                         axes_2: List[FourVector],
                         beta: float = 1.0,
                         R: float = 0.4) -> float:
    """计算 τ_{2,1} = τ_2 / τ_1.

    用于区分 1-prong (quark/gluon) 和 2-prong (W/Z → qq') 喷注.

    τ_{2,1} 小 → 更像 2-prong (如 W 喷注)
    τ_{2,1} 大 → 更像 1-prong (如 QCD 喷注)
    """
    tau_1 = n_subjettiness(constituents, axes_1, beta, R)
    tau_2 = n_subjettiness(constituents, axes_2, beta, R)

    if tau_1 < 1e-15:
        return 1.0  # 避免除零

    return tau_2 / tau_1


# ─────────────────────────────────────────────────────────────────────────────
# 能量关联函数 (Energy Correlation Functions, ECF)
# ─────────────────────────────────────────────────────────────────────────────

def energy_correlation_2(constituents: List[FourVector],
                         alpha: float = 1.0) -> float:
    """二体能量关联函数 e2(α).

    e2(α) = Σ_{i<j} z_i z_j (ΔR_{ij})^α

    其中 z_i = p_{Ti} / p_{T,jet} 为横向动量份额.

    e2 小 → 能量集中于少数粒子 (准直)
    e2 大 → 能量分散 (多辐射)
    """
    n = len(constituents)
    if n < 2:
        return 0.0

    sum_pT = sum(c.pT for c in constituents)
    if sum_pT < 1e-15:
        return 0.0

    e2 = 0.0
    for i in range(n):
        zi = constituents[i].pT / sum_pT
        for j in range(i + 1, n):
            zj = constituents[j].pT / sum_pT
            dr = constituents[i].delta_R(constituents[j])
            e2 += zi * zj * (dr ** alpha)

    return e2


def energy_correlation_3(constituents: List[FourVector],
                         alpha: float = 1.0) -> float:
    """三体能量关联函数 e3(α).

    e3(α) = Σ_{i<j<k} z_i z_j z_k (ΔR_{ij} · ΔR_{jk} · ΔR_{ik})^α

    时间复杂度 O(n³), 对大规模 constituents 使用随机采样近似.
    """
    n = len(constituents)
    if n < 3:
        return 0.0

    sum_pT = sum(c.pT for c in constituents)
    if sum_pT < 1e-15:
        return 0.0

    # 对 n > 50 使用随机采样 (减少计算量)
    max_triples = 10000
    total_triples = n * (n - 1) * (n - 2) // 6

    if total_triples > max_triples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=(max_triples, 3), replace=False)
        e3 = 0.0
        for idx in indices:
            i, j, k = idx
            zi = constituents[i].pT / sum_pT
            zj = constituents[j].pT / sum_pT
            zk = constituents[k].pT / sum_pT
            dr_ij = constituents[i].delta_R(constituents[j])
            dr_jk = constituents[j].delta_R(constituents[k])
            dr_ik = constituents[i].delta_R(constituents[k])
            prod_dr = dr_ij * dr_jk * dr_ik
            e3 += zi * zj * zk * (prod_dr ** alpha)
        # 乘以采样比例修正
        e3 *= total_triples / max_triples
        return e3
    else:
        from itertools import combinations
        e3 = 0.0
        for i, j, k in combinations(range(n), 3):
            zi = constituents[i].pT / sum_pT
            zj = constituents[j].pT / sum_pT
            zk = constituents[k].pT / sum_pT
            dr_ij = constituents[i].delta_R(constituents[j])
            dr_jk = constituents[j].delta_R(constituents[k])
            dr_ik = constituents[i].delta_R(constituents[k])
            prod_dr = dr_ij * dr_jk * dr_ik
            e3 += zi * zj * zk * (prod_dr ** alpha)
        return e3


def D2_ratio(constituents: List[FourVector],
             alpha: float = 1.0) -> float:
    """D₂ 观测量: D₂ = e3 · e1³ / e2³.

    D₂ 是一个红外安全 (IRC-safe) 的观测量, 用于:
    - 区分 quark 喷注 (低 D₂) 和 gluon 喷注 (高 D₂)
    - 识别 W/Z/H → qq' 的 boosted 喷注

    理论预期:
      quark jet: D₂ ~ O(α_s)
      gluon jet: D₂ ~ O(1)
      2-prong: D₂ 有明确的峰值
    """
    n = len(constituents)
    if n < 3:
        return 0.0

    # e1 = Σ z_i² = Σ (p_{Ti}/p_{T,jet})²
    sum_pT = sum(c.pT for c in constituents)
    if sum_pT < 1e-15:
        return 0.0
    e1 = sum((c.pT / sum_pT) ** 2 for c in constituents)

    e2 = energy_correlation_2(constituents, alpha)
    e3 = energy_correlation_3(constituents, alpha)

    if e2 < 1e-30:
        return 0.0

    return e3 * (e1 ** 3) / (e2 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# 能量密度径向分布 ρ(r) (源自 1233 交叉相关分析)
# ─────────────────────────────────────────────────────────────────────────────

def energy_density_profile(constituents: List[FourVector],
                           jet_axis: FourVector,
                           R: float = 0.4,
                           n_bins: int = 20) -> tuple:
    """计算喷注内部能量密度径向分布 ρ(r).

    ρ(r) = (1 / (2π r Δr)) Σ_{i: r < ΔR_i < r+Δr} p_{Ti} / p_{T,jet}

    这是 jet substructure 分析中最基本的分布函数.
    在微扰 QCD 中:
      ρ(r) ~ α_s / π · P(z) / r  (小 r 极限, 单胶子辐射)

    Returns:
        (r_centers, rho_values)
    """
    if len(constituents) == 0:
        return (np.zeros(n_bins), np.zeros(n_bins))

    sum_pT = sum(c.pT for c in constituents)
    if sum_pT < 1e-15:
        return (np.linspace(0, R, n_bins + 1)[:-1] + R / (2 * n_bins),
                np.zeros(n_bins))

    dr = R / n_bins
    r_edges = np.linspace(0, R, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    rho = np.zeros(n_bins)

    for c in constituents:
        r = c.delta_R(jet_axis)
        if r < R:
            bin_idx = int(r / dr)
            bin_idx = min(bin_idx, n_bins - 1)
            # 归一化: 每单位面积的能量份额
            area = 2.0 * math.pi * r_centers[bin_idx] * dr
            if area > 1e-15:
                rho[bin_idx] += (c.pT / sum_pT) / area

    return (r_centers, rho)


# ─────────────────────────────────────────────────────────────────────────────
# 互相关函数 (源自 1233 DAS 交叉相关)
# ─────────────────────────────────────────────────────────────────────────────

def cross_correlation_2d(constituents: List[FourVector],
                         jet_axis: FourVector,
                         R: float = 0.4,
                         n_bins_eta: int = 10,
                         n_bins_phi: int = 10) -> np.ndarray:
    """计算喷注内部的二维互相关函数 C(Δη, Δφ).

    C(Δη, Δφ) = (1/N_pairs) Σ_{i≠j} ε_i · ε_j · δ(Δη-Δη_{ij}) · δ(Δφ-Δφ_{ij})

    其中 ε_i = p_{Ti} / p_{T,jet} 为能量份额.

    互相关函数揭示喷注内部的角关联结构:
    - 2-prong 喷注: 在 ΔR ~ R/2 处有峰值
    - QCD 喷注: 在 ΔR ~ 0 处有强烈的峰 (collinear singularity)

    源自 1233 中用于 DAS 信号分析的 XCorr 方法.
    """
    if len(constituents) < 2:
        return np.zeros((n_bins_eta, n_bins_phi))

    sum_pT = sum(c.pT for c in constituents)
    if sum_pT < 1e-15:
        return np.zeros((n_bins_eta, n_bins_phi))

    C = np.zeros((n_bins_eta, n_bins_phi))
    deta_range = 2 * R
    dphi_range = 2 * R
    d_deta = deta_range / n_bins_eta
    d_dphi = dphi_range / n_bins_phi

    n = len(constituents)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            deta = constituents[i].eta - constituents[j].eta
            dphi = constituents[i].phi - constituents[j].phi
            # 折叠 Δφ 到 [-π, π]
            while dphi > math.pi:
                dphi -= 2 * math.pi
            while dphi < -math.pi:
                dphi += 2 * math.pi

            # 映射到 bin
            bin_eta = int((deta + R) / d_deta)
            bin_phi = int((dphi + R) / d_dphi)

            if 0 <= bin_eta < n_bins_eta and 0 <= bin_phi < n_bins_phi:
                ei = constituents[i].pT / sum_pT
                ej = constituents[j].pT / sum_pT
                C[bin_eta, bin_phi] += ei * ej

    # 归一化
    total = np.sum(C)
    if total > 1e-15:
        C /= total

    return C


# ─────────────────────────────────────────────────────────────────────────────
# Merger tree 分析 (源自 1131 rockstar merger analysis)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mass_drop(tree_node) -> dict:
    """计算 mass drop 变量 (用于 top-tagging).

    Mass Drop: μ = m_{子喷注} / m_{母喷注}
    条件: μ < μ_cut (典型 μ_cut = 0.67)

    同时检查对称性: z = min(pT1, pT2)/(pT1+pT2) > z_cut (典型 0.09)

    Returns:
        dict: {'mu': 质量比, 'z': 对称性, 'passed': 是否通过条件}
    """
    if tree_node.is_leaf:
        return {'mu': 0.0, 'z': 0.5, 'passed': False}

    m_parent = tree_node.mass
    if m_parent < 1e-10:
        return {'mu': 0.0, 'z': 0.5, 'passed': False}

    # 递归找到最大的子喷注质量
    m_left = tree_node.left.mass if tree_node.left else 0.0
    m_right = tree_node.right.mass if tree_node.right else 0.0
    m_sub = max(m_left, m_right)

    mu = m_sub / m_parent if m_parent > 1e-10 else 0.0
    z = tree_node.z

    # Mass Drop + Symmetry 条件
    mu_cut = 0.67
    z_cut = 0.09
    passed = (mu < mu_cut) and (z > z_cut)

    return {'mu': mu, 'z': z, 'passed': passed}


def pruning_mask(constituents: List[FourVector],
                 jet: FourVector,
                 z_cut: float = 0.1,
                 R_cut: float = 0.2,
                 R: float = 0.4) -> List[bool]:
    """Soft Drop / Pruning 修剪条件.

    对聚类树的每次分裂检查:
      z > z_cut · (2/R)^(β) · ΔR_{12}^β

    对 β=0 (pruning): z > z_cut
    对 β=1 (Soft Drop): z > z_cut · (ΔR/R)

    Returns:
        布尔掩码: True 表示该 constituent 通过修剪
    """
    n = len(constituents)
    mask = [True] * n
    sum_pT = sum(c.pT for c in constituents)

    for i, c in enumerate(constituents):
        # 简单修剪: 移除低 pT 且远离喷注轴的 constituent
        z = c.pT / sum_pT if sum_pT > 1e-15 else 0.0
        dr = c.delta_R(jet)
        if dr > R:
            mask[i] = False
            continue
        # Soft Drop 条件
        threshold = z_cut * (dr / R)
        if z < threshold and dr > R_cut:
            mask[i] = False

    return mask
