"""
phase_field_damage.py
=====================
相场断裂损伤演化模块。
整合自：
  - 435_fitzhugh_nagumo_ode：快慢变量动力学思想用于相场变量演化

物理背景：
  在增材制造结构中，疲劳裂纹萌生与扩展是核心失效模式。
  相场断裂模型将不连续的裂纹面正则化为连续的损伤场 φ(x,t) ∈ [0,1]，
  其中 φ=0 表示完好材料，φ=1 表示完全断裂。

核心方程（受 FitzHugh-Nagumo 快慢动力学启发）：

  1. 相场演化方程（类似 FHN 快变量 v）：
       ṗ = -M · ∂Ψ/∂φ
     其中 M 为迁移率，Ψ 为总自由能泛函：
       Ψ(φ, ∇φ, ε) = ∫_Ω [g(φ)·Ψ_e(ε) + Ψ_c(φ) + (G_c/c_0)·(φ/l_0 + l_0·|∇φ|²)] dΩ

     g(φ) = (1-φ)² + k          （弹性退化函数，k 为残余刚度）
     Ψ_c(φ) = (G_c/c_0·l_0) · φ

  2. 采用双势阱正则化（FHN 双稳态思想）：
       F(φ) = φ²(1-φ)²          （标准双势阱）
     演化方程：
       ṗ = -M·[2(1-k)·φ·Ψ_e - G_c/c_0·(1/l_0 - 2·l_0·∇²φ) + ∂F/∂φ]

  3. 临界能量释放率 G_c 与相场特征长度 l_0 的关系：
       c_0 = 8/3  （二维）
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 1. 相场模型参数与势函数
# =============================================================================

def degradation_function(phi: np.ndarray, k_res: float = 1e-6) -> np.ndarray:
    """
    弹性退化函数：
        g(φ) = (1 - φ)² + k_res
    保证当 φ=1 时仍有极小残余刚度，避免刚度矩阵奇异。
    """
    return (1.0 - phi)**2 + k_res


def degradation_derivative(phi: np.ndarray) -> np.ndarray:
    """
    g'(φ) = -2(1 - φ)
    """
    return -2.0 * (1.0 - phi)


def double_well_potential(phi: np.ndarray) -> np.ndarray:
    """
    双势阱函数（FHN 思想）：
        F(φ) = φ² (1 - φ)²
    在 φ=0 和 φ=1 处取极小值，在 φ=0.5 处取极大值。
    """
    return phi**2 * (1.0 - phi)**2


def double_well_derivative(phi: np.ndarray) -> np.ndarray:
    """
    F'(φ) = 2φ(1-φ)(1-2φ)
    """
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


# =============================================================================
# 2. 弹性应变能驱动断裂
# =============================================================================

def compute_elastic_strain_energy(U: np.ndarray, K_dense: np.ndarray) -> float:
    """
    弹性应变能：
        Ψ_e = 1/2 · U^T · K · U
    """
    return 0.5 * np.dot(U, K_dense @ U)


def split_strain_energy(U: np.ndarray, node_xy: np.ndarray,
                        element_node: np.ndarray, E: float, nu: float,
                        plane_stress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    应变能分解：拉伸部分（驱动断裂）和压缩部分（不驱动断裂）。

    对每个单元，计算主应变：
        ε_{1,2} = (ε_xx + ε_yy)/2 ± √[((ε_xx - ε_yy)/2)² + (ε_xy/2)²]

    拉伸应变能密度（只取正特征值）：
        Ψ_e^+ = 1/2 · λ · <tr(ε)>_+² + μ · Σ <ε_i>_+²
    压缩应变能密度：
        Ψ_e^- = 1/2 · λ · <tr(ε)>_-² + μ · Σ <ε_i>_-²

    其中 <x>_+ = max(x,0), <x>_- = min(x,0)。
    """
    n_elements = element_node.shape[0]
    psi_pos = np.zeros(n_elements, dtype=np.float64)
    psi_neg = np.zeros(n_elements, dtype=np.float64)

    # TODO Hole 3: 实现应变能分解（拉伸部分驱动断裂，压缩部分不驱动）
    # 步骤：
    #  1. 根据 plane_stress 计算 Lamé 常数 λ, μ
    #  2. 遍历每个单元，计算应变 ε = B @ u_e
    #  3. 求主应变 ε1, ε2 = (ε_xx+ε_yy)/2 ± √[((ε_xx-ε_yy)/2)² + ε_xy²]
    #  4. 分解正/负部分：<x>_+ = max(x,0), <x>_- = min(x,0)
    #  5. 计算拉伸/压缩应变能密度：
    #     ψ^+ = 0.5·λ·<tr(ε)>_+² + μ·(<ε1>_+² + <ε2>_+²)
    #     ψ^- = 0.5·λ·<tr(ε)>_-² + μ·(<ε1>_-² + <ε2>_-²)
    raise NotImplementedError("Hole 3: split_strain_energy 未实现")
    return psi_pos, psi_neg


# =============================================================================
# 3. 相场演化求解
# =============================================================================

def solve_phase_field_evolution(element_node: np.ndarray, node_xy: np.ndarray,
                                 psi_e_pos: np.ndarray, G_c: float,
                                 l_0: float, n_iter: int = 50,
                                 mobility: float = 1.0) -> np.ndarray:
    """
    求解静态相场方程（历史场方法，类似 FHN 快变量稳态）。

    离散化的相场方程（单元中心）：
        ∂Ψ/∂φ_e = 2·φ_e·ψ_{e,hist} - G_c/l_0 + (G_c·l_0)·(L·φ)_e = 0

    其中 L 为离散 Laplacian，ψ_{e,hist} = max(ψ_e^+, 历史最大值)。

    采用不动点迭代（类似 FHN 慢变量 w 的准静态处理）：
        [diag(2·ψ_hist + G_c/l_0) + G_c·l_0·L] · φ = ψ_hist
    """
    n_elements = element_node.shape[0]
    phi = np.zeros(n_elements, dtype=np.float64)

    # 历史应变能（防止裂纹愈合）
    psi_hist = psi_e_pos.copy()

    # 构建单元邻接 Laplacian（基于共享节点）
    from fem_core import build_vtoe
    vtoe_ptr, vtoe = build_vtoe(element_node, node_xy.shape[0])

    # 构建单元邻接图
    adjacency = [set() for _ in range(n_elements)]
    for v in range(node_xy.shape[0]):
        cells = vtoe[vtoe_ptr[v]:vtoe_ptr[v+1]]
        for ci in cells:
            for cj in cells:
                if ci != cj:
                    adjacency[ci].add(cj)

    # 单元中心
    centers = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        centers[e] = np.mean(node_xy[element_node[e, :], :], axis=0)

    # 单元面积（近似）
    areas = np.zeros(n_elements, dtype=np.float64)
    for e in range(n_elements):
        x = node_xy[element_node[e, :], 0]
        y = node_xy[element_node[e, :], 1]
        areas[e] = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

    for _ in range(n_iter):
        phi_new = np.zeros_like(phi)
        for e in range(n_elements):
            # Laplacian 近似：Σ_{j∈adj(e)} (φ_j - φ_e) / |c_j - c_e|² · A_j / A_e
            lap = 0.0
            total_weight = 0.0
            for j in adjacency[e]:
                dist2 = np.sum((centers[j] - centers[e])**2)
                if dist2 > 1e-14:
                    weight = 1.0 / dist2
                    lap += weight * (phi[j] - phi[e])
                    total_weight += weight
            if total_weight > 1e-14:
                lap /= total_weight
            else:
                lap = 0.0

            # 相场更新（隐式欧拉近似）
            coeff = 2.0 * psi_hist[e] + G_c / l_0 + mobility * G_c * l_0 * total_weight
            rhs = 2.0 * psi_hist[e] + mobility * G_c * l_0 * lap
            if coeff > 1e-14:
                phi_new[e] = rhs / coeff
            else:
                phi_new[e] = 0.0

            # 边界截断
            phi_new[e] = max(0.0, min(1.0, phi_new[e]))

        # 收敛检查
        if np.max(np.abs(phi_new - phi)) < 1e-6:
            break
        phi = phi_new

    return phi


def compute_crack_driving_force(node_xy: np.ndarray, element_node: np.ndarray,
                                 U: np.ndarray, E: float, nu: float,
                                 G_c: float, l_0: float,
                                 plane_stress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算裂纹驱动力、相场分布和等效断裂韧性。

    Returns
    -------
    psi_pos : ndarray
        各单元拉伸应变能密度。
    phi : ndarray
        相场损伤变量。
    J_integral : float
        等效 J 积分（能量释放率）。
    """
    psi_pos, psi_neg = split_strain_energy(
        U, node_xy, element_node, E, nu, plane_stress)
    phi = solve_phase_field_evolution(element_node, node_xy, psi_pos, G_c, l_0)

    # 等效 J 积分估算：沿裂纹路径的 G_c 积分
    # 简化：J ≈ Σ_e G_c · (φ_e / l_0) · A_e
    areas = np.zeros(len(psi_pos), dtype=np.float64)
    for e in range(len(psi_pos)):
        x = node_xy[element_node[e, :], 0]
        y = node_xy[element_node[e, :], 1]
        areas[e] = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

    J_integral = np.sum(G_c * (phi / l_0) * areas)
    return psi_pos, phi, J_integral
