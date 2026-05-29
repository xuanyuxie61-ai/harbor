"""
topological_invariant.py
拓扑不变量计算：Weyl荷、Chern数与Z2指标

凝聚态物理核心公式：

Weyl节点是Berry曲率的磁单极子：
    \nabla_k · Omega(k) = 2*pi * Q * delta^{(3)}(k - k_Weyl)

其中Q = ±1是Weyl荷（chirality）。

Chern数（对二维闭曲面S）：
    C = (1/2*pi) \oint_S Omega · dS

对于Weyl半金属，在固定kz的二维截面上：
    C(kz) = sum_{k_Weyl} Q_i * theta(kz - kz_i)

其中theta是Heaviside阶跃函数。当kz经过Weyl节点时，Chern数发生跳变。

因此Weyl节点可以看作Chern数的"源"和"汇"：
    dC/dkz = sum_i Q_i * delta(kz - kz_i)

Z2拓扑指标（时间反演不变系统）：
    nu = (1/2*pi) [oint_{C} A · dk - \int_{S} Omega · dS]  mod 2

基于种子项目491_grid_display中的网格处理思想，
在k空间网格上计算拓扑量。
"""

import numpy as np
from typing import Tuple, List
from berry_curvature import berry_curvature_numeric, chern_number_2d_slice, weyl_charge_surface_integral


def compute_chern_numbers_vs_kz(ham, kz_values: np.ndarray,
                                 kx_range: Tuple[float, float],
                                 ky_range: Tuple[float, float],
                                 grid_size: int = 30,
                                 band_index: int = 0) -> np.ndarray:
    """
    计算一系列kz截面上的Chern数
    
    C(kz) = (1/2*pi) \int_{T^2} Omega_{xy}(k) dkx dky
    
    Parameters
    ----------
    ham : WeylHamiltonian
    kz_values : np.ndarray
        一系列kz值
    kx_range, ky_range : tuple
    grid_size : int
    band_index : int
    
    Returns
    -------
    chern_numbers : np.ndarray
    """
    chern_numbers = np.zeros(len(kz_values))
    
    for i, kz in enumerate(kz_values):
        c = chern_number_2d_slice(ham, kx_range, ky_range, kz, grid_size, band_index)
        # 取最接近的整数（数值误差修正）
        chern_numbers[i] = round(c)
    
    return chern_numbers


def locate_weyl_nodes_from_chern_jump(kz_values: np.ndarray,
                                       chern_numbers: np.ndarray) -> np.ndarray:
    """
    从Chern数的跳变定位Weyl节点
    
    Weyl节点位置对应Chern数发生整数跳变的位置。
    
    Parameters
    ----------
    kz_values : np.ndarray
    chern_numbers : np.ndarray
    
    Returns
    -------
    node_positions : np.ndarray
        Weyl节点的kz位置
    node_charges : np.ndarray
        对应的Weyl荷
    """
    jumps = np.diff(chern_numbers)
    
    node_positions = []
    node_charges = []
    
    for i in range(len(jumps)):
        if abs(jumps[i]) > 0.5:
            # Chern数跳变，存在Weyl节点
            pos = 0.5 * (kz_values[i] + kz_values[i + 1])
            charge = int(round(jumps[i]))
            node_positions.append(pos)
            node_charges.append(charge)
    
    return np.array(node_positions), np.array(node_charges)


def compute_weyl_charges_spherical(ham, weyl_nodes: np.ndarray,
                                    radius: float = 0.3,
                                    n_theta: int = 16,
                                    n_phi: int = 16,
                                    band_index: int = 0) -> np.ndarray:
    """
    对每个Weyl节点计算其Weyl荷
    
    Q_i = (1/2*pi) \oint_{S_i} Omega · dS
    
    Parameters
    ----------
    ham : WeylHamiltonian
    weyl_nodes : np.ndarray, shape (N, 3)
    radius : float
        包围球半径
    n_theta, n_phi : int
    band_index : int
    
    Returns
    -------
    charges : np.ndarray, shape (N,)
    """
    n_nodes = weyl_nodes.shape[0] if weyl_nodes.ndim > 1 else 1
    charges = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        node = weyl_nodes[i] if weyl_nodes.ndim > 1 else weyl_nodes
        q = weyl_charge_surface_integral(ham, node, radius, n_theta, n_phi, band_index)
        # 取最接近的整数
        charges[i] = round(q)
    
    return charges


def nielsen_ninomiya_theorem_check(charges: np.ndarray) -> bool:
    """
    验证Nielsen-Ninomiya定理
    
    定理：在周期性晶格中，所有Weyl节点的荷之和为零：
        sum_i Q_i = 0
    
    这意味着Weyl节点必须成对出现（+-配对）。
    
    Parameters
    ----------
    charges : np.ndarray
    
    Returns
    -------
    satisfied : bool
    """
    total = np.sum(charges)
    return abs(total) < 0.5


def berry_phase_wilson_loop(ham, kx_line: np.ndarray, ky_fixed: float,
                            kz_fixed: float, band_index: int = 0) -> float:
    """
    使用Wilson loop计算一维Berry相位
    
    Wilson loop定义：
        W = prod_{i=1}^{N} <u_i | u_{i+1}>
        gamma = -Im ln W
    
    Parameters
    ----------
    ham : WeylHamiltonian
    kx_line : np.ndarray
        kx取值
    ky_fixed, kz_fixed : float
    band_index : int
    
    Returns
    -------
    phase : float
    """
    n_points = len(kx_line)
    if n_points < 2:
        return 0.0
    
    # 收集本征矢
    vectors = []
    for kx in kx_line:
        k = np.array([kx, ky_fixed, kz_fixed])
        _, eigvecs = ham.eigenproblem(k)
        vec = eigvecs[:, band_index].copy()
        # 规范固定
        if abs(vec[0]) > 1e-14:
            vec *= np.exp(-1.0j * np.angle(vec[0]))
        vectors.append(vec)
    
    # Wilson loop乘积
    prod = 1.0 + 0.0j
    for i in range(n_points - 1):
        overlap = np.vdot(vectors[i], vectors[i + 1])
        prod *= overlap
    
    # 闭合回路
    overlap = np.vdot(vectors[-1], vectors[0])
    prod *= overlap
    
    phase = -np.angle(prod)
    return phase


def compute_z2_index(ham, kx_values: np.ndarray, ky_values: np.ndarray,
                     kz_fixed: float, band_index: int = 0) -> int:
    """
    计算时间反演不变系统中的Z2拓扑指标
    
    简化方法：计算Berry相位的Wilson loop
    若gamma = pi (mod 2*pi)，则Z2 = 1；否则Z2 = 0。
    
    注意：本函数仅演示框架，严格的Z2计算需要时间反演对称性约束。
    
    Parameters
    ----------
    ham : WeylHamiltonian
    kx_values, ky_values : np.ndarray
    kz_fixed : float
    band_index : int
    
    Returns
    -------
    z2 : int
        0 或 1
    """
    # 对ky的每个值计算Wilson loop的Berry相位
    phases = []
    for ky in ky_values:
        phase = berry_phase_wilson_loop(ham, kx_values, ky, kz_fixed, band_index)
        phases.append(phase)
    
    phases = np.array(phases)
    
    # Z2 = (1/pi) * sum_i (phase_i - phase_{i-1}) mod 2
    # 简化为判断是否存在pi跳变
    z2 = 0
    for p in phases:
        if abs(abs(p) - np.pi) < 0.3 * np.pi:
            z2 = 1
            break
    
    return z2
