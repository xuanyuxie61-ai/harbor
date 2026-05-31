"""
instability_analysis.py
=======================
ICF 内爆不稳定性分析模块。

融合原项目 655_leaf_chaos（迭代函数系统 IFS 生成分形表面粗糙度）与
1205_test_digraph_arc（有向图弧结构与连通性分析）的核心思想，
对靶丸表面的 Rayleigh-Taylor (RT) 不稳定性及 Richtmyer-Meshkov (RM)
不稳定性进行建模、增长分析与能量流网络追踪。

物理模型：
1. Rayleigh-Taylor 增长（烧蚀前缘）:
     gamma = sqrt( A_t * g * k ) - beta * k * v_abl
   其中 A_t = (rho_high - rho_low) / (rho_high + rho_low) 为 Atwood 数,
   k = l / R 为波数, v_abl 为烧蚀速度, beta 为烧蚀致稳系数。

2. Richtmyer-Meshkov 增长（冲击界面）:
     eta(t) = eta_0 * (1 + Delta_v * A_t * k * t)

3. 表面粗糙度模型（IFS）:
   基于 leaf_chaos 的迭代函数系统，生成靶丸表面随机扰动模式。

4. 能量流网络（digraph_arc）:
   追踪激光能量在不同模式间的不稳定性耦合转移路径。
"""

import numpy as np
from typing import Tuple, List, Dict
from icf_parameters import NP, TP
from utils import clamp


# ========================================================================
# IFS 表面粗糙度模型（基于原项目 655_leaf_chaos）
# ========================================================================

def generate_surface_perturbation_ifs(n_points: int = 1000,
                                      amplitude: float = NP.PERTURBATION_AMPLITUDE,
                                      mode: int = NP.PERTURBATION_MODE) -> np.ndarray:
    """
    使用迭代函数系统（IFS）生成靶丸表面粗糙度扰动。

    通过随机选择线性仿射变换迭代生成扰动谱系，
    再叠加到指定球谐模式上。
    """
    # 定义 IFS 变换（类似 leaf_chaos 的随机矩阵选择）
    transforms = [
        (np.array([[0.8, 0.0], [0.0, 0.8]]), np.array([0.1, 0.04])),
        (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.25, 0.4])),
        (np.array([[0.355, -0.355], [0.355, 0.355]]), np.array([0.266, 0.078])),
        (np.array([[0.355, 0.355], [-0.355, 0.355]]), np.array([0.378, 0.434])),
    ]

    x = np.array([0.5, 0.5])
    points = []

    # 瞬态丢弃
    for _ in range(100):
        idx = np.random.randint(0, len(transforms))
        A, b = transforms[idx]
        x = A @ x + b

    for _ in range(n_points):
        idx = np.random.randint(0, len(transforms))
        A, b = transforms[idx]
        x = A @ x + b
        points.append(x.copy())

    points = np.array(points)

    # 映射到球谐模式振幅
    theta = np.linspace(0.0, 2.0 * np.pi, mode + 1)[:-1]
    perturbation = np.zeros(mode)
    for i in range(mode):
        # 使用 IFS 点的统计特性确定模式振幅
        sector_mask = (points[:, 0] >= i / mode) & (points[:, 0] < (i + 1) / mode)
        if np.any(sector_mask):
            perturbation[i] = amplitude * np.std(points[sector_mask, 1])
        else:
            perturbation[i] = amplitude * 0.1

    return perturbation


# ========================================================================
# Rayleigh-Taylor 不稳定性增长
# ========================================================================

def atwood_number(rho_high: float, rho_low: float) -> float:
    """
    Atwood 数:
        A_t = (rho_high - rho_low) / (rho_high + rho_low)
    """
    denom = rho_high + rho_low
    if denom < 1.0e-30:
        return 0.0
    return (rho_high - rho_low) / denom


def rayleigh_taylor_growth_rate(rho_ablation: float, rho_corona: float,
                                 acceleration: float, mode_l: int,
                                 radius: float, v_ablation: float,
                                 beta_stabilization: float = 3.0) -> float:
    """
    烧蚀致稳的 Rayleigh-Taylor 线性增长率:

        gamma = sqrt( A_t * k * g ) - beta * k * v_abl

    其中 k = l / R 为波数。

    参数
    ----
    rho_ablation, rho_corona : float
        烧蚀层与冕区密度 [kg/m^3]
    acceleration : float
        界面加速度 [m/s^2]
    mode_l : int
        球谐模式数
    radius : float
        界面半径 [m]
    v_ablation : float
        烧蚀速度 [m/s]
    beta_stabilization : float
        烧蚀致稳系数

    返回
    ----
    gamma : float
        增长率 [s^-1]，若为负则返回 0（稳定）
    """
    A_t = atwood_number(rho_ablation, rho_corona)
    if A_t <= 0.0 or radius <= 1.0e-15 or acceleration <= 0.0:
        return 0.0

    k = mode_l / radius
    g = acceleration

    term_classical = np.sqrt(max(A_t * k * g, 0.0))
    term_ablative = beta_stabilization * k * v_ablation

    gamma = term_classical - term_ablative
    return max(gamma, 0.0)


def richtmyer_meshkov_amplitude(eta_0: float, delta_v: float,
                                 A_t: float, mode_l: int, radius: float,
                                 t: float) -> float:
    """
    Richtmyer-Meshkov 振幅增长:
        eta(t) = eta_0 * (1 + k * A_t * Delta_v * t)
    冲击后线性增长模型（Impulsive model）。
    """
    if radius <= 1.0e-15 or t < 0.0:
        return eta_0
    k = mode_l / radius
    return eta_0 * (1.0 + k * A_t * delta_v * t)


def compute_mode_growth_spectrum(rho_profile: np.ndarray,
                                 r_cells: np.ndarray,
                                 u_nodes: np.ndarray,
                                 mode_range: range = range(1, 25)) -> Dict[int, float]:
    """
    计算各球谐模式的 RT 增长率谱。

    参数
    ----
    rho_profile : np.ndarray
        单元密度
    r_cells : np.ndarray
        单元中心半径
    u_nodes : np.ndarray
        节点速度
    mode_range : range
        模式数范围

    返回
    ----
    growth_rates : dict
        {mode_l: gamma}
    """
    growth_rates = {}
    n_cells = len(r_cells)
    if n_cells < 2:
        return growth_rates

    # 找到密度梯度最大处作为 RT 界面
    interface_idx = 0
    max_grad = 0.0
    for i in range(1, n_cells - 1):
        dr = r_cells[i + 1] - r_cells[i - 1]
        if dr < 1.0e-15:
            continue
        grad = abs(rho_profile[i + 1] - rho_profile[i - 1]) / dr
        if grad > max_grad:
            max_grad = grad
            interface_idx = i

    if interface_idx == 0:
        return growth_rates

    rho_high = rho_profile[interface_idx]
    rho_low = rho_profile[min(interface_idx + 1, n_cells - 1)]
    R_int = r_cells[interface_idx]

    # 加速度近似: a = du/dt ≈ u^2 / R (向心近似)
    u_int = 0.5 * (u_nodes[interface_idx] + u_nodes[interface_idx + 1])
    accel = u_int**2 / max(R_int, 1.0e-15)

    # 烧蚀速度（密度下降速率近似）
    v_abl = abs(u_int) * 0.1

    for l in mode_range:
        gamma = rayleigh_taylor_growth_rate(
            rho_high, rho_low, accel, l, R_int, v_abl
        )
        growth_rates[l] = gamma

    return growth_rates


# ========================================================================
# 有向图能量流网络（基于原项目 1205_test_digraph_arc）
# ========================================================================

def build_energy_flow_digraph() -> Tuple[np.ndarray, List[str]]:
    """
    构建 ICF 内爆能量转移有向图。

    节点：
      0: 激光能量 (Laser)
      1: 电子热能 (E_thermal)
      2: 离子热能 (I_thermal)
      3: 辐射能 (Radiation)
      4: 流体动能 (Kinetic)
      5: 聚变能 (Fusion)
      6: 中子逃逸 (Neutron_loss)
      7: X射线逃逸 (Xray_loss)

    弧 (i->j) 表示能量从 i 流向 j。
    """
    node_names = [
        "Laser", "E_thermal", "I_thermal", "Radiation",
        "Kinetic", "Fusion", "Neutron_loss", "Xray_loss"
    ]

    # 弧列表: (source, target)
    arcs = [
        (0, 1),   # Laser -> E_thermal (逆轫致吸收)
        (1, 2),   # E_thermal -> I_thermal (电子离子 equilibration)
        (1, 3),   # E_thermal -> Radiation (轫致辐射)
        (1, 4),   # E_thermal -> Kinetic (烧蚀压驱动)
        (2, 5),   # I_thermal -> Fusion (聚变点火)
        (3, 7),   # Radiation -> Xray_loss
        (5, 2),   # Fusion -> I_thermal (alpha 加热)
        (5, 6),   # Fusion -> Neutron_loss
        (4, 2),   # Kinetic -> I_thermal (激波加热)
    ]

    n_nodes = len(node_names)
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in arcs:
        adjacency_matrix[i, j] = 1

    return adjacency_matrix, node_names


def energy_flow_pagerank(adjacency: np.ndarray, damping: float = 0.85,
                         tol: float = 1.0e-8, max_iter: int = 100) -> np.ndarray:
    """
    使用 PageRank 思想计算能量流网络中各节点的重要性排序。

    基于原项目 1205_test_digraph_arc（Moler 网络示例）的图结构思想。

    PR(i) = (1-d)/N + d * sum_{j->i} PR(j) / outdegree(j)
    """
    n = adjacency.shape[0]
    out_degrees = np.sum(adjacency, axis=1)
    # 将出度为 0 的节点设为全连接（随机跳转）
    transition = np.zeros((n, n))
    for j in range(n):
        if out_degrees[j] > 0:
            transition[:, j] = adjacency[j, :] / out_degrees[j]
        else:
            transition[:, j] = 1.0 / n

    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1.0 - damping) / n + damping * transition @ pr
        if np.linalg.norm(pr_new - pr, ord=1) < tol:
            break
        pr = pr_new

    return pr


def analyze_instability_feedthrough(mode_growth: Dict[int, float],
                                    perturbation_spectrum: np.ndarray) -> float:
    """
    综合表面粗糙度与 RT 增长率，评估总不稳定性馈通因子。

    总馈通因子 = sum_l (eta_l * exp(gamma_l * t))**2
    """
    total = 0.0
    for l, gamma in mode_growth.items():
        if l - 1 < len(perturbation_spectrum):
            eta_l = perturbation_spectrum[l - 1]
        else:
            eta_l = NP.PERTURBATION_AMPLITUDE / l
        total += (eta_l * gamma)**2
    return np.sqrt(total)
