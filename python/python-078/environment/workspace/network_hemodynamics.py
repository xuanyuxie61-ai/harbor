"""
network_hemodynamics.py
动脉网络血流分配与PageRank类比分析

融合来源:
- 1405_web_matrix: 网页链接转移矩阵、幂迭代法、PageRank特征向量计算
- 345_exm (pagerank): 马尔可夫链稳态分布

科学背景:
主动脉弓及其分支构成一个复杂的有向网络：
    升主动脉 → 主动脉弓 → 降主动脉
                ↓
        头臂干 → 右颈总 / 右锁骨下
        左颈总动脉
        左锁骨下动脉

血液在网络中的分配遵循质量守恒（Kirchhoff定律）和压力梯度驱动。
类比PageRank：将流量分配视为马尔可夫链的稳态分布。

对于血管网络，我们构造转移矩阵T：
    T_{ij} = 从节点i流向节点j的流量比例

稳态分布π满足: π = T π
这正好是有向图的PageRank方程（无damping因子时）。

Murray定律在最优网络中:
    r_parent³ = Σ r_children³
    流量分配比例与半径立方成正比
"""

import numpy as np
from typing import Tuple, List, Dict


# ======================================================================
# 来自 1405_web_matrix 的转移矩阵与幂迭代
# ======================================================================

def incidence_to_transition(adjacency: np.ndarray) -> np.ndarray:
    """
    将邻接矩阵转换为转移概率矩阵。

    步骤:
    1. 对邻接矩阵A的每一行归一化（出度倒数）
    2. dangling node（无出链节点）处理：自环概率为1
    3. 转置得到列随机矩阵：T = (A_row_normalized)^T

    参数:
        adjacency: (N, N) 邻接矩阵，A[i,j]=1 表示节点i有链接到j

    返回:
        T: (N, N) 列随机转移矩阵
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    row_sums = A.sum(axis=1)

    T_row = np.zeros_like(A)
    for i in range(n):
        if row_sums[i] > 0:
            T_row[i, :] = A[i, :] / row_sums[i]
        else:
            # dangling node: 自环
            T_row[i, i] = 1.0

    T = T_row.T
    return T


def power_rank(T: np.ndarray, max_iter: int = 200,
               tol: float = 1e-10) -> np.ndarray:
    """
    幂迭代法计算转移矩阵的稳态分布（PageRank向量）。

    迭代格式:
        x_{k+1} = T @ x_k
        x_0 = (1/N, ..., 1/N)^T

    收敛判据:
        ||x_{k+1} - x_k||_∞ < tol

    参数:
        T: (N, N) 列随机转移矩阵
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        x: (N,) 稳态分布向量（已归一化）
    """
    n = T.shape[0]
    x = np.ones(n) / n

    for it in range(max_iter):
        x_new = T @ x
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            break

    # 确保归一化
    s = x.sum()
    if s > 1e-15:
        x = x / s
    return x


def page_rank_with_damping(adjacency: np.ndarray,
                           damping: float = 0.85,
                           max_iter: int = 200,
                           tol: float = 1e-10) -> np.ndarray:
    """
    带阻尼因子的PageRank（标准Google PageRank）。

    方程:
        π = d T π + (1-d)/N · 1

    参数:
        adjacency: 邻接矩阵
        damping: 阻尼因子（通常0.85）
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        π: PageRank向量
    """
    T = incidence_to_transition(adjacency)
    n = T.shape[0]
    x = np.ones(n) / n

    for it in range(max_iter):
        x_new = damping * (T @ x) + (1.0 - damping) / n
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            break

    s = x.sum()
    if s > 1e-15:
        x = x / s
    return x


# ======================================================================
# 动脉网络建模
# ======================================================================

class ArterialNetwork:
    """
    简化的主动脉弓动脉网络模型。

    节点编号:
        0: 升主动脉 (Ascending Aorta)
        1: 主动脉弓 (Aortic Arch)
        2: 降主动脉 (Descending Aorta)
        3: 头臂干 (Brachiocephalic Trunk)
        4: 右颈总 (Right Common Carotid)
        5: 右锁骨下 (Right Subclavian)
        6: 左颈总 (Left Common Carotid)
        7: 左锁骨下 (Left Subclavian)

    边: 有向边表示血流方向
    """
    def __init__(self):
        self.node_names = [
            "Ascending_Aorta", "Aortic_Arch", "Descending_Aorta",
            "Brachiocephalic", "R_Common_Carotid", "R_Subclavian",
            "L_Common_Carotid", "L_Subclavian"
        ]
        self.n_nodes = len(self.node_names)

        # 构建邻接矩阵（简化拓扑）
        self.adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        edges = [
            (0, 1),   # 升主动脉 → 主动脉弓
            (1, 2),   # 主动脉弓 → 降主动脉
            (1, 3),   # 主动脉弓 → 头臂干
            (1, 6),   # 主动脉弓 → 左颈总
            (1, 7),   # 主动脉弓 → 左锁骨下
            (3, 4),   # 头臂干 → 右颈总
            (3, 5),   # 头臂干 → 右锁骨下
        ]
        for i, j in edges:
            self.adjacency[i, j] = 1

        # 各段血管的半径 [m]（近似值）
        self.radii = np.array([
            0.014, 0.012, 0.010,
            0.006, 0.004, 0.004,
            0.004, 0.004
        ])

        # 各段长度 [m]
        self.lengths = np.array([
            0.05, 0.08, 0.30,
            0.04, 0.12, 0.20,
            0.12, 0.20
        ])

    def compute_flow_distribution(self, total_flow: float = 5.0e-5) -> Dict[str, float]:
        """
        基于Murray定律和PageRank类比计算网络中的流量分配。

        总流量默认: 5.0e-5 m³/s ≈ 3000 mL/min（心输出量）

        流量分配比例由稳态分布π决定：
            Q_i = total_flow · π_i / π_root

        返回:
            节点到流量的字典 [m³/s]
        """
        pi = page_rank_with_damping(self.adjacency, damping=0.85)

        # 根节点（升主动脉）的流量等于总流量
        root_idx = 0
        scale = total_flow / (pi[root_idx] + 1e-15)

        flows = {}
        for i, name in enumerate(self.node_names):
            flows[name] = float(pi[i] * scale)

        return flows

    def compute_wss_from_flow(self, flow_dict: Dict[str, float],
                              blood_viscosity_pa_s: float = 0.0035) -> Dict[str, float]:
        """
        由流量估算壁面剪切应力（WSS）。

        对于圆管层流（Poiseuille流）:
            τ_w = 4 μ Q / (π R³)

        参数:
            flow_dict: 流量字典 [m³/s]
            blood_viscosity_pa_s: 血液动力粘度 [Pa·s]

        返回:
            WSS字典 [Pa]
        """
        wss = {}
        for i, name in enumerate(self.node_names):
            Q = flow_dict.get(name, 0.0)
            R = self.radii[i]
            if R > 1e-6:
                tau_w = 4.0 * blood_viscosity_pa_s * Q / (np.pi * R ** 3)
            else:
                tau_w = 0.0
            wss[name] = float(tau_w)
        return wss

    def network_resistance(self, blood_viscosity_pa_s: float = 0.0035) -> Dict[str, float]:
        """
        计算各段的Poiseuille流阻。

        R = 8 μ L / (π R⁴)
        """
        resistances = {}
        for i, name in enumerate(self.node_names):
            mu = blood_viscosity_pa_s
            L = self.lengths[i]
            R = self.radii[i]
            if R > 1e-6:
                resistances[name] = float(8.0 * mu * L / (np.pi * R ** 4))
            else:
                resistances[name] = float('inf')
        return resistances

    def womersley_numbers(self, heart_rate_bpm: float = 72.0,
                          kinematic_viscosity: float = 3.3e-6) -> Dict[str, float]:
        """
        计算各段的Womersley数。

        α = R * sqrt(ω / ν),  ω = 2π f, f = HR/60
        """
        f = heart_rate_bpm / 60.0
        omega = 2.0 * np.pi * f
        alpha_dict = {}
        for i, name in enumerate(self.node_names):
            R = self.radii[i]
            alpha = R * np.sqrt(omega / kinematic_viscosity)
            alpha_dict[name] = float(alpha)
        return alpha_dict


def bifurcation_flow_split(r_parent: float, r_child1: float,
                           r_child2: float) -> Tuple[float, float]:
    """
    基于Murray定律计算分叉处的流量分配比例。

    Murray定律:
        Q_parent = Q_child1 + Q_child2
        r_p³ = r_1³ + r_2³

    假设压力梯度相同，流量与半径⁴成正比（Poiseuille定律），
    但Murray优化给出 r ∝ Q^{1/3}，因此:
        Q_1 / Q_total = r_1³ / (r_1³ + r_2³)
    """
    r1_cubed = r_child1 ** 3
    r2_cubed = r_child2 ** 3
    total = r1_cubed + r2_cubed + 1e-15
    q1_ratio = r1_cubed / total
    q2_ratio = r2_cubed / total
    return q1_ratio, q2_ratio
