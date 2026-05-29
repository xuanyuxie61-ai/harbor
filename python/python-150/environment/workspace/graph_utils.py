"""
graph_utils.py
==============
图分析与高维几何工具

融合种子项目:
  - 850_partition_greedy      : 贪心图划分
  - 899_polyomino_parity      : Diophantine 约束与奇偶性分析
  - 556_hypercube_distance    : 高维距离统计
  - 420_fermat_factor         : Fermat 整数分解（用于图哈希）
  - 185_circles               : 参数化圆/球基函数

科学背景:
  1. 图划分：将大分子图划分为子图进行 mini-batch 消息传递。
  2. 奇偶性约束：用于分子对称性计数（如立体异构体数目）。
  3. 高维距离统计：分子描述符空间的距离分布分析。
  4. 整数分解：生成图的拓扑指纹。
  5. 球谐/圆基：原子轨道角度部分的参数化表示。
"""

import numpy as np
from typing import List, Tuple


# ------------------------------------------------------------------
# 1. 贪心图划分 (源自 partition_greedy)
# ------------------------------------------------------------------

def greedy_graph_partition(weights: np.ndarray, adjacency: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    对分子图进行二部贪心划分，使得割边权重最小化。
    算法：按节点度降序，将每个节点分配到当前总权重较小的子集。

    Parameters
    ----------
    weights : np.ndarray, shape (n,)
        节点权重（如原子重要性）。
    adjacency : np.ndarray, shape (n, n)
        邻接矩阵（稠密）。

    Returns
    -------
    partition : np.ndarray
        0/1 划分标签。
    sum0, sum1 : float
        两子集的总权重。
    """
    n = len(weights)
    partition = np.zeros(n, dtype=np.int32)
    # 按节点加权度排序
    degrees = adjacency.sum(axis=1) * weights
    order = np.argsort(-degrees)
    sum0, sum1 = 0.0, 0.0
    for idx in order:
        if sum0 <= sum1:
            partition[idx] = 0
            sum0 += weights[idx]
        else:
            partition[idx] = 1
            sum1 += weights[idx]
    return partition, sum0, sum1


# ------------------------------------------------------------------
# 2. Fermat 整数分解 (源自 fermat_factor)
# ------------------------------------------------------------------

def fermat_factor(n: int) -> Tuple[int, int]:
    """
    Fermat 因数分解: N = A² - B² = (A+B)(A-B)。
    用于生成分子图的拓扑哈希指纹。
    """
    if n < 2:
        return (1, n)
    a = int(np.floor(np.sqrt(n)))
    if a * a == n:
        return (a, a)
    while True:
        a += 1
        b2 = a * a - n
        if b2 < 0:
            continue
        b = int(np.sqrt(b2))
        if b * b == b2:
            return (a - b, a + b)
        if a > n:
            return (1, n)


def graph_hash_fingerprint(n_nodes: int, n_edges: int) -> int:
    """
    基于 Fermat 分解的图拓扑指纹:
        hash = |factor(n_nodes * n_edges + 1) - n_nodes|
    """
    val = n_nodes * n_edges + 1
    f1, f2 = fermat_factor(val)
    return abs(f1 - n_nodes)


# ------------------------------------------------------------------
# 3. Diophantine 与奇偶性约束 (源自 polyomino_parity)
# ------------------------------------------------------------------

def diophantine_nonnegative_solutions(target: int, n_vars: int) -> List[np.ndarray]:
    """
    枚举 target = x_1 + ... + x_n_vars 的所有非负整数解。
    用于分子对称群的阶数分解（如 n 次旋转对称轴的阶数分解）。
    """
    solutions = []
    def backtrack(remain, start, current):
        if start == n_vars - 1:
            current.append(remain)
            solutions.append(np.array(current, dtype=np.int32))
            current.pop()
            return
        for v in range(remain + 1):
            current.append(v)
            backtrack(remain - v, start + 1, current)
            current.pop()
    backtrack(target, 0, [])
    return solutions


def parity_violation_check(atom_counts: np.ndarray, required_parity: int = 0) -> bool:
    """
    检查分子组成是否满足奇偶性约束。
    例如：某些分子性质要求特定原子数的奇偶性（如 Handedness）。
    返回 True 若存在奇偶违反。
    """
    total = np.sum(atom_counts)
    return (total % 2) != required_parity


# ------------------------------------------------------------------
# 4. 高维距离统计 (源自 hypercube_distance)
# ------------------------------------------------------------------

def hypercube_distance_stats(descriptors: np.ndarray, n_pairs: int = 500) -> Tuple[float, float]:
    """
    从分子描述符集合中随机采样点对，计算欧氏距离的均值与方差。
    用于评估描述符空间的均匀性与可区分度。
    """
    n = descriptors.shape[0]
    if n < 2:
        return 0.0, 0.0
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        d = np.linalg.norm(descriptors[i] - descriptors[j])
        distances.append(d)
    if not distances:
        return 0.0, 0.0
    dists = np.array(distances, dtype=np.float64)
    return float(dists.mean()), float(dists.var())


def descriptor_space_uniformity(descriptors: np.ndarray) -> float:
    """
    基于距离方差的描述符空间均匀性度量:
        U = 1 / (1 + σ²_d / μ_d²)
    U → 1 表示分布均匀。
    """
    mu, var = hypercube_distance_stats(descriptors, n_pairs=min(1000, len(descriptors) * 10))
    if mu < 1e-12:
        return 0.0
    return 1.0 / (1.0 + var / (mu ** 2))


# ------------------------------------------------------------------
# 5. 参数化圆/球基函数 (源自 circles)
# ------------------------------------------------------------------

def spherical_basis_angles(n_points: int = 16, rotation: float = 0.0) -> np.ndarray:
    """
    生成二维圆上均匀分布的角度基，可用于原子轨道的角度分解。
    返回 shape (n_points, 2) 的单位向量。
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False) + np.radians(rotation)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def angular_descriptor(atoms: np.ndarray, center_idx: int, n_angles: int = 8) -> np.ndarray:
    """
    计算以 center_idx 为中心、沿多个角度方向的邻居原子投影和。
    用于捕捉局部各向异性环境。
    """
    n = atoms.shape[0]
    if center_idx >= n:
        return np.zeros(n_angles)
    dirs = spherical_basis_angles(n_angles)
    center = atoms[center_idx]
    desc = np.zeros(n_angles, dtype=np.float64)
    for i in range(n):
        if i == center_idx:
            continue
        dr = atoms[i] - center
        r = np.linalg.norm(dr)
        if r < 1e-6:
            continue
        dr_u = dr / r
        # 投影到每个角度方向并加权 1/r
        for a in range(n_angles):
            proj = np.dot(dr_u[:2], dirs[a])
            desc[a] += proj / r
    # 归一化
    norm = np.linalg.norm(desc)
    return desc / norm if norm > 1e-12 else desc
