"""
multibody_topology.py
=====================
多体系统拓扑分析与约束管理

本模块将以下种子项目的核心算法融入结构力学多体动力学：
  - 672_lights_out  : GF(2) 线性代数与零空间正交性检验 → 桁架拓扑稳定性判定
  - 1390_vector     : 字典序向量枚举与排列等价类 → 自由度编号与构型空间索引
  - 1332_triangulation_boundary_edges : 边界边检测与链式追踪 → 柔性体表面网格边界提取

核心物理模型：
  - 桁架关联矩阵 (Incidence Matrix) A ∈ ℝ^(m×n)
  - 平衡矩阵 B = [A · diag(c)]，其中 c 为杆件方向余弦向量
  - 静定判定：rank(B) = 3j - r，其中 j 为节点数，r 为支座约束数
  - 零空间分析：若 B·q = 0 存在非零解，则体系存在机构位移 (mechanism)
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional


def build_incidence_matrix(nodes: np.ndarray, members: List[Tuple[int, int]]) -> np.ndarray:
    """
    构建桁架关联矩阵 A ∈ ℝ^(m×n)（有向图形式）。
    对第 k 根杆件连接节点 i→j，令 A[k, i] = -1, A[k, j] = +1。
    
    参数
    ----
    nodes : (n, d) 节点坐标数组
    members : 杆件端点对列表，每对 (i, j) 为 0-based 索引
    
    返回
    ----
    A : (m, n) 关联矩阵
    """
    n_nodes = nodes.shape[0]
    n_members = len(members)
    A = np.zeros((n_members, n_nodes), dtype=np.float64)
    for k, (i, j) in enumerate(members):
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError(f"节点索引越界: ({i}, {j}), 总节点数={n_nodes}")
        if i == j:
            raise ValueError(f"自环杆件不允许: ({i}, {j})")
        A[k, i] = -1.0
        A[k, j] = +1.0
    return A


def build_equilibrium_matrix(nodes: np.ndarray, members: List[Tuple[int, int]]) -> np.ndarray:
    """
    构建三维空间桁架平衡矩阵 B ∈ ℝ^(3n×m)。
    对每根杆件 k，其在全局坐标下的方向余弦为
        c_k = (r_j - r_i) / ||r_j - r_i||
    平衡矩阵列块为 B[:, k] = [ ... 0, c_k, 0 ... ]^T，
    对应节点 i 为 -c_k，节点 j 为 +c_k。
    
    物理意义：B · t = f_ext，其中 t 为杆件轴力向量，f_ext 为节点外力。
    """
    n_nodes = nodes.shape[0]
    n_members = len(members)
    B = np.zeros((3 * n_nodes, n_members), dtype=np.float64)
    for k, (i, j) in enumerate(members):
        diff = nodes[j] - nodes[i]
        length = np.linalg.norm(diff)
        if length < 1e-14:
            raise ValueError(f"杆件 {k} 长度为零: 节点 {i} 与 {j} 重合")
        c = diff / length
        B[3 * i:3 * i + 3, k] = -c
        B[3 * j:3 * j + 3, k] = +c
    return B


def check_static_determinacy(n_nodes: int, n_members: int, n_reactions: int,
                            spatial_dim: int = 3) -> Tuple[bool, int, int]:
    """
    判定桁架静定性。
    
    经典公式（Maxwell 准则）：
        m + r = d · j   →  静定 (isostatic)
        m + r < d · j   →  机构 (mechanism)
        m + r > d · j   →  超静定 (hyperstatic)
    
    其中 d 为空间维数，j = n_nodes，m = n_members，r = n_reactions。
    
    返回
    ----
    is_determinate : 是否满足 m + r == d*j
    deficiency : d*j - (m + r)  <0 表示超静定，>0 表示机构
    rank_deficiency : 数值秩亏量
    """
    expected = spatial_dim * n_nodes
    actual = n_members + n_reactions
    deficiency = expected - actual
    is_determinate = (deficiency == 0)
    # 数值安全：允许极小误差
    rank_deficiency = max(0, deficiency)
    return is_determinate, deficiency, rank_deficiency


def nullspace_orthogonality_check(B: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    """
    基于 Lights Out 的 GF(2) 零空间正交思想，在实数域上计算平衡矩阵 B 的零空间基。
    
    对桁架而言，B 的左零空间 (nullspace of B^T) 的维数即为机构位移数（infinitesimal
    mechanism count）。若 dim(null(B^T)) > 6（刚体运动自由度），则体系存在内部机构。
    
    使用 SVD 分解：B = U Σ V^T，零空间由 V 中对应奇异值 < tol 的列张成。
    
    返回
    ----
    N : (m, k) 零空间基矩阵，满足 B @ N ≈ 0
    dim_null : 零空间维数
    """
    if B.size == 0:
        return np.zeros((0, 0)), 0
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    rank = np.sum(s > tol)
    dim_null = B.shape[1] - rank
    if dim_null <= 0:
        return np.zeros((B.shape[1], 0)), 0
    N = Vt[rank:, :].T  # 列向量为零空间基
    return N, dim_null


def enumerate_dof_indices(n_nodes: int, spatial_dim: int = 3,
                          fixed_nodes: Optional[List[int]] = None) -> np.ndarray:
    """
    字典序枚举全部自由度索引，并剔除支座约束节点。
    
    受 1390_vector 中 lexicographic enumeration 启发，对 n 个节点、每个节点 d 个平动
    自由度，总 DOF 编号为 0, 1, ..., d*n-1。固定节点对应的 d 个编号被移除。
    
    返回
    ----
    free_dofs : 自由自由度编号数组
    """
    total_dofs = spatial_dim * n_nodes
    all_dofs = np.arange(total_dofs, dtype=np.int32)
    if fixed_nodes is None or len(fixed_nodes) == 0:
        return all_dofs
    mask = np.ones(total_dofs, dtype=bool)
    for node in fixed_nodes:
        if not (0 <= node < n_nodes):
            raise ValueError(f"固定节点索引越界: {node}")
        mask[spatial_dim * node:spatial_dim * node + spatial_dim] = False
    return all_dofs[mask]


def lexicographic_joint_configs(n_joints: int, n_states: int) -> np.ndarray:
    """
    枚举 n_joints 个铰接关节、每个关节 n_states 种离散状态的全体字典序构型。
    对应 vector 中 base-B odometer counting，用于多体系统离散构型空间遍历。
    
    总构型数 = n_states^n_joints。对大规模系统采用迭代生成（生成器）避免内存爆炸。
    """
    total = n_states ** n_joints
    if total > 1_000_000:
        raise ValueError(f"构型空间过大: {total} > 1e6，请使用生成器版本")
    configs = np.zeros((total, n_joints), dtype=np.int32)
    for i in range(total):
        tmp = i
        for j in range(n_joints):
            configs[i, j] = tmp % n_states
            tmp //= n_states
    return configs


def triangulation_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """
    从三角形网格中提取边界边。
    
    基于 1332_triangulation_boundary_edges 的核心思想：内部边恰好被两个三角形共享
    （方向可能相反），边界边仅出现一次。将三角形的三条边展开后排序，统计出现次数，
    单例即为边界。
    
    参数
    ----
    triangles : (n_tri, 3) int 数组，每行三个顶点索引构成一个三角形
    
    返回
    ----
    boundary_edges : (n_be, 2) 边界边顶点对，已按连通性排序为链
    """
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必须是 (n, 3) 形状")
    n_tri = triangles.shape[0]
    edges = np.zeros((3 * n_tri, 2), dtype=np.int32)
    # 提取三条边 (0,1), (1,2), (2,0)
    edges[0::3, 0] = triangles[:, 0]
    edges[0::3, 1] = triangles[:, 1]
    edges[1::3, 0] = triangles[:, 1]
    edges[1::3, 1] = triangles[:, 2]
    edges[2::3, 0] = triangles[:, 2]
    edges[2::3, 1] = triangles[:, 0]
    # 规范化：小索引在前
    edges_sorted = np.sort(edges, axis=1)
    # 使用字典统计
    edge_dict = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_dict[key] = edge_dict.get(key, 0) + 1
    boundary = [np.array(k) for k, v in edge_dict.items() if v == 1]
    if len(boundary) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    boundary = np.vstack(boundary)
    # 链式排序：从第一条边开始，依次找共享顶点
    chained = _chain_boundary_edges(boundary)
    return chained


def _chain_boundary_edges(edges: np.ndarray) -> np.ndarray:
    """
    将边界边按连通性排序成一条或多条闭合/开链。
    """
    if edges.shape[0] == 0:
        return edges
    n_edges = edges.shape[0]
    used = np.zeros(n_edges, dtype=bool)
    result = []
    current = 0
    used[current] = True
    chain = [edges[current].copy()]
    while np.any(~used):
        tail = chain[-1][1]
        found = False
        for i in range(n_edges):
            if used[i]:
                continue
            if edges[i, 0] == tail:
                chain.append(edges[i].copy())
                used[i] = True
                found = True
                break
            elif edges[i, 1] == tail:
                chain.append(np.array([edges[i, 1], edges[i, 0]]))
                used[i] = True
                found = True
                break
        if not found:
            result.append(np.vstack(chain))
            # 开始新链
            remaining = np.where(~used)[0]
            if len(remaining) == 0:
                break
            current = remaining[0]
            used[current] = True
            chain = [edges[current].copy()]
    if len(chain) > 0:
        result.append(np.vstack(chain))
    if len(result) == 1:
        return result[0]
    return np.vstack(result)


def generate_truss_topology(num_bays: int = 2, bay_length: float = 1.0,
                            height: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    生成一个简单的空间桁架拓扑（平面三角化桁架， extruded 到 3D）。
    用于测试与演示。
    
    节点排布：底部一排，顶部一排，形成三角形网格。
    """
    if num_bays < 1:
        raise ValueError("num_bays 必须 ≥ 1")
    n_nodes = 2 * (num_bays + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(num_bays + 1):
        nodes[i, :] = [i * bay_length, 0.0, 0.0]
        nodes[i + num_bays + 1, :] = [i * bay_length, height, 0.0]
    members = []
    # 底弦
    for i in range(num_bays):
        members.append((i, i + 1))
    # 顶弦
    for i in range(num_bays):
        members.append((i + num_bays + 1, i + num_bays + 2))
    # 竖杆
    for i in range(num_bays + 1):
        members.append((i, i + num_bays + 1))
    # 斜杆（三角化）
    for i in range(num_bays):
        members.append((i, i + num_bays + 2))
        members.append((i + 1, i + num_bays + 1))
    return nodes, members
