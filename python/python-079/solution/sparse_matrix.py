"""
稀疏矩阵存储与线性代数运算模块

基于种子项目：
  - 986_r8ncf：坐标格式 (COO) 稀疏矩阵存储与运算
  - 1349_triangulation_rcm：Reverse Cuthill-McKee 带宽缩减重排序

核心功能：
  1. R8NCF 坐标格式稀疏矩阵：存储 (row, col, value) 三元组，
     对角元优先排列以支持不完全分解预条件子。
  2. 稀疏矩阵-向量乘法 (SpMV) 及其转置。
  3. 从有限差分/有限元模板自动生成稀疏矩阵（二维 Laplacian、
     质量矩阵等）。
  4. RCM 重排序：基于三角剖分邻接图构建稀疏结构，执行 BFS 
     伪周长根搜索与层次遍历，生成最小化带宽的节点排列。
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import compute_bandwidth


# ======================================================================
# 1. R8NCF 坐标格式稀疏矩阵 (源自 986_r8ncf)
# ======================================================================

class R8NCFSparseMatrix:
    """
    坐标格式 (COO) 稀疏矩阵类，兼容 R8NCF 规范。
    内部存储：
        rowcol : (2, nz_num) 的 int 数组，每列为 (row, col)
        values : (nz_num,) 的 float 数组
    对角元必须位于前 n 个位置（n = min(m, n_rows)）。
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nz_num = 0
        self.rowcol = np.zeros((2, 0), dtype=int)
        self.values = np.zeros(0, dtype=float)

    def add_entry(self, row: int, col: int, value: float):
        """添加单个非零元。"""
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            raise IndexError("行列索引越界")
        new_rc = np.zeros((2, self.nz_num + 1), dtype=int)
        new_val = np.zeros(self.nz_num + 1, dtype=float)
        if self.nz_num > 0:
            new_rc[:, : self.nz_num] = self.rowcol
            new_val[: self.nz_num] = self.values
        new_rc[0, self.nz_num] = row
        new_rc[1, self.nz_num] = col
        new_val[self.nz_num] = value
        self.rowcol = new_rc
        self.values = new_val
        self.nz_num += 1

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法：y = A x
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_cols:
            raise ValueError("向量维度与矩阵列数不匹配")
        y = np.zeros(self.n_rows, dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[i] += self.values[k] * x[j]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵转置-向量乘法：y = A^T x
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_rows:
            raise ValueError("向量维度与矩阵行数不匹配")
        y = np.zeros(self.n_cols, dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[j] += self.values[k] * x[i]
        return y

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵（仅用于小规模调试）。"""
        A = np.zeros((self.n_rows, self.n_cols), dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            A[i, j] += self.values[k]
        return A

    def permute_rows_cols(self, perm: List[int]) -> "R8NCFSparseMatrix":
        """
        对行列同时应用相同的排列 perm（新索引 → 旧索引）。
        用于 RCM 重排序后的矩阵变换。
        """
        inv = [0] * len(perm)
        for new_idx, old_idx in enumerate(perm):
            inv[old_idx] = new_idx
        B = R8NCFSparseMatrix(self.n_rows, self.n_cols)
        for k in range(self.nz_num):
            i_old = self.rowcol[0, k]
            j_old = self.rowcol[1, k]
            B.add_entry(inv[i_old], inv[j_old], self.values[k])
        return B


def build_laplacian_2d_sparse(
    nx: int, ny: int, dx: float, dy: float
) -> R8NCFSparseMatrix:
    """
    构建二维五点 Laplacian 离散稀疏矩阵：
        -∇²u ≈ (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}) / (dx*dy)
    内部节点编号按行优先：idx = i*ny + j。
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx 和 ny 至少为 3")
    n = nx * ny
    A = R8NCFSparseMatrix(n, n)
    denom = dx * dy
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            # 对角元
            A.add_entry(idx, idx, 4.0 / denom)
            # 左邻
            if i > 0:
                A.add_entry(idx, (i - 1) * ny + j, -1.0 / denom)
            # 右邻
            if i < nx - 1:
                A.add_entry(idx, (i + 1) * ny + j, -1.0 / denom)
            # 下邻
            if j > 0:
                A.add_entry(idx, i * ny + (j - 1), -1.0 / denom)
            # 上邻
            if j < ny - 1:
                A.add_entry(idx, i * ny + (j + 1), -1.0 / denom)
    return A


def build_second_difference_1d_sparse(n: int) -> R8NCFSparseMatrix:
    """
    构建一维二阶差分矩阵（三对角 [-1, 2, -1]）的稀疏表示。
    """
    if n < 2:
        raise ValueError("n 至少为 2")
    A = R8NCFSparseMatrix(n, n)
    for i in range(n):
        A.add_entry(i, i, 2.0)
        if i > 0:
            A.add_entry(i, i - 1, -1.0)
        if i < n - 1:
            A.add_entry(i, i + 1, -1.0)
    return A


# ======================================================================
# 2. RCM 重排序 (源自 1349_triangulation_rcm)
# ======================================================================

def build_adjacency_from_triangulation(
    triangles: List[Tuple[int, int, int]], n_nodes: int
) -> List[List[int]]:
    """
    从三角剖分构建节点邻接图（无向）。
    """
    adj = [set() for _ in range(n_nodes)]
    for tri in triangles:
        for k in range(3):
            a = tri[k]
            b = tri[(k + 1) % 3]
            if 0 <= a < n_nodes and 0 <= b < n_nodes:
                adj[a].add(b)
                adj[b].add(a)
    return [sorted(list(s)) for s in adj]


def rcm_reorder(adj: List[List[int]]) -> List[int]:
    """
    Reverse Cuthill-McKee 重排序算法。
    返回排列 perm（新索引 → 旧索引），使得重排序后的矩阵带宽最小化。
    算法步骤：
      1. 对每个未访问的连通分量，找到伪周长根节点。
      2. 从根出发进行 BFS 层次遍历，每层内按度排序。
      3. 将遍历顺序反转得到 RCM 排列。
    """
    n = len(adj)
    if n == 0:
        return []
    visited = [False] * n
    perm = []

    for start in range(n):
        if visited[start]:
            continue
        # 找到该连通分量的伪周长根
        root = _find_pseudo_peripheral(adj, start)
        # 层次遍历
        level_order = []
        queue = [root]
        visited[root] = True
        while queue:
            # 按度排序当前层
            queue.sort(key=lambda x: len(adj[x]))
            next_queue = []
            for node in queue:
                level_order.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        next_queue.append(nb)
            queue = next_queue
        # 反转得到 RCM 顺序
        perm.extend(reversed(level_order))
    return perm


def _find_pseudo_peripheral(adj: List[List[int]], start: int) -> int:
    """
    通过反复扩展层次结构找到伪周长节点。
    算法：从 start 出发构建层次结构，取最后一层中度最小的节点，
    重复直到层次数不再增加。
    """
    n = len(adj)
    current = start
    while True:
        levels, _ = _build_level_structure(adj, current)
        last_level = levels[-1]
        # 取最后一层中度最小的节点
        min_deg = n + 1
        candidate = current
        for node in last_level:
            deg = len(adj[node])
            if deg < min_deg:
                min_deg = deg
                candidate = node
        if candidate == current:
            break
        new_levels, _ = _build_level_structure(adj, candidate)
        if len(new_levels) <= len(levels):
            break
        current = candidate
    return current


def _build_level_structure(
    adj: List[List[int]], root: int
) -> Tuple[List[List[int]], List[int]]:
    """
    从 root 出发构建层次结构（BFS）。
    返回 levels（每层节点列表）和 level_idx（每个节点的层次编号）。
    """
    n = len(adj)
    level_idx = [-1] * n
    queue = [root]
    level_idx[root] = 0
    levels = [[root]]
    while queue:
        next_queue = []
        for node in queue:
            for nb in adj[node]:
                if level_idx[nb] == -1:
                    level_idx[nb] = len(levels)
                    next_queue.append(nb)
        if not next_queue:
            break
        levels.append(next_queue)
        queue = next_queue
    return levels, level_idx


def apply_rcm_to_matrix(
    A: R8NCFSparseMatrix, triangles: List[Tuple[int, int, int]], n_nodes: int
) -> Tuple[R8NCFSparseMatrix, List[int], int, int]:
    """
    对稀疏矩阵 A 应用 RCM 重排序。
    返回 (A_perm, perm, bw_before, bw_after)。
    """
    adj = build_adjacency_from_triangulation(triangles, n_nodes)
    bw_before = compute_bandwidth(adj, list(range(n_nodes)))
    perm = rcm_reorder(adj)
    bw_after = compute_bandwidth(adj, perm)
    A_perm = A.permute_rows_cols(perm)
    return A_perm, perm, bw_before, bw_after
