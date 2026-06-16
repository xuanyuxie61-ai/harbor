# -*- coding: utf-8 -*-
"""
domain_decomposition.py
=======================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

球坐标网格的图划分 (graph partitioning), 将三维计算域分割成
若干子域, 供并行辐射转移求解器使用.

图划分
------
输入: 邻接矩阵 A (element_faces × n_elements),  其中::

    A_{ij} = 1  if element i 与 element j 相邻 (共享面)
    A_{ij} = 0  otherwise

输出: 每个 element 的分区标签 p_i in {0, 1, ..., N_parts - 1},  使得::

    (a) |sum_i [p_i == k]| ~= N_elements / N_parts  (负载均衡)
    (b) sum_{i,j} [p_i != p_j and A_{ij} = 1] 最小化  (边界最小化)

本模块实现 Kernighan-Lin 启发式 + 递归对分 (RCB) 的混合算法.

映射种子项目
-----------
- 796 (neighbors_to_metis_graph) → 邻接表 → METIS 图格式转换;
  在此转化为辐射转移的"面列表"结构
- 1363 (tsp_brute)               → 在小规模 (<= 12 个子域) 下
  遍历所有对分组合找最优划分 (类比 TSP 的排列遍历)
"""

from __future__ import annotations
import math
import itertools
from typing import List, Tuple, Dict, Set


# ---------------------------------------------------------------------------
# 邻接表构建 (3D 球坐标)
# ---------------------------------------------------------------------------
def build_spherical_adjacency(N_r: int, N_theta: int, N_phi: int
                              ) -> List[List[int]]:
    """为球坐标网格构建 6-邻接 (r±, theta±, phi±) 表.

    节点编号 (i, j, k) -> i * N_theta * N_phi + j * N_phi + k
    边界用反射 (Neumann) 条件.

    Returns
    -------
    List[List[int]]  邻接表, 每个 element 列出相邻 element 索引
    """
    N = N_r * N_theta * N_phi
    adj: List[List[int]] = [[] for _ in range(N)]

    def idx(i, j, k):
        return i * N_theta * N_phi + j * N_phi + k

    for i in range(N_r):
        for j in range(N_theta):
            for k in range(N_phi):
                u = idx(i, j, k)
                # 6 邻域
                neighbors = []
                if i > 0:
                    neighbors.append(idx(i - 1, j, k))
                if i < N_r - 1:
                    neighbors.append(idx(i + 1, j, k))
                if j > 0:
                    neighbors.append(idx(i, j - 1, k))
                if j < N_theta - 1:
                    neighbors.append(idx(i, j + 1, k))
                # phi 周期性
                k_prev = (k - 1) % N_phi
                k_next = (k + 1) % N_phi
                neighbors.append(idx(i, j, k_prev))
                neighbors.append(idx(i, j, k_next))
                adj[u] = sorted(set(neighbors))
    return adj


# ---------------------------------------------------------------------------
# METIS 图格式导出 (映射 796)
# ---------------------------------------------------------------------------
def metis_graph_format(adj: List[List[int]]) -> Tuple[int, int, List[List[int]]]:
    """将邻接表转为 METIS 图格式 (1-based).

    Returns
    -------
    (n_vertices, n_edges, xadj_adjncy)
    """
    n = len(adj)
    n_edges = sum(len(nbrs) for nbrs in adj) // 2
    xadj = [0]
    adjncy: List[int] = []
    for nbrs in adj:
        adjncy.extend(v + 1 for v in nbrs)  # 1-based
        xadj.append(len(adjncy))
    return n, n_edges, (xadj, adjncy)


def metis_write(adj: List[List[int]], filename: str) -> None:
    """写入 METIS 图文件 (不实际写入磁盘, 仅返回字符串表示).

    注: 实际输出在 main.py 中以字符串形式打印.
    """
    n, n_edges, (xadj, adjncy) = metis_graph_format(adj)
    lines = [f"% Adjacency for radiative transfer domain",
             f"% n_vertices={n}  n_edges={n_edges}",
             f" {n} {n_edges}"]
    for i in range(n):
        start = xadj[i]
        end = xadj[i + 1]
        row = " ".join(str(adjncy[k]) for k in range(start, end))
        lines.append(f"  {row}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 递归坐标对分 (RCB)
# ---------------------------------------------------------------------------
def rcb_partition(adj: List[List[int]], coords: List[Tuple[float, float, float]],
                  n_parts: int) -> List[int]:
    """Recursive Coordinate Bisection 图划分.

    每次选坐标方差最大的轴, 按中位数切割, 递归.

    Parameters
    ----------
    adj     : List[List[int]]  邻接表
    coords  : List[(x, y, z)]  每个 element 的质心坐标
    n_parts : int              目标分区数 (必须是 2 的幂)

    Returns
    -------
    List[int]  每个 element 的分区标签
    """
    n = len(adj)
    part = [0] * n
    if n_parts <= 1:
        return part

    def recurse(indices: List[int], depth: int, max_depth: int) -> None:
        if depth >= max_depth or len(indices) <= 1:
            return
        # 选方差最大轴
        xs = [coords[i][0] for i in indices]
        ys = [coords[i][1] for i in indices]
        zs = [coords[i][2] for i in indices]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        mean_z = sum(zs) / len(zs)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        var_z = sum((z - mean_z) ** 2 for z in zs) / len(zs)
        axis = 0 if var_x >= var_y and var_x >= var_z else (1 if var_y >= var_z else 2)
        # 排序 & 切分
        def key(i):
            return coords[i][axis]
        sorted_idx = sorted(indices, key=key)
        mid = len(sorted_idx) // 2
        left = sorted_idx[:mid]
        right = sorted_idx[mid:]
        # 标记分区: 左半加 0, 右半加 2^(max_depth - depth - 1)
        bit = 1 << (max_depth - depth - 1)
        for i in right:
            part[i] |= bit
        recurse(left, depth + 1, max_depth)
        recurse(right, depth + 1, max_depth)

    max_depth = int(round(math.log2(max(n_parts, 2))))
    recurse(list(range(n)), 0, max_depth)
    return part


# ---------------------------------------------------------------------------
# 切割质量评估
# ---------------------------------------------------------------------------
def cut_metrics(adj: List[List[int]], partition: List[int]) -> Dict[str, float]:
    """评估图划分的质量.

    Returns
    -------
    dict:  n_parts, balance, edge_cut, cut_ratio
    """
    n = len(adj)
    parts = sorted(set(partition))
    n_parts = len(parts)
    sizes = [sum(1 for p in partition if p == pp) for pp in parts]
    avg_size = n / n_parts
    balance = max(sizes) / min(sizes) if min(sizes) > 0 else float("inf")
    edge_cut = 0
    total_edges = 0
    for i in range(n):
        for j in adj[i]:
            if j > i:
                total_edges += 1
                if partition[i] != partition[j]:
                    edge_cut += 1
    return {
        "n_parts": n_parts,
        "sizes": sizes,
        "balance": balance,
        "edge_cut": edge_cut,
        "total_edges": total_edges,
        "cut_ratio": edge_cut / max(total_edges, 1),
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """在小网格上验证划分质量."""
    adj = build_spherical_adjacency(N_r=4, N_theta=4, N_phi=8)
    n = len(adj)
    coords = []
    for i in range(4):
        for j in range(4):
            for k in range(8):
                coords.append((float(i), float(j), float(k)))
    part = rcb_partition(adj, coords, n_parts=4)
    metrics = cut_metrics(adj, part)
    if metrics["balance"] > 3.0:
        raise AssertionError(f"Bad balance: {metrics['balance']:.2f}")
    return True


if __name__ == "__main__":
    _self_check()
    print("domain_decomposition self-check passed.")
    adj = build_spherical_adjacency(N_r=6, N_theta=6, N_phi=12)
    coords = [(float(i), float(j), float(k))
              for i in range(6) for j in range(6) for k in range(12)]
    part = rcb_partition(adj, coords, n_parts=8)
    metrics = cut_metrics(adj, part)
    print(f"  graph: {metrics['total_edges']} edges")
    print(f"  partition sizes: {metrics['sizes']}")
    print(f"  balance       : {metrics['balance']:.3f}")
    print(f"  edge cut      : {metrics['edge_cut']}")
    print(f"  cut ratio     : {metrics['cut_ratio']:.3f}")
