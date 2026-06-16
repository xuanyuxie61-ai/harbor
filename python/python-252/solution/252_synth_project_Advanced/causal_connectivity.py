#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_connectivity.py  ——  MHD 网格因果连通性分析 (Alfvén / 光锥)

融合种子项目:
  - 1134_watcl-lab_graphalgosimulation : BFS / DFS 图遍历算法
  - 755_mesh_etoe : 单元邻接关系

核心思想:
  在 MHD 网格上, 每个单元通过 Alfvén / 快磁声信号影响邻居.
  构造因果图: 节点 = 单元, 边 = 信号可传播.
  BFS 遍历得到某单元的影响域 (domain of dependence / influence).

  Alfvén 锥:  dt/dr = 1 / (|v_r| + vA_r)
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Set, Tuple
from mhd_grid import MHDGrid
from mhd_equations import MHDState
from mhd_constants import MHDConfig


class CausalGraph:
    """
    MHD 网格的因果图.
    节点: 扁平化索引 (i * Nt * Np + j * Np + k).
    边: 如果单元 A 能在一个时间步内影响单元 B, 则存在有向边 A -> B.
    """

    def __init__(self, grid: MHDGrid, cfg: MHDConfig,
                 state: MHDState, dt: float) -> None:
        self.grid = grid
        self.cfg = cfg
        self.dt = dt
        self.Nr, self.Nt, self.Np = grid.Nr - 1, grid.Nt - 1, grid.Np

        # 构造邻接表 (融合 1134 的图遍历)
        self.adj: List[List[int]] = [[] for _ in range(self.Nr * self.Nt * self.Np)]
        self._build_causal_edges(state)

    def _flat(self, i: int, j: int, k: int) -> int:
        return i * self.Nt * self.Np + j * self.Np + k

    def _unflat(self, idx: int) -> Tuple[int, int, int]:
        k = idx % self.Np
        j = (idx // self.Np) % self.Nt
        i = idx // (self.Nt * self.Np)
        return i, j, k

    def _build_causal_edges(self, state: MHDState) -> None:
        """
        对每个单元, 计算其 Alfvén / 快磁声速度,
        确定一个时间步内能影响哪些邻居.
        """
        cf = np.maximum(
            np.sqrt(state.vr**2 + state.vt**2 + state.vp**2)
            + self.cfg.plasma.fast_magnetosonic(
                state.press, state.rho,
                state.Br**2 + state.Bt**2 + state.Bp**2),
            1e-30)

        for i in range(self.Nr):
            for j in range(self.Nt):
                for k in range(self.Np):
                    idx = self._flat(i, j, k)
                    c = cf[i, j, k]
                    # 径向邻居
                    if i > 0 and self.dt * c > self.grid.dr[i - 1]:
                        self.adj[idx].append(self._flat(i - 1, j, k))
                    if i < self.Nr - 1 and self.dt * c > self.grid.dr[i]:
                        self.adj[idx].append(self._flat(i + 1, j, k))
                    # 极向邻居
                    if j > 0 and self.dt * c > self.grid.dtheta[j - 1]:
                        self.adj[idx].append(self._flat(i, j - 1, k))
                    if j < self.Nt - 1 and self.dt * c > self.grid.dtheta[j]:
                        self.adj[idx].append(self._flat(i, j + 1, k))
                    # 方位向 (周期)
                    if self.Np > 1:
                        self.adj[idx].append(self._flat(i, j, (k - 1) % self.Np))
                        self.adj[idx].append(self._flat(i, j, (k + 1) % self.Np))

    def bfs_reachable(self, source: int) -> Set[int]:
        """
        BFS 从 source 出发, 返回所有可达节点 (融合 1134 reference_bfs).
        """
        n_total = self.Nr * self.Nt * self.Np
        visited = [False] * n_total
        queue = deque()
        queue.append(source)
        visited[source] = True
        while queue:
            node = queue.popleft()
            for neighbor in self.adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return set(i for i, v in enumerate(visited) if v)

    def dfs_path(self, source: int, target: int) -> List[int]:
        """
        DFS 找一条从 source 到 target 的路径 (融合 1134 reference_dfs).
        """
        n_total = self.Nr * self.Nt * self.Np
        visited = [False] * n_total
        parent = [-1] * n_total
        stack = [source]
        visited[source] = True
        found = False
        while stack:
            node = stack.pop()
            if node == target:
                found = True
                break
            for neighbor in self.adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    stack.append(neighbor)
        if not found:
            return []
        # 回溯路径
        path = []
        cur = target
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]

    def influence_domain_size(self, source: int) -> int:
        """返回 source 的影响域大小."""
        return len(self.bfs_reachable(source))

    def causal_diameter(self) -> int:
        """
        因果图直径 (最大最短路径).
        用 BFS 从边界点估计.
        """
        # 简化: 从 (0,0,0) 做 BFS
        source = self._flat(0, 0, 0)
        reachable = self.bfs_reachable(source)
        return len(reachable)
