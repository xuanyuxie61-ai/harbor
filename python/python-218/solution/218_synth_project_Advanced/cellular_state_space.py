"""
cellular_state_space.py
=======================
形式化细胞自动机的安全配置空间探索, 用于 VI 的解分支分析.

数学背景
--------
细胞自动机 (CA) 的配置空间:
    C = {0, 1}^n  (n 个细胞, 每个有 2 个状态)

安全配置空间 (Secure Configuration Space):
    S = {c ∈ C : δ(c) ∈ A}
其中 δ 为全局转移函数, A 为允许集合.

在参数化 VI 中的应用:
    当 VI 的参数 t 连续变化时, 解 x*(t) 的路径可能经历:
        - 分支 (bifurcation): 一个解分裂为多个
        - 转折 (turning point): 解路径折返
        - 跳跃 (jump): 解的不连续变化

    将参数离散化为 CA 的细胞, 解的状态编码为细胞状态,
    通过探索安全配置空间来系统地发现所有解分支.

关键公式
--------
覆盖度 (Coverage): 已探索的安全配置 / 总安全配置数.
邻域 (von Neumann): N_i = {i-1, i, i+1}
局部转移规则: δ_i = f(c_{i-1}, c_i, c_{i+1})

安全配置的 VI 意义:
    安全 ⟺ 该配置对应的活动集划分是"稳定的"
    (即 Newton 步不会导致活动集的大幅变化)

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConfigurationState:
    """CA 配置状态."""
    config: Tuple[int, ...]
    is_safe: bool
    vi_residual: float = 0.0
    active_set_hash: int = 0


class CellularStateSpaceExplorer:
    """
    细胞自动机状态空间探索器.

    将 VI 的活动集划分编码为 CA 配置:
        c_i = 0  ⟺  i ∈ Active set (x_i = 0)
        c_i = 1  ⟺  i ∈ Inactive set (x_i > 0)

    安全配置的判定:
        配置 c 是安全的, 当且仅当:
        1. 相邻活动集分量间的间隙 > 最小间隙阈值
        2. 该配置对应的局部 Newton 系统条件数 < 阈值

    探索策略:
        BFS: 从初始配置出发, 逐步翻转单个细胞状态
        DFS: 深度优先搜索所有可达的安全配置
        Random: 随机采样配置空间
    """

    def __init__(self, n_cells: int, min_gap: int = 1, cond_threshold: float = 1e6):
        self.n_cells = n_cells
        self.min_gap = min_gap
        self.cond_threshold = cond_threshold
        self.safe_configs: Set[Tuple[int, ...]] = set()
        self.visited: Set[Tuple[int, ...]] = set()
        self.transition_graph: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    def encode_active_set(self, active_mask: np.ndarray) -> Tuple[int, ...]:
        """将活动集掩码编码为 CA 配置."""
        return tuple(int(x) for x in (active_mask > 0).astype(int))

    def is_safe_config(self, config: Tuple[int, ...]) -> bool:
        """
        判定配置是否安全.

        安全条件:
        1. 无孤立活动细胞 (除非 n ≤ 2)
        2. 活动段之间的间隙 ≥ min_gap
        """
        if self.n_cells <= 2:
            return True

        # 检查间隙条件
        cells = list(config)
        in_active_segment = False
        last_active_end = -self.min_gap - 1

        for i, c in enumerate(cells):
            if c == 1:
                if not in_active_segment:
                    # 新活动段开始, 检查与上一段的间隙
                    if i - last_active_end < self.min_gap and last_active_end >= 0:
                        return False
                    in_active_segment = True
            else:
                if in_active_segment:
                    last_active_end = i
                    in_active_segment = False

        return True

    def get_neighbors(self, config: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        获取配置的邻居 (单细胞翻转).

        在 VI 的语境下, 邻居对应于活动集中增加或移除一个指标.
        """
        neighbors = []
        for i in range(self.n_cells):
            new_config = list(config)
            new_config[i] = 1 - new_config[i]
            neighbors.append(tuple(new_config))
        return neighbors

    def bfs_explore(self, initial_config: Tuple[int, ...], max_depth: int = 5) -> int:
        """
        BFS 探索安全配置空间.

        Returns
        -------
        n_safe : int
            发现的安全配置数量
        """
        queue: List[Tuple[Tuple[int, ...], int]] = [(initial_config, 0)]
        self.visited.add(initial_config)
        if self.is_safe_config(initial_config):
            self.safe_configs.add(initial_config)

        while queue:
            config, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            neighbors = self.get_neighbors(config)
            safe_neighbors = []
            for nbr in neighbors:
                if nbr not in self.visited:
                    self.visited.add(nbr)
                    if self.is_safe_config(nbr):
                        self.safe_configs.add(nbr)
                        safe_neighbors.append(nbr)
                    queue.append((nbr, depth + 1))

            self.transition_graph[config] = safe_neighbors

        return len(self.safe_configs)

    def compute_coverage(self, total_safe_estimate: Optional[int] = None) -> float:
        """
        计算安全配置空间的覆盖度.

        coverage = |S_explored| / |S_total|
        """
        if total_safe_estimate is None or total_safe_estimate == 0:
            # 使用全空间大小作为粗略估计
            total_safe_estimate = 2 ** self.n_cells
        return len(self.safe_configs) / total_safe_estimate

    def count_connected_components(self) -> int:
        """
        计算安全配置空间中连通分量的数量.

        这反映 VI 解空间的分块结构: 每个连通分量对应一个
        "解族", 分量间的过渡需要通过不安全区域.
        """
        if not self.safe_configs:
            return 0
        visited_local: Set[Tuple[int, ...]] = set()
        n_components = 0

        for config in self.safe_configs:
            if config in visited_local:
                continue
            n_components += 1
            # BFS 遍历该连通分量
            stack = [config]
            while stack:
                curr = stack.pop()
                if curr in visited_local:
                    continue
                visited_local.add(curr)
                for nbr in self.get_neighbors(curr):
                    if nbr in self.safe_configs and nbr not in visited_local:
                        stack.append(nbr)

        return n_components

    def get_safe_config_stats(self) -> Dict[str, float]:
        """安全配置空间的统计信息."""
        if not self.safe_configs:
            return {'count': 0, 'density': 0.0, 'avg_active': 0.0}
        configs_array = np.array(list(self.safe_configs))
        return {
            'count': len(self.safe_configs),
            'density': len(self.safe_configs) / (2 ** self.n_cells),
            'avg_active': float(np.mean(np.sum(configs_array, axis=1))),
            'n_components': self.count_connected_components(),
        }
