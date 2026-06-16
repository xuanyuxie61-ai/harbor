"""
active_set_manager.py
=====================
基于连通分量标记的活动集识别与管理.

数学背景
--------
在变分不等式的求解中, 活动集 (active set) 方法是最经典的精确方法之一.
对于 NCP: x ≥ 0, F(x) ≥ 0, x^T F(x) = 0,
活动集定义为:
    A(x) = {i : x_i = 0}     (active / 约束紧)
    I(x) = {i : x_i > 0}     (inactive / 约束松)

在 I 上, F_i(x) = 0 (等式约束)
在 A 上, F_i(x) ≥ 0 (不等式约束)

本模块的创新: 使用连通分量标记 (connected component labeling) 的思想
来识别活动集中的"块结构". 在接触力学中, 接触区域 (active set)
通常形成空间连通的斑块, 识别这些连通分量可以:
    1. 加速 Newton 迭代 (块对角结构)
    2. 提供物理洞察 (接触区域的拓扑)
    3. 支持并行求解 (不同分量独立处理)

关键公式
--------
活动集判据 (基于阈值):
    A_k = {i : x_i^k < ε_active}
    I_k = {i : x_i^k ≥ ε_active}

连通分量标记 (一维):
    对于二值数组 A[1..n], 将连续的 1 标记为同一分量.
    C[i] = k 当且仅当 A[i] = 1 且 i 属于第 k 个连续段.

分量间的间隙 (gap):
    gap_k = |{i : A[i] = 0, i 在第 k 和 k+1 个分量之间}|

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ActiveSetComponent:
    """活动集的一个连通分量."""
    component_id: int
    indices: np.ndarray  # 属于该分量的索引
    size: int
    min_index: int
    max_index: int
    local_F_values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ActiveSetPartition:
    """活动集的完整划分."""
    active_indices: np.ndarray
    inactive_indices: np.ndarray
    components: List[ActiveSetComponent]
    n_components: int
    gap_sizes: List[int]


class ConnectedComponentLabeler1D:
    """
    一维连通分量标记器.

    算法 (来自 components_1d 的思想):
        输入: 二值数组 A[1..n] (0 = 背景, 1 = 前景)
        输出: 标记数组 C[1..n], C[i] = 分量编号 (0 表示背景)

        扫描 A, 当 A[i] = 1 且 A[i-1] = 0 时开始新分量;
        当 A[i] = 1 且 A[i-1] = 1 时继续当前分量.

    复杂度: O(n) 时间, O(n) 空间.
    """

    @staticmethod
    def label(binary_array: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        对一维二值数组进行连通分量标记.

        Parameters
        ----------
        binary_array : ndarray (n,)
            0/1 二值数组

        Returns
        -------
        labels : ndarray (n,)
            分量标记, 0 表示背景
        n_components : int
        """
        n = len(binary_array)
        labels = np.zeros(n, dtype=int)
        n_components = 0
        in_component = False

        for i in range(n):
            if binary_array[i] > 0:
                if not in_component:
                    n_components += 1
                    in_component = True
                labels[i] = n_components
            else:
                in_component = False

        return labels, n_components

    @staticmethod
    def get_component_indices(labels: np.ndarray, n_components: int) -> List[np.ndarray]:
        """提取每个分量的索引列表."""
        indices_list = []
        for k in range(1, n_components + 1):
            indices_list.append(np.where(labels == k)[0])
        return indices_list


class ConnectedComponentLabeler2D:
    """
    二维连通分量标记器 (用于网格化 VI 问题).

    算法: 两遍扫描 (two-pass) + Union-Find
        Pass 1: 扫描网格, 为每个前景像素分配临时标记,
                使用 Union-Find 合并等价标记.
        Pass 2: 将临时标记转换为连续编号.

    邻接关系: 4-邻接 (上下左右) 或 8-邻接 (含对角).
    """

    def __init__(self, connectivity: int = 4):
        if connectivity not in (4, 8):
            raise ValueError(f"邻接类型必须为 4 或 8, 得到 {connectivity}")
        self.connectivity = connectivity

    def label(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        对二维网格进行连通分量标记.

        Parameters
        ----------
        grid : ndarray (H, W)
            0/1 二值网格

        Returns
        -------
        labels : ndarray (H, W)
        n_components : int
        """
        H, W = grid.shape
        labels = np.zeros((H, W), dtype=int)
        parent = {}  # Union-Find
        next_label = 1

        # Pass 1
        for i in range(H):
            for j in range(W):
                if grid[i, j] == 0:
                    continue
                neighbors = self._get_neighbors(i, j, H, W)
                neighbor_labels = set()
                for ni, nj in neighbors:
                    if labels[ni, nj] > 0:
                        neighbor_labels.add(labels[ni, nj])

                if not neighbor_labels:
                    labels[i, j] = next_label
                    parent[next_label] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbor_labels)
                    labels[i, j] = min_label
                    for lab in neighbor_labels:
                        self._union(parent, lab, min_label)

        # Pass 2: 展平标记
        unique_labels = set()
        for i in range(H):
            for j in range(W):
                if labels[i, j] > 0:
                    root = self._find(parent, labels[i, j])
                    unique_labels.add(root)
                    labels[i, j] = root

        # 重新编号为 1..K
        label_map = {old: new + 1 for new, old in enumerate(sorted(unique_labels))}
        for i in range(H):
            for j in range(W):
                if labels[i, j] > 0:
                    labels[i, j] = label_map[labels[i, j]]

        n_components = len(unique_labels)
        return labels, n_components

    def _get_neighbors(self, i: int, j: int, H: int, W: int) -> List[Tuple[int, int]]:
        """获取 (i,j) 的已扫描邻居 (仅上方和左方)."""
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if self.connectivity == 8:
            if i > 0 and j > 0:
                neighbors.append((i - 1, j - 1))
            if i > 0 and j < W - 1:
                neighbors.append((i - 1, j + 1))
        return neighbors

    def _find(self, parent: Dict[int, int], x: int) -> int:
        """带路径压缩的 Find."""
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def _union(self, parent: Dict[int, int], x: int, y: int) -> None:
        """Union 操作."""
        rx, ry = self._find(parent, x), self._find(parent, y)
        if rx != ry:
            parent[rx] = ry


class ActiveSetManager:
    """
    变分不等式求解中的活动集管理器.

    功能:
        1. 基于阈值的活动集检测
        2. 活动集分量的连通标记
        3. 活动集变化的检测与跟踪
        4. 基于活动集分量的分块 Newton 步

    活动集更新策略:
        - 保守策略: ε_active 递减 (从松到紧)
        - 激进策略: 一步到位使用最小阈值
        - 自适应策略: 基于残差动态调整 ε
    """

    def __init__(
        self,
        n: int,
        epsilon_active: float = 1e-6,
        grid_shape: Optional[Tuple[int, ...]] = None,
        connectivity: int = 4,
    ):
        self.n = n
        self.epsilon_active = epsilon_active
        self.grid_shape = grid_shape
        self.connectivity = connectivity
        self.labeler_1d = ConnectedComponentLabeler1D()
        if grid_shape is not None and len(grid_shape) == 2:
            self.labeler_2d = ConnectedComponentLabeler2D(connectivity)
        else:
            self.labeler_2d = None

        self.previous_partition: Optional[ActiveSetPartition] = None
        self.change_history: List[int] = []

    def detect_active_set(self, x: np.ndarray, F_x: np.ndarray) -> ActiveSetPartition:
        """
        检测当前迭代的活动集并标记连通分量.

        判据:
            x_i < ε_active  → i ∈ Active set
            x_i ≥ ε_active  → i ∈ Inactive set

        当 F_i(x) 也很小时, 表示处于"退化"状态 (degenerate point).
        """
        if len(x) != self.n:
            raise ValueError(f"输入维度 {len(x)} != {self.n}")

        is_active = (x < self.epsilon_active).astype(int)
        active_idx = np.where(is_active > 0)[0]
        inactive_idx = np.where(is_active == 0)[0]

        # 连通分量标记
        if self.grid_shape is not None and self.labeler_2d is not None:
            grid = is_active.reshape(self.grid_shape)
            labels, n_comp = self.labeler_2d.label(grid)
            labels_flat = labels.flatten()
        else:
            labels_flat, n_comp = self.labeler_1d.label(is_active)

        # 构建分量对象
        components = []
        gap_sizes = []
        for k in range(1, n_comp + 1):
            comp_idx = np.where(labels_flat == k)[0]
            comp = ActiveSetComponent(
                component_id=k,
                indices=comp_idx,
                size=len(comp_idx),
                min_index=int(comp_idx[0]),
                max_index=int(comp_idx[-1]),
                local_F_values=F_x[comp_idx] if len(F_x) == self.n else np.array([]),
            )
            components.append(comp)
            if k < n_comp:
                gap = int(components[-1].indices[-1] - comp.min_index - len(comp_idx) + 1)
                gap_sizes.append(max(0, gap))

        partition = ActiveSetPartition(
            active_indices=active_idx,
            inactive_indices=inactive_idx,
            components=components,
            n_components=n_comp,
            gap_sizes=gap_sizes,
        )

        # 跟踪变化
        if self.previous_partition is not None:
            n_changed = int(np.sum(
                np.abs(
                    (x < self.epsilon_active).astype(int) -
                    (self.previous_partition.active_indices.__len__() > 0)
                )
            ))
            self.change_history.append(n_changed)

        self.previous_partition = partition
        return partition

    def build_block_system(
        self,
        partition: ActiveSetPartition,
        J: np.ndarray,
        residual: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        基于活动集分量构建分块线性系统.

        对每个分量 k, 提取子矩阵 J[A_k, A_k] 和子向量 r[A_k],
        解局部系统 J_k d_k = -r_k.

        Returns
        -------
        blocks : list of (indices, J_block, r_block)
        """
        blocks = []
        for comp in partition.components:
            idx = comp.indices
            J_block = J[np.ix_(idx, idx)]
            r_block = residual[idx]
            blocks.append((idx, J_block, r_block))
        return blocks

    def adaptive_threshold(self, iteration: int, residual_norm: float) -> float:
        """
        自适应调整活动集阈值.

        策略: ε_k = max(ε_min, ε_0 · exp(-α · k) · ||r_k||)

        这确保阈值随迭代递减, 最终达到精确的活动集识别.
        """
        eps_min = 1e-12
        alpha = 0.1
        eps_new = self.epsilon_active * np.exp(-alpha * iteration) * max(residual_norm, 1e-10)
        self.epsilon_active = max(eps_new, eps_min)
        return self.epsilon_active
