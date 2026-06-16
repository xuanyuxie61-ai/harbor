#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mode_selector.py  ——  动态规划最优模态选择

融合种子项目:
  - 624_knapsack_dynamic : 0/1 背包动态规划 → 在计算预算内选择最不稳定模态

核心问题:
  给定 M 个候选模态 (MRI, Alfvén, 磁声...), 每个有:
    - 增长率 gamma_i (价值)
    - 计算代价 c_i (权重: 正比于分辨率需求)
  在总计算预算 C 内, 选择模态子集使总增长率最大.

  等价于 0/1 背包:
    max  sum gamma_i x_i
    s.t. sum c_i x_i <= C,  x_i in {0, 1}
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from mhd_constants import MHDConfig


# ============================================================
# 0/1 背包求解 (复刻 624_knapsack_dynamic)
# ============================================================
def knapsack_dp(values: np.ndarray, weights: np.ndarray, capacity: int
                 ) -> Tuple[List[int], np.ndarray]:
    """
    0/1 背包动态规划 (复刻 624_knapsack_dynamic.m).
    values: 物品价值 (N,).
    weights: 物品重量 (N,), 整数.
    capacity: 背包容量.
    返回 (selected_indices, dp_table).
    """
    n = len(values)
    # DP 表: m[i, w] = 前 i 个物品, 容量 w 的最大价值
    m = np.zeros((n + 1, capacity + 1), dtype=float)
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            m[i, w] = m[i - 1, w]
            if weights[i - 1] <= w:
                val = m[i - 1, w - weights[i - 1]] + values[i - 1]
                if val > m[i, w]:
                    m[i, w] = val
    # 回溯
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if m[i, w] > m[i - 1, w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()
    return selected, m


# ============================================================
# MHD 模态选择器
# ============================================================
class ModeSelector:
    """
    从候选模态中选择最不稳定且计算可行的子集.
    """

    def __init__(self, cfg: MHDConfig) -> None:
        self.cfg = cfg

    def select_modes(self, candidate_modes: List[dict],
                      budget: int = 100) -> List[dict]:
        """
        candidate_modes: [{"name": str, "growth_rate": float, "cost": int}, ...]
        budget: 总计算预算.
        返回选中的模态列表.
        """
        if not candidate_modes:
            return []
        n = len(candidate_modes)
        values = np.array([m["growth_rate"] for m in candidate_modes])
        weights = np.array([max(int(m["cost"]), 1) for m in candidate_modes])
        capacity = max(budget, 1)
        # 确保重量为整数
        weights = np.clip(weights, 1, capacity)
        selected_idx, dp_table = knapsack_dp(values, weights, capacity)
        return [candidate_modes[i] for i in selected_idx]

    def rank_modes_by_efficiency(self, candidate_modes: List[dict]
                                   ) -> List[dict]:
        """
        按效率 (growth_rate / cost) 排序.
        """
        modes = []
        for m in candidate_modes:
            eff = m["growth_rate"] / max(m["cost"], 1)
            modes.append({**m, "efficiency": eff})
        modes.sort(key=lambda x: -x["efficiency"])
        return modes
