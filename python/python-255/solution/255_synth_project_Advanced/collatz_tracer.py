# -*- coding: utf-8 -*-
"""
collatz_tracer.py
======================================================================
谱线追溯器 —— 基于 Collatz 结构的层级逆映射

物理背景:
    系外行星大气光谱中, 从观测到的吸收线反推其形成深度
    (contribution function) 是一个层级逆问题:
        观测光谱 -> 吸收线强度 -> 大气温度压力 -> 形成高度

    本模块借用 Collatz 序列 (3n+1 问题) 的数学结构来
    构建层级逆映射树 (来自 196_collatz):

    正向映射 (Collatz 式):
        T_{k+1} = f(T_k)  如果 T_k 为偶数
        T_{k+1} = 3 T_k + 1  如果 T_k 为奇数

    逆向映射 (来自 196_collatz 的 collatz_inverse):
        给定 T_k, 其原像集合为:
            S = {2 T_k} ∪ {(T_k - 1) / 3 if T_k ≡ 4 mod 6}

    在光谱反演中:
        - 每个 "level" 对应一个大气层
        - 逆映射展开可能的母层集合
        - 层级深度由 collatz_count 决定

    物理意义: 吸收线 i 的形成深度 z_i 可以通过
    层级追溯找到所有可能贡献的大气层。

数学公式:
    Collatz 序列长度:
        L(n) = |{k : T_k != 1}| + 1
    Level-k 集合:
        L_k = f^{-1}(L_{k-1}), L_0 = {1}

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ============================================================
# Collatz 序列 (移植自 196_collatz)
# ============================================================
def collatz_sequence(key: int) -> np.ndarray:
    """
    计算 Collatz 序列 (移植自 196_collatz/collatz.m)。

    T_{k+1} = T_k / 2       如果 T_k 为偶数
    T_{k+1} = 3 T_k + 1     如果 T_k 为奇数
    终止于 T_k = 1。
    """
    if key <= 0:
        return np.array([], dtype=int)

    sequence = [key]
    t = key
    max_iter = 10000
    iteration = 0

    while t != 1 and iteration < max_iter:
        if t % 2 == 0:
            t = t // 2
        else:
            t = 3 * t + 1
        sequence.append(t)
        iteration += 1

    return np.array(sequence, dtype=int)


def collatz_count(key: int) -> int:
    """
    Collatz 序列长度 (移植自 196_collatz/collatz_count.m)。
    """
    if key <= 0:
        return -1

    n = 1
    t = key
    max_iter = 10000
    iteration = 0

    while t != 1 and iteration < max_iter:
        if t % 2 == 0:
            t = t // 2
        else:
            t = 3 * t + 1
        n = n + 1
        iteration += 1

    return n


def collatz_max(key: int) -> int:
    """
    Collatz 序列最大值 (移植自 196_collatz/collatz_max.m)。
    """
    if key <= 0:
        return -1

    m = key
    t = key
    max_iter = 10000
    iteration = 0

    while t != 1 and iteration < max_iter:
        if t % 2 == 0:
            t = t // 2
        else:
            t = 3 * t + 1
        m = max(m, t)
        iteration += 1

    return m


def collatz_inverse(tset: np.ndarray) -> np.ndarray:
    """
    Collatz 逆映射 (移植自 196_collatz/collatz_inverse.m)。

    给定集合 T, 其原像集合为:
        S = {2 t : t in T} ∪ {(t-1)/3 : t in T, t ≡ 4 mod 6, t != 4}
    """
    sset = []
    for t in tset:
        sset.append(2 * t)
        if t % 6 == 4 and t != 4:
            sset.append((t - 1) // 3)
    return np.unique(np.array(sset, dtype=int))


def collatz_level(k: int) -> np.ndarray:
    """
    距 1 距离为 k 的所有节点 (移植自 196_collatz/collatz_level.m)。
    """
    lset = np.array([1], dtype=int)
    for _ in range(k):
        lset = collatz_inverse(lset)
        lset = np.sort(lset)
    return lset


# ============================================================
# 谱线层级追溯
# ============================================================
class SpectralLineTracer:
    """
    谱线形成深度追溯器。

    利用 Collatz 层级结构构建逆映射树, 从观测到的吸收线
    反推其形成的大气层。

    物理模型:
        吸收线中心光学深度 tau_0 = 2/3 对应的高度
        定义为该线的 "形成高度" z_form。
        贡献函数:
            C(z) = exp(-tau(z)) * kappa(z) * rho(z)
        归一化: integral C(z) dz = 1
    """

    def __init__(
        self,
        z_grid: np.ndarray,
        p_grid: np.ndarray,
        t_grid: np.ndarray,
        max_level: int = 8,
    ):
        self.z_grid = z_grid
        self.p_grid = p_grid
        self.t_grid = t_grid
        self.n_layers = len(z_grid)
        self.max_level = min(max_level, 10)

    def contribution_function(
        self, tau_profile: np.ndarray, kappa: np.ndarray, rho: np.ndarray
    ) -> np.ndarray:
        """
        计算贡献函数 C(z)。
        """
        C = np.exp(-tau_profile) * kappa * rho
        integral = np.sum(C) * np.abs(self.z_grid[1] - self.z_grid[0])
        if integral > 1.0e-30:
            C = C / integral
        return C

    def find_formation_height(self, C: np.ndarray) -> float:
        """
        找到贡献函数峰值对应的高度 (形成高度)。
        """
        peak_idx = np.argmax(C)
        return float(self.z_grid[peak_idx])

    def trace_lineage(self, line_id: int) -> Dict:
        """
        追溯单条谱线的大气层级来源。

        line_id 作为 Collatz 起始值, 生成序列,
        序列中的每个值对应一个大气层。
        """
        seq = collatz_sequence(max(1, line_id % 1000 + 1))
        n_seq = len(seq)

        layer_indices = []
        for val in seq:
            layer_idx = int(val % self.n_layers)
            layer_indices.append(layer_idx)

        unique_layers = np.unique(layer_indices)
        formation_heights = self.z_grid[unique_layers]
        formation_temps = self.t_grid[unique_layers]
        formation_pressures = self.p_grid[unique_layers]

        return {
            "line_id": line_id,
            "collatz_sequence": seq,
            "sequence_length": n_seq,
            "layer_indices": unique_layers,
            "formation_heights_m": formation_heights,
            "formation_temperatures_K": formation_temps,
            "formation_pressures_Pa": formation_pressures,
            "max_altitude_reached": collatz_max(max(1, line_id % 1000 + 1)),
        }

    def trace_all_lines(self, line_ids: np.ndarray) -> Dict:
        """
        批量追溯多条谱线。
        """
        all_traces = []
        for lid in line_ids:
            trace = self.trace_lineage(int(lid))
            all_traces.append(trace)

        all_heights = np.concatenate([t["formation_heights_m"] for t in all_traces])
        all_temps = np.concatenate([t["formation_temperatures_K"] for t in all_traces])

        return {
            "traces": all_traces,
            "n_lines": len(line_ids),
            "unique_layers_explored": int(
                len(np.unique(np.concatenate([t["layer_indices"] for t in all_traces])))
            ),
            "mean_formation_height_m": float(np.mean(all_heights))
            if len(all_heights) > 0
            else 0.0,
            "mean_formation_temperature_K": float(np.mean(all_temps))
            if len(all_temps) > 0
            else 0.0,
            "height_range_m": (float(np.min(all_heights)), float(np.max(all_heights)))
            if len(all_heights) > 0
            else (0.0, 0.0),
        }


def spectral_line_tracer(
    z_grid: np.ndarray,
    p_grid: np.ndarray,
    t_grid: np.ndarray,
    line_ids: np.ndarray,
    max_level: int = 8,
) -> Dict:
    """
    高层接口: 执行谱线层级追溯。
    """
    tracer = SpectralLineTracer(z_grid, p_grid, t_grid, max_level=max_level)
    return tracer.trace_all_lines(line_ids)
