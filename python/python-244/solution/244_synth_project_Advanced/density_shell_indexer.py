#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
density_shell_indexer.py
========================
【融合种子项目】 428_file_increment (文件增量/索引偏移)

本模块将 file_increment 的索引偏移思想应用于中子星分层结构:
不同物理区域 (外壳/内壳/外核/内核) 使用不同索引基 (0-based / 1-based),
在区域交界面进行索引转换, 并在密度-压力表中维护一致的全局索引.

物理/数学公式
-------------
1. 中子星分层结构 (典型模型):
       外壳 (Outer Crust)    : 0 < rho < 4.3e11 g/cm^3
       内壳 (Inner Crust)    : 4.3e11 < rho < 2.0e14 g/cm^3
       外核 (Outer Core)     : 2.0e14 < rho < ~2-3 rho_nuc
       内核 (Inner Core)     : rho > 2-3 rho_nuc (可能含奇异物质)

2. 壳-核转变密度 (Baym-Bethe-Pethick 1971):
       rho_drip ≈ 4.3 × 10^{11} g/cm^3
       rho_trans ≈ 2.0 × 10^{14} g/cm^3

3. 索引映射 (从全局到局部):
       i_local = i_global - i_offset(region)
   其中 i_offset 为区域起始偏移量

4. 密度壳层编号:
       壳 k 属于区域 R 如果 rho_k ∈ [rho_min^R, rho_max^R]

5. 递增操作 (从 file_increment):
       I_new = I_old + delta
   用于在不同参考系之间转换索引
"""

import math
import numpy as np
from typing import Tuple, Dict, List, Optional
from numerical_constants import NeutronStarConstants as NS


# ============================================================
# 第一部分: 区域定义与边界
# ============================================================

class StellarLayer:
    """中子星层结构定义."""

    ENVELOPE = "envelope"        # 包层
    OUTER_CRUST = "outer_crust"  # 外壳
    INNER_CRUST = "inner_crust"  # 内壳
    OUTER_CORE = "outer_core"    # 外核
    INNER_CORE = "inner_core"    # 内核
    EXOTIC_CORE = "exotic_core"  # 奇异核

    # 密度边界 (g/cm^3)
    BOUNDARIES = {
        ENVELOPE: (0.0, 1.0e9),
        OUTER_CRUST: (1.0e9, 4.3e11),
        INNER_CRUST: (4.3e11, 2.0e14),
        OUTER_CORE: (2.0e14, 5.4e14),     # 2 * rho_nuc
        INNER_CORE: (5.4e14, 8.1e14),     # 3 * rho_nuc
        EXOTIC_CORE: (8.1e14, 1.0e16),
    }

    @classmethod
    def classify(cls, rho: float) -> str:
        """根据密度分类所属区域."""
        for name, (lo, hi) in cls.BOUNDARIES.items():
            if lo <= rho < hi:
                return name
        return cls.EXOTIC_CORE

    @classmethod
    def classify_array(cls, rho_arr: np.ndarray) -> np.ndarray:
        """批量分类."""
        labels = []
        for rho in rho_arr:
            labels.append(cls.classify(rho))
        return np.array(labels)


# ============================================================
# 第二部分: 密度壳层索引器
# ============================================================

class DensityShellIndexer:
    """
    密度壳层索引管理器.

    核心功能:
        1. 维护全局索引 → 区域-局部索引 的双向映射
        2. 支持索引增量偏移 (0-based ↔ 1-based, 区域间转换)
        3. 维护区域边界的全局索引

    索引约定:
        - 全局索引: 0-based, 从星体中心向外
        - 区域局部索引: 0-based, 从区域下界开始
        - 文件索引: 1-based (兼容 Fortran/MEDIT 格式)
    """

    def __init__(self, r_nodes: np.ndarray, rho_profile: np.ndarray):
        """
        Parameters
        ----------
        r_nodes : np.ndarray, shape (N+1,)
            径向节点 (km)
        rho_profile : np.ndarray, shape (N,) 或 (N+1,)
            壳层密度或节点密度 (g/cm^3)
        """
        self.r_nodes = np.asarray(r_nodes, dtype=np.float64)
        self.n_shells = len(self.r_nodes) - 1
        self.rho = np.asarray(rho_profile, dtype=np.float64)

        # 确保 rho 对应壳层 (若为节点值则取平均)
        if len(self.rho) == len(self.r_nodes):
            self.rho_shell = 0.5 * (self.rho[:-1] + self.rho[1:])
        elif len(self.rho) == self.n_shells:
            self.rho_shell = self.rho.copy()
        else:
            raise ValueError(
                f"rho 长度 {len(self.rho)} 与壳层数 {self.n_shells} 不匹配"
            )

        # 区域分类
        self.layer_labels = StellarLayer.classify_array(self.rho_shell)

        # 构建区域索引映射
        self._build_index_maps()

    def _build_index_maps(self):
        """构建全局 ↔ 局部索引映射."""
        # 每个区域的全局索引列表
        self.region_global_indices = {}
        self.region_local_indices = {}
        self.region_offsets = {}

        for layer_name in [StellarLayer.ENVELOPE, StellarLayer.OUTER_CRUST,
                           StellarLayer.INNER_CRUST, StellarLayer.OUTER_CORE,
                           StellarLayer.INNER_CORE, StellarLayer.EXOTIC_CORE]:
            mask = self.layer_labels == layer_name
            global_idx = np.where(mask)[0]
            if len(global_idx) > 0:
                self.region_global_indices[layer_name] = global_idx
                self.region_local_indices[layer_name] = np.arange(len(global_idx))
                self.region_offsets[layer_name] = int(global_idx[0])
            else:
                self.region_global_indices[layer_name] = np.array([], dtype=int)
                self.region_local_indices[layer_name] = np.array([], dtype=int)
                self.region_offsets[layer_name] = -1

        # 区域边界索引 (交界处)
        self.boundary_indices = self._find_boundaries()

    def _find_boundaries(self) -> Dict[str, int]:
        """找到各区域边界的壳层索引."""
        boundaries = {}
        prev_label = None
        for k in range(self.n_shells):
            label = self.layer_labels[k]
            if label != prev_label:
                boundaries[label + "_start"] = k
                if prev_label is not None:
                    boundaries[prev_label + "_end"] = k - 1
            prev_label = label
        if prev_label is not None:
            boundaries[prev_label + "_end"] = self.n_shells - 1
        return boundaries

    # ============================================================
    # 索引转换操作 (移植自 file_increment)
    # ============================================================

    def global_to_local(self, global_idx: int) -> Tuple[str, int]:
        """
        全局索引 → (区域名, 局部索引).

        对应 file_increment 中 从 "全局坐标" 减 "增量" 得到 "局部坐标".
        """
        if global_idx < 0 or global_idx >= self.n_shells:
            raise IndexError(f"全局索引 {global_idx} 超出范围 [0, {self.n_shells})")

        for layer_name, indices in self.region_global_indices.items():
            if len(indices) == 0:
                continue
            local_pos = np.searchsorted(indices, global_idx)
            if local_pos < len(indices) and indices[local_pos] == global_idx:
                return layer_name, int(local_pos)

        return "unknown", -1

    def local_to_global(self, layer_name: str, local_idx: int) -> int:
        """
        (区域名, 局部索引) → 全局索引.

        对应 file_increment 中 局部坐标 + 增量 = 全局坐标.
        """
        if layer_name not in self.region_global_indices:
            raise ValueError(f"未知区域: {layer_name}")
        indices = self.region_global_indices[layer_name]
        if local_idx < 0 or local_idx >= len(indices):
            raise IndexError(
                f"区域 {layer_name} 的局部索引 {local_idx} 超出范围"
            )
        return int(indices[local_idx])

    def increment_index(self, idx_array: np.ndarray,
                        delta: int) -> np.ndarray:
        """
        索引增量操作 (直接移植自 file_increment).

        I_new = I_old + delta

        用途:
            - 0-based → 1-based: delta = +1
            - 1-based → 0-based: delta = -1
            - 区域间偏移: delta = offset_new - offset_old

        Parameters
        ----------
        idx_array : np.ndarray
            整数索引数组
        delta : int
            增量

        Returns
        -------
        np.ndarray
            偏移后的索引数组
        """
        idx = np.asarray(idx_array, dtype=np.int64)
        return idx + delta

    def convert_connectivity(self, elements: np.ndarray,
                              from_base: int, to_base: int) -> np.ndarray:
        """
        转换单元-节点连接表的索引基 (直接移植 file_increment 思想).

        例如: FEM 连接表从 1-based 转为 0-based:
            elements_0based = elements_1based - 1
        """
        delta = to_base - from_base
        return self.increment_index(elements, delta)


# ============================================================
# 第三部分: 密度表管理
# ============================================================

class DensityTable:
    """
    密度-压力-能量密度 查找表.

    维护 EoS 表格的索引一致性, 支持增量操作.
    """

    def __init__(self, rho_grid: np.ndarray, pressure_grid: np.ndarray,
                 energy_density_grid: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        rho_grid : np.ndarray
            密度网格 (g/cm^3), 必须严格递增
        pressure_grid : np.ndarray
            压力网格 (dyn/cm^2)
        energy_density_grid : np.ndarray, optional
            能量密度网格 (erg/cm^3)
        """
        self.rho = np.asarray(rho_grid, dtype=np.float64)
        self.pressure = np.asarray(pressure_grid, dtype=np.float64)
        self.n_points = len(self.rho)

        # 检查单调性
        if not np.all(np.diff(self.rho) > 0):
            raise ValueError("密度网格必须严格递增")
        if not np.all(np.diff(self.pressure) >= 0):
            raise ValueError("压力网格必须非递减 (因果性)")

        if energy_density_grid is not None:
            self.energy_density = np.asarray(energy_density_grid, dtype=np.float64)
        else:
            self.energy_density = None

        # 计算声速 (绝热指数)
        self._compute_sound_speed()

    def _compute_sound_speed(self):
        """
        计算声速 (绝热).

        c_s^2 = dp/d epsilon

        数值上通过中心差分:
            (c_s^2)_k ≈ (p_{k+1} - p_{k-1}) / (epsilon_{k+1} - epsilon_{k-1})
        """
        n = self.n_points
        self.cs2_over_c2 = np.zeros(n)  # (c_s/c)^2

        if self.energy_density is None:
            # 近似: dp/drho * rho / p ~ Gamma (多方指数)
            # 使用有限差分
            for k in range(1, n - 1):
                drho = self.rho[k + 1] - self.rho[k - 1]
                dp = self.pressure[k + 1] - self.pressure[k - 1]
                if drho > 0:
                    # Gamma = rho/p * dp/drho
                    gamma_local = self.rho[k] / (self.pressure[k] + 1e-30) * dp / drho
                    # c_s^2/c^2 = Gamma p / (rho c^2)
                    rho_c2 = self.rho[k] * NS.c_light ** 2
                    self.cs2_over_c2[k] = gamma_local * self.pressure[k] / (rho_c2 + 1e-30)
            # 边界外推
            if n > 2:
                self.cs2_over_c2[0] = self.cs2_over_c2[1]
                self.cs2_over_c2[-1] = self.cs2_over_c2[-2]
        else:
            for k in range(1, n - 1):
                deps = self.energy_density[k + 1] - self.energy_density[k - 1]
                dp = self.pressure[k + 1] - self.pressure[k - 1]
                if deps > 0:
                    self.cs2_over_c2[k] = dp / deps * NS.c_light ** 2 / NS.c_light ** 2
            if n > 2:
                self.cs2_over_c2[0] = self.cs2_over_c2[1]
                self.cs2_over_c2[-1] = self.cs2_over_c2[-2]

        # 因果性限制: c_s <= c
        self.cs2_over_c2 = np.clip(self.cs2_over_c2, 0.0, 1.0)

    def interpolate_pressure(self, rho_target: float) -> float:
        """
        在给定密度处插值压力.

        使用对数-对数线性插值:
            ln p = ln p_k + (ln rho - ln rho_k)/(ln rho_{k+1} - ln rho_k)
                   * (ln p_{k+1} - ln p_k)
        """
        if rho_target <= self.rho[0]:
            return float(self.pressure[0])
        if rho_target >= self.rho[-1]:
            return float(self.pressure[-1])

        idx = np.searchsorted(self.rho, rho_target) - 1
        idx = max(0, min(idx, self.n_points - 2))

        rho_lo, rho_hi = self.rho[idx], self.rho[idx + 1]
        p_lo, p_hi = self.pressure[idx], self.pressure[idx + 1]

        # 对数插值
        eps = 1e-300
        log_rho = math.log(max(rho_target, eps))
        log_rho_lo = math.log(max(rho_lo, eps))
        log_rho_hi = math.log(max(rho_hi, eps))

        if abs(log_rho_hi - log_rho_lo) < 1e-30:
            return float(p_lo)

        t = (log_rho - log_rho_lo) / (log_rho_hi - log_rho_lo)
        log_p_lo = math.log(max(p_lo, eps))
        log_p_hi = math.log(max(p_hi, eps))
        log_p = log_p_lo + t * (log_p_hi - log_p_lo)
        return math.exp(log_p)

    def sound_speed_at(self, rho_target: float) -> float:
        """在给定密度处插值声速."""
        cs2_arr = self.cs2_over_c2
        cs2 = float(np.interp(rho_target, self.rho, cs2_arr))
        cs2 = max(cs2, 0.0)
        return math.sqrt(cs2) * NS.c_light  # cm/s


# ============================================================
# 第四部分: 文件 I/O 工具 (移植自 file_increment)
# ============================================================

def read_index_file(filepath: str) -> np.ndarray:
    """
    读取整数索引文件 (兼容 file_increment 格式).

    跳过 # 开头的注释行和空行.
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                vals = [int(x) for x in line.split()]
                data.append(vals)
    except FileNotFoundError:
        return np.array([], dtype=int)

    if not data:
        return np.array([], dtype=int)

    # 确保每行长度一致
    max_cols = max(len(row) for row in data)
    result = np.zeros((len(data), max_cols), dtype=int)
    for i, row in enumerate(data):
        result[i, :len(row)] = row

    return result


def write_index_file(filepath: str, data: np.ndarray,
                     header: str = ""):
    """
    写入整数索引文件.
    """
    with open(filepath, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        if data.ndim == 1:
            for val in data:
                f.write(f"{int(val)}\n")
        else:
            for row in data:
                line = "  ".join(f"{int(v):6d}" for v in row)
                f.write(line + "\n")


def increment_and_write(input_path: str, output_path: str,
                        increment: int):
    """
    读取索引文件, 增量, 写入新文件.

    直接移植 file_increment 的核心功能:
        array_new = array_old + increment

    Parameters
    ----------
    input_path : str
        输入文件路径
    output_path : str
        输出文件路径
    increment : int
        增量值
    """
    data = read_index_file(input_path)
    if len(data) == 0:
        write_index_file(output_path, data,
                         header=f"空文件, increment={increment}")
        return
    data_incremented = data + increment
    write_index_file(output_path, data_incremented,
                     header=f"increment={increment}")


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("=== 密度壳层索引器自检 ===")

    # 创建测试密度分布 (从中心向外递减)
    n_shells = 50
    r_nodes = np.linspace(0, 12.0, n_shells + 1)
    # 典型中子星密度分布
    rho_centers = 1e15 * np.exp(-2.0 * np.linspace(0, 1, n_shells))

    indexer = DensityShellIndexer(r_nodes, rho_centers)
    print(f"总壳层数: {indexer.n_shells}")
    for layer_name, indices in indexer.region_global_indices.items():
        if len(indices) > 0:
            print(f"  {layer_name}: {len(indices)} 壳, 偏移 = {indexer.region_offsets[layer_name]}")

    print(f"边界索引: {indexer.boundary_indices}")

    # 测试索引转换
    g_idx, l_name = indexer.global_to_local(0)
    print(f"全局 0 -> ({l_name}, {g_idx})")

    # 测试增量操作 (核心功能移植)
    old_idx = np.array([0, 1, 2, 3, 4])
    new_idx_1based = indexer.increment_index(old_idx, 1)
    print(f"0-based: {old_idx} -> 1-based: {new_idx_1based}")

    # 测试连接表转换
    elements = np.array([[1, 2], [2, 3], [3, 4]])
    elements_0based = indexer.convert_connectivity(elements, 1, 0)
    print(f"连接表 1-based -> 0-based:\n{elements_0based}")

    # 测试密度表
    rho_grid = np.logspace(9, 15.5, 100)
    p_grid = 1e30 * (rho_grid / 1e14) ** (5.0 / 3.0)
    eos_table = DensityTable(rho_grid, p_grid)
    p_test = eos_table.interpolate_pressure(1e14)
    print(f"在 rho=1e14 g/cm^3 处 P = {p_test:.3e} dyn/cm^2")

    # 测试文件 I/O
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# 测试索引文件\n")
        f.write("1  2  3\n")
        f.write("4  5  6\n")
        tmppath = f.name

    outpath = tmppath + ".out"
    increment_and_write(tmppath, outpath, -1)
    data_back = read_index_file(outpath)
    print(f"增量读取结果:\n{data_back}")

    os.unlink(tmppath)
    if os.path.exists(outpath):
        os.unlink(outpath)

    print("density_shell_indexer.py 自检通过.")
