# -*- coding: utf-8 -*-
"""
data_io.py
==========

数据 I/O 模块: 相空间分布、谱数据、探测器配置的读写

对应种子项目:
  - 490_grf_io: 图数据的文件读写 (节点、边、邻接表)
  - 1234_scRNA-seq: 矩阵数据加载与质控 (质量筛选流程)

科学应用
--------
1. 相空间分布函数 f(x, v) 的二进制/text I/O
2. 反冲谱 dR/dE 的保存与加载
3. 探测器配置 (节点-边图结构) 的 GRF 格式读写
4. 质控: 检查数据完整性、NaN 检测、范围验证
"""

from __future__ import annotations
import numpy as np
import json
import os
import struct
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# 第一部分: GRF 图格式 I/O (源自 490_grf_io)
# ---------------------------------------------------------------------------

class GRFData:
    """
    GRF (Graph Format) 数据结构的 Python 实现。

    GRF 文件格式 (源自 490_grf_io):
      # 注释行
      node_id  x  y  neighbor_1 neighbor_2 ...

    应用于探测器模块图:
      节点 = 探测器模块
      边   = 模块间连接/串扰
      坐标 = 物理位置
    """

    def __init__(self):
        self.edge_pointer: List[int] = []  # 每个节点的边起始指针
        self.edge_data: List[int] = []     # 邻接数据
        self.xy: np.ndarray = np.array([]) # 节点坐标

    def write(self, filename: str, node_coords: np.ndarray, adjacency: List[List[int]]):
        """
        写入 GRF 文件 (源自 grf_data_write)。

        Parameters
        ----------
        filename : str
            输出文件名
        node_coords : ndarray, shape (N, 2)
            节点坐标
        adjacency : list of list
            邻接表
        """
        n_nodes = len(node_coords)
        with open(filename, 'w') as f:
            f.write(f"# GRF file: detector geometry\n")
            f.write(f"# nodes: {n_nodes}\n")
            for i in range(n_nodes):
                neighbors = adjacency[i] if i < len(adjacency) else []
                line = f"{i} {node_coords[i, 0]:.6f} {node_coords[i, 1]:.6f}"
                for nb in neighbors:
                    line += f" {nb}"
                f.write(line + "\n")

    def read(self, filename: str) -> Dict:
        """
        读取 GRF 文件 (源自 grf_data_read)。

        返回 {edge_pointer, edge_data, xy}
        """
        edge_pointer = [0]
        edge_data = []
        xy_list = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [float(x) for x in line.split()]
                node_i = int(parts[0])
                x_coord = parts[1]
                y_coord = parts[2]
                while len(xy_list) <= node_i:
                    xy_list.append([0.0, 0.0])
                xy_list[node_i] = [x_coord, y_coord]

                neighbors = [int(p) for p in parts[3:]]
                edge_data.extend(neighbors)
                edge_pointer.append(edge_pointer[-1] + len(neighbors))

        self.edge_pointer = edge_pointer
        self.edge_data = edge_data
        self.xy = np.array(xy_list)

        return {
            'edge_pointer': edge_pointer,
            'edge_data': edge_data,
            'xy': self.xy,
        }


# ---------------------------------------------------------------------------
# 第二部分: 相空间数据 I/O (源自 1234 矩阵加载)
# ---------------------------------------------------------------------------

class PhaseSpaceIO:
    """
    相空间分布函数 f(x, v) 的 I/O (类比 1234 中的 10x mtx 加载)。

    支持格式:
      - text: ASCII 空格分隔
      - binary: numpy .npy
      - json: 含元数据的 JSON
    """

    @staticmethod
    def save_phase_space(
        filename: str,
        f: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        metadata: Optional[Dict] = None,
        fmt: str = 'npy',
    ):
        """保存相空间分布。"""
        data = {
            'f': f,
            'x': x,
            'v': v,
            'metadata': metadata or {},
        }
        if fmt == 'npy':
            np.save(filename + '.npy', data, allow_pickle=True)
        elif fmt == 'text':
            header = f"# phase_space: {f.shape[0]} x {f.shape[1]}\n"
            if metadata:
                header += f"# metadata: {json.dumps(metadata)}\n"
            np.savetxt(filename + '.txt', f, header=header)
        elif fmt == 'json':
            with open(filename + '.json', 'w') as fp:
                json.dump({
                    'f': f.tolist(),
                    'x': x.tolist(),
                    'v': v.tolist(),
                    'metadata': metadata or {},
                }, fp)
        else:
            raise ValueError(f"未知格式: {fmt}")

    @staticmethod
    def load_phase_space(filename: str, fmt: str = 'npy') -> Dict:
        """加载相空间分布。"""
        if fmt == 'npy':
            data = np.load(filename, allow_pickle=True).item()
            return data
        elif fmt == 'text':
            f = np.loadtxt(filename)
            return {'f': f, 'x': None, 'v': None, 'metadata': {}}
        elif fmt == 'json':
            with open(filename, 'r') as fp:
                raw = json.load(fp)
            return {
                'f': np.array(raw['f']),
                'x': np.array(raw['x']),
                'v': np.array(raw['v']),
                'metadata': raw.get('metadata', {}),
            }
        else:
            raise ValueError(f"未知格式: {fmt}")


# ---------------------------------------------------------------------------
# 第三部分: 质控 (源自 1234 质量控制)
# ---------------------------------------------------------------------------

class DataQualityControl:
    """
    数据质控模块 (类比 1234 scRNA-seq 的 QC 流程)。

    检查:
      1. NaN / Inf 检测
      2. 物理范围验证 (f ≥ 0, E ≥ 0)
      3. 归一化一致性
      4. 缺失值填充
    """

    def __init__(self):
        self.issues: List[str] = []

    def check_phase_space(self, f: np.ndarray, x: np.ndarray, v: np.ndarray) -> bool:
        """
        质控相空间分布数据。

        返回 True 表示通过所有检查。
        """
        self.issues = []

        # NaN 检查
        if np.any(np.isnan(f)):
            self.issues.append(f"NaN detected in f: {np.sum(np.isnan(f))} values")
        if np.any(np.isinf(f)):
            self.issues.append(f"Inf detected in f: {np.sum(np.isinf(f))} values")

        # 非负检查 (分布函数)
        if np.any(f < 0):
            self.issues.append(f"Negative values in f: min={np.min(f)}")

        # 网格一致性
        if f.shape != (len(x), len(v)):
            self.issues.append(
                f"Shape mismatch: f={f.shape}, x={len(x)}, v={len(v)}"
            )

        # 单调网格检查
        if len(x) > 1 and not np.all(np.diff(x) > 0):
            self.issues.append("x grid not monotonically increasing")
        if len(v) > 1 and not np.all(np.diff(v) > 0):
            self.issues.append("v grid not monotonically increasing")

        return len(self.issues) == 0

    def check_spectrum(self, E: np.ndarray, rate: np.ndarray) -> bool:
        """质控反冲能谱。"""
        self.issues = []
        if np.any(np.isnan(rate)):
            self.issues.append("NaN in rate spectrum")
        if np.any(rate < 0):
            self.issues.append("Negative rate values")
        if len(E) != len(rate):
            self.issues.append("E and rate length mismatch")
        return len(self.issues) == 0

    def sanitize_spectrum(self, rate: np.ndarray) -> np.ndarray:
        """清洗谱数据: NaN → 0, 负值 → 0。"""
        rate = np.where(np.isnan(rate), 0.0, rate)
        rate = np.where(np.isinf(rate), 0.0, rate)
        rate = np.maximum(rate, 0.0)
        return rate


# ---------------------------------------------------------------------------
# 第四部分: 配置与结果管理
# ---------------------------------------------------------------------------

class ExperimentConfig:
    """
    实验配置 I/O: 参数集的序列化。
    """

    @staticmethod
    def save_config(filename: str, config: Dict):
        """保存实验配置为 JSON。"""
        # 转换 numpy 类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2, default=convert)

    @staticmethod
    def load_config(filename: str) -> Dict:
        """加载实验配置。"""
        with open(filename, 'r') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# 第五部分: 探测器节点-边图构建
# ---------------------------------------------------------------------------

def build_detector_graph(n_modules: int, geometry: str = 'cylinder') -> Dict:
    """
    构建探测器模块的图结构。

    geometry:
      'cylinder': 圆柱分层 (径向 + 轴向)
      'cube': 立方体网格

    返回: {coords, adjacency, n_nodes}
    """
    if geometry == 'cylinder':
        # 简化的分层圆柱
        coords = np.zeros((n_modules, 2))
        adjacency = [[] for _ in range(n_modules)]
        for i in range(n_modules):
            r = 0.5 + 0.3 * (i % 5)
            z = -2.0 + 0.5 * (i // 5)
            coords[i] = [r, z]
            # 连接相邻层
            if i > 0 and (i - 1) // 5 == i // 5:
                adjacency[i].append(i - 1)
                adjacency[i - 1].append(i)
            if i >= 5:
                adjacency[i].append(i - 5)
                adjacency[i - 5].append(i)
    else:
        # 方形网格
        side = int(np.ceil(np.sqrt(n_modules)))
        coords = np.zeros((n_modules, 2))
        adjacency = [[] for _ in range(n_modules)]
        for i in range(n_modules):
            coords[i] = [i % side, i // side]
            if i % side > 0:
                adjacency[i].append(i - 1)
            if i % side < side - 1 and i + 1 < n_modules:
                adjacency[i].append(i + 1)
            if i >= side:
                adjacency[i].append(i - side)
            if i + side < n_modules:
                adjacency[i].append(i + side)

    return {
        'coords': coords,
        'adjacency': adjacency,
        'n_nodes': n_modules,
    }
