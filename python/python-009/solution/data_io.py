"""
data_io.py
科学数据读写与格式转换模块。

融合原始项目：1197_tec_io（TEC科学数据文件解析）
             1320_triangle_to_fem（网格格式转换）

在天体物理光谱反演中，需要将模拟或观测光谱数据以结构化格式读写，
同时处理大气网格数据在不同数值格式间的转换。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class TecDataset:
    """
    模拟 TECPLOT 格式的科学数据集容器。

    数据结构:
        - 变量名列表（如 ['WAVELENGTH', 'FLUX', 'ERROR']）
        - 节点数据：N个节点，每个节点有 NVAR 个变量值
        - 单元数据：描述网格拓扑连接关系
    """

    def __init__(self, title: str = "", variables: Optional[List[str]] = None):
        self.title = title
        self.variables = variables or []
        self.node_data: Optional[np.ndarray] = None  # shape (n_nodes, n_vars)
        self.element_nodes: Optional[np.ndarray] = None  # shape (n_elements, element_order)
        self.zone_info: Dict[str, Any] = {}

    def add_variable(self, name: str, values: np.ndarray):
        """向数据集添加一个新变量列。"""
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.node_data is None:
            self.node_data = values.copy()
            self.variables.append(name)
        else:
            if values.shape[0] != self.node_data.shape[0]:
                raise ValueError(f"变量长度不匹配: 已有 {self.node_data.shape[0]} 行，新数据 {values.shape[0]} 行")
            self.node_data = np.hstack([self.node_data, values])
            self.variables.append(name)

    def get_variable(self, name: str) -> np.ndarray:
        """按变量名提取数据列。"""
        if name not in self.variables:
            raise KeyError(f"变量 '{name}' 不存在，可用变量: {self.variables}")
        idx = self.variables.index(name)
        return self.node_data[:, idx]

    def write_ascii(self, filename: str):
        """
        写入简化版 ASCII 数据文件。

        格式:
            TITLE = "..."
            VARIABLES = "var1", "var2", ...
            ZONE N=... E=... ZONETYPE=...
            [节点数据行]
            [单元连接行]
        """
        with open(filename, 'w') as f:
            f.write(f'TITLE = "{self.title}"\n')
            f.write('VARIABLES = ' + ', '.join(f'"{v}"' for v in self.variables) + '\n')
            n_nodes = self.node_data.shape[0] if self.node_data is not None else 0
            n_elements = self.element_nodes.shape[0] if self.element_nodes is not None else 0
            elem_order = self.element_nodes.shape[1] if self.element_nodes is not None else 0
            f.write(f'ZONE N={n_nodes} E={n_elements} ZONETYPE=FETRIANGLE\n')
            if self.node_data is not None:
                for i in range(n_nodes):
                    line = ' '.join(f'{self.node_data[i, j]:.8e}' for j in range(self.node_data.shape[1]))
                    f.write(line + '\n')
            if self.element_nodes is not None:
                for i in range(n_elements):
                    line = ' '.join(str(int(self.element_nodes[i, j])) for j in range(elem_order))
                    f.write(line + '\n')

    @staticmethod
    def read_ascii(filename: str) -> "TecDataset":
        """从 ASCII 文件读取数据集。"""
        ds = TecDataset()
        with open(filename, 'r') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('TITLE'):
                ds.title = line.split('=', 1)[1].strip().strip('"')
            elif line.startswith('VARIABLES'):
                var_part = line.split('=', 1)[1]
                ds.variables = [v.strip().strip('"\'') for v in var_part.split(',')]
            elif line.startswith('ZONE'):
                data_start = i + 1
                break

        n_vars = len(ds.variables)
        data_lines = []
        elem_lines = []
        in_elements = False
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == n_vars and not in_elements:
                try:
                    row = [float(p) for p in parts]
                    data_lines.append(row)
                except ValueError:
                    in_elements = True
                    elem_lines.append([int(p) for p in parts])
            else:
                in_elements = True
                elem_lines.append([int(p) for p in parts])

        if data_lines:
            ds.node_data = np.array(data_lines, dtype=np.float64)
        if elem_lines:
            ds.element_nodes = np.array(elem_lines, dtype=np.int64)
        return ds


def convert_triangle_to_fem(node_coords: np.ndarray, element_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 TRIANGLE 格式网格转换为标准 FEM 格式。

    TRIANGLE 格式特点：
        - 节点编号从 0 或 1 开始
        - 可能包含属性列和边界标记

    FEM 标准格式：
        - 节点坐标数组: shape (n_nodes, dim)
        - 单元节点连接数组: shape (n_elements, element_order)，0-based 索引

    参数:
        node_coords: 节点坐标，shape (n_nodes, dim)
        element_nodes: 单元节点索引，shape (n_elements, element_order)

    返回:
        fem_nodes: 清洗后的节点坐标
        fem_elements: 0-based 单元连接
    """
    node_coords = np.asarray(node_coords, dtype=np.float64)
    element_nodes = np.asarray(element_nodes, dtype=np.int64)

    n_nodes = node_coords.shape[0]

    # 检测并转换为 0-based 索引
    min_idx = element_nodes.min()
    if min_idx == 1:
        element_nodes = element_nodes - 1
    elif min_idx < 0:
        raise ValueError(f"单元节点索引出现负值: {min_idx}")

    max_idx = element_nodes.max()
    if max_idx >= n_nodes:
        raise ValueError(f"单元节点索引越界: 最大索引 {max_idx}，节点总数 {n_nodes}")

    # 去除重复节点
    unique_nodes, inverse = np.unique(np.round(node_coords, decimals=12),
                                       axis=0, return_inverse=True)
    if unique_nodes.shape[0] < n_nodes:
        # 需要重映射单元索引
        new_elements = inverse[element_nodes]
        return unique_nodes, new_elements

    return node_coords, element_nodes


def write_spectrum_ascii(wavelength: np.ndarray, flux: np.ndarray,
                         error: Optional[np.ndarray] = None,
                         filename: str = "spectrum.dat"):
    """
    写入光谱数据为 ASCII 表格。

    格式:
        # WAVELENGTH    FLUX    ERROR(optional)
        value1          value1  value1
        ...
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if error is not None:
        error = np.asarray(error, dtype=np.float64)

    if wavelength.shape != flux.shape:
        raise ValueError("波长和流量数组形状必须一致")
    if error is not None and error.shape != wavelength.shape:
        raise ValueError("误差数组形状必须与波长一致")

    with open(filename, 'w') as f:
        if error is not None:
            f.write("# WAVELENGTH_FLUX_ERROR\n")
            for i in range(len(wavelength)):
                f.write(f"{wavelength[i]:.6e} {flux[i]:.6e} {error[i]:.6e}\n")
        else:
            f.write("# WAVELENGTH_FLUX\n")
            for i in range(len(wavelength)):
                f.write(f"{wavelength[i]:.6e} {flux[i]:.6e}\n")


def read_spectrum_ascii(filename: str) -> Dict[str, np.ndarray]:
    """读取光谱 ASCII 数据文件。"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])

    if not data:
        raise ValueError(f"文件 {filename} 无数据")

    arr = np.array(data, dtype=np.float64)
    result = {}
    if arr.shape[1] >= 2:
        result['wavelength'] = arr[:, 0]
        result['flux'] = arr[:, 1]
    if arr.shape[1] >= 3:
        result['error'] = arr[:, 2]
    return result


def save_json_metadata(metadata: Dict[str, Any], filename: str):
    """保存元数据为 JSON 文件。"""
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def load_json_metadata(filename: str) -> Dict[str, Any]:
    """加载 JSON 元数据。"""
    with open(filename, 'r') as f:
        return json.load(f)
