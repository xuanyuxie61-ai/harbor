"""
io_utils.py
===========
基于 tec_io (1197_tec_io) 的数据读写框架，
为海气耦合模式提供自定义格式网格数据的输入输出支持。

科学背景
--------
海洋模式输出通常包含：
- 网格节点坐标（经度、纬度、深度）；
- 物理场变量（温度、盐度、流速、温跃层深度等）；
- 元数据（时间戳、模式版本、参数设置等）。

本模块实现一种轻量化的结构化数据格式（类似简化版 TECPLOT），
支持二维/三维有限元/有限差分网格数据的读写，
便于耦合模式各组件间的数据交换与结果存档。

核心公式
--------
1. 数据文件格式（文本）：
   
   TITLE = "ENSO_Coupled_Model_Output"
   VARIABLES = "lon", "lat", "SST", "h_anom", "u_zonal"
   ZONE N=1000, E=500, DATAPACKING=POINT, ZONETYPE=FETRIANGLE
   [节点数据：每行 N 个变量值]
   [单元数据：每行 element_order 个节点索引]

2. 数据校验和（CRC32 简化）：
   checksum = Σ_{i} (i+1) * hash(data_i) mod 2^32

3. NetCDF 风格维度映射：
   对于 (time, depth, lat, lon) 四维场，
   数据布局为行优先：index = t * (nz*ny*nx) + z * (ny*nx) + y * nx + x
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import hashlib


def compute_checksum(data: np.ndarray) -> int:
    """
    计算数据校验和（简化版）。

    使用 NumPy 数组的字节表示的 MD5 哈希值取模。
    """
    byte_data = data.tobytes()
    hash_val = int(hashlib.md5(byte_data).hexdigest(), 16)
    return hash_val % (2 ** 32)


def write_mesh_data(filename: str,
                    title: str,
                    variable_names: List[str],
                    node_coords: np.ndarray,
                    node_data: np.ndarray,
                    element_nodes: Optional[np.ndarray] = None,
                    element_type: str = "FETRIANGLE") -> None:
    """
    将网格数据写入自定义格式文件。

    参数
    ----
    filename : str
        输出文件路径。
    title : str
        数据标题。
    variable_names : List[str]
        变量名列表（前 dim_num 个为坐标变量）。
    node_coords : np.ndarray, shape (dim_num, n_nodes)
        节点坐标。
    node_data : np.ndarray, shape (n_data_vars, n_nodes)
        节点数据。
    element_nodes : np.ndarray, optional
        单元-节点连接关系，shape (element_order, n_elements)。
    element_type : str
        单元类型。
    """
    dim_num = node_coords.shape[0]
    n_nodes = node_coords.shape[1]
    n_data_vars = node_data.shape[0]

    if node_data.shape[1] != n_nodes:
        raise ValueError("node_data and node_coords must have same number of nodes")

    with open(filename, 'w') as f:
        # Title
        f.write(f'TITLE = "{title}"\n')

        # Variables
        f.write('VARIABLES = ')
        vars_all = list(variable_names)
        f.write(', '.join([f'"{v}"' for v in vars_all]))
        f.write('\n')

        # Zone info
        n_elements = 0 if element_nodes is None else element_nodes.shape[1]
        element_order = 0 if element_nodes is None else element_nodes.shape[0]
        f.write(f'ZONE N={n_nodes}, E={n_elements}, DATAPACKING=POINT, ZONETYPE={element_type}\n')

        # Node data
        for i in range(n_nodes):
            line_parts = []
            for d in range(dim_num):
                line_parts.append(f"{node_coords[d, i]:.8e}")
            for v in range(n_data_vars):
                line_parts.append(f"{node_data[v, i]:.8e}")
            f.write(' '.join(line_parts) + '\n')

        # Element connectivity
        if element_nodes is not None:
            for e in range(n_elements):
                line = ' '.join([str(int(element_nodes[o, e]) + 1)  # 1-based indexing
                                 for o in range(element_order)])
                f.write(line + '\n')


def read_mesh_data(filename: str) -> Dict:
    """
    从自定义格式文件读取网格数据。

    参数
    ----
    filename : str
        输入文件路径。

    返回
    ----
    data : dict
        包含 title, variable_names, node_coords, node_data, element_nodes 等。
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # 解析 TITLE
    title = ""
    variable_names = []
    zone_info = {}
    data_start_idx = 0

    for idx, line in enumerate(lines):
        if line.startswith('TITLE'):
            title = line.split('=', 1)[1].strip().strip('"')
        elif line.startswith('VARIABLES'):
            var_part = line.split('=', 1)[1].strip()
            # 解析引号分隔的变量名
            import re
            variable_names = re.findall(r'"([^"]*)"', var_part)
        elif line.startswith('ZONE'):
            # 解析 N=..., E=..., ZONETYPE=...
            parts = line.replace('ZONE', '').strip().split(',')
            for part in parts:
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    zone_info[k.strip()] = v.strip()
            data_start_idx = idx + 1
            break

    dim_num = 2  # 默认二维
    if 'ZONETYPE' in zone_info:
        if zone_info['ZONETYPE'] == 'FETRIANGLE':
            element_order = 3
        elif zone_info['ZONETYPE'] == 'FEQUADRILATERAL':
            element_order = 4
        elif zone_info['ZONETYPE'] == 'FETETRAHEDRON':
            element_order = 4
            dim_num = 3
        elif zone_info['ZONETYPE'] == 'FEBRICK':
            element_order = 8
            dim_num = 3
        else:
            element_order = 3
    else:
        element_order = 3

    n_nodes = int(zone_info.get('N', 0))
    n_elements = int(zone_info.get('E', 0))

    # 推断数据变量数
    n_total_vars = len(variable_names)
    n_data_vars = n_total_vars - dim_num

    # 读取节点数据
    node_coords = np.zeros((dim_num, n_nodes), dtype=float)
    node_data = np.zeros((n_data_vars, n_nodes), dtype=float)

    for i in range(n_nodes):
        line_idx = data_start_idx + i
        if line_idx >= len(lines):
            break
        values = lines[line_idx].split()
        if len(values) < n_total_vars:
            continue
        for d in range(dim_num):
            node_coords[d, i] = float(values[d])
        for v in range(n_data_vars):
            node_data[v, i] = float(values[dim_num + v])

    # 读取单元连接
    element_nodes = None
    if n_elements > 0:
        element_nodes = np.zeros((element_order, n_elements), dtype=int)
        elem_start = data_start_idx + n_nodes
        for e in range(n_elements):
            line_idx = elem_start + e
            if line_idx >= len(lines):
                break
            vals = lines[line_idx].split()
            if len(vals) < element_order:
                continue
            for o in range(element_order):
                element_nodes[o, e] = int(vals[o]) - 1  # 转回 0-based

    return {
        "title": title,
        "variable_names": variable_names,
        "node_coords": node_coords,
        "node_data": node_data,
        "element_nodes": element_nodes,
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "element_order": element_order,
    }


def write_sst_timeseries(filename: str,
                         times: np.ndarray,
                         nino34: np.ndarray,
                         metadata: Optional[Dict] = None) -> None:
    """
    将 Niño 3.4 时间序列写入文本文件。

    格式：
    # metadata key=value
    time_months  nino34_index
    0.0          0.52
    1.0          0.61
    ...
    """
    with open(filename, 'w') as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}={v}\n")
        f.write("time_months nino34_index\n")
        for t, n in zip(times, nino34):
            f.write(f"{t:.4f} {n:.6f}\n")


def read_sst_timeseries(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    从文本文件读取 Niño 3.4 时间序列。

    返回
    ----
    times, nino34, metadata
    """
    metadata = {}
    times = []
    nino = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                kv = line[1:].strip().split('=', 1)
                if len(kv) == 2:
                    metadata[kv[0].strip()] = kv[1].strip()
            elif line.startswith('time'):
                continue  # header
            else:
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    nino.append(float(parts[1]))

    return np.array(times), np.array(nino), metadata
