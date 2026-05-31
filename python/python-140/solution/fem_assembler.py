"""
fem_assembler.py
有限元数据组装与 I/O 模块
负责网格数据的读写、有限元矩阵组装、以及结果数据的格式化输出。
原项目映射:
  - 351_fd_to_tec (有限差分数据到 TECPLOT 格式转换)
  - 1198_tec_to_fem (TECPLOT 到 FEM 模型转换)
"""

import numpy as np
import os


def write_node_file(filename, node_coord):
    """
    写入节点坐标文件。
    映射自 fd_to_tec / tec_to_fem 的数据输出格式。
    """
    node_coord = np.asarray(node_coord, dtype=np.float64)
    if node_coord.ndim == 1:
        node_coord = node_coord.reshape(-1, 1)
    with open(filename, 'w') as f:
        for i in range(node_coord.shape[0]):
            line = "  ".join(f"{node_coord[i, j]:18.10e}" for j in range(node_coord.shape[1]))
            f.write(line + "\n")


def write_element_file(filename, element_node):
    """
    写入单元连接文件。
    """
    element_node = np.asarray(element_node, dtype=np.int64)
    if element_node.ndim == 1:
        element_node = element_node.reshape(1, -1)
    with open(filename, 'w') as f:
        for i in range(element_node.shape[0]):
            line = "  ".join(f"{element_node[i, j] + 1:12d}" for j in range(element_node.shape[1]))
            f.write(line + "\n")


def write_value_file(filename, values):
    """
    写入节点值文件。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    with open(filename, 'w') as f:
        for i in range(values.shape[0]):
            line = "  ".join(f"{values[i, j]:18.10e}" for j in range(values.shape[1]))
            f.write(line + "\n")


def read_node_file(filename):
    """
    读取节点坐标文件。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])
    return np.array(data, dtype=np.float64)


def read_element_file(filename):
    """
    读取单元连接文件（转换为 0-based 索引）。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([int(p) - 1 for p in parts])
    return np.array(data, dtype=np.int64)


def read_value_file(filename):
    """
    读取节点值文件。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])
    return np.array(data, dtype=np.float64)


def assemble_fem_data(prefix, node_coord, element_node, node_values):
    """
    组装 FEM 数据并写入文件。
    映射自 tec_to_fem 的数据组装流程。
    """
    node_file = prefix + "_nodes.txt"
    element_file = prefix + "_elements.txt"
    value_file = prefix + "_values.txt"

    write_node_file(node_file, node_coord)
    write_element_file(element_file, element_node)
    write_value_file(value_file, node_values)

    return node_file, element_file, value_file


def write_tecplot_ascii(filename, node_coord, element_node, node_values, var_names=None):
    """
    写入 TECPLOT ASCII 格式数据文件。
    映射自 fd_to_tec 的输出格式。
    
    格式:
        TITLE = "..."
        VARIABLES = "X", "Y", ..., "T", ...
        ZONE N=... E=... F=FEPOINT ET=TRIANGLE
        [节点数据]
        [单元连接]
    """
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    node_values = np.asarray(node_values, dtype=np.float64)

    if node_coord.ndim == 1:
        node_coord = node_coord.reshape(-1, 1)
    if node_values.ndim == 1:
        node_values = node_values.reshape(-1, 1)

    dim = node_coord.shape[1]
    n_nodes = node_coord.shape[0]
    n_elements = element_node.shape[0]
    n_vars = dim + node_values.shape[1]

    if var_names is None:
        var_names = [f"Var{i + 1}" for i in range(n_vars)]

    with open(filename, 'w') as f:
        f.write(f'TITLE = "{filename}"\n')
        f.write('VARIABLES = ' + ", ".join(f'"{v}"' for v in var_names) + '\n')
        f.write(f'ZONE N={n_nodes} E={n_elements} F=FEPOINT ET=TRIANGLE\n')

        # 节点数据
        for i in range(n_nodes):
            coords = "  ".join(f"{node_coord[i, j]:18.10e}" for j in range(dim))
            vals = "  ".join(f"{node_values[i, j]:18.10e}" for j in range(node_values.shape[1]))
            f.write(coords + "  " + vals + "\n")

        # 单元连接（1-based）
        for i in range(n_elements):
            conn = "  ".join(f"{element_node[i, j] + 1:12d}" for j in range(element_node.shape[1]))
            f.write(conn + "\n")


def compute_fem_mass_matrix(node_coord, element_node):
    """
    计算质量矩阵（一致质量矩阵的 lumped 近似）。
    
    对于三角形单元，面积 A = 0.5 * |det([x2-x1, x3-x1])|
    lumped 质量: M_i = Σ_e (A_e / 3) for each node i in element e
    """
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    n_nodes = node_coord.shape[0]
    M = np.zeros(n_nodes, dtype=np.float64)

    for e in element_node:
        p1, p2, p3 = node_coord[e[0]], node_coord[e[1]], node_coord[e[2]]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        for idx in e:
            M[idx] += area / 3.0

    return M


def compute_fem_stiffness_matrix(node_coord, element_node, kappa_element):
    """
    计算刚度矩阵（泊松方程 -∇·(κ∇u) = f 的 Galerkin 离散）。
    
    对于线性三角形单元，刚度矩阵元素:
        K_{ij} = A_e * κ_e * (∇N_i · ∇N_j)
    其中 ∇N_i = [b_i, c_i] / (2A_e)
    """
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    n_nodes = node_coord.shape[0]
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for idx_e, e in enumerate(element_node):
        p1, p2, p3 = node_coord[e[0]], node_coord[e[1]], node_coord[e[2]]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1e-14:
            continue
        # 形状函数梯度系数
        b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]], dtype=np.float64)
        c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]], dtype=np.float64)
        kappa = kappa_element[idx_e] if hasattr(kappa_element, '__len__') else kappa_element
        for i in range(3):
            for j in range(3):
                K[e[i], e[j]] += kappa * (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)

    return K
