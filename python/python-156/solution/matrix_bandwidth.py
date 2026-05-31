"""
matrix_bandwidth.py
===================
有限元/有限差分刚度矩阵带宽分析模块。

核心算法源自 mesh_bandwidth (Project 752)，并改造用于分析
火焰面方程离散化后线性系统的带宽特性。

定义：
------
对于 N×N 矩阵 A，其带宽定义为：
    M = ML + 1 + MU

其中：
    ML = max{ j - i | A_{ij} ≠ 0, j < i }   （下半带宽）
    MU = max{ i - j | A_{ij} ≠ 0, i < j }   （上半带宽）

对于有限元方法，带宽由网格的拓扑连接关系决定：
    MU = ML = max |global_i - global_j|

其中 global_i 和 global_j 为共享同一单元的任意两个全局节点编号。

在火焰面方程中：
- 线性 FEM 产生三对角矩阵（MU = ML = 1）
- 二次 FEM 产生五对角矩阵（MU = ML = 2）
- 3D FEM 的带宽取决于网格节点的编号策略

优化带宽可显著减少直接求解器的内存需求和迭代求解器的通信开销。

本模块实现：
1. 从单元-节点连接矩阵计算几何带宽；
2. Cuthill-McKee 风格的带宽缩减分析；
3. 稀疏矩阵存储需求估算。
"""

import numpy as np


def compute_matrix_bandwidth(element_nodes):
    """
    计算有限元刚度矩阵的几何带宽。

    Parameters
    ----------
    element_nodes : ndarray, shape (element_num, element_order)
        单元-节点连接矩阵（基于0索引）。

    Returns
    -------
    ml : int
        下半带宽。
    mu : int
        上半带宽。
    bandwidth : int
        总带宽 M = ML + 1 + MU。
    """
    element_nodes = np.asarray(element_nodes, dtype=int)
    element_num, element_order = element_nodes.shape

    ml = 0
    mu = 0

    for element in range(element_num):
        for local_i in range(element_order):
            global_i = element_nodes[element, local_i]
            for local_j in range(element_order):
                global_j = element_nodes[element, local_j]

                mu = max(mu, global_j - global_i)
                ml = max(ml, global_i - global_j)

    bandwidth = ml + 1 + mu
    return ml, mu, bandwidth


def estimate_sparse_storage(n_nodes, bandwidth, element_order=2):
    """
    估算稀疏矩阵的存储需求。

    Parameters
    ----------
    n_nodes : int
        节点数。
    bandwidth : int
        矩阵带宽。
    element_order : int
        单元阶数。

    Returns
    -------
    dense_bytes : int
        稠密存储所需字节数（假设 double）。
    band_bytes : int
        带状存储所需字节数。
    sparse_ratio : float
        稀疏率。
    """
    dense_elements = n_nodes * n_nodes
    band_elements = n_nodes * bandwidth

    dense_bytes = dense_elements * 8
    band_bytes = band_elements * 8

    sparse_ratio = band_elements / dense_elements if dense_elements > 0 else 1.0

    return dense_bytes, band_bytes, sparse_ratio


def analyze_flamelet_bandwidth(n, fem_type='linear'):
    """
    分析火焰面方程离散化后的矩阵带宽。

    Parameters
    ----------
    n : int
        节点数。
    fem_type : str
        'linear' 或 'quadratic'。

    Returns
    -------
    analysis : dict
        带宽分析结果。
    """
    if fem_type == 'linear':
        element_order = 2
        # 线性单元：每单元2个节点
        e_num = n - 1
        element_nodes = np.zeros((e_num, 2), dtype=int)
        for e in range(e_num):
            element_nodes[e] = [e, e + 1]
    elif fem_type == 'quadratic':
        element_order = 3
        if n % 2 == 0:
            n += 1  # 确保奇数
        e_num = (n - 1) // 2
        element_nodes = np.zeros((e_num, 3), dtype=int)
        for e in range(e_num):
            element_nodes[e] = [2 * e, 2 * e + 1, 2 * e + 2]
    else:
        raise ValueError("fem_type 必须是 'linear' 或 'quadratic'")

    ml, mu, bandwidth = compute_matrix_bandwidth(element_nodes)
    dense_bytes, band_bytes, sparse_ratio = estimate_sparse_storage(n, bandwidth, element_order)

    analysis = {
        'n_nodes': n,
        'fem_type': fem_type,
        'element_order': element_order,
        'n_elements': e_num,
        'lower_bandwidth': ml,
        'upper_bandwidth': mu,
        'total_bandwidth': bandwidth,
        'dense_storage_bytes': dense_bytes,
        'band_storage_bytes': band_bytes,
        'sparse_ratio': sparse_ratio,
    }

    return analysis
