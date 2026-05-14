# -*- coding: utf-8 -*-
"""
io_utils.py
基于 1322_triangle_to_xml 与 719_matlab_compiler 合成
恒星演化数据的序列化、网格格式转换与文件I/O。
同时包含数值测试用的特殊矩阵生成（魔方阵）。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class IOUtils:
    """
    恒星演化数据 I/O 工具。
    支持自定义二进制格式和文本格式的读写。
    """

    @staticmethod
    def serialize_stellar_model(model_data: Dict[str, Any], filename: str):
        """
        将恒星模型数据序列化为 numpy npz 格式。
        """
        np.savez(filename, **model_data)

    @staticmethod
    def deserialize_stellar_model(filename: str) -> Dict[str, Any]:
        """反序列化恒星模型数据。"""
        data = np.load(filename, allow_pickle=True)
        return {k: data[k] for k in data.files}

    @staticmethod
    def write_evolution_track(times: np.ndarray, luminosities: np.ndarray,
                              radii: np.ndarray, temperatures: np.ndarray,
                              filename: str):
        """
        写入演化轨迹文本文件。
        列：时间[yr] 光度[L_sun] 半径[R_sun] 有效温度[K]
        """
        header = "# time[yr]  L[L_sun]  R[R_sun]  Teff[K]"
        data = np.column_stack([times, luminosities, radii, temperatures])
        np.savetxt(filename, data, header=header, fmt='%.6e')

    @staticmethod
    def read_evolution_track(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """读取演化轨迹。"""
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    @staticmethod
    def grid_to_mass_coordinates(node_r: np.ndarray, element_nodes: np.ndarray,
                                 node_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将径向网格节点和单元连接关系转换到质量坐标。
        基于 1322_triangle_to_xml 的网格拓扑管理思想。
        
        node_r : 节点径向坐标
        element_nodes : 单元-节点连接 (N_element, N_node_per_element)
        node_mass : 节点质量坐标
        """
        node_r = np.asarray(node_r, dtype=np.float64)
        node_mass = np.asarray(node_mass, dtype=np.float64)
        element_nodes = np.asarray(element_nodes, dtype=int)
        # 0-based 索引安全检查
        max_idx = element_nodes.max()
        if max_idx >= len(node_r):
            element_nodes = element_nodes - 1  # 尝试 1-based 转 0-based
        return node_mass, element_nodes

    @staticmethod
    def magic_square(n: int) -> np.ndarray:
        """
        生成 n×n 魔方阵（基于 719_matlab_compiler 的 magic 思想）。
        用于数值测试和稳定性验证矩阵。
        
        魔方阵性质：每行、每列、对角线之和为 n(n²+1)/2。
        Siamese 方法（适用于奇数阶）：
          从顶行中间开始，向右上移动；
          若越界则绕回；若已占则向下移一格。
        """
        if n < 3 or n % 2 == 0:
            # 对偶数阶使用简单填充
            return np.arange(1, n * n + 1).reshape(n, n)
        M = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            M[i, j] = num
            new_i, new_j = (i - 1) % n, (j + 1) % n
            if M[new_i, new_j] != 0:
                i = (i + 1) % n
            else:
                i, j = new_i, new_j
        return M

    @staticmethod
    def test_matrix_condition(n: int = 5) -> Tuple[np.ndarray, float]:
        """
        生成测试矩阵并计算条件数（用于验证数值稳定性）。
        """
        M = IOUtils.magic_square(n).astype(np.float64)
        cond = np.linalg.cond(M)
        return M, cond
