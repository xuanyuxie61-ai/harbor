"""
mesh_io.py — 格点数据的读写与格式转换
============================================================

本模块处理量子霍尔系统中的格点数据 I/O:

1. 读取: 从文本/二进制文件读取格点坐标和波函数
2. 写出: 将格点数据导出为标准格式
3. 格式转换: 在不同格点表示之间转换

源自 ice_to_medit 项目的网格数据读写框架,
但适配了量子霍尔问题的特殊需求.

支持的格式:
    - 内部格式: NumPy .npz 文件
    - MESH 格式: MEDIT 网格格式 (用于可视化软件)
    - XYZ 格式: 简单的坐标+属性文件
    - VTK 格式: 用于 ParaView 等可视化工具

参考文献:
    [1] Frey, P. "MEDIT: An interactive mesh visualization software"
        INRIA RT-0253 (2001)
"""

import numpy as np
import os
from typing import Dict, Any, Tuple, List, Optional


def save_wavefunction_data(filename: str,
                            x_grid: np.ndarray,
                            y_grid: np.ndarray,
                            eigenvalues: np.ndarray,
                            eigenvectors: np.ndarray,
                            metadata: Optional[Dict] = None) -> None:
    """将波函数数据保存为 NumPy .npz 格式

    Args:
        filename: 输出文件名
        x_grid: x 坐标
        y_grid: y 坐标
        eigenvalues: 本征值
        eigenvectors: 本征态
        metadata: 额外元数据
    """
    save_dict = {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
    }
    if metadata:
        for key, val in metadata.items():
            save_dict[f'meta_{key}'] = np.array([val])

    np.savez(filename, **save_dict)


def load_wavefunction_data(filename: str) -> Dict[str, Any]:
    """读取波函数数据

    Args:
        filename: .npz 文件名
    Returns:
        数据字典
    """
    data = np.load(filename, allow_pickle=True)
    result = {}
    for key in data.files:
        if key.startswith('meta_'):
            result[key[5:]] = data[key][0]
        else:
            result[key] = data[key]
    return result


def export_mesh_format(filename: str,
                        vertices: np.ndarray,
                        triangles: Optional[np.ndarray] = None,
                        vertex_data: Optional[np.ndarray] = None,
                        dim: int = 2) -> None:
    """导出 MEDIT MESH 格式 (源自 ice_to_medit)

    MESH 格式结构:
        MeshVersionFormatted 1
        Dimension
        2
        Vertices
        N
        x1 y1 label1
        x2 y2 label2
        ...
        Triangles (optional)
        M
        v1 v2 v3 label
        ...
        End

    Args:
        filename: 输出文件名
        vertices: 顶点坐标 (N × dim)
        triangles: 三角形连接 (M × 3), 可选
        vertex_data: 顶点标量数据 (N,), 可选
        dim: 空间维度
    """
    N = len(vertices)

    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n\n")
        f.write(f"Dimension\n{dim}\n\n")

        f.write(f"Vertices\n{N}\n")
        for i in range(N):
            coords = ' '.join(f"{vertices[i, d]:.8f}" for d in range(dim))
            label = 1
            f.write(f"  {coords}  {label}\n")

        if triangles is not None:
            M = len(triangles)
            f.write(f"\nTriangles\n{M}\n")
            for i in range(M):
                v = ' '.join(str(triangles[i, j] + 1) for j in range(3))
                f.write(f"  {v}  1\n")

        f.write("\nEnd\n")


def export_xyz_format(filename: str,
                       positions: np.ndarray,
                       values: np.ndarray,
                       header: str = "QHE lattice data") -> None:
    """导出 XYZ 格式

    XYZ 格式:
        N
        header
        x1 y1 z1 value1
        x2 y2 z2 value2
        ...

    Args:
        filename: 输出文件名
        positions: 坐标 (N × 3 或 N × 2)
        values: 标量值 (N,)
        header: 文件头
    """
    N = len(positions)
    dim = positions.shape[1]

    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        f.write(f"{header}\n")
        for i in range(N):
            if dim == 2:
                f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} 0.0 "
                        f"{values[i]:.10e}\n")
            else:
                f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} "
                        f"{positions[i,2]:.6f} {values[i]:.10e}\n")


def lattice_to_mesh(pos_A: np.ndarray, pos_B: np.ndarray,
                     bonds: List) -> Tuple[np.ndarray, np.ndarray]:
    """将格点数据转换为 MESH 格式

    将蜂窝格点的位置和键转换为顶点和三角形.
    每个三角形由三个键连接的顶点构成.

    Args:
        pos_A: A 子晶格位置
        pos_B: B 子晶格位置
        bonds: 键列表
    Returns:
        (vertices, triangles)
    """
    vertices = np.vstack([pos_A, pos_B])
    N_A = len(pos_A)

    # 从键构建三角形 (简单的 Delaunay 近似)
    triangles = []
    bond_set = set(bonds)

    for i, j in bonds:
        # 找共享顶点的第三个键
        for k, l in bonds:
            if k == i and l != j:
                # 检查 (l, j) 或 (j, l) 是否存在
                if (l, j) in bond_set or (j, l) in bond_set:
                    tri = tuple(sorted([i, j, l]))
                    if len(set(tri)) == 3:
                        triangles.append(tri)

    # 去重
    triangles = list(set(triangles))
    triangles = np.array(triangles) if triangles else np.zeros((0, 3), dtype=int)

    return vertices, triangles


def generate_report(results: Dict[str, Any], output_file: str) -> None:
    """生成计算报告文件

    Args:
        results: 计算结果字典
        output_file: 输出文件名
    """
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  量子霍尔效应数值对角化 — 计算报告\n")
        f.write("=" * 60 + "\n\n")

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                f.write(f"{key}: shape={value.shape}, dtype={value.dtype}\n")
                if value.size <= 10:
                    f.write(f"  values: {value}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for k2, v2 in value.items():
                    f.write(f"  {k2}: {v2}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("  报告生成完毕\n")
        f.write("=" * 60 + "\n")
