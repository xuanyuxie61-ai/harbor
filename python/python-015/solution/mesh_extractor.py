"""
mesh_extractor.py
2D/3D网格数据提取与格式转换

凝聚态物理应用：
在能带计算中，需要将k空间数据转换为标准网格格式，
以便进行有限元分析或后处理。

本模块提供：
1. 2D网格提取（基于种子项目789_navier_stokes_mesh2d）
2. 3D网格提取（基于种子项目790_navier_stokes_mesh3d）
3. 节点值到元素值的平均（基于种子项目1340_triangulation_node_to_element）
4. 网格格式转换（基于种子项目1322_triangle_to_xml）

核心数据结构：
- 节点坐标 nodeco[node_num, dim]
- 元素连接 elnode[element_num, nodes_per_element]
- 边界信息 bdynde
"""

import numpy as np
from typing import Tuple, Optional


def mesh2d_extract(nodeco: np.ndarray, elnode: np.ndarray,
                    bdynde: Optional[np.ndarray] = None) -> dict:
    """
    提取2D网格信息
    
    基于种子项目789_navier_stokes_mesh2d中的mesh2d_extract。
    
    Parameters
    ----------
    nodeco : np.ndarray, shape (node_num, 2)
        节点坐标
    elnode : np.ndarray, shape (element_num, 3)
        三角形元素（3节点）
    bdynde : np.ndarray, optional, shape (bdy_num, 2)
        边界边
    
    Returns
    -------
    mesh : dict
        包含网格信息的字典
    """
    node_num = nodeco.shape[0]
    element_num = elnode.shape[0]
    
    mesh = {
        'dim': 2,
        'node_num': node_num,
        'element_num': element_num,
        'nodeco': nodeco,
        'elnode': elnode,
        'bdynde': bdynde,
        'element_order': 3
    }
    
    return mesh


def mesh3d_extract(nodeco: np.ndarray, elnode: np.ndarray,
                    bdynde: Optional[np.ndarray] = None) -> dict:
    """
    提取3D网格信息
    
    基于种子项目790_navier_stokes_mesh3d中的mesh3d_extract。
    
    Parameters
    ----------
    nodeco : np.ndarray, shape (node_num, 3)
    elnode : np.ndarray, shape (element_num, element_order)
        3D元素（如4节点四面体、8节点六面体等）
    bdynde : np.ndarray, optional
    
    Returns
    -------
    mesh : dict
    """
    node_num = nodeco.shape[0]
    element_num = elnode.shape[0]
    element_order = elnode.shape[1]
    
    mesh = {
        'dim': 3,
        'node_num': node_num,
        'element_num': element_num,
        'nodeco': nodeco,
        'elnode': elnode,
        'bdynde': bdynde,
        'element_order': element_order
    }
    
    return mesh


def node_values_to_elements(node_values: np.ndarray, elnode: np.ndarray,
                            element_order: int = 3) -> np.ndarray:
    """
    将节点值平均到元素上
    
    基于种子项目1340_triangulation_node_to_element的核心思想。
    
    公式：V_element = (1/element_order) * sum_{i=1}^{element_order} V_node_i
    
    Parameters
    ----------
    node_values : np.ndarray, shape (node_num,) 或 (node_num, D)
    elnode : np.ndarray, shape (element_num, element_order)
    element_order : int
    
    Returns
    -------
    element_values : np.ndarray, shape (element_num,) 或 (element_num, D)
    """
    element_num = elnode.shape[0]
    node_values = np.asarray(node_values)
    
    if node_values.ndim == 1:
        element_values = np.zeros(element_num)
        for i in range(element_num):
            element_values[i] = np.mean(node_values[elnode[i]])
    else:
        D = node_values.shape[1]
        element_values = np.zeros((element_num, D))
        for i in range(element_num):
            element_values[i] = np.mean(node_values[elnode[i]], axis=0)
    
    return element_values


def mesh_to_xml_string(mesh: dict) -> str:
    """
    将网格数据转换为DOLFIN XML格式字符串
    
    基于种子项目1322_triangle_to_xml中的xml_mesh2d_write。
    
    Parameters
    ----------
    mesh : dict
    
    Returns
    -------
    xml_str : str
    """
    dim = mesh['dim']
    node_num = mesh['node_num']
    element_num = mesh['element_num']
    nodeco = mesh['nodeco']
    elnode = mesh['elnode']
    element_order = mesh.get('element_order', 3)
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append('<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">')
    
    if dim == 2:
        lines.append('  <mesh celltype="triangle" dim="2">')
    else:
        lines.append(f'  <mesh celltype="tetrahedron" dim="{dim}">')
    
    lines.append(f'    <vertices size="{node_num}">')
    for node in range(node_num):
        if dim == 2:
            lines.append(f'      <vertex index="{node}" x="{nodeco[node, 0]:.10g}" y="{nodeco[node, 1]:.10g}"/>')
        else:
            lines.append(f'      <vertex index="{node}" x="{nodeco[node, 0]:.10g}" y="{nodeco[node, 1]:.10g}" z="{nodeco[node, 2]:.10g}"/>')
    lines.append('    </vertices>')
    
    lines.append(f'    <cells size="{element_num}">')
    for elem in range(element_num):
        if dim == 2 and element_order == 3:
            lines.append(f'      <triangle index="{elem}" v0="{elnode[elem, 0]}" v1="{elnode[elem, 1]}" v2="{elnode[elem, 2]}"/>')
        elif dim == 3 and element_order == 4:
            lines.append(f'      <tetrahedron index="{elem}" v0="{elnode[elem, 0]}" v1="{elnode[elem, 1]}" v2="{elnode[elem, 2]}" v3="{elnode[elem, 3]}"/>')
        else:
            # 通用格式
            v_str = ' '.join([f'v{i}="{elnode[elem, i]}"' for i in range(element_order)])
            lines.append(f'      <cell index="{elem}" {v_str}/>')
    lines.append('    </cells>')
    lines.append('  </mesh>')
    lines.append('</dolfin>')
    
    return '\n'.join(lines)


def create_fermi_surface_mesh(k_points: np.ndarray, energies: np.ndarray,
                               e_fermi: float, tolerance: float = 0.1) -> dict:
    """
    从k点数据构建Fermi面网格
    
    选择能量接近E_F的点，构建2D投影网格。
    
    Parameters
    ----------
    k_points : np.ndarray, shape (N, 3)
    energies : np.ndarray, shape (N,)
    e_fermi : float
    tolerance : float
        能量容差
    
    Returns
    -------
    mesh : dict
        2D网格字典
    """
    mask = np.abs(energies - e_fermi) < tolerance
    fs_points = k_points[mask, :2]  # 投影到kx-ky平面
    
    if len(fs_points) < 3:
        return {
            'dim': 2,
            'node_num': len(fs_points),
            'element_num': 0,
            'nodeco': fs_points,
            'elnode': np.zeros((0, 3), dtype=int),
            'bdynde': None,
            'element_order': 3
        }
    
    from triangulation_mesh import delaunay_triangulate_2d
    triangles = delaunay_triangulate_2d(fs_points)
    
    mesh = {
        'dim': 2,
        'node_num': len(fs_points),
        'element_num': len(triangles),
        'nodeco': fs_points,
        'elnode': triangles,
        'bdynde': None,
        'element_order': 3
    }
    
    return mesh
