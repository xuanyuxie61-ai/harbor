"""
反应坐标空间网格生成与I/O工具
基于 mesh2d_write 核心思想：在二维反应坐标空间生成结构化网格并导出。

在蛋白质折叠自由能景观分析中，常用的反应坐标包括：
- Q : 天然接触分数 (fraction of native contacts)
- RMSD : 与天然态的均方根偏差
- Rg : 回转半径

本模块提供反应坐标计算、2D网格生成及文件导出功能。
"""

import numpy as np
from typing import Tuple, Optional


def compute_rmsd(coords: np.ndarray, native_coords: np.ndarray) -> float:
    """
    计算当前构象与天然态的均方根偏差 (RMSD)。
    
    数学定义:
        RMSD = sqrt( (1/N) * sum_{i=1}^{N} |r_i - r_i^{native}|^2 )
    
    其中 N 为残基数，r_i 为第 i 个残基的坐标。
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, d)
        当前构象坐标，N 个残基，d 维空间。
    native_coords : np.ndarray, shape (N, d)
        天然态构象坐标。
    
    Returns
    -------
    rmsd : float
        非负的 RMSD 值。
    """
    if coords.shape != native_coords.shape:
        raise ValueError("coords and native_coords must have the same shape")
    diff = coords - native_coords
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_radius_of_gyration(coords: np.ndarray, masses: Optional[np.ndarray] = None) -> float:
    """
    计算回转半径 (Radius of Gyration, Rg)。
    
    数学定义:
        Rg^2 = (1/M) * sum_{i=1}^{N} m_i |r_i - r_cm|^2
    
    其中 r_cm 为质心坐标，M = sum m_i 为总质量。
    若所有质量相等，则简化为:
        Rg^2 = (1/N) * sum_{i=1}^{N} |r_i - r_cm|^2
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, d)
        残基坐标。
    masses : np.ndarray, shape (N,), optional
        残基质量，默认为等质量。
    
    Returns
    -------
    rg : float
        回转半径，非负。
    """
    if masses is None:
        masses = np.ones(coords.shape[0])
    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError("Total mass must be positive")
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    diff = coords - center_of_mass
    rg_sq = np.sum(masses * np.sum(diff ** 2, axis=1)) / total_mass
    return float(np.sqrt(max(rg_sq, 0.0)))


def compute_native_contact_fraction(coords: np.ndarray, native_coords: np.ndarray,
                                    contact_cutoff: float = 1.2,
                                    native_cutoff: float = 1.5) -> float:
    """
    计算天然接触分数 Q (fraction of native contacts)。
    
    定义:
        在天然态中，若残基 i 和 j (|i-j| > 2) 的距离小于 native_cutoff，则称 (i,j)
        为一个天然接触对。
        在当前构象中，若该接触对的距离小于 contact_cutoff * d_{ij}^{native}，
        则认为该接触被保持。
    
        Q = (当前保持的接触数) / (总天然接触数)
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, d)
        当前构象坐标。
    native_coords : np.ndarray, shape (N, d)
        天然态坐标。
    contact_cutoff : float
        判定接触是否保持的相对阈值，默认 1.2。
    native_cutoff : float
        判定天然接触的距离阈值，默认 1.5。
    
    Returns
    -------
    q : float
        天然接触分数，范围 [0, 1]。
    """
    N = coords.shape[0]
    native_dists = np.linalg.norm(native_coords[:, np.newaxis, :] - native_coords[np.newaxis, :, :], axis=2)
    current_dists = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    native_contacts = (native_dists < native_cutoff) & (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 2)
    # 排除自身
    np.fill_diagonal(native_contacts, False)
    
    total_native = np.count_nonzero(native_contacts) // 2
    if total_native == 0:
        return 0.0
    
    formed = current_dists < contact_cutoff * native_dists
    formed_contacts = formed & native_contacts
    count_formed = np.count_nonzero(formed_contacts) // 2
    return float(count_formed / total_native)


def generate_reaction_coordinate_grid(q_min: float, q_max: float, nq: int,
                                      rmsd_min: float, rmsd_max: float, nrmsd: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    在二维反应坐标空间 (Q, RMSD) 上生成结构化矩形网格。
    
    网格节点坐标返回为 (nodes, elements)，其中：
    - nodes : shape (nq * nrmsd, 2)，每行为 (Q, RMSD)
    - elements : shape ((nq-1)*(nrmsd-1), 4)，每个矩形单元由4个节点索引组成
    
    注：本模块返回四边形单元；若需三角形单元，可进一步剖分每个四边形为2个三角形。
    
    Parameters
    ----------
    q_min, q_max : float
        Q 坐标范围。
    nq : int
        Q 方向节点数。
    rmsd_min, rmsd_max : float
        RMSD 坐标范围。
    nrmsd : int
        RMSD 方向节点数。
    
    Returns
    -------
    nodes : np.ndarray
        网格节点数组。
    elements : np.ndarray
        四边形单元连接表（0-based索引）。
    """
    if nq < 2 or nrmsd < 2:
        raise ValueError("Grid dimensions must be at least 2 in each direction")
    q_vals = np.linspace(q_min, q_max, nq)
    rmsd_vals = np.linspace(rmsd_min, rmsd_max, nrmsd)
    Q_grid, R_grid = np.meshgrid(q_vals, rmsd_vals, indexing='ij')
    nodes = np.column_stack((Q_grid.ravel(), R_grid.ravel()))
    
    elements = []
    for i in range(nq - 1):
        for j in range(nrmsd - 1):
            n0 = i * nrmsd + j
            n1 = (i + 1) * nrmsd + j
            n2 = (i + 1) * nrmsd + (j + 1)
            n3 = i * nrmsd + (j + 1)
            elements.append([n0, n1, n2, n3])
    elements = np.array(elements, dtype=int)
    return nodes, elements


def write_grid_to_file(nodes: np.ndarray, elements: np.ndarray, label: str,
                       output_dir: str = ".") -> None:
    """
    将反应坐标网格节点和单元信息写入文本文件。
    基于 mesh2d_write 的核心功能，输出 R8MAT 和 I4MAT 格式。
    
    Parameters
    ----------
    nodes : np.ndarray, shape (np, 2)
        节点坐标矩阵。
    elements : np.ndarray, shape (nt, 4)
        四边形单元连接表（或三角形，任意列数）。
    label : str
        文件名前缀。
    output_dir : str
        输出目录。
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    node_file = os.path.join(output_dir, f"{label}_nodes.txt")
    elem_file = os.path.join(output_dir, f"{label}_elements.txt")
    
    np.savetxt(node_file, nodes, fmt="%.8e", header=f"{nodes.shape[0]} {nodes.shape[1]}", comments='')
    np.savetxt(elem_file, elements, fmt="%d", header=f"{elements.shape[0]} {elements.shape[1]}", comments='')


def compute_end_to_end_distance(coords: np.ndarray) -> float:
    """
    计算蛋白质链的末端距离 (End-to-End Distance)。
    
    定义:
        d_ee = |r_N - r_1|
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, d)
        残基坐标。
    
    Returns
    -------
    d : float
        末端距离，非负。
    """
    return float(np.linalg.norm(coords[-1] - coords[0]))


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算四个点定义的二面角 (dihedral angle)，单位为弧度。
    
    定义:
        b1 = p1 - p0,  b2 = p2 - p1,  b3 = p3 - p2
        n1 = normalize(b1 x b2)
        n2 = normalize(b2 x b3)
        m1 = n1 x normalize(b2)
        x = dot(n1, n2)
        y = dot(m1, n2)
        angle = atan2(y, x)
    
    Parameters
    ----------
    p0, p1, p2, p3 : np.ndarray, shape (3,)
        四个连续残基的 C_alpha 坐标（三维）。
    
    Returns
    -------
    angle : float
        二面角，范围 [-pi, pi]。
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-12)
    
    n1 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    
    n2 = np.cross(b2, b3)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    
    m1 = np.cross(n1, b2_norm)
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))
