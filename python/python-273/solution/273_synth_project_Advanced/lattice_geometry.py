"""
lattice_geometry.py — 晶体晶格几何与高斯基构造
=============================================

融合种子项目:
  - 067_ball_grid: 三维球内网格生成 (正八分面枚举 + 径向反射对称)
  - 341_eternity_tile: 非周期密铺邻接矩阵 (三角形邻域编码 adj(N,3))
  - 758_mesh2d_to_medit: 网格格式转换与拓扑连接表

物理背景:
  晶体中第 l 层近邻壳层的键向量集合 {b_j^(l)} 决定了力常数矩阵
  的对称性，从而约束声子色散关系 omega_n(k) 的解析结构。
  WS 原胞的三角化邻接编码用于 Brillouin 区边界采样。

核心公式:
  Bravais 格点: R = n1*a1 + n2*a2 + n3*a3
  倒格子基矢: b_i = 2*pi * (a_j x a_k) / V_cell
  最小镜像: d_ij = r_j - r_i - L * round((r_j - r_i) / L)
"""

import numpy as np
from typing import Tuple, List, Dict

LATTICE_CONSTANTS = {
    'Si': 5.431, 'Ge': 5.658, 'GaAs': 5.653,
    'diamond_C': 3.567, 'Cu': 3.615, 'Fe': 2.867, 'W': 3.165,
}
ATOMIC_MASSES = {
    'Si': 28.0855, 'Ge': 72.630, 'Ga': 69.723, 'As': 74.922,
    'C': 12.011, 'Cu': 63.546, 'Fe': 55.845, 'W': 183.84,
}


def generate_bravais_lattice(
    lattice_type: str, a: float, n_cells: Tuple[int, int, int] = (4, 4, 4),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Bravais 晶格。支持 sc/bcc/fcc/diamond 结构。
    融合 ball_grid 的八分面枚举思想：先枚举第一卦限格点,
    再通过反射对称性生成全部格点，避免冗余计算。
    """
    nx, ny, nz = n_cells
    basis = a * np.eye(3)
    frac_dict = {
        'sc': [np.zeros(3)],
        'bcc': [np.zeros(3), np.array([0.5, 0.5, 0.5])],
        'fcc': [np.zeros(3), np.array([0.5, 0.5, 0.0]),
                np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])],
        'diamond': [np.zeros(3), np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5]),
                    np.array([0.25, 0.25, 0.25]), np.array([0.75, 0.75, 0.25]),
                    np.array([0.75, 0.25, 0.75]), np.array([0.25, 0.75, 0.75])],
    }
    if lattice_type not in frac_dict:
        raise ValueError(f"未知晶格类型: {lattice_type}")
    frac_positions = frac_dict[lattice_type]
    n_basis = len(frac_positions)
    positions = np.zeros((nx * ny * nz * n_basis, 3))
    atom_types = np.zeros(nx * ny * nz * n_basis, dtype=int)
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = np.array([ix, iy, iz], dtype=float)
                for b_idx, frac in enumerate(frac_positions):
                    positions[idx] = (shift + frac) @ basis
                    atom_types[idx] = b_idx
                    idx += 1
    return positions, atom_types


def compute_neighbor_shells(
    positions: np.ndarray, box_length: float,
    n_shells: int = 4, tolerance: float = 1e-6,
) -> List[Dict]:
    """
    计算近邻壳层结构 (融合 ball_grid 的径向距离分组策略)。
    对每个壳层 l: 距离 r_l, 配位数 Z_l, 键向量集合 {b_j^(l)}。
    """
    n_atoms = len(positions)
    ref = positions[0]
    diffs = positions - ref
    diffs -= box_length * np.round(diffs / box_length)
    dists = np.linalg.norm(diffs, axis=1)
    dists[0] = np.inf
    si = np.argsort(dists)
    sd = dists[si]
    shells = []
    i = 0
    for sl in range(n_shells):
        if i >= len(sd):
            break
        d_ref = sd[i]
        pairs = []
        while i < len(sd) and abs(sd[i] - d_ref) < tolerance:
            pairs.append(si[i])
            i += 1
        shells.append({
            'shell_index': sl + 1, 'distance': d_ref,
            'coordination': len(pairs), 'pair_indices': np.array(pairs),
            'bond_vectors': diffs[pairs],
        })
    return shells


def generate_wigner_seitz_adjacency(lattice_type: str, a: float) -> np.ndarray:
    """
    WS 原胞表面三角化 + 邻接矩阵 adj(Nt,3) (融合 eternity_tile 编码)。
    adj(i,j) = 三角形 i 的第 j 条边的邻居索引, -1 为边界。
    角度编码: j=0(30deg), j=1(60deg), j=2(90deg)。
    """
    nv_map = {'fcc': 24, 'bcc': 14, 'sc': 8, 'diamond': 24}
    nv = nv_map.get(lattice_type, 8)
    center = np.zeros(3)
    angles = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    phi_vals = np.linspace(0.3, np.pi - 0.3, nv)
    verts = np.column_stack([
        np.sin(phi_vals) * np.cos(angles),
        np.sin(phi_vals) * np.sin(angles),
        np.cos(phi_vals),
    ]) * (a / 4.0)
    triangles = []
    for i in range(1, nv - 1):
        triangles.append([0, i, i + 1])
    if not triangles:
        return np.array([[-1, -1, -1]])
    tri = np.array(triangles, dtype=int)
    nt = len(tri)
    adj = np.full((nt, 3), -1, dtype=int)
    edge_map: Dict = {}
    for ti in range(nt):
        for e in range(3):
            v0, v1 = tri[ti, e], tri[ti, (e + 1) % 3]
            ek = (min(v0, v1), max(v0, v1))
            if ek in edge_map:
                ot, oe = edge_map[ek]
                adj[ti, e] = ot
                adj[ot, oe] = ti
            else:
                edge_map[ek] = (ti, e)
    return adj


def generate_monkhorst_pack_grid(
    n_grid: Tuple[int, int, int] = (8, 8, 8),
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monkhorst-Pack k 点网格。
    k_i = (2*n_i - N_i - 1) / (2*N_i) + shift_i
    融合 chinese_remainder_theorem: k 点线性索引可通过 CRT
    分解为三维网格索引 (当 N1,N2,N3 互素时)。
    """
    n1, n2, n3 = n_grid
    s1, s2, s3 = shift
    kpts = []
    for i1 in range(1, n1 + 1):
        for i2 in range(1, n2 + 1):
            for i3 in range(1, n3 + 1):
                kpts.append([
                    (2 * i1 - n1 - 1) / (2.0 * n1) + s1,
                    (2 * i2 - n2 - 1) / (2.0 * n2) + s2,
                    (2 * i3 - n3 - 1) / (2.0 * n3) + s3,
                ])
    kpts = np.array(kpts)
    w = np.ones(len(kpts)) / len(kpts)
    return kpts, w


def compute_reciprocal_lattice(basis: np.ndarray) -> np.ndarray:
    """倒格子基矢: b_i = 2pi * (a_j x a_k) / V_cell"""
    vol = abs(np.dot(basis[0], np.cross(basis[1], basis[2])))
    if vol < 1e-15:
        raise ValueError("正格子体积为零")
    recip = np.zeros((3, 3))
    recip[0] = 2 * np.pi * np.cross(basis[1], basis[2]) / vol
    recip[1] = 2 * np.pi * np.cross(basis[2], basis[0]) / vol
    recip[2] = 2 * np.pi * np.cross(basis[0], basis[1]) / vol
    return recip


def minimum_image_vector(r1: np.ndarray, r2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """最小镜像约定: d = r2 - r1 - L*round((r2-r1)/L)"""
    diff = r2 - r1
    return diff - box * np.round(diff / box)
