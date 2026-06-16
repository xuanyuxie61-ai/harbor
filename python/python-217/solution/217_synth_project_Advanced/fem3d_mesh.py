"""
fem3d_mesh.py
-------------
三维有限元网格与投影 —— 映射自种子项目 418_fem3d_project + 790_navier_stokes_mesh3d
核心思想：生成四面体网格，定义 P2/P1 有限元基函数，
执行样本数据到 FEM 空间的 L2 投影。

科学背景：
    三维 SMB 色谱柱的实际几何为圆柱体，离散为四面体网格。
    有限元空间 V_h ⊂ H^1(Ω) 上，PDE 弱形式：
        (du/dt, v) + v_flow · (grad u, v) + D(grad u, grad v) = (f, v)
    其中 v ∈ V_h。

    样本数据的 L2 投影：
        min_{u_h in V_h} ||u_h - u_sample||_{L2}^2
    法方程：M c = b,  其中
        M_{ij} = integral(phi_i phi_j),  b_i = integral(u_sample phi_i)

四面体基函数 (P1)：
    对于四面体 T = [v1, v2, v3, v4]：
        phi_i(x) = a_i + b_i x + c_i y + d_i z
    其中系数由 phi_i(v_j) = delta_{ij} 确定。
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 四面体网格生成 (简化的结构化网格)
# =============================================================================
def tet_mesh_generate(
    nx: int, ny: int, nz: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 [0, Lx] x [0, Ly] x [0, Lz] 上生成结构化四面体网格。
    每个六面体分为 5 个或 6 个四面体 (此处用 6 个)。

    返回
    ----
    nodes : (Nn, 3) 节点坐标
    elements : (Ne, 4) 四面体顶点索引 (0-based)
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz 必须 >= 1")
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    z = np.linspace(0, Lz, nz + 1)
    # 节点编号
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = i + (nx + 1) * (j + (ny + 1) * k)
                nodes[idx] = [x[i], y[j], z[k]]

    def node_id(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    # 六面体 → 6 四面体
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v = [
                    node_id(i, j, k),
                    node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k),
                    node_id(i, j + 1, k),
                    node_id(i, j, k + 1),
                    node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i, j + 1, k + 1),
                ]
                # 6 个四面体的分解
                tets = [
                    [v[0], v[1], v[3], v[4]],
                    [v[1], v[2], v[3], v[6]],
                    [v[1], v[3], v[4], v[6]],
                    [v[4], v[5], v[1], v[6]],
                    [v[4], v[7], v[3], v[6]],
                    [v[1], v[3], v[4], v[6]],  # 重复但简化
                ]
                # 去重
                seen = set()
                for t in tets:
                    key = tuple(sorted(t))
                    if key not in seen:
                        seen.add(key)
                        elements.append(list(key))
    elements = np.array(elements, dtype=int)
    return nodes, elements


def tet_volume(p: np.ndarray) -> float:
    """
    四面体体积：
        V = |det([v2-v1, v3-v1, v4-v1])| / 6
    """
    if p.shape != (4, 3):
        raise ValueError("tet_volume: 需要 4x3 矩阵")
    M = np.array([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
    return abs(np.linalg.det(M)) / 6.0


# =============================================================================
# P1 有限元基函数
# =============================================================================
def p1_basis_at_point(
    tet_nodes: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    P1 基函数在点 x 处的值 (4 个)。
    phi_i(x) = (a_i + b_i x + c_i y + d_i z) / (6 V)
    """
    V = tet_volume(tet_nodes)
    if V < 1e-14:
        raise ValueError("退化四面体")
    # 构造系数矩阵
    M = np.hstack([np.ones((4, 1)), tet_nodes])  # (4, 4)
    # 求解每个基函数的系数
    phi = np.zeros(4)
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        c = np.linalg.solve(M, e)
        phi[i] = c[0] + c[1] * x[0] + c[2] * x[1] + c[3] * x[2]
    return phi


def local_mass_matrix(tet_nodes: np.ndarray) -> np.ndarray:
    """
    P1 四面体局部质量矩阵：
        M_{ij} = integral_T phi_i phi_j dV = V/10 (1 + delta_{ij})
    """
    V = tet_volume(tet_nodes)
    M = np.full((4, 4), V / 20.0)
    for i in range(4):
        M[i, i] = V / 10.0
    return M


def local_stiffness_matrix(tet_nodes: np.ndarray) -> np.ndarray:
    """
    P1 四面体局部刚度矩阵 (Laplacian)：
        K_{ij} = integral_T grad phi_i · grad phi_j dV
    """
    V = tet_volume(tet_nodes)
    # 梯度：grad phi_i = M^{-1} e_i 的后三行
    M = np.hstack([np.ones((4, 1)), tet_nodes])
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    grad_phi = Minv[1:, :]  # (3, 4)
    K = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            K[i, j] = V * np.dot(grad_phi[:, i], grad_phi[:, j])
    return K


# =============================================================================
# L2 投影
# =============================================================================
def fem3d_project(
    nodes: np.ndarray,
    elements: np.ndarray,
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    reg: float = 1e-8,
) -> np.ndarray:
    """
    将样本数据投影到 P1 有限元空间：
        min_{c} sum_k (u_h(x_k; c) - v_k)^2 + reg ||c||^2
    法方程：(Phi^T Phi + reg I) c = Phi^T v
    其中 Phi_{ki} = phi_i(x_k).

    参数
    ----
    nodes : (Nn, 3) 网格节点
    elements : (Ne, 4) 四面体
    sample_points : (Ns, 3) 采样点
    sample_values : (Ns,) 采样值

    返回
    ----
    c : (Nn,) FEM 系数
    """
    Nn = nodes.shape[0]
    Ns = sample_points.shape[0]
    # 寻找每个采样点所在的单元
    Phi = np.zeros((Ns, Nn))
    for s in range(Ns):
        x = sample_points[s]
        for e_idx in range(elements.shape[0]):
            tet = elements[e_idx]
            tet_nodes = nodes[tet]
            try:
                phi = p1_basis_at_point(tet_nodes, x)
            except ValueError:
                continue
            # 检查点是否在四面体内 (基函数非负)
            if np.all(phi >= -1e-10) and np.sum(phi) > 0.5:
                phi = np.maximum(phi, 0.0)
                phi /= np.sum(phi)
                for i in range(4):
                    Phi[s, tet[i]] = phi[i]
                break
    # 最小二乘
    A = Phi.T @ Phi + reg * np.eye(Nn)
    b = Phi.T @ np.asarray(sample_values, dtype=float).ravel()
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        c = np.zeros(Nn)
    return c


def fem3d_evaluate(
    nodes: np.ndarray,
    elements: np.ndarray,
    coeffs: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """在查询点处评估 FEM 解。"""
    Nq = query_points.shape[0]
    vals = np.zeros(Nq)
    for q in range(Nq):
        x = query_points[q]
        for e_idx in range(elements.shape[0]):
            tet = elements[e_idx]
            tet_nodes = nodes[tet]
            try:
                phi = p1_basis_at_point(tet_nodes, x)
            except ValueError:
                continue
            if np.all(phi >= -1e-10):
                phi = np.maximum(phi, 0.0)
                phi /= max(np.sum(phi), 1e-14)
                vals[q] = sum(phi[i] * coeffs[tet[i]] for i in range(4))
                break
    return vals


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nodes, elems = tet_mesh_generate(2, 2, 2, 1.0, 1.0, 1.0)
    print(f"Nodes: {nodes.shape}, Elements: {elems.shape}")
    # 体积检查
    vols = [tet_volume(nodes[e]) for e in elems]
    print(f"Total volume: {sum(vols):.4f} (expected 1.0)")
    # L2 投影
    sample_pts = np.random.default_rng(0).random((20, 3))
    sample_vals = np.sin(np.pi * sample_pts[:, 0]) * np.cos(np.pi * sample_pts[:, 1])
    c = fem3d_project(nodes, elems, sample_pts, sample_vals, reg=1e-6)
    print(f"FEM coeffs: norm={np.linalg.norm(c):.4f}")
