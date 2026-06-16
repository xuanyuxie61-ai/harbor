"""
sheath_simplicial_topology.py
=============================
单纯复形拓扑分析模块。

本模块融合种子项目 1208_fergal-murphy_simplicial_emergence_hypergraphs
的单纯复形 (simplicial complex) 和超图 (hypergraph) 理论，
用于分析等离子体鞘层相空间的拓扑结构。

物理背景：
    等离子体鞘层的相空间 (x, v_x) 具有复杂的拓扑结构：
    - 捕获粒子区域形成"岛"结构
    - 相空间流形成特定的连通分量
    - 拓扑不变量（Betti 数）表征结构复杂性

    通过构造单纯复形并计算同调群，可以量化
    鞘层相空间结构的拓扑特征。

核心概念：
    单纯复形 K:
        - 0-单纯形: 顶点 (相空间点)
        - 1-单纯形: 边 (相邻点连接)
        - 2-单纯形: 面 (三角化)

    Betti 数：
        β₀ = 连通分量数
        β₁ = 独立回路数
        β₂ = 空腔数

    Euler 特征：
        χ = Σ (-1)^k β_k = V - E + F

    边界算子：
        ∂_k: C_k → C_{k-1}
        H_k = ker(∂_k) / im(∂_{k+1})
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations
from typing import Tuple, List, Dict, Set, Optional
import math


class SimplicialComplex:
    """
    单纯复形类

    支持：
        - 单纯形添加
        - 边界算子构造
        - Betti 数计算
        - Laplacian 谱分析
    """

    def __init__(self):
        self.simplices: Dict[int, List[tuple]] = {0: [], 1: [], 2: []}
        self._face_map: Dict[tuple, int] = {}

    def add_vertex(self, v: tuple):
        """添加 0-单纯形（顶点）"""
        if len(v) != 1:
            v = (v,) if isinstance(v, (int, float)) else tuple(v[:1])
        if v not in self.simplices[0]:
            self.simplices[0].append(v)

    def add_edge(self, u, v):
        """添加 1-单纯形（边）"""
        edge = tuple(sorted([u, v]))
        if edge not in self.simplices[1]:
            self.simplices[1].append(edge)
            self.add_vertex((u,))
            self.add_vertex((v,))

    def add_triangle(self, u, v, w):
        """添加 2-单纯形（三角形）"""
        tri = tuple(sorted([u, v, w]))
        if tri not in self.simplices[2]:
            self.simplices[2].append(tri)
            self.add_edge(u, v)
            self.add_edge(v, w)
            self.add_edge(u, w)

    def n_simplices(self, k: int) -> int:
        """k-单纯形数量"""
        return len(self.simplices.get(k, []))

    def boundary_matrix(self, k: int) -> np.ndarray:
        """
        构造第 k 阶边界算子矩阵

        ∂_k: C_k → C_{k-1}

        对于 1-边界 (边 → 顶点):
            ∂₁(e_{ij}) = v_j - v_i

        对于 2-边界 (面 → 边):
            ∂₂(f_{ijk}) = e_{jk} - e_{ik} + e_{ij}

        返回：
            B_k: shape (n_{k-1}, n_k) 边界矩阵
        """
        if k <= 0 or k > 2:
            return np.zeros((0, 0))

        n_upper = self.n_simplices(k)
        n_lower = self.n_simplices(k - 1)

        if n_upper == 0 or n_lower == 0:
            return np.zeros((max(n_lower, 1), max(n_upper, 1)))

        B = np.zeros((n_lower, n_upper))

        if k == 1:
            # 边到顶点
            vertex_idx = {v: i for i, v in enumerate(self.simplices[0])}
            for j, edge in enumerate(self.simplices[1]):
                if len(edge) >= 2:
                    v1 = (edge[0],)
                    v2 = (edge[1],)
                    if v1 in vertex_idx:
                        B[vertex_idx[v1], j] = -1.0
                    if v2 in vertex_idx:
                        B[vertex_idx[v2], j] = 1.0

        elif k == 2:
            # 面到边
            edge_idx = {e: i for i, e in enumerate(self.simplices[1])}
            for j, tri in enumerate(self.simplices[2]):
                if len(tri) >= 3:
                    # ∂₂(σ_{012}) = σ_{12} - σ_{02} + σ_{01}
                    edges = [
                        tuple(sorted([tri[1], tri[2]])),
                        tuple(sorted([tri[0], tri[2]])),
                        tuple(sorted([tri[0], tri[1]])),
                    ]
                    signs = [1, -1, 1]
                    for e, sign in zip(edges, signs):
                        if e in edge_idx:
                            B[edge_idx[e], j] = sign

        return B

    def betti_numbers(self) -> Tuple[int, int, int]:
        """
        计算 Betti 数 β₀, β₁, β₂

        β_k = dim(ker ∂_k) - dim(im ∂_{k+1})
            = n_k - rank(∂_k) - rank(∂_{k+1})

        返回：
            (beta_0, beta_1, beta_2)
        """
        betti = []
        for k in range(3):
            n_k = self.n_simplices(k)
            if n_k == 0:
                betti.append(0)
                continue

            rank_k = 0
            rank_kp1 = 0

            if k > 0:
                B_k = self.boundary_matrix(k)
                if B_k.size > 0:
                    rank_k = np.linalg.matrix_rank(B_k, tol=1e-10)

            if k < 2:
                B_kp1 = self.boundary_matrix(k + 1)
                if B_kp1.size > 0:
                    rank_kp1 = np.linalg.matrix_rank(B_kp1, tol=1e-10)

            beta_k = n_k - rank_k - rank_kp1
            betti.append(max(0, beta_k))

        return tuple(betti)

    def euler_characteristic(self) -> int:
        """
        Euler 特征数
            χ = β₀ - β₁ + β₂ = V - E + F
        """
        beta = self.betti_numbers()
        return beta[0] - beta[1] + beta[2]

    def laplacian_spectrum(self, k: int = 0) -> np.ndarray:
        """
        k-阶 combinatorial Laplacian 谱

        L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k

        零特征值的数量 = β_k

        参数：
            k: 阶数

        返回：
            eigenvalues: Laplacian 特征值
        """
        n_k = self.n_simplices(k)
        if n_k == 0:
            return np.array([])

        L = np.zeros((n_k, n_k))

        if k < 2:
            B_kp1 = self.boundary_matrix(k + 1)
            if B_kp1.size > 0 and B_kp1.shape[0] == n_k:
                L += B_kp1 @ B_kp1.T

        if k > 0:
            B_k = self.boundary_matrix(k)
            if B_k.size > 0 and B_k.shape[1] == n_k:
                L += B_k.T @ B_k

        eigenvalues = la.eigvalsh(L)
        return np.sort(np.real(eigenvalues))


def build_sheath_phase_space_complex(
    f_dist: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    threshold: float = 0.1,
) -> SimplicialComplex:
    """
    从鞘层相空间分布函数构造单纯复形

    使用阈值化 + Delaunay 三角化近似

    参数：
        f_dist: shape (Nx, Nv) 分布函数
        x: 空间网格
        v: 速度网格
        threshold: 分布函数阈值

    返回：
        K: 单纯复形
    """
    K = SimplicialComplex()

    Nx = len(x)
    Nv = len(v)

    # 找高分布函数区域
    active = f_dist > threshold * np.max(f_dist)

    # 添加顶点
    vertex_map = {}
    vid = 0
    for i in range(Nx):
        for j in range(Nv):
            if active[i, j]:
                K.add_vertex((vid,))
                vertex_map[(i, j)] = vid
                vid += 1

    # 添加边和面 (4-邻接 + 三角化)
    for (i, j), v_id in vertex_map.items():
        # 右邻
        if (i + 1, j) in vertex_map:
            K.add_edge(v_id, vertex_map[(i + 1, j)])
        # 上邻
        if (i, j + 1) in vertex_map:
            K.add_edge(v_id, vertex_map[(i, j + 1)])
        # 对角
        if (i + 1, j + 1) in vertex_map:
            K.add_edge(v_id, vertex_map[(i + 1, j + 1)])
        # 三角化
        if ((i + 1, j) in vertex_map and (i, j + 1) in vertex_map):
            K.add_triangle(
                v_id,
                vertex_map[(i + 1, j)],
                vertex_map[(i, j + 1)]
            )
        if ((i + 1, j) in vertex_map and (i + 1, j + 1) in vertex_map):
            K.add_triangle(
                v_id,
                vertex_map[(i + 1, j)],
                vertex_map[(i + 1, j + 1)]
            )

    return K


def analyze_topology_emergence(
    f_snapshots: List[np.ndarray],
    x: np.ndarray,
    v: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> dict:
    """
    分析鞘层相空间拓扑演化

    （源自种子项目 1208 的 simplicial emergence 概念）

    跟踪 Betti 数随时间的变化，检测拓扑相变

    参数：
        f_snapshots: 分布函数时间序列
        x, v: 网格
        thresholds: 阈值列表

    返回：
        结果字典
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.5, 10)

    n_snap = len(f_snapshots)
    n_thresh = len(thresholds)

    betti_history = np.zeros((n_snap, n_thresh, 3), dtype=int)
    euler_history = np.zeros((n_snap, n_thresh), dtype=int)

    for t_idx, f in enumerate(f_snapshots):
        for th_idx, th in enumerate(thresholds):
            K = build_sheath_phase_space_complex(f, x, v, threshold=th)
            beta = K.betti_numbers()
            betti_history[t_idx, th_idx, :] = beta
            euler_history[t_idx, th_idx] = K.euler_characteristic()

    # 检测拓扑相变 (Betti 数突变)
    transitions = []
    for th_idx in range(n_thresh):
        for t_idx in range(1, n_snap):
            db0 = abs(betti_history[t_idx, th_idx, 0] - betti_history[t_idx - 1, th_idx, 0])
            db1 = abs(betti_history[t_idx, th_idx, 1] - betti_history[t_idx - 1, th_idx, 1])
            if db0 + db1 > 0:
                transitions.append((t_idx, th_idx, db0, db1))

    return {
        'betti_history': betti_history,
        'euler_history': euler_history,
        'thresholds': thresholds,
        'n_transitions': len(transitions),
        'transitions': transitions[:20],
        'final_betti': betti_history[-1, :, :] if n_snap > 0 else np.zeros((n_thresh, 3), dtype=int),
    }


def persistent_homology_summary(
    K: SimplicialComplex,
) -> dict:
    """
    单纯复形的持续同调摘要

    返回 Laplacian 谱信息和拓扑摘要

    参数：
        K: 单纯复形

    返回：
        摘要字典
    """
    beta = K.betti_numbers()
    chi = K.euler_characteristic()

    # Laplacian 谱
    spectra = {}
    for k in range(3):
        spec = K.laplacian_spectrum(k)
        if len(spec) > 0:
            n_zero = np.sum(spec < 1e-8)
            spectra[k] = {
                'n_simplices': len(spec),
                'n_zero_eigenvalues': int(n_zero),
                'spectral_gap': float(np.min(spec[spec >= 1e-8])) if n_zero < len(spec) else 0.0,
                'max_eigenvalue': float(np.max(spec)),
            }

    return {
        'betti_numbers': beta,
        'euler_characteristic': chi,
        'n_vertices': K.n_simplices(0),
        'n_edges': K.n_simplices(1),
        'n_triangles': K.n_simplices(2),
        'laplacian_spectra': spectra,
    }
