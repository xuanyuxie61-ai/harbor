"""
molecular_graph.py
==================
分子图构建与谱分析模块

融合种子项目:
  - 756_mesh_vtoe    : 顶点-单元关联映射 (CSR 风格稀疏邻接)
  - 1405_web_matrix  : 稀疏三元组格式、关联矩阵→转移矩阵、幂迭代特征向量

科学背景:
  将分子视为图 G=(V,E)，原子为节点，化学键为边。
  计算图拉普拉斯 L = D - A 及其归一化形式 Ḻ = 2L/λ_max - I，
  为后续 Chebyshev 谱图卷积提供基础。
"""

import numpy as np
from typing import Tuple, List, Dict


class MolecularGraph:
    """
    分子图对象，封装原子特征、键连接、稀疏邻接及谱属性。
    """

    def __init__(self, atoms: np.ndarray, bonds: List[Tuple[int, int, float]],
                 atom_features: np.ndarray):
        """
        Parameters
        ----------
        atoms : np.ndarray, shape (n_atoms, 3)
            原子笛卡尔坐标 (Å)。
        bonds : list of (i, j, order)
            化学键列表，order 为键级 (单键 1.0, 双键 2.0, 三键 3.0, 芳香键 1.5)。
        atom_features : np.ndarray, shape (n_atoms, d)
            原子类型编码（如 one-hot 元素、电负性、范德华半径等）。
        """
        self.atoms = np.asarray(atoms, dtype=np.float64)
        self.bonds = bonds
        self.atom_features = np.asarray(atom_features, dtype=np.float64)
        self.n_atoms = self.atoms.shape[0]
        self.n_bonds = len(bonds)

        # 构建稀疏邻接
        self.adj_coo = self._build_adj_coo()
        self.degree = self._build_degree()
        self.laplacian_coo = self._build_laplacian_coo()
        self.lmax = self._estimate_lmax_power()
        self.normalized_laplacian = self._build_normalized_laplacian()

        # 基于 web_matrix 的关联矩阵与转移矩阵
        self.incidence = self._build_incidence()
        self.transition = self._incidence_to_transition(self.incidence)

        # 构建邻接转移矩阵用于幂迭代 (square matrix, n_atoms × n_atoms)
        self.adj_transition = self._build_adjacency_transition()

        # 幂迭代求主导特征向量 (PageRank 风格原子重要性)
        self.atom_importance = self._power_rank(self.adj_transition, max_iter=100, tol=1e-10)

    # ------------------------------------------------------------------
    # 1. 稀疏邻接 (源自 web_matrix 的 triplet 思想)
    # ------------------------------------------------------------------
    def _build_adj_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 COO 格式的稀疏邻接矩阵 (i, j, v)。
        权重按键级 / r_ij^2 计算，以反映键强度随距离衰减。
        """
        i_list, j_list, v_list = [], [], []
        for (a, b, order) in self.bonds:
            r = np.linalg.norm(self.atoms[a] - self.atoms[b])
            if r < 1e-6:
                r = 1e-6
            w = order / (r ** 2)
            # 无向图，双向存储
            i_list.extend([a, b])
            j_list.extend([b, a])
            v_list.extend([w, w])
        return (np.array(i_list, dtype=np.int32),
                np.array(j_list, dtype=np.int32),
                np.array(v_list, dtype=np.float64))

    def _build_degree(self) -> np.ndarray:
        """度矩阵对角线元素 D_ii = Σ_j A_ij。"""
        deg = np.zeros(self.n_atoms, dtype=np.float64)
        i, j, v = self.adj_coo
        for idx in range(len(v)):
            deg[i[idx]] += v[idx]
        # 防止孤立节点度为零
        deg = np.where(deg < 1e-12, 1e-12, deg)
        return deg

    # ------------------------------------------------------------------
    # 2. 图拉普拉斯 (谱图神经网络基础)
    # ------------------------------------------------------------------
    def _build_laplacian_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        L = D - A，仍以 COO 格式存储。
        """
        i, j, v = self.adj_coo
        # 对角线: D
        diag_i = np.arange(self.n_atoms, dtype=np.int32)
        diag_j = diag_i.copy()
        diag_v = self.degree
        # 非对角线: -A
        off_v = -v
        Li = np.concatenate([diag_i, i])
        Lj = np.concatenate([diag_j, j])
        Lv = np.concatenate([diag_v, off_v])
        return Li, Lj, Lv

    def _estimate_lmax_power(self) -> float:
        """
        用幂迭代估计拉普拉斯最大特征值上界 λ_max。
        迭代公式:  x_{k+1} = L x_k / ||L x_k||
        Rayleigh 商给出特征值估计。
        """
        x = np.random.randn(self.n_atoms)
        x = x / np.linalg.norm(x)
        Li, Lj, Lv = self.laplacian_coo
        for _ in range(80):
            y = np.zeros(self.n_atoms, dtype=np.float64)
            for idx in range(len(Lv)):
                y[Li[idx]] += Lv[idx] * x[Lj[idx]]
            norm = np.linalg.norm(y)
            if norm < 1e-15:
                break
            x = y / norm
        # Rayleigh quotient
        y = np.zeros(self.n_atoms, dtype=np.float64)
        for idx in range(len(Lv)):
            y[Li[idx]] += Lv[idx] * x[Lj[idx]]
        lam = float(np.dot(x, y))
        # 留足 margin
        return max(lam * 1.01, 1e-6)

    def _build_normalized_laplacian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        归一化拉普拉斯用于 Chebyshev 谱卷积:
            Ḻ = 2L / λ_max - I
        这样其特征值落在 [-1, 1]。
        """
        Li, Lj, Lv = self.laplacian_coo
        Lv_norm = 2.0 * Lv / self.lmax
        # 减去单位矩阵: 对角线元素 -1
        mask_diag = (Li == Lj)
        Lv_norm[mask_diag] -= 1.0
        return Li, Lj, Lv_norm

    # ------------------------------------------------------------------
    # 3. 关联矩阵与转移矩阵 (源自 web_matrix)
    # ------------------------------------------------------------------
    def _build_incidence(self) -> np.ndarray:
        """
        节点-边关联矩阵 B (n_atoms × n_bonds)，
        B_{i,e} = +1 若 i 为边 e 的起点, -1 若终点, 0 否则。
        方向按原子序号递增确定。
        """
        B = np.zeros((self.n_atoms, self.n_bonds), dtype=np.float64)
        for eidx, (a, b, _) in enumerate(self.bonds):
            B[a, eidx] = 1.0
            B[b, eidx] = -1.0
        return B

    @staticmethod
    def _incidence_to_transition(B: np.ndarray) -> np.ndarray:
        """
        将关联矩阵转换为列随机转移矩阵 (column-stochastic)。
        先对行归一化再转置，用于幂迭代计算节点重要性。
        T = (D_r^{-1} B)^{T}, 其中 D_r 为行和。
        """
        row_sums = np.abs(B).sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        B_norm = B / row_sums
        return B_norm.T

    def _build_adjacency_transition(self) -> np.ndarray:
        """
        从加权邻接矩阵构建列随机转移矩阵 (square, n_atoms × n_atoms)。
        T_{ji} = A_{ij} / D_{ii}，满足 Σ_j T_{ji} = 1。
        """
        T = np.zeros((self.n_atoms, self.n_atoms), dtype=np.float64)
        for idx in range(len(self.adj_coo[0])):
            i = self.adj_coo[0][idx]
            j = self.adj_coo[1][idx]
            v = self.adj_coo[2][idx]
            T[j, i] = v / self.degree[i]
        return T

    @staticmethod
    def _power_rank(T: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
        """
        幂迭代求主导特征向量 (PageRank 风格原子重要性)。
        收敛判据: ||x_{k+1} - x_k||_2 < tol。
        """
        n = T.shape[0]
        x = np.ones(n, dtype=np.float64) / n
        for _ in range(max_iter):
            x_next = T.dot(x)
            norm = np.linalg.norm(x_next)
            if norm > 0:
                x_next = x_next / norm
            if np.linalg.norm(x_next - x) < tol:
                break
            x = x_next
        # 保证非负
        x = np.abs(x)
        s = x.sum()
        return x / s if s > 0 else x

    # ------------------------------------------------------------------
    # 4. 公共接口
    # ------------------------------------------------------------------
    def apply_normalized_laplacian(self, x: np.ndarray) -> np.ndarray:
        """
        计算 Ḻ @ x，其中 x.shape = (n_atoms, d)。
        """
        Li, Lj, Lv = self.normalized_laplacian
        out = np.zeros_like(x)
        for idx in range(len(Lv)):
            out[Li[idx]] += Lv[idx] * x[Lj[idx]]
        return out

    def adjacency_dense(self) -> np.ndarray:
        """返回稠密邻接矩阵 (仅用于小体系验证)。"""
        A = np.zeros((self.n_atoms, self.n_atoms), dtype=np.float64)
        i, j, v = self.adj_coo
        for idx in range(len(v)):
            A[i[idx], j[idx]] = v[idx]
        return A


def build_demo_molecules() -> List[MolecularGraph]:
    """
    构建若干演示分子: H2O, CH4, C6H6 (苯环简化)。
    返回 MolecularGraph 列表。
    """
    molecules = []

    # --- H2O ---
    # O at (0,0,0), H at (0.96Å, 0, 0) and (-0.24, 0.93, 0)
    atoms_h2o = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ], dtype=np.float64)
    bonds_h2o = [(0, 1, 1.0), (0, 2, 1.0)]
    feats_h2o = np.array([
        [8.0, 3.44, 1.52],   # O: Z=8, EN=3.44, vdw=1.52
        [1.0, 2.20, 1.20],   # H
        [1.0, 2.20, 1.20]
    ], dtype=np.float64)
    molecules.append(MolecularGraph(atoms_h2o, bonds_h2o, feats_h2o))

    # --- CH4 ---
    # 四面体构型，键长 1.09Å
    a = 1.09
    atoms_ch4 = np.array([
        [0.0, 0.0, 0.0],
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=np.float64) / np.sqrt(3.0)
    bonds_ch4 = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)]
    feats_ch4 = np.array([
        [6.0, 2.55, 1.70],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20]
    ], dtype=np.float64)
    molecules.append(MolecularGraph(atoms_ch4, bonds_ch4, feats_ch4))

    # --- C6H6 (苯环, 平面 z=0) ---
    R = 1.40  # C-C 键长近似
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    carbons = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(6)], axis=1)
    hydrogens = np.stack([2.48 * np.cos(angles), 2.48 * np.sin(angles), np.zeros(6)], axis=1)
    atoms_benzene = np.vstack([carbons, hydrogens]).astype(np.float64)
    bonds_benzene = []
    # C-C 环
    for i in range(6):
        bonds_benzene.append((i, (i + 1) % 6, 1.5))  # 芳香键级 1.5
    # C-H
    for i in range(6):
        bonds_benzene.append((i, 6 + i, 1.0))
    feats_benzene = np.vstack([
        np.tile([6.0, 2.55, 1.70], (6, 1)),
        np.tile([1.0, 2.20, 1.20], (6, 1))
    ]).astype(np.float64)
    molecules.append(MolecularGraph(atoms_benzene, bonds_benzene, feats_benzene))

    return molecules
