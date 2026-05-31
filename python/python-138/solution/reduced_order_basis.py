"""
基于 Gram-Schmidt 正交化的微反应器降阶模型基函数构造
=====================================================
对微反应器快照数据 (snapshot) 进行正交化，构造 Proper Orthogonal Decomposition (POD)
基函数，用于加速多参数反应器模拟。

数学框架：
    给定 m 个快照向量 {u_1, u_2, ..., u_m} ∈ ℝ^N，构造标准正交基
    {φ_1, φ_2, ..., φ_r} 使得投影能量最大：

        max Σ_{k=1}^r Σ_{j=1}^m |<u_j, φ_k>|²
        s.t.  <φ_i, φ_j> = δ_{ij}

    等价于对快照矩阵 U = [u_1, ..., u_m] 进行 QR 分解或 SVD 截断。

    本模块实现 Modified Gram-Schmidt (MGS) 以获得更好的数值稳定性：

        for k = 1 .. m:
            v_k = u_k
            for j = 1 .. k-1:
                v_k = v_k - <v_k, φ_j> φ_j
            if ||v_k|| > tol:
                φ_k = v_k / ||v_k||

    正交性误差控制：
        || I - Φ^T Φ ||_F < ε_machine · cond(U)
"""

import numpy as np
from typing import Tuple, List, Optional


class ReducedOrderBasisBuilder:
    """
    基于 Modified Gram-Schmidt 的 POD 基函数构造器。
    """

    def __init__(self, tolerance: float = 1.0e-12):
        self.tol = max(tolerance, 1.0e-15)

    def modified_gram_schmidt(
        self, snapshots: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        对快照矩阵执行 MGS 正交化。

        输入:
            snapshots: 形状 (N, m)，每列为一个快照

        返回:
            basis: 形状 (N, r) 的正交基矩阵
            rank:  有效秩
            orth_error: 正交性误差 ||I - Φ^T Φ||_F
        """
        N, m = snapshots.shape
        if N == 0 or m == 0:
            return np.zeros((N, 0)), 0, 0.0

        # 列归一化初始向量以改善数值稳定性
        U = snapshots.copy().astype(float)
        basis_list = []

        for k in range(m):
            v = U[:, k].copy()
            for j in range(len(basis_list)):
                phi = basis_list[j]
                coeff = np.dot(v, phi)
                v -= coeff * phi
            norm_v = np.linalg.norm(v)
            if norm_v > self.tol:
                phi_k = v / norm_v
                basis_list.append(phi_k)

        rank = len(basis_list)
        if rank == 0:
            return np.zeros((N, 0)), 0, 0.0

        basis = np.column_stack(basis_list)
        # 正交性检验
        G = basis.T @ basis
        I = np.eye(rank)
        orth_error = np.linalg.norm(G - I, "fro")

        return basis, rank, orth_error

    def compute_pod_modes_svd(
        self, snapshots: np.ndarray, n_modes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        通过经济型 SVD 计算 POD 模态，作为 MGS 的交叉验证。

            U = Φ Σ Ψ^T

        取前 r 个左奇异向量作为 POD 基。

        返回:
            pod_basis: (N, r)
            singular_values: (r,)
            energy_ratio: 前 r 阶能量占比 Σ_{i=1}^r σ_i² / Σ σ_i²
        """
        N, m = snapshots.shape
        if N == 0 or m == 0:
            return np.zeros((N, 0)), np.array([]), 0.0

        U_mat, s, Vh = np.linalg.svd(snapshots, full_matrices=False)
        total_energy = np.sum(s ** 2)
        if total_energy < 1.0e-20:
            return U_mat[:, :0], s[:0], 0.0

        if n_modes is None:
            # 自动截断：保留 99.9% 能量
            cum_energy = np.cumsum(s ** 2) / total_energy
            n_modes = int(np.searchsorted(cum_energy, 0.999)) + 1
            n_modes = min(n_modes, len(s))

        n_modes = min(n_modes, len(s))
        pod_basis = U_mat[:, :n_modes]
        energy_ratio = np.sum(s[:n_modes] ** 2) / total_energy
        return pod_basis, s[:n_modes], energy_ratio

    def project_onto_basis(
        self, field: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        """
        将场向量投影到降阶基上：
            a = Φ^T u
        """
        if basis.size == 0:
            return np.array([])
        coeffs = basis.T @ field
        return coeffs

    def reconstruct_from_basis(
        self, coeffs: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        """
        由降阶系数重构场：
            u_r = Φ a
        """
        if basis.size == 0 or len(coeffs) == 0:
            return np.zeros(basis.shape[0])
        return basis @ coeffs

    def compute_reduction_error(
        self, snapshots: np.ndarray, basis: np.ndarray
    ) -> float:
        """
        计算降阶重构的相对误差：
            ε = (1/m) Σ_j ||u_j - Φ Φ^T u_j|| / ||u_j||
        """
        if basis.size == 0:
            return 1.0
        _, m = snapshots.shape
        total_err = 0.0
        for j in range(m):
            u = snapshots[:, j]
            u_proj = self.reconstruct_from_basis(
                self.project_onto_basis(u, basis), basis
            )
            norm_u = np.linalg.norm(u)
            if norm_u > 1.0e-14:
                total_err += np.linalg.norm(u - u_proj) / norm_u
        return total_err / m

    def build_operator_rom(
        self, A_full: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        """
        将全阶算子 A_full 投影到降阶子空间：
            A_r = Φ^T A_full Φ
        """
        if basis.size == 0:
            return np.zeros((0, 0))
        A_rom = basis.T @ A_full @ basis
        return A_rom
