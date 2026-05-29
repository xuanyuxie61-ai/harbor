# -*- coding: utf-8 -*-
"""
stiffness_matrix.py
===================
有限元刚度矩阵组装、稀疏格式转换、带状矩阵高效求解与残差分析模块。

源自种子项目：
  - 508_hb_to_mm（Harwell-Boeing ↔ Matrix Market 稀疏格式转换）
  - 972_r8but（带状上三角矩阵专用运算）
  - 688_linpack_bench_backslash（稠密线性系统求解与残差分析）

科学背景：
---------
在复合材料渐进损伤分析中，每个载荷步都需要求解线性化系统：
  K_tan(d) * Δu = R(u, d)
其中 K_tan 为含损伤的切线刚度矩阵，R 为残差力向量。

对于一维杆单元（两节点线性形函数），单元刚度矩阵为：
  k_e = (E_e * A / h_e) * [[ 1, -1],
                            [-1,  1]]
其中 E_e = E_0 * (1 - d_e)^2 + η 为含损伤退化后的模量。

全局刚度矩阵通过 Direct Stiffness Method 组装：
  K = Σ_e C_e^T * k_e * C_e
其中 C_e 为单元-全局自由度映射矩阵。

本模块提供：
  1. 一维 FEM 刚度矩阵组装（含损伤退化）；
  2. 稀疏矩阵 CSC/CSR/COO 格式转换（类比 HB↔MM）；
  3. 带状上三角矩阵存储与专用回代求解器（类比 r8but）；
  4. 线性系统求解残差分析与条件数估计（类比 linpack bench）。
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, norm as spnorm
from typing import Optional, Tuple


class StiffnessMatrixAssembler1D:
    """
    一维有限元刚度矩阵组装器。
    """

    def __init__(self, nodes: np.ndarray, A: float = 1.0, E0: float = 100e9):
        """
        Parameters
        ----------
        nodes : np.ndarray
            节点坐标数组，shape (n_nodes,)。
        A : float
            横截面积 [m²].
        E0 : float
            无损弹性模量 [Pa].
        """
        self.nodes = np.asarray(nodes)
        self.n_nodes = len(nodes)
        self.A = A
        self.E0 = E0
        self.n_elements = self.n_nodes - 1
        self.elem_sizes = np.diff(nodes)
        if np.any(self.elem_sizes <= 0):
            raise ValueError("Node coordinates must be strictly increasing.")

    def assemble_global_stiffness(self, damage: Optional[np.ndarray] = None,
                                   E_override: Optional[np.ndarray] = None) -> csr_matrix:
        """
        组装全局刚度矩阵 K。

        Parameters
        ----------
        damage : np.ndarray or None
            单元损伤变量 d_e，shape (n_elements,)；None 表示无损。
        E_override : np.ndarray or None
            直接指定各单元弹性模量，覆盖默认计算。

        Returns
        -------
        K : csr_matrix
            全局刚度矩阵 (n_nodes, n_nodes)。
        """
        ne = self.n_elements
        nn = self.n_nodes

        if E_override is not None:
            E_elem = np.asarray(E_override)
        else:
            E_elem = np.full(ne, self.E0)
            if damage is not None:
                damage = np.clip(np.asarray(damage), 0.0, 1.0)
                g_d = (1.0 - damage) ** 2 + 1e-6
                E_elem *= g_d

        rows = []
        cols = []
        data = []

        for e in range(ne):
            h_e = self.elem_sizes[e]
            k = E_elem[e] * self.A / h_e
            # 单元刚度矩阵 [k, -k; -k, k]
            local_dofs = [e, e + 1]
            local_k = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
            for i in range(2):
                for j in range(2):
                    rows.append(local_dofs[i])
                    cols.append(local_dofs[j])
                    data.append(local_k[i, j])

        K_coo = coo_matrix((data, (rows, cols)), shape=(nn, nn))
        return K_coo.tocsr()

    def apply_boundary_conditions(self, K: csr_matrix, F: np.ndarray,
                                   fixed_dofs: np.ndarray,
                                   penalty: float = 1e12) -> Tuple[csr_matrix, np.ndarray]:
        """
        使用罚函数法施加位移边界条件。
        对固定自由度 dof：K[dof,dof] += penalty，F[dof] += penalty * u_prescribed。
        这里假设固定位移为 0。

        Parameters
        ----------
        K : csr_matrix
        F : np.ndarray
        fixed_dofs : np.ndarray
            固定自由度索引。
        penalty : float
            罚因子（应远大于刚度矩阵元素量级）。

        Returns
        -------
        K_bc, F_bc
        """
        K = K.copy()
        F = F.copy()
        for dof in fixed_dofs:
            K[dof, dof] += penalty
            F[dof] += penalty * 0.0
        return K, F


class SparseMatrixConverter:
    """
    稀疏矩阵格式转换器（类比 hb_to_mm 的核心思想）。
    支持 COO ↔ CSR ↔ CSC 的互转，以及简单的 Harwell-Boeing 风格解析。
    """

    @staticmethod
    def to_coo(K: csr_matrix) -> coo_matrix:
        return K.tocoo()

    @staticmethod
    def to_csc(K: csr_matrix) -> csc_matrix:
        return K.tocsc()

    @staticmethod
    def to_matrix_market_string(K: coo_matrix) -> str:
        """
        将 COO 矩阵输出为 Matrix Market 坐标格式字符串（仅实数非对称）。
        """
        nnz = K.nnz
        lines = ["%%MatrixMarket matrix coordinate real general",
                 f"{K.shape[0]} {K.shape[1]} {nnz}"]
        for i, j, v in zip(K.row, K.col, K.data):
            lines.append(f"{i+1} {j+1} {v:.16e}")
        return "\n".join(lines)

    @staticmethod
    def from_matrix_market_string(mm_string: str) -> coo_matrix:
        """解析 Matrix Market 坐标格式字符串。"""
        lines = mm_string.strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("%")]
        header = data_lines[0].split()
        n_rows, n_cols, nnz = int(header[0]), int(header[1]), int(header[2])
        rows = []
        cols = []
        data = []
        for line in data_lines[1:]:
            parts = line.split()
            rows.append(int(parts[0]) - 1)
            cols.append(int(parts[1]) - 1)
            data.append(float(parts[2]))
        return coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    @staticmethod
    def bandwidth(K: csr_matrix) -> Tuple[int, int]:
        """计算矩阵的下半带宽和上半带宽。"""
        K_coo = K.tocoo()
        lower = 0
        upper = 0
        for i, j in zip(K_coo.row, K_coo.col):
            if i > j:
                lower = max(lower, i - j)
            elif j > i:
                upper = max(upper, j - i)
        return lower, upper


class BandedUpperTriangularSolver:
    """
    带状上三角矩阵（R8BUT 风格）专用求解器。
    适用于一维 FEM 经 LU 分解后的上三角因子。
    """

    def __init__(self, n: int, mu: int):
        """
        Parameters
        ----------
        n : int
            矩阵阶数。
        mu : int
            上半带宽（主对角线以上包含 mu 条对角线）。
        """
        self.n = n
        self.mu = mu
        # 紧凑存储：(mu+1) x n 数组
        self.data = np.zeros((mu + 1, n))

    def from_dense(self, U: np.ndarray):
        """从稠密上三角矩阵提取带状存储。"""
        U = np.asarray(U)
        if U.shape != (self.n, self.n):
            raise ValueError("Dense matrix shape mismatch.")
        for j in range(self.n):
            for i in range(max(0, j - self.mu), j + 1):
                self.data[self.mu + i - j, j] = U[i, j]

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        带状上三角回代求解 U x = b。
        数值稳定，带除零保护。
        """
        b = np.asarray(b, dtype=float).copy()
        x = np.zeros(self.n)
        for j in range(self.n - 1, -1, -1):
            diag = self.data[self.mu, j]
            if abs(diag) < 1e-30:
                diag = 1e-30 * np.sign(diag) if diag != 0 else 1e-30
            # 确保切片长度匹配，避免 j 接近末尾时越界
            end_idx = min(j + 1 + self.mu, self.n)
            band_len = end_idx - (j + 1)
            if band_len > 0:
                x[j] = (b[j] - np.dot(self.data[self.mu - band_len:self.mu, j], x[j + 1:end_idx])) / diag
            else:
                x[j] = b[j] / diag
        return x

    def determinant(self) -> float:
        """行列式 = 对角线元素乘积。"""
        return np.prod(self.data[self.mu, :])

    def mv(self, x: np.ndarray) -> np.ndarray:
        """矩阵-向量乘法 U @ x。"""
        x = np.asarray(x)
        y = np.zeros(self.n)
        for j in range(self.n):
            start = max(0, j - self.mu)
            for i in range(start, j + 1):
                y[i] += self.data[self.mu + i - j, j] * x[j]
        return y


class LinearSolverAnalysis:
    """
    线性系统求解与残差分析（类比 linpack_bench_backslash）。
    """

    @staticmethod
    def solve_and_analyze(K: csr_matrix, F: np.ndarray) -> dict:
        """
        求解 K u = F 并返回详细分析结果。

        Returns
        -------
        result : dict
            包含解向量、残差、条件数估计、相对误差等。
        """
        u = spsolve(K, F)

        # 残差
        residual = F - K @ u
        residual_norm = np.linalg.norm(residual)
        rhs_norm = np.linalg.norm(F)
        sol_norm = np.linalg.norm(u)
        matrix_norm = spnorm(K, ord=2)

        # 机器精度（双精度）
        eps_machine = np.finfo(float).eps
        relative_residual = residual_norm / (matrix_norm * sol_norm + 1e-30)
        normalized_residual = relative_residual / eps_machine

        # 简单条件数估计（通过逆矩阵范数）
        # 对稀疏矩阵，用幂法估计最大奇异值，用随机向量估计最小奇异值
        cond_est = LinearSolverAnalysis._estimate_condition_number(K)

        return {
            "solution": u,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "normalized_residual": normalized_residual,
            "condition_number_estimate": cond_est,
            "rhs_norm": rhs_norm,
            "solution_norm": sol_norm,
        }

    @staticmethod
    def _estimate_condition_number(K: csr_matrix, num_iter: int = 5) -> float:
        """
        用幂法估计 2-范数条件数 cond_2(K) ≈ σ_max / σ_min。
        对对称正定矩阵，σ_max ≈ λ_max，σ_min ≈ λ_min。
        """
        n = K.shape[0]
        # 估计最大特征值（幂法）
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        for _ in range(num_iter):
            x = K @ x
            x = x / np.linalg.norm(x)
        sigma_max = np.linalg.norm(K @ x)

        # 估计最小特征值（逆幂法，使用 spsolve）
        try:
            y = np.random.randn(n)
            y = y / np.linalg.norm(y)
            for _ in range(num_iter):
                y = spsolve(K, y)
                y = y / np.linalg.norm(y)
            sigma_min = 1.0 / np.linalg.norm(spsolve(K, y))
        except Exception:
            sigma_min = 1e-16

        return sigma_max / (sigma_min + 1e-30)


if __name__ == "__main__":
    # 自测试
    nodes = np.linspace(0.0, 1.0, 21)
    assembler = StiffnessMatrixAssembler1D(nodes, A=1e-4, E0=100e9)

    # 无损刚度矩阵
    K0 = assembler.assemble_global_stiffness()
    F = np.zeros(assembler.n_nodes)
    F[-1] = 1000.0  # 右端拉力

    K_bc, F_bc = assembler.apply_boundary_conditions(K0, F, fixed_dofs=np.array([0]))
    result = LinearSolverAnalysis.solve_and_analyze(K_bc, F_bc)
    print("Displacement at free end:", result["solution"][-1])
    print("Normalized residual:", result["normalized_residual"])
    print("Condition number estimate:", result["condition_number_estimate"])

    # 有损刚度矩阵
    damage = np.linspace(0.0, 0.5, assembler.n_elements)
    K_damaged = assembler.assemble_global_stiffness(damage=damage)
    Kd_bc, Fd_bc = assembler.apply_boundary_conditions(K_damaged, F, fixed_dofs=np.array([0]))
    result_d = LinearSolverAnalysis.solve_and_analyze(Kd_bc, Fd_bc)
    print("Damaged displacement at free end:", result_d["solution"][-1])

    # 带状求解器测试
    U = np.triu(np.random.rand(10, 10)) + np.eye(10)
    band_solver = BandedUpperTriangularSolver(n=10, mu=9)
    band_solver.from_dense(U)
    b = np.random.rand(10)
    x_band = band_solver.solve(b)
    x_dense = np.linalg.solve(U, b)
    print("Banded solver error:", np.max(np.abs(x_band - x_dense)))
