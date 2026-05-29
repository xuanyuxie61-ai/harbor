"""
stiffness_assembly.py
层合板有限元刚度矩阵组装、稀疏矩阵处理与带状求解器。
原项目映射：
  - 508_hb_to_mm 的稀疏矩阵结构与存储格式
  - 972_r8but 的带状上三角矩阵求解算法
  - 688_linpack_bench_backslash 的稠密矩阵求解与残差分析
  - 1385_vandermonde_interp_2d 的矩阵构建思想用于形函数
科学背景：
  经典层合板理论（CLT）中，A-B-D 刚度矩阵为：
    A_ij = Σ_{k=1}^N (Q̄_ij)_k * (z_k - z_{k-1})
    B_ij = 1/2 Σ_{k=1}^N (Q̄_ij)_k * (z_k^2 - z_{k-1}^2)
    D_ij = 1/3 Σ_{k=1}^N (Q̄_ij)_k * (z_k^3 - z_{k-1}^3)
  对于对称层合板，B_ij = 0。
  有限元离散后，全局刚度矩阵 K 由各单元刚度矩阵 k_e 组装：
    K = Σ_e C_e^T k_e C_e
  其中 C_e 为单元连接矩阵。
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from utils import validate_positive, compute_condition_number, compute_normalized_residual


class LaminateStiffness:
    """层合板 A-B-D 刚度矩阵计算。"""

    def __init__(self, plies, thicknesses, material):
        """
        plies: 铺层角度列表 (度)
        thicknesses: 各层厚度列表 (mm)
        material: CompositeMaterial 对象
        """
        if len(plies) != len(thicknesses):
            raise ValueError("plies and thicknesses must have same length.")
        self.n_plies = len(plies)
        self.plies = [float(p) for p in plies]
        self.thicknesses = [float(t) for t in thicknesses]
        self.material = material
        self._compute_abd()

    def _compute_abd(self):
        """计算 A, B, D 矩阵。"""
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))

        # 计算中面坐标
        z = [-sum(self.thicknesses) / 2.0]
        for t in self.thicknesses:
            z.append(z[-1] + t)
        self.z_coords = np.array(z)

        for k in range(self.n_plies):
            theta = self.plies[k]
            Q_bar = self.material.compute_transformed_stiffness(theta)
            h_k = self.thicknesses[k]
            z_k = z[k + 1]
            z_k1 = z[k]

            self.A += Q_bar * (z_k - z_k1)
            self.B += 0.5 * Q_bar * (z_k ** 2 - z_k1 ** 2)
            self.D += (1.0 / 3.0) * Q_bar * (z_k ** 3 - z_k1 ** 3)

        # 组装6x6 ABD矩阵
        self.ABD = np.block([
            [self.A, self.B],
            [self.B, self.D]
        ])

    def compute_degraded_abd(self, damage_by_ply):
        """
        根据各层损伤状态计算退化后的ABD矩阵。
        damage_by_ply: list of DamageState，长度等于层数
        """
        A_deg = np.zeros((3, 3))
        B_deg = np.zeros((3, 3))
        D_deg = np.zeros((3, 3))
        z = self.z_coords

        # === HOLE 2 ===
        # TODO: Compute degraded ABD matrices given per-ply damage states.
        # For each ply k:
        #   1. Extract damage state d from damage_by_ply[k]
        #   2. Compute the degraded stiffness matrix consistent with the
        #      continuum damage mechanics model used in material_model.py
        #   3. Apply coordinate transformation for the ply angle theta
        #   4. Accumulate contributions to A_deg, B_deg, D_deg using CLT formulas:
        #        A += Q_bar * (z_k - z_{k-1})
        #        B += 0.5 * Q_bar * (z_k^2 - z_{k-1}^2)
        #        D += (1/3) * Q_bar * (z_k^3 - z_{k-1}^3)
        # The implementation must be consistent with how material_model.py
        # defines stiffness degradation.
        raise NotImplementedError("Hole 2: compute_degraded_abd core loop needs implementation.")

        return A_deg, B_deg, D_deg

    def get_total_thickness(self):
        return sum(self.thicknesses)


class SparseStiffnessAssembler:
    """稀疏全局刚度矩阵组装器（从 hb_to_mm 的稀疏结构思想迁移）。"""

    def __init__(self, n_nodes, ndof_per_node=2):
        self.n_nodes = n_nodes
        self.ndof = ndof_per_node
        self.n_dof_total = n_nodes * ndof_per_node
        self.K_data = []
        self.K_row = []
        self.K_col = []

    def add_element_stiffness(self, element_nodes, k_e):
        """
        将单元刚度矩阵 k_e 组装到全局稀疏矩阵。
        element_nodes: 单元节点编号列表
        k_e: 单元刚度矩阵 (nen*ndof, nen*ndof)
        """
        nen = len(element_nodes)
        dof_map = []
        for node in element_nodes:
            for d in range(self.ndof):
                dof_map.append(node * self.ndof + d)

        for i_local, i_global in enumerate(dof_map):
            for j_local, j_global in enumerate(dof_map):
                val = k_e[i_local, j_local]
                if abs(val) > 1e-16:
                    self.K_data.append(val)
                    self.K_row.append(i_global)
                    self.K_col.append(j_global)

    def get_csr_matrix(self):
        """返回组装后的 CSR 格式稀疏刚度矩阵。"""
        K = csr_matrix((self.K_data, (self.K_row, self.K_col)),
                       shape=(self.n_dof_total, self.n_dof_total))
        return K

    def get_csc_matrix(self):
        """返回组装后的 CSC 格式稀疏刚度矩阵。"""
        K = csc_matrix((self.K_data, (self.K_row, self.K_col)),
                       shape=(self.n_dof_total, self.n_dof_total))
        return K


class BandedUpperTriangularSolver:
    """
    带状上三角矩阵求解器（从 972_r8but 的 r8but_sl 与 r8but_mv 迁移）。
    用于层合板厚度方向的分层平衡求解。
    """

    def __init__(self, n, mu):
        """
        n: 矩阵阶数
        mu: 上带宽
        """
        validate_positive(n, "n")
        validate_positive(mu, "mu")
        self.n = int(n)
        self.mu = int(mu)

    def solve(self, A_band, b):
        """
        求解 A x = b，其中 A 以带状上三角格式存储。
        A_band shape: (mu+1, n)
          对角线在第 mu 行
          第 k 上对角线在第 mu-k 行，列从 k+1 到 n
        """
        if A_band.shape != (self.mu + 1, self.n):
            raise ValueError(f"A_band shape must be ({self.mu+1}, {self.n}), got {A_band.shape}")
        b = np.asarray(b, dtype=float)
        if len(b) != self.n:
            raise ValueError("b length mismatch.")

        x = b.copy()
        for j in range(self.n - 1, -1, -1):
            diag_row = j - j + self.mu
            diag_val = A_band[diag_row, j]
            if abs(diag_val) < 1e-15:
                diag_val = 1e-12  # 正则化
            x[j] = x[j] / diag_val
            jlo = max(0, j - self.mu)
            for i in range(jlo, j):
                row_idx = i - j + self.mu
                x[i] -= A_band[row_idx, j] * x[j]
        return x

    def multiply(self, A_band, x_vec):
        """计算 b = A * x。"""
        x_vec = np.asarray(x_vec, dtype=float)
        b = np.zeros(self.n)
        for i in range(self.n):
            for j in range(i, min(self.n, i + self.mu + 1)):
                row_idx = i - j + self.mu
                b[i] += A_band[row_idx, j] * x_vec[j]
        return b


def solve_equilibrium_dense(K, F):
    """
    稠密线性系统求解与残差分析（从 linpack_bench_backslash 迁移）。
    K: 稠密刚度矩阵
    F: 载荷向量
    返回: displacement, residual_norm, normalized_residual, condition_number
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    if len(F) != K.shape[0]:
        raise ValueError("F length must match K dimension.")

    cond_K = compute_condition_number(K)

    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        # 若奇异，使用最小二乘
        U, _, _, _ = np.linalg.lstsq(K, F, rcond=None)

    r_norm = np.linalg.norm(F - K @ U, ord=np.inf)
    norm_res = compute_normalized_residual(K, U, F)

    return U, r_norm, norm_res, cond_K


def solve_equilibrium_sparse(K_csr, F):
    """稀疏线性系统求解。"""
    from scipy.sparse.linalg import spsolve
    F = np.asarray(F, dtype=float)
    U = spsolve(K_csr, F)
    if U is None:
        raise ValueError("Sparse solver failed.")
    U = np.asarray(U)
    r_norm = np.linalg.norm(F - K_csr @ U, ord=np.inf)
    return U, r_norm
