#!/usr/bin/env python3
"""
linear_solver.py
================
线性代数求解器模块，融合种子项目:
  - [972] r8but: 带状上三角矩阵运算 (mv, det, solve)
  - [982] r8ge_np: 一般稠密矩阵非主元 LU 分解
  - [405] fem2d_heat_sparse: 稀疏矩阵组装与求解

数学基础:
  带状上三角矩阵 (R8BUT):
    对角线在第 mu+1 行, 第 j 列
    上对角线在第 mu 行, 第 j+1 列, ...
    det(A) = Π a_{mu+1,j}
    求解: 回代法 (back substitution)

  非主元 LU 分解 (R8GE_NP):
    A = L U, 无行交换
    适用于严格对角占优或对称正定矩阵
    若遇到零主元, 分解失败

  稀疏 FEM 组装:
    K_ij = Σ_e ∫_e κ ∇φ_i · ∇φ_j dΩ
    M_ij = Σ_e ∫_e ρ φ_i φ_j dΩ
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class BandedUpperTriangular:
    """
    带状上三角矩阵 (R8BUT 格式, 融合 [972])。

    存储格式:
      A 为 (mu+1) × N 数组
      对角线: A[mu, j] (0-indexed)
      第 k 上对角线: A[mu-k, j+k]
      其余位置为零

    示例 (N=5, mu=2):
      A11 A12 A13   0   0
        0 A22 A23 A24   0
        0   0 A33 A34 A35
        0   0   0 A44 A45
        0   0   0   0 A55
    """

    def __init__(self, N, mu):
        """
        参数:
            N: 矩阵阶数
            mu: 上带宽
        """
        self.N = N
        self.mu = mu
        self.data = np.zeros((mu + 1, N), dtype=np.float64)

    def set_diagonal(self, diag):
        """设置对角线元素"""
        self.data[self.mu, :] = diag

    def set_superdiagonal(self, k, vals):
        """设置第 k 条上对角线 (k=1,2,...,mu)"""
        if k < 1 or k > self.mu:
            raise ValueError(f"上对角线索引超出范围: k={k}, mu={self.mu}")
        self.data[self.mu - k, k:self.N] = vals[:self.N - k]

    def matvec(self, x):
        """
        矩阵-向量乘积 b = A x。
        对应 [972] r8but_mv。
        """
        b = np.zeros(self.N)
        for i in range(self.N):
            for j in range(i, min(self.N, i + self.mu + 1)):
                b[i] += self.data[i - j + self.mu, j] * x[j]
        return b

    def determinant(self):
        """
        行列式 = 对角线元素之积。
        对应 [972] r8but_det。
        """
        det = 1.0
        for j in range(self.N):
            det *= self.data[self.mu, j]
        return det

    def solve(self, b):
        """
        求解上三角系统 A x = b (回代)。
        对应 [972] r8but_sl。

        x_j = (b_j - Σ_{k=j+1}^{min(N,j+mu)} a_{j,k} x_k) / a_{j,j}
        """
        x = b.copy().astype(np.float64)
        for j in range(self.N - 1, -1, -1):
            x[j] /= self.data[self.mu, j]
            jlo = max(0, j - self.mu)
            for i in range(jlo, j):
                x[i] -= self.data[i - j + self.mu, j] * x[j]
        return x

    def log_determinant(self):
        """对数行列式 (防止溢出)"""
        log_det = 0.0
        for j in range(self.N):
            diag_val = self.data[self.mu, j]
            if abs(diag_val) < 1e-300:
                return -np.inf
            log_det += np.log(abs(diag_val))
        return log_det


class DenseNonPivotingLU:
    """
    非主元高斯消去 LU 分解 (R8GE_NP, 融合 [982])。

    A = L U
    其中 L 为单位下三角, U 为上三角。
    存储在同一个 N×N 数组中 (LINPACK 风格)。

    注意: 无主元选取, 仅适用于非奇异且无零主元的矩阵。
    若遇到零主元, 分解失败 (info ≠ 0)。
    """

    def __init__(self, N):
        self.N = N
        self.LU = None
        self.info = 0

    def factor(self, A):
        """
        LU 分解 (非主元)。
        对应 [982] r8ge_np_fa。

        参数:
            A: (N, N) 矩阵
        返回:
            info: 0=成功, k=第k步出现零主元
        """
        A = np.asarray(A, dtype=np.float64)
        self.LU = A.copy()
        self.info = 0

        for k in range(self.N - 1):
            if abs(self.LU[k, k]) < 1e-300:
                self.info = k + 1
                return self.info

            self.LU[k + 1:, k] = -self.LU[k + 1:, k] / self.LU[k, k]
            for j in range(k + 1, self.N):
                self.LU[k + 1:, j] += self.LU[k + 1:, k] * self.LU[k, j]

        if abs(self.LU[self.N - 1, self.N - 1]) < 1e-300:
            self.info = self.N
        return self.info

    def solve(self, b, transposed=False):
        """
        求解 LU 分解后的系统。
        对应 [982] r8ge_np_sl。

        参数:
            b: 右端向量
            transposed: 若 True, 求解 A^T x = b
        """
        if self.LU is None or self.info != 0:
            raise RuntimeError("矩阵未成功分解")

        x = b.copy().astype(np.float64)

        if not transposed:
            # 前代: L y = b
            for k in range(self.N - 1):
                x[k + 1:] += self.LU[k + 1:, k] * x[k]
            # 回代: U x = y
            for k in range(self.N - 1, -1, -1):
                x[k] /= self.LU[k, k]
                x[:k] -= self.LU[:k, k] * x[k]
        else:
            # A^T x = b
            # U^T L^T x = b
            for k in range(self.N):
                x[k] = (x[k] - np.dot(x[:k], self.LU[:k, k])) / self.LU[k, k]
            for k in range(self.N - 2, -1, -1):
                x[k] += np.dot(self.LU[k + 1:, k], x[k + 1:])

        return x

    def determinant(self):
        """行列式 = 对角线元素之积"""
        if self.LU is None:
            raise RuntimeError("矩阵未分解")
        det = 1.0
        for i in range(self.N):
            det *= self.LU[i, i]
        return det

    def log_determinant(self):
        """对数行列式"""
        if self.LU is None:
            raise RuntimeError("矩阵未分解")
        log_det = 0.0
        sign = 1
        for i in range(self.N):
            diag = self.LU[i, i]
            if abs(diag) < 1e-300:
                return -np.inf, 0
            log_det += np.log(abs(diag))
            if diag < 0:
                sign *= -1
        return log_det, sign


class SparseFEMAssembler:
    """
    稀疏有限元组装器 (融合 [405] fem2d_heat_sparse)。
    用于组装 ICF 内爆模拟中的热传导/粘性矩阵。

    热方程弱形式:
      ∫ ρ c_v (∂T/∂t) v dΩ + ∫ κ ∇T · ∇v dΩ = ∫ Q v dΩ
    半离散:
      M dT/dt + K T = F
    其中:
      M_ij = ∫ ρ c_v φ_i φ_j dΩ  (质量矩阵)
      K_ij = ∫ κ ∇φ_i · ∇φ_j dΩ  (刚度矩阵)
      F_i = ∫ Q φ_i dΩ           (载荷向量)
    """

    @staticmethod
    def assemble_1d_diffusion(N, dx, kappa):
        """
        组装一维扩散方程的刚度矩阵 (稀疏)。
        K_ij = ∫ κ φ_i' φ_j' dx
        对线性元:
          K_{i,i} = 2κ/h, K_{i,i+1} = K_{i,i-1} = -κ/h
        """
        if callable(kappa):
            kappa_vals = np.array([kappa(x) for x in np.linspace(0, (N - 1) * dx, N)])
        else:
            kappa_vals = np.full(N, kappa)

        rows, cols, vals = [], [], []
        for i in range(1, N - 1):
            k_avg = 0.5 * (kappa_vals[i] + kappa_vals[i - 1])
            rows.append(i); cols.append(i - 1); vals.append(-k_avg / dx)
            rows.append(i); cols.append(i); vals.append((kappa_vals[i - 1] + kappa_vals[i]) / dx)
            k_avg = 0.5 * (kappa_vals[i] + kappa_vals[i + 1])
            rows.append(i); cols.append(i + 1); vals.append(-k_avg / dx)

        K = csr_matrix((vals, (rows, cols)), shape=(N, N))
        return K

    @staticmethod
    def assemble_1d_mass(N, dx, rho_cv=1.0):
        """
        组装一维质量矩阵 (集中质量或一致质量)。
        一致质量: M_{i,i} = 2h/3, M_{i,i±1} = h/6
        集中质量: M_{i,i} = h (对角)
        """
        rows, cols, vals = [], [], []
        for i in range(N):
            rows.append(i); cols.append(i); vals.append(rho_cv * dx)
        # 使用集中质量 (对角)
        M = csr_matrix((vals, (rows, cols)), shape=(N, N))
        return M

    @staticmethod
    def assemble_2d_laplacian_tri(nodes, elements, kappa=1.0):
        """
        组装二维三角形网格上的 Laplace 矩阵。
        K_ij = Σ_e κ ∫_e ∇φ_i · ∇φ_j dA

        对线性三角形:
          ∇φ_i = (1/(2A_e)) [y_j - y_k, x_k - x_j]^T
          K^e_ij = κ/(4A_e) [(y_j-y_k)(y_l-y_m) + (x_k-x_j)(x_m-x_l)]
        """
        N_nodes = nodes.shape[0]
        N_elem = elements.shape[0]
        K = lil_matrix((N_nodes, N_nodes), dtype=np.float64)

        for e in range(N_elem):
            idx = elements[e]
            xy = nodes[idx]  # (3, 2)
            # 三角形面积
            A = 0.5 * abs((xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1]) -
                          (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1]))
            if A < 1e-30:
                continue

            # 梯度: ∇φ_i
            grad_phi = np.zeros((3, 2))
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                grad_phi[i, 0] = (xy[j, 1] - xy[k, 1]) / (2 * A)
                grad_phi[i, 1] = (xy[k, 0] - xy[j, 0]) / (2 * A)

            # 单元刚度矩阵
            for i in range(3):
                for j in range(3):
                    kij = kappa * A * np.dot(grad_phi[i], grad_phi[j])
                    K[idx[i], idx[j]] += kij

        return K.tocsr()

    @staticmethod
    def apply_dirichlet_bc(K, F, bc_nodes, bc_values):
        """
        施加 Dirichlet 边界条件。
        大数法: K_{ii} = BIG, F_i = BIG * bc_value_i
        """
        K = K.tolil()
        BIG = 1.0e30
        for i, node in enumerate(bc_nodes):
            K[node, :] = 0
            K[node, node] = BIG
            F[node] = BIG * bc_values[i]
        return K.tocsr(), F


def solve_linear_system(A, b, method='sparse'):
    """
    统一线性系统求解接口。

    参数:
        A: 系数矩阵
        b: 右端向量
        method: 'sparse', 'dense', 'banded'
    返回:
        x: 解向量
    """
    if method == 'sparse':
        if not isinstance(A, csr_matrix):
            A = csr_matrix(A)
        return spsolve(A, b)
    elif method == 'dense':
        lu = DenseNonPivotingLU(A.shape[0])
        info = lu.factor(A)
        if info != 0:
            raise RuntimeError(f"LU 分解失败, info={info}")
        return lu.solve(b)
    else:
        raise ValueError(f"未知方法: {method}")
