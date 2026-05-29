"""
rom_analysis.py
===============
基于奇异值分解 (SVD) 的降阶模型 (ROM) 分析

融合原始项目：
  - 1184_svd_basis：从 PDE 解数据中提取主导模态 (POD / SVD)

数学物理模型：
  1. 给定 M 个空间离散点上的 N 个快照（snapshots），构成数据矩阵：
       A ∈ ℝ^{M×N},   A_{ij} = u_i(t_j)

  2. 奇异值分解：
       A = U Σ V^T
     其中 U ∈ ℝ^{M×M} 为正交矩阵（空间模态），
           Σ ∈ ℝ^{M×N} 为对角矩阵（奇异值 σ₁ ≥ σ₂ ≥ ... ≥ 0），
           V ∈ ℝ^{N×N} 为正交矩阵（时间系数）。

  3. 截断能量准则：
       取前 L 个模态使得：
         Σ_{k=1}^L σ_k² / Σ_{k=1}^{min(M,N)} σ_k² ≥ η
       通常 η = 0.99（保留 99% 能量）。

  4. 降阶重构：
       Ã = U_L Σ_L V_L^T = Σ_{k=1}^L σ_k u_k v_k^T
     重构误差（Frobenius 范数）：
       ||A - Ã||_F = √( Σ_{k=L+1}^{min(M,N)} σ_k² )

  5. 在细胞迁移模型中的应用：
       - 对 chemoattractant 浓度场进行 POD，提取主导空间模态
       - 用少量模态系数快速重构完整浓度场，加速参数扫描
"""

import numpy as np


def compute_pod_basis(data_matrix: np.ndarray, energy_threshold: float = 0.99):
    """
    对数据矩阵执行 POD / SVD，提取主导模态。

    参数
    ----
    data_matrix : np.ndarray, shape (M, N)
        每一列为一个快照
    energy_threshold : float
        能量保留阈值 (0,1]

    返回
    ----
    U : np.ndarray, shape (M, L)
        前 L 个空间模态（左奇异向量）
    sigma : np.ndarray, shape (L,)
        对应奇异值
    Vt : np.ndarray, shape (L, N)
        时间系数矩阵（右奇异向量的转置）
    L : int
        保留的模态数
    energy : np.ndarray
        累计能量比例
    """
    A = np.asarray(data_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("compute_pod_basis: data_matrix 必须为二维数组")
    M, N = A.shape

    # 去均值（可选但推荐）
    mean_vec = A.mean(axis=1, keepdims=True)
    A_centered = A - mean_vec

    # 执行经济型 SVD
    try:
        U_full, sigma_full, Vt_full = np.linalg.svd(A_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # 如果 SVD 失败，尝试对小的矩阵使用更稳定的方法
        U_full, sigma_full, Vt_full = np.linalg.svd(A_centered + 1e-12 * np.random.randn(M, N), full_matrices=False)

    # 计算累计能量
    sigma_sq = sigma_full ** 2
    total_energy = np.sum(sigma_sq)
    if total_energy < 1e-30:
        # 数据接近零矩阵，返回空基
        return np.zeros((M, 1)), np.array([0.0]), np.zeros((1, N)), 1, np.array([1.0])

    cum_energy = np.cumsum(sigma_sq) / total_energy
    L = int(np.searchsorted(cum_energy, energy_threshold)) + 1
    L = min(L, min(M, N))
    L = max(1, L)

    return U_full[:, :L], sigma_full[:L], Vt_full[:L, :], L, cum_energy[:L]


def reconstruct_from_pod(U, sigma, Vt):
    """
    由 POD 模态重构数据矩阵。

    公式：
        A_recon = U · diag(σ) · Vt
    """
    if sigma.size == 0:
        return np.zeros((U.shape[0], Vt.shape[1]))
    return U @ np.diag(sigma) @ Vt


def pod_galerkin_projection(rhs_func, U, initial_coeff, dt, n_steps):
    """
    在 POD 模态张成的子空间中进行 Galerkin 投影时间推进。

    假设状态变量 u ≈ U · a，其中 a(t) ∈ ℝ^L 为模态系数。
    原系统 du/dt = f(u) 投影后：
        da/dt = U^T f(U a)

    这里使用向前 Euler：
        a_{n+1} = a_n + dt · U^T f(U a_n)

    参数
    ----
    rhs_func : callable
        f(u) -> du/dt，完整空间维度的右端项
    U : np.ndarray, shape (M, L)
        POD 模态矩阵
    initial_coeff : np.ndarray, shape (L,)
        初始模态系数
    dt : float
    n_steps : int

    返回
    ----
    coeff_history : np.ndarray, shape (L, n_steps+1)
    """
    L = U.shape[1]
    a = np.asarray(initial_coeff, dtype=float).copy()
    history = np.zeros((L, n_steps + 1))
    history[:, 0] = a
    for n in range(n_steps):
        u_full = U @ a
        f_full = rhs_func(u_full)
        a = a + dt * (U.T @ f_full)
        history[:, n + 1] = a
    return history


class ChemotaxisROM:
    """
    趋化因子浓度场的降阶模型。

    将三维浓度场拉平为一维向量后执行 POD，
    提供快速重构与 Galerkin 投影接口。
    """

    def __init__(self, snapshot_list=None):
        """
        参数
        ----
        snapshot_list : list of np.ndarray
            每个元素为一个三维浓度场数组 (nx, ny, nz)
        """
        self.U = None
        self.sigma = None
        self.Vt = None
        self.L = 0
        self.mean_vec = None
        self.original_shape = None
        self.energy = None
        if snapshot_list is not None and len(snapshot_list) > 0:
            self.build_basis(snapshot_list)

    def build_basis(self, snapshot_list, energy_threshold=0.99):
        """
        从快照列表构建 POD 基。
        """
        self.original_shape = snapshot_list[0].shape
        M = int(np.prod(self.original_shape))
        N = len(snapshot_list)
        A = np.zeros((M, N), dtype=float)
        for j, snap in enumerate(snapshot_list):
            A[:, j] = snap.flatten()
        self.U, self.sigma, self.Vt, self.L, self.energy = compute_pod_basis(A, energy_threshold)
        self.mean_vec = A.mean(axis=1)

    def reconstruct(self, coeff):
        """
        由模态系数重构三维浓度场。

        参数
        ----
        coeff : np.ndarray, shape (L,)

        返回
        ----
        field : np.ndarray, shape original_shape
        """
        if self.U is None:
            raise RuntimeError("ChemotaxisROM: 尚未构建 POD 基")
        coeff = np.asarray(coeff, dtype=float)
        flat = self.U @ coeff + self.mean_vec
        return flat.reshape(self.original_shape)

    def project(self, field):
        """
        将完整浓度场投影到 POD 子空间，得到模态系数。

        公式：
            a = U^T (field - mean)
        """
        if self.U is None:
            raise RuntimeError("ChemotaxisROM: 尚未构建 POD 基")
        flat = np.asarray(field, dtype=float).flatten() - self.mean_vec
        return self.U.T @ flat

    def relative_error(self, field):
        """
        计算用当前 ROM 重构该场的相对误差。
        """
        a = self.project(field)
        recon = self.reconstruct(a)
        norm_true = np.linalg.norm(field)
        if norm_true < 1e-15:
            return 0.0
        return float(np.linalg.norm(field - recon) / norm_true)

    def summary(self):
        """打印 ROM 信息摘要。"""
        if self.U is None:
            return "ChemotaxisROM: 未构建基"
        lines = []
        lines.append("=" * 60)
        lines.append("Chemotaxis ROM 摘要")
        lines.append("  原始场维度 M = %d" % self.U.shape[0])
        lines.append("  保留模态数 L = %d" % self.L)
        lines.append("  压缩比 M/L = %.2f" % (self.U.shape[0] / max(1, self.L)))
        lines.append("  主导奇异值: " + ", ".join("%.4g" % s for s in self.sigma[:min(5, self.sigma.size)]))
        lines.append("  保留能量: %.4f" % (self.energy[-1] if self.energy.size > 0 else 0.0))
        lines.append("=" * 60)
        return "\n".join(lines)
