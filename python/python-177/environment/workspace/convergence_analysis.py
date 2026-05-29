# -*- coding: utf-8 -*-
"""
convergence_analysis.py
=======================
收敛性分析、误差估计与降阶模型模块。

融合原始项目:
  - 1190_svd_powers: SVD 降维、奇异值分析、主成分提取

核心数学公式
------------
1. 数值误差度量:
   L2 误差:   e_2 = ||φ_num - φ_exact||_2 = √(∫ |φ_num - φ_exact|² dx)^{1/2}
   L∞ 误差:   e_∞ = max |φ_num - φ_exact|
   H1 半范数: e_H1 = √(∫ |∇(φ_num - φ_exact)|² dx)^{1/2}

2. 收敛阶数估计:
   若误差满足 e_h = C h^p，则对两组网格 h1, h2:
   p = log(e_{h1} / e_{h2}) / log(h1 / h2)

3. SVD 降阶模型（POD）:
   对快照矩阵 A = [φ_1, φ_2, ..., φ_m] ∈ R^{n×m}，
   做经济 SVD: A = U Σ V^T
   取前 r 个左奇异向量构成 POD 基:
   φ(t) ≈ Σ_{i=1}^r α_i(t) u_i
   其中 u_i 为 U 的第 i 列。

4. 能量截断准则:
   保留的能量比例 = Σ_{i=1}^r σ_i² / Σ_{i=1}^{min(n,m)} σ_i²
   通常取 r 使得能量比例 > 99%。

5. 快照矩阵的预处理:
   去均值: ā = mean(A, axis=1)
   A' = A - ā 1^T
   再做 SVD，提高低阶模态的物理意义。
"""

import numpy as np


class ConvergenceAnalysis:
    """
    收敛性分析与误差估计工具。
    """

    @staticmethod
    def l2_error(phi_num, phi_exact, dx, dy):
        """
        L2 范数误差。
        """
        diff = phi_num - phi_exact
        return np.sqrt(np.sum(diff ** 2) * dx * dy)

    @staticmethod
    def linf_error(phi_num, phi_exact):
        """
        L∞ 范数误差。
        """
        return np.max(np.abs(phi_num - phi_exact))

    @staticmethod
    def h1_seminorm_error(phi_num, phi_exact, dx, dy):
        """
        H1 半范数误差: ||∇(φ_num - φ_exact)||_2
        """
        nx, ny = phi_num.shape
        diff = phi_num - phi_exact
        grad_sq = np.zeros_like(diff)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dx_diff = (diff[i + 1, j] - diff[i - 1, j]) / (2.0 * dx)
                dy_diff = (diff[i, j + 1] - diff[i, j - 1]) / (2.0 * dy)
                grad_sq[i, j] = dx_diff ** 2 + dy_diff ** 2

        return np.sqrt(np.sum(grad_sq) * dx * dy)

    @staticmethod
    def convergence_order(errors, resolutions):
        """
        计算收敛阶数。
        参数:
            errors      : list of float, 各分辨率下的误差
            resolutions : list of float, 对应的网格尺寸 h
        返回:
            orders      : list of float, 相邻两组间的收敛阶
        """
        orders = []
        for i in range(len(errors) - 1):
            e1, e2 = errors[i], errors[i + 1]
            h1, h2 = resolutions[i], resolutions[i + 1]
            if e1 <= 0 or e2 <= 0 or h1 <= h2:
                orders.append(np.nan)
            else:
                p = np.log(e1 / e2) / np.log(h1 / h2)
                orders.append(p)
        return orders


class ReducedOrderModel:
    """
    基于 POD-SVD 的降阶模型（融入 1190_svd_powers 思想）。
    """

    def __init__(self, snapshots):
        """
        参数:
            snapshots : ndarray, shape (nx*ny, num_snapshots)
        """
        self.snapshots = np.asarray(snapshots, dtype=np.float64)
        self.mean_vec = None
        self.U = None
        self.S = None
        self.Vt = None
        self.coefficients = None

    def compute_pod_basis(self, energy_threshold=0.99):
        """
        计算 POD 基。
        步骤:
        1. 去均值
        2. SVD 分解
        3. 按能量阈值截断
        """
        A = self.snapshots.copy()
        self.mean_vec = np.mean(A, axis=1)
        A_centered = A - self.mean_vec[:, np.newaxis]

        # 经济 SVD
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        self.U = U
        self.S = S
        self.Vt = Vt

        # 能量截断
        total_energy = np.sum(S ** 2)
        cumsum = np.cumsum(S ** 2)
        r = np.searchsorted(cumsum / total_energy, energy_threshold) + 1
        r = max(1, min(r, len(S)))
        self.r = r
        self.Ur = U[:, :r]

        # 计算模态系数
        self.coefficients = self.Ur.T @ A_centered
        return self.Ur, self.S[:r]

    def reconstruct(self, mode_indices=None):
        """
        用 POD 基重构快照。
        """
        if self.Ur is None:
            raise ValueError("Must call compute_pod_basis first")
        if mode_indices is None:
            Ur_use = self.Ur
            coeffs_use = self.coefficients
        else:
            Ur_use = self.Ur[:, mode_indices]
            coeffs_use = self.coefficients[mode_indices, :]

        A_recon = Ur_use @ coeffs_use + self.mean_vec[:, np.newaxis]
        return A_recon

    def project_state(self, phi):
        """
        将状态投影到 POD 子空间。
        α = U_r^T (φ - φ_mean)
        """
        phi_centered = phi - self.mean_vec
        return self.Ur.T @ phi_centered

    def reconstruct_from_coefficients(self, alpha):
        """
        从系数重构状态。
        φ ≈ U_r α + φ_mean
        """
        return self.Ur @ alpha + self.mean_vec

    def get_mode_energy(self):
        """
        返回各模态的能量占比。
        """
        if self.S is None:
            return None
        total = np.sum(self.S ** 2)
        return self.S ** 2 / total

    def reduced_galerkin_matrix(self, L_full):
        """
        将全阶算子 L 投影到降阶子空间:
        L_r = U_r^T L_full U_r

        参数:
            L_full : ndarray, (N, N)
        返回:
            L_reduced : ndarray, (r, r)
        """
        if self.Ur is None:
            raise ValueError("Must call compute_pod_basis first")
        L_reduced = self.Ur.T @ L_full @ self.Ur
        return L_reduced
